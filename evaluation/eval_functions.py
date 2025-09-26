# Description: Evaluation functions for 2D and 3D molecules
import copy
import numpy as np
import pickle
import os
try:
    import torch  # optional for CPU eval
except Exception:
    torch = None
from rdkit import Chem
from .jodo.rdkit_metric import eval_rdmol
# from evaluation.jodo.mose_metric import compute_intermediate_statistics, mapper, get_smiles, reconstruct_mol, MeanProperty
# from fcd_torch import FCD as FCDMetric
from multiprocessing import Pool
# from moses.metrics.metrics import SNNMetric, FragMetric, ScafMetric, internal_diversity, \
#     fraction_passes_filters, weight, logP, SA, QED
from .jodo.stability import bond_list, allowed_fc_bonds, stability_bonds
from rdkit.Geometry import Point3D
from .jodo.bond_analyze import get_bond_order, geom_predictor, allowed_bonds, allowed_fc_bonds
from .jodo.cal_geometry import load_target_geometry, compute_geo_mmd, cal_bond_distance, cal_bond_angle, cal_dihedral_angle
from tqdm import tqdm
from rdkit.Chem import AllChem


def _safe_sanitize(mol: Chem.Mol) -> Chem.Mol:
    """Try to sanitize the molecule; always at least update property cache."""
    if mol is None:
        return mol
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Keep going with partially-sanitized mol to avoid RDKit precondition crashes later
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
    return mol


def check_2D_stability(rdmol):
    """Convert the generated tensors to rdkit mols and check stability."""
    # Ensure properties/implicit valence are computed before AddHs()
    rdmol = Chem.Mol(rdmol)
    rdmol = _safe_sanitize(rdmol)

    rdmol = Chem.AddHs(rdmol)
    atom_num = rdmol.GetNumAtoms()

    new_mol = copy.deepcopy(rdmol)
    try:
        Chem.Kekulize(new_mol)
    except Exception:
        print("Can't Kekulize mol.")
        pass

    nr_bonds = np.zeros(atom_num, dtype='int')
    for bond in new_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        order = stability_bonds[bond_type]
        nr_bonds[start] += order
        nr_bonds[end] += order

    nr_stable_bonds = 0
    atom_types_str = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    formal_charges = [atom.GetFormalCharge() for atom in rdmol.GetAtoms()]
    for atom_type_i, nr_bonds_i, fc_i in zip(atom_types_str, nr_bonds, formal_charges):
        possible_bonds = allowed_fc_bonds[atom_type_i]
        if isinstance(possible_bonds, int):
            is_stable = (possible_bonds == nr_bonds_i)
        elif isinstance(possible_bonds, dict):
            expected_bonds = possible_bonds[fc_i] if fc_i in possible_bonds.keys(
            ) else possible_bonds[0]
            is_stable = (expected_bonds == nr_bonds_i) if isinstance(
                expected_bonds, int) else (nr_bonds_i in expected_bonds)
        else:
            is_stable = (nr_bonds_i in possible_bonds)
        nr_stable_bonds += int(is_stable)

    molecule_stable = (nr_stable_bonds == atom_num)
    return molecule_stable, nr_stable_bonds, atom_num


def get_2D_edm_metric(predict_mols, train_mols=None):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]
        train_smiles = [Chem.CanonSmiles(s) for s in train_smiles]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    for mol in tqdm(predict_mols):
        try:
            validity_res = check_2D_stability(mol)
        except Exception:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms()]
        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])

    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)

    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    rdkit_dict = eval_rdmol(predict_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict


def check_3D_stability(positions, atoms, dataset_name, debug=False, rdmol=None, use_mmff=False):
    """Reconstruct bonds from 3D coordinates with valence-safe capping and build an RDKit Mol."""
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    if use_mmff:
        try:
            AllChem.MMFFOptimizeMolecule(rdmol, confId=0, maxIters=200)
            positions = rdmol.GetConformer(0).GetPositions()
        except Exception:
            print('MMFF failed, use original coordinates.')

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # create atoms
    rwmol = Chem.RWMol()
    for sym in atoms:
        rwmol.AddAtom(Chem.Atom(sym))

    # add coordinates
    conf = Chem.Conformer(rwmol.GetNumAtoms())
    for i in range(rwmol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(positions[i][0]), float(
            positions[i][1]), float(positions[i][2])))
    rwmol.AddConformer(conf)

    def _can_accept(total_now: int, sym: str, add_order: int) -> bool:
        allowed = allowed_bonds[sym]
        target = int(total_now) + int(add_order)
        if isinstance(allowed, int):
            return target <= allowed
        try:
            mx = max(allowed) if len(allowed) > 0 else 0
        except Exception:
            mx = 0
        return (target in allowed) or (target <= mx)

    # propose bond order, downscale if needed to avoid over-valence, then add bond
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]], dtype=float)
            p2 = np.array([x[j], y[j], z[j]], dtype=float)
            dist = float(np.linalg.norm(p1 - p2))
            a1, a2 = atoms[i], atoms[j]
            pair = tuple(sorted([a1, a2]))

            if 'QM9' in dataset_name:
                order = int(get_bond_order(a1, a2, dist))
            elif 'Geom' in dataset_name:
                order = int(geom_predictor(pair, dist))
            else:
                raise ValueError('Fail to get dataset bond info.')

            o = max(0, order)
            while o > 0 and (not _can_accept(nr_bonds[i], a1, o) or not _can_accept(nr_bonds[j], a2, o)):
                o -= 1

            if o > 0:
                rwmol.AddBond(i, j, bond_list[o])
                nr_bonds[i] += o
                nr_bonds[j] += o

    # finalize to Mol and sanitize so downstream AddHs/validity checks don't crash
    mol = rwmol.GetMol()
    mol = _safe_sanitize(mol)

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atoms, nr_bonds):
        possible_bonds = allowed_bonds[atom_type_i]
        if isinstance(possible_bonds, int):
            is_stable = (nr_bonds_i == possible_bonds)
        else:
            is_stable = (nr_bonds_i in possible_bonds)
        if not is_stable and debug:
            print(
                f"Invalid bonds for atom {atom_type_i} with {nr_bonds_i} total order")
        nr_stable_bonds += int(is_stable)

    molecule_stable = (nr_stable_bonds == len(x))
    return molecule_stable, nr_stable_bonds, len(x), mol


def get_3D_edm_metric(predict_mols, train_mols=None, dataset_name='QM9', use_mmff=False):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    for mol in tqdm(predict_mols):
        pos = mol.GetConformer(0).GetPositions()
        pos = pos - pos.mean(axis=0)
        atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
        try:
            validity_res = check_3D_stability(
                pos, atom_type, dataset_name, rdmol=mol, use_mmff=use_mmff, debug=False)
        except Exception:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms(), mol]

        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])
        rd_mols.append(validity_res[3])

    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)

    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict, rd_mols


def get_3D_edm_metric_batch(predict_mols, train_mols=None, dataset_name='QM9'):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    predict_mols = [predict_mols[i:i+10]
                    for i in range(0, len(predict_mols), 10)]
    for mol_list in tqdm(predict_mols):
        validity_res_list = []
        smiles = [Chem.MolToSmiles(mol) for mol in mol_list]
        assert len(set(smiles)) == 1

        for mol in mol_list:
            pos = mol.GetConformer(0).GetPositions()
            pos = pos - pos.mean(axis=0)
            atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
            validity_res = check_3D_stability(
                pos, atom_type, dataset_name, rdmol=mol)
            validity_res_list.append(validity_res)
        max_validity_res = max(validity_res_list, key=lambda x: x[0])
        molecule_stable += int(max_validity_res[0])
        nr_stable_bonds += int(max_validity_res[1])
        n_atoms += int(max_validity_res[2])
        rd_mols.append(max_validity_res[3])

    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)

    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict


# def get_moses_metrics(test_mols, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None, cache_path=None):
#     ...
#     return moses_metrics


def get_sub_geometry_metric(test_mols, dataset_info, root_path):
    tar_geo_stat = load_target_geometry(test_mols, dataset_info, root_path)

    def sub_geometry_metric(gen_mols):
        bond_length_dict = compute_geo_mmd(
            gen_mols, tar_geo_stat, cal_bond_distance, dataset_info[
                'top_bond_sym'], mean_name='bond_length_mean'
        )
        bond_angle_dict = compute_geo_mmd(
            gen_mols, tar_geo_stat, cal_bond_angle, dataset_info[
                'top_angle_sym'], mean_name='bond_angle_mean'
        )
        dihedral_angle_dict = compute_geo_mmd(
            gen_mols, tar_geo_stat, cal_dihedral_angle, dataset_info[
                'top_dihedral_sym'], mean_name='dihedral_angle_mean'
        )
        metric = {**bond_length_dict, **bond_angle_dict, **dihedral_angle_dict}
        return metric

    return sub_geometry_metric
