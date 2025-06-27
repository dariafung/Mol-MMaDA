import pandas as pd
import pyarrow.parquet as pq
import json
from selfies import encoder 
from rdkit import Chem 
from rdkit.Chem import AllChem
import numpy as np
import os

# --- 新增的 atom_to_id 函数 ---
def atom_to_id(symbol: str) -> int:
    """
    将原子符号映射到其原子序数 (Z)。
    如果符号无法识别，返回 0（或你定义的未知原子 ID）。
    """
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        return 0 # 对于无法识别的原子符号，返回0

# --- 移除这个旧的 ATOM_TO_ID 字典，或者确保它不会被使用 ---
# ATOM_TO_ID = {
#     'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
#     'unknown': 0
# }

def smiles_to_selfies(smiles_string):
    # ... (此函数保持不变)
    try:
        if pd.isna(smiles_string):
            return None
        return encoder(str(smiles_string)) 
    except Exception as e:
        print(f"Error converting SMILES '{smiles_string}' to SELFIES: {e}")
        return None

def get_3d_coords_and_features(mol):
    """
    从 RDKit Mol 对象中提取 3D 坐标和原子特征、键信息。
    假设 Mol 对象已包含 3D 构象。
    """
    if mol is None:
        return None, None, None, None, None, None # 没有有效的 Mol 对象

    # 尝试获取第一构象
    if mol.GetNumConformers() == 0:
        # 如果没有构象，尝试生成一个（这会比较慢，请注意性能）
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
            if mol.GetNumConformers() == 0:
                return None, None, None, None, None, None # 即使尝试生成也失败了
        except Exception as e:
            print(f"Failed to generate 3D conformer: {e}")
            return None, None, None, None, None, None

    conf = mol.GetConformer(0) # 获取第一构象
    
    atom_vec = [] # 原子类型 ID
    coords = []   # 3D 坐标 (N, 3)

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        # --- 这里修改为调用新的 atom_to_id 函数 ---
        atom_vec.append(atom_to_id(atom_symbol)) 
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    coords = np.array(coords, dtype=np.float32)
    atom_vec = np.array(atom_vec, dtype=np.int64)

    num_atoms = mol.GetNumAtoms()
    
    # 距离矩阵 (N, N)
    dist_matrix = Chem.Get3DDistanceMatrix(mol, confId=conf.GetId())
    dist_matrix = np.array(dist_matrix, dtype=np.float32)

    # 键类型和邻接矩阵
    bond_type_mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
        # 可以根据需要添加其他键类型
    }
    edge_type = np.zeros((num_atoms, num_atoms), dtype=np.int64) # 简单的邻接（是否有键）
    bond_type = np.zeros((num_atoms, num_atoms), dtype=np.int64) # 键的类型

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_type[i, j] = edge_type[j, i] = 1 
        bt = bond_type_mapping.get(bond.GetBondType(), 0) # 0 for unknown bond type
        bond_type[i, j] = bond_type[j, i] = bt

    rdmol2selfies = None # 保持 None，除非你决定实现这个复杂部分

    return atom_vec, coords, edge_type, bond_type, dist_matrix, rdmol2selfies

def process_m3_entry(row):
    # ... (此函数保持不变)
    smiles = row['smiles'] 
    text_description = row['Description'] 

    selfies = smiles_to_selfies(smiles)
    if selfies is None:
        return None 

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Could not parse SMILES '{smiles}' to RDKit Mol, skipping.")
        return None

    atom_vec, coords, edge_type, bond_type, dist_matrix, rdmol2selfies = get_3d_coords_and_features(mol)

    if coords is None or len(coords) == 0: 
        return None 
    
    return {
        'id': row.get('cid', str(row.name)), 
        'selfies_string': selfies,
        'text_description': text_description if pd.notna(text_description) else "",
        'atom_vec_str': json.dumps(atom_vec.tolist()),
        'coordinates_str': json.dumps(coords.tolist()),
        'edge_type_str': json.dumps(edge_type.tolist()),
        'bond_type_str': json.dumps(bond_type.tolist()),
        'dist_str': json.dumps(dist_matrix.tolist()),
        'rdmol2selfies_str': json.dumps(rdmol2selfies.tolist()) if rdmol2selfies is not None else json.dumps([])
    }

def convert_m3_dataset(input_csv_path, output_parquet_path, num_rows_to_process=None):
    # ... (此函数保持不变)
    print(f"Loading CSV from {input_csv_path}...")
    df_m3 = pd.read_csv(input_csv_path) 
    
    if num_rows_to_process is not None:
        df_m3 = df_m3.head(num_rows_to_process)
        print(f"Processing only the first {num_rows_to_process} rows for testing.")

    processed_records = []
    skipped_count = 0
    for index, row in df_m3.iterrows():
        record = process_m3_entry(row)
        if record:
            processed_records.append(record)
        else:
            skipped_count += 1
        
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} records, Skipped: {skipped_count}")

    final_df = pd.DataFrame(processed_records)
    print(f"Finished processing. Total processed: {len(final_df)} records, Skipped: {skipped_count}.")
    
    print(f"Saving {len(final_df)} records to {output_parquet_path}...")
    final_df.to_parquet(output_parquet_path, index=False)
    print("Conversion complete!")
    print(f"Output Parquet file saved at: {output_parquet_path}")


if __name__ == '__main__':
    # --- 请修改以下路径为你的实际文件路径 ---
    input_csv_file = "/home/exouser/data/m3-20m/M^3-Datasets/M^3_Multi.csv" 
    output_parquet_file = "/home/exouser/MMaDA/m3_molecular_data.parquet" # 建议保存到 MMaDA 目录或一个更大的数据卷

    # --- 运行转换 ---
    # 先处理少量数据进行测试，例如前 1000 行
    convert_m3_dataset(input_csv_file, output_parquet_file, num_rows_to_process=1000)
    
    # 确认测试无误后，再处理整个数据集 (注释掉 num_rows_to_process 参数)
    # convert_m3_dataset(input_csv_file, output_parquet_file)
