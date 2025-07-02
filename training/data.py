# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import math
import os
import random
import re
import pandas as pd
from functools import partial
from typing import List, Optional, Union

from PIL import Image

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

person_token = ["a person", "someone", "somebody"]

def replace_person_token(t):
    "Used for CC12M - handles all case variations of <person> tag"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t, flags=re.IGNORECASE)
    
    person_pattern = re.compile(r"<person>", re.IGNORECASE)
    while person_pattern.search(t):
        match = person_pattern.search(t)
        t = t[:match.start()] + f" {random.choice(person_token)} " + t[match.end():]
    
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None, src=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample.keys():
            print(f"fname not in filesample.keys(): {filesample}")
            print(f"src: {src}")
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler) # [{fname,data,__url__}, ...]  __url__ 字段标识当前读取的文件来自哪个 tar 包
    samples = group_by_keys_nothrow(files, handler=handler, src=src)
    return samples



def filter_long_samples(sample):
    return sample.get('input_ids') is not None


if __name__ == '__main__':
    pass
