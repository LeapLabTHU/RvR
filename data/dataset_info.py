# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from .interleave_datasets import RefineIterableDataset
from .data_paths import get_data_root
from .t2i_dataset import T2IIterableDatasetRvR
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'rvr': RefineIterableDataset,
    'blip3o': T2IIterableDatasetRvR,
}


# Single source of truth for the dataset root. Override at runtime with the
# DATA_ROOT env var (set in scripts/train.sh).
DATA_ROOT = get_data_root()


RVR_GENERAL = {
    'rvr': {
        'data_dir': f'{DATA_ROOT}/rvr/data',
        'num_files': 10,
        'num_total_samples': 117736,
        'parquet_info_path': f'{DATA_ROOT}/rvr/json/parquet_info.json',
    },
}


DATASET_INFO = {
    'vlm_sft': {
        'llava_ov': {
            'data_dir': f'{DATA_ROOT}/vlm/images',
            'jsonl_path': f'{DATA_ROOT}/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000,
        },
    },
    'rvr': deepcopy(RVR_GENERAL),
    'blip3o': {
        'blip3o60k': {
            'data_dir': f'{DATA_ROOT}/blip3o60k/data',
            'num_files': 10,
            'num_total_samples': 58859,
        },
    },
}
