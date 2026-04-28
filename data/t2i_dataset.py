# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import pyarrow.parquet as pq
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from .data_paths import resolve_data_path

Image.MAX_IMAGE_PIXELS = 20_000_000


def _first_str(value):
    """Return the first string element from a list-like or pass through a scalar string.

    Supports both schemas seen in the wild for T2I parquets:
        - list-like (Python list / numpy ndarray / pyarrow array) of strings -> take [0]
        - plain string -> return as-is
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    try:
        first = value[0]
    except (TypeError, IndexError, KeyError):
        return value
    if isinstance(first, bytes):
        return first.decode('utf-8', errors='ignore')
    return first


class T2IIterableDatasetRvR(DistributedIterableDataset):
    """T2I dataset reading parquet files. Tolerates two on-disk schemas:

        Schema A (list form):
            image_list: list<string>      # length 1, image path
            user_prompt: list<string>     # length 1, caption / prompt

        Schema B (scalar form):
            cache_image: string           # image path
            user_prompt: string           # caption / prompt
    """

    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            try:
                                if 'image_list' in row:
                                    image_field = row['image_list']
                                elif 'cache_image' in row:
                                    image_field = row['cache_image']
                                else:
                                    raise KeyError(
                                        "parquet row has neither 'image_list' nor 'cache_image'"
                                    )
                                image_path = resolve_data_path(_first_str(image_field))
                                image = pil_img2rgb(Image.open(image_path))
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                raise

                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens = width * height // transform_stride ** 2

                            prompt = _first_str(row['user_prompt'])
                            caption_token = self.tokenizer.encode(prompt)
                            num_tokens += len(caption_token)

                            sequence_plan = [
                                {
                                    'type': 'text',
                                    'enable_cfg': 1,
                                    'loss': 0,
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                },
                                {
                                    'type': 'vae_image',
                                    'enable_cfg': 0,
                                    'loss': 1,
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                },
                            ]

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=[caption_token],
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                },
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
