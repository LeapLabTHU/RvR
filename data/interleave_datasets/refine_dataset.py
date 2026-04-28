# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..data_paths import resolve_data_path


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


REFINE_PROMPT = '''Here is the user's prompt: {user_prompt}. '''

REFINE_SYSTEM_PROMPT = '''Analyze the potential misalignment between the generated image and the user's prompt. Then, re-generate the image to precisely align with the user's prompt.
'''


class RefineIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_list = row["image_list"]
        if isinstance(image_list, str):
            try:
                import ast
                image_list = ast.literal_eval(image_list)
            except Exception:
                raise ValueError(f"Failed to parse image_list string: {image_list}")

        image_num = len(image_list)
        if image_num != 2:
            raise ValueError("Only two images are supported for alignment")

        image_list = [resolve_data_path(p) for p in image_list]

        start_idx = 0
        end_idx = 1

        data = self._init_data()

        data = self._add_text(data, REFINE_SYSTEM_PROMPT, need_loss=False, enable_cfg=False)

        data = self._add_image(
            data,
            pil_img2rgb(Image.open(image_list[start_idx])),
            enable_cfg=False,
            need_loss=False,
            need_vae=False,
            need_vit=True,
        )

        data = self._add_text(
            data,
            REFINE_PROMPT.format(user_prompt=row["user_prompt"][0]),
            need_loss=False,
            enable_cfg=False,
        )

        data = self._add_image(
            data,
            pil_img2rgb(Image.open(image_list[end_idx])),
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )

        return data
