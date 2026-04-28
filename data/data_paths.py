# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for resolving release-relative paths.

Release datasets (parquets, jsonl, parquet_info.json) store image paths as
strings RELATIVE to a single release root (DATA_ROOT). At runtime the
strings are joined with the current root so the data is location-agnostic and
only one env var has to be updated when the data is moved between disks.

Usage
-----
    from data.data_paths import get_data_root, resolve_data_path

    data_root = get_data_root()                       # /xxx/.../training_data
    abs_path = resolve_data_path('rvr/images/xxx.jpg')
    # -> /xxx/.../training_data/rvr/images/xxx.jpg

If a stored string is already an absolute path, `resolve_data_path` returns
it untouched (preserves backward compatibility with old absolute-path data).
"""

from __future__ import annotations

import os


_DEFAULT_DATA_ROOT = (
    ''
)


def get_data_root() -> str:
    """Return the active dataset root.

    Reads DATA_ROOT from the environment; falls back to an empty string
    (i.e., relative paths resolve against the current working directory).
    """
    return os.environ.get('DATA_ROOT', _DEFAULT_DATA_ROOT)


def resolve_data_path(p: str | None) -> str | None:
    """Return an absolute path; passthrough for already-absolute or None."""
    if p is None:
        return None
    if not isinstance(p, str):
        return p
    if os.path.isabs(p):
        return p
    return os.path.join(get_data_root(), p)
