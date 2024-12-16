# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass extension of TIR analysis passes."""
from .extract_prim_func_info import extract_prim_func_info
from .ensure_well_formed import ensure_well_formed
from .get_mask_associated_dtype import get_mask_associated_dtype
from .has_perf_record_tick import has_perf_record_tick
