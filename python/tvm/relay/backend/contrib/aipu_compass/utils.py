# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Common AIPU Compass utilities."""
import os
from tvm import runtime
from .config import AipuCompassConfig
from . import _ffi_api


X86_DESIRED_LAYOUTS = {
    "nn.conv2d": ["NCHW", "OIHW"],
    "nn.conv3d": ["NCDHW", "OIDHW"],
    "qnn.conv2d": ["NCHW", "OIHW"],
    "nn.conv2d_transpose": ["NCHW", "IOHW"],
    "nn.max_pool2d": ["NCHW"],
    "nn.avg_pool2d": ["NCHW"],
    "nn.global_avg_pool2d": ["NCHW"],
    "image.resize2d": ["NCHW"],
    "image.grid_sample": ["NCHW"],
    "vision.roi_pool": ["NCHW", "default"],
    "nn.deformable_conv2d": ["NCHW", "OIHW"],
}


def convert_to_tuple(x):
    """Helper function to convert the given argument to a tuple object."""
    if isinstance(x, runtime.NDArray):
        return (x,)

    if isinstance(x, list):
        return tuple(value for value in x)

    if isinstance(x, runtime.container.ADT):
        if x.tag == 0:
            return tuple(field for field in x)

    raise RuntimeError(f'Can\'t convert type "{type(x)}" to tuple.')


def relative_symlink_in_dir(src_paths, dst_dir):
    """Create relative symbolic links to the given paths in the given directory."""
    if not isinstance(src_paths, (list, tuple)):
        src_paths = (src_paths,)

    src_paths = tuple(x for x in src_paths if os.path.exists(x))
    if len(src_paths) == 0:
        return

    os.makedirs(dst_dir, exist_ok=True)
    for src_path in src_paths:
        link_target = os.path.relpath(src_path, dst_dir)
        os.symlink(link_target, f"{dst_dir}/{os.path.basename(os.path.normpath(src_path))}")


def create_aipu_compass_module(aipu_bin, func_name="", target=None):
    """The function used to construct object of class AipuBmModuleNode or AipuCompassModuleNode."""
    cfg = AipuCompassConfig.get()
    target = target or cfg.gbuilder["target"]

    if cfg.common.get("bare_metal", "false") == "true":
        return _ffi_api.AipuBmModuleNode(aipu_bin, func_name, target)

    gb_dtcm_sz = cfg.gbuilder.get("tcm_size", None)
    # The size in GBuilder is kBytes, the size in UMD is MBytes.
    umd_dtcm_sz = str(int(gb_dtcm_sz) // 1024) if gb_dtcm_sz else ""
    return _ffi_api.AipuCompassModuleNode(aipu_bin, func_name, target, umd_dtcm_sz)
