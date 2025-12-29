# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Common Compass utilities."""
import os
import numpy as np
from tvm import runtime, relax


X86_DESIRED_LAYOUTS = {
    "relax.nn.conv2d": ["NCHW", "OIHW"],
    # "nn.conv3d": ["NCDHW", "OIDHW"],
    "relax.nn.conv2d_transpose": ["NCHW", "IOHW"],
    "relax.nn.max_pool2d": ["NCHW"],
    # "nn.avg_pool2d": ["NCHW"],
    # "nn.global_avg_pool2d": ["NCHW"],
    # "image.resize2d": ["NCHW"],
    # "image.grid_sample": ["NHWC"],
    # "vision.roi_pool": ["NCHW", "default"],
    # "nn.deformable_conv2d": ["NCHW", "OIHW"],
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


def compute_cos_distance(x, y, precision_dtype="float32", keep_decimal=3, judge_divisor_0=False):
    """Get cosine similarity."""
    x = x.astype(precision_dtype)
    y = y.astype(precision_dtype)
    divisor = np.linalg.norm(x) * np.linalg.norm(y)
    divisor = 1e-10 if judge_divisor_0 and divisor == 0 else divisor
    similarity = np.dot(x.flatten(), y.flatten()) / divisor
    return float(format(similarity, f".{keep_decimal}f"))


def unpack_commutative_args(call, rhs_name="const"):
    """Unpack arguments of the binary operators consider commutative, ensure the
    right hand side operand is the expected one."""
    assert isinstance(call, relax.Call)
    valid_ops = ("relax.add", "relax.multiply", "relax.maximum", "relax.minimum")

    lhs, rhs = call.args
    if rhs_name == "const":
        if isinstance(rhs, relax.Constant):
            return lhs, rhs
        assert call.op.name in valid_ops
        assert isinstance(lhs, relax.Constant)
        return rhs, lhs

    if isinstance(rhs, relax.Call) and rhs.op.name == rhs_name:
        return lhs, rhs

    assert call.op.name in valid_ops
    assert isinstance(lhs, relax.Call) and lhs.op.name == rhs_name
    return rhs, lhs
