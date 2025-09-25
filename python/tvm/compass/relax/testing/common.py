# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name,comparison-with-itself
"""General functions for testing."""
import os


def lower_framework(framework):
    if framework.lower() in ["tf", "tensorflow"]:
        return "tf"
    return framework.lower()


def convert_to_list(x):
    """Convert to list."""
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


def get_cutted_string(string, delimiter, index):
    """Get the substring after cutting."""
    return string.strip().split(delimiter)[index].strip()


def is_number(s):
    """Return true if s is a number else false."""

    def is_NaN(num):
        return num != num

    try:
        x = float(s)
        if is_NaN(x):
            return False
        return True
    except ValueError:
        pass

    try:
        # pylint: disable=import-outside-toplevel
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_subgraph_and_op_count(ir_path: str):
    """Get the number of NPU subgraph and the number of other OP.

    Args:
        ir_path (str): the path to the relax ir.

    Returns:
        int, int: number of NPU subgraph, number of other OP.
    """
    assert os.path.isfile(ir_path), f"{ir_path} is Not Exists."

    npu_subgraph_count = 0
    other_op_count = 0

    with open(ir_path, "r") as f:
        main_body = []
        is_body = False
        for line in f.readlines():
            if "def main" in line:
                is_body = True
                continue
            if is_body:
                main_body.append(line.strip())

    for line in main_body:
        if " = " not in line:
            continue
        rvalue = line.split(" = ")[1]
        if rvalue.startswith("cls.tvm_compass_"):
            npu_subgraph_count += 1
        elif rvalue.startswith(("R.", "cls.post")):
            other_op_count += 1
    return npu_subgraph_count, other_op_count


def is_all_npu_subgraph(ir_path: str):
    _, op_count = get_subgraph_and_op_count(ir_path)
    assert op_count == 0, f"There are {op_count} operators not divided into NPU."


def is_npu_subgraph_number_matched(ir_path: str, expect_npu_subgraph_number: int):
    npu_count, _ = get_subgraph_and_op_count(ir_path)

    def assert_info(type_, real_number, expect_number):
        return (
            f"Mismatched Number of {type_} Subgraphs: Real {real_number} vs Expect {expect_number}"
        )

    assert npu_count == expect_npu_subgraph_number, assert_info(
        "NPU", npu_count, expect_npu_subgraph_number
    )
