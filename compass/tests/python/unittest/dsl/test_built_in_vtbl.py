# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vtbl(vdtype, i_vdtype, prim_name):
    @S.prim_func
    def vtbl_gentype_abcd_i(
        a: S.ptr(vdtype, "global"),
        b: S.ptr(vdtype, "global"),
        c: S.ptr(vdtype, "global"),
        d: S.ptr(vdtype, "global"),
        i: S.ptr(i_vdtype, "global"),
        out: S.ptr(vdtype, "global"),
    ):
        out[0] = S.vtbl(table=(a[0], b[0], c[0], d[0]), indices=i[0])  # tables is tuple

    @S.prim_func
    def vtbl_gentype_abc_i(
        a: S.ptr(vdtype, "global"),
        b: S.ptr(vdtype, "global"),
        c: S.ptr(vdtype, "global"),
        i: S.ptr(i_vdtype, "global"),
        out: S.ptr(vdtype, "global"),
    ):
        tables = (a[0], b[0], c[0])
        out[0] = S.vtbl(table=tables, indices=i[0])  # tables is tuple

    @S.prim_func
    def vtbl_gentype_ab_i(
        a: S.ptr(vdtype, "global"),
        b: S.ptr(vdtype, "global"),
        i: S.ptr(i_vdtype, "global"),
        out: S.ptr(vdtype, "global"),
    ):
        out[0] = S.vtbl(table=[a[0], b[0]], indices=i[0])  # tables is list

    if prim_name == "vtbl_gentype_abcd_i":
        return vtbl_gentype_abcd_i
    elif prim_name == "vtbl_gentype_abc_i":
        return vtbl_gentype_abc_i
    else:
        return vtbl_gentype_ab_i


def get_gt_output(sub_tables, i):
    table = np.concatenate(sub_tables)
    table_indices = list(range(len(table)))
    gt_out = [table[idx] if idx in table_indices else 0 for idx in i]
    gt_out = np.array(gt_out, dtype=sub_tables[0].dtype)
    return gt_out


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
@pytest.mark.parametrize("is_index_signed", (True, False))
@pytest.mark.parametrize(
    "prim_name",
    ("vtbl_gentype_abcd_i", "vtbl_gentype_abc_i", "vtbl_gentype_ab_i"),
)
def test_vtbl_gentype(dtype, is_index_signed, prim_name):
    vdtype = hw_native_vdtype(dtype)
    lane = vdtype.lanes
    a = rand(lane, dtype)
    b = rand(lane, dtype)
    c = rand(lane, dtype)
    d = rand(lane, dtype)
    if prim_name == "vtbl_gentype_abcd_i":
        sub_tables = [a, b, c, d]
    elif prim_name == "vtbl_gentype_abc_i":
        sub_tables = [a, b, c]
    elif prim_name == "vtbl_gentype_ab_i":
        sub_tables = [a, b]
    else:
        assert False, f"Unsupported prim name:{prim_name}"
    i_dtype = vdtype.with_int().element_of if is_index_signed else vdtype.with_uint().element_of
    i_vdtype = hw_native_vdtype(i_dtype)
    i = rand(lane, i_dtype)
    gt_out = get_gt_output(sub_tables=sub_tables, i=i)

    f_vtbl = gen_vtbl(vdtype, i_vdtype, prim_name)
    bm = BuildManager()
    ex = bm.build(f_vtbl)

    if prim_name == "vtbl_gentype_ab_i":
        expect = "out[0] = __vperm(a[0], b[0]"
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    py_out = np.empty(lane, dtype)
    f_vtbl(*sub_tables, i, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(lane, dtype)
    ex(*sub_tables, i, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vtbl_gentype(dtype="int8", is_index_signed=True, prim_name="vtbl_gentype_abcd_i")
    test_vtbl_gentype(dtype="float32", is_index_signed=True, prim_name="vtbl_gentype_abc_i")
    test_vtbl_gentype(dtype="float16", is_index_signed=False, prim_name="vtbl_gentype_ab_i")
    test_vtbl_gentype(dtype="bfloat16", is_index_signed=False, prim_name="vtbl_gentype_ab_i")
