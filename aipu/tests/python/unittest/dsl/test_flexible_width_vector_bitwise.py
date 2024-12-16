# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand, hw_native_vdtype


def gen_scalar_func(op_name, n, lanes, dtype, mask):
    @S.prim_func
    def scalar_xor_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] ^ b[0:lanes]
        c[lanes:n] = S.vxor(a[lanes:n], b[lanes:n], mask=mask)

    return locals()[f"scalar_{op_name}_func"]


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("op_name", ("xor",))
def test_scalar_flexible_width_vector(op_name, dtype):
    lanes = hw_native_vdtype(dtype).lanes + 3
    n = lanes * 2
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = np.concatenate((np.array([True] * lanes), rand(lanes, "bool")))
    gt_out = np.where(mask, a ^ b, 0).astype(dtype)

    py_func = gen_scalar_func(op_name, n, lanes, dtype, mask[lanes:])
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def gen_vshift_func(dtype, hw_lanes, op_name):
    sdot_func = {"shift_right": S.vsr, "shift_left": S.vsl}[op_name]

    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vshift_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(sdot_func(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(sdot_func(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(sdot_func(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(sdot_func(va3, vb3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(sdot_func(va4, vb4), cur_out)

    return vshift_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("op_name", ("shift_right", "shift_left"))
def test_vshift(dtype, op_name):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype, low=0, high=vdtype.bits)
    gt_out = a >> b if op_name == "shift_right" else a << b

    py_func = gen_vshift_func(dtype, hw_lanes, op_name)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vror_func(dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vror_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vror(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vror(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vror(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vror(va3, vb3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(S.vror(va4, vb4), cur_out)

    return vror_func


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "uint32"))
def test_vror(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype, low=0, high=vdtype.bits)
    gt_out = (a >> b) | (a << (b.itemsize * 8 - b))

    py_func = gen_vror_func(dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_single_input_func(dtype, mask, hw_lanes, func_name):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    cur_offset = 0
    mask0 = mask[cur_offset : cur_offset + lanes0]
    cur_offset += lanes0
    mask1 = mask[cur_offset : cur_offset + lanes1]
    cur_offset += lanes1
    mask2 = mask[cur_offset : cur_offset + lanes2]
    cur_offset += lanes2
    mask3 = mask[cur_offset : cur_offset + lanes3]
    cur_offset += lanes3
    mask4 = mask[cur_offset : cur_offset + lanes4]

    sdot_func = getattr(S, func_name)

    @S.prim_func
    def vcls_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_out = a, out
        cur_offset = 0
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(sdot_func(va0, mask0), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        cur_offset += lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(sdot_func(va1, mask1), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        cur_offset += lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(sdot_func(va2, mask2), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        cur_offset += lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(sdot_func(va3, mask3), cur_out)

        cur_a, cur_out = cur_a + lanes3, cur_out + lanes3
        cur_offset += lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        S.vstore(sdot_func(va4, mask4), cur_out)

    return vcls_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vcls(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    mask = rand(n, "bool")

    x_binary_list = [np.binary_repr(y, vdtype.bits) for y in a.tolist()]
    x_cls_list = np.array([len(s) - len(s.lstrip(s[0])) - 1 for s in x_binary_list], dtype=dtype)
    gt_out = np.where(mask, x_cls_list, 0)

    py_func = gen_single_input_func(dtype, mask, hw_lanes, "vcls")
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_clz(dtype):
    vdtype = hw_native_vdtype(dtype)
    u_dtype = vdtype.with_uint().element_of

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    mask = rand(n, "bool")

    bin_str_list = [f"{item:0{vdtype.bits}b}" for item in a.view(u_dtype)]
    x_clz_list = np.array([len(item) - len(item.lstrip("0")) for item in bin_str_list], dtype=dtype)
    gt_out = np.where(mask, x_clz_list, 0)

    py_func = gen_single_input_func(dtype, mask, hw_lanes, "clz")
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vinv(dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    mask = rand(n, "bool")

    gt_out = np.where(mask, ~a, 0)

    py_func = gen_single_input_func(dtype, mask, hw_lanes, "vinv")
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vbrevs(dtype):
    vdtype = hw_native_vdtype(dtype)
    u_dtype = vdtype.with_uint().element_of
    hw_lanes = vdtype.lanes
    bits = vdtype.bits
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    mask = rand(n, "bool")

    reversed_arr = np.array([int(f"{x:0{bits}b}"[::-1], 2) for x in a.view(u_dtype)])
    gt_out = np.where(mask, reversed_arr.astype(dtype), 0)

    py_func = gen_single_input_func(dtype, mask, hw_lanes, "vbrevs")
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vpcnt(dtype):
    vdtype = hw_native_vdtype(dtype)
    u_dtype = vdtype.with_uint().element_of
    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    mask = rand(n, "bool")

    cnt_arr = np.array([bin(data).count("1") for data in a.view(u_dtype)])
    gt_out = np.where(mask, cnt_arr.astype(dtype), 0)

    py_func = gen_single_input_func(dtype, mask, hw_lanes, "vpcnt")
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def gen_vand_vor_func(func_name, dtype, hw_lanes):
    sdot_func = {"vand": S.vand, "vor": S.vor}[func_name]
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vsdot0 = sdot_func(va0, vb0)
        mask0 = sdot_func(va0 > 0, vb0 > 0)
        S.vstore(S.vsel(vsdot0, 0, mask0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vsdot1 = sdot_func(va1, vb1)
        mask1 = sdot_func(va1 > 0, vb1 > 0)
        S.vstore(S.vsel(vsdot1, 0, mask1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vsdot2 = sdot_func(va2, vb2)
        mask2 = sdot_func(va2 > 0, vb2 > 0)
        S.vstore(S.vsel(vsdot2, 0, mask2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vsdot3 = sdot_func(va3, vb3)
        mask3 = sdot_func(va3 > 0, vb3 > 0)
        S.vstore(S.vsel(vsdot3, 0, mask3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        vsdot4 = sdot_func(va4, vb4)
        mask4 = sdot_func(va4 > 0, vb4 > 0)
        S.vstore(S.vsel(vsdot4, 0, mask4), cur_out)

    return func


@pytest.mark.parametrize("func_name", ("vand", "vor"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vand_vor(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")

    if func_name == "vand":
        gt_out = np.where((a > 0) & (b > 0), a & b, 0)
    elif func_name == "vor":
        gt_out = np.where((a > 0) | (b > 0), a | b, 0)

    py_func = gen_vand_vor_func(func_name, dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def gen_vall_vany_func(dtype, hw_lanes, func_name):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    sdot_func = getattr(S, func_name)

    @S.prim_func
    def vall_vany_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a = a
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        mask = va0 > 0
        if sdot_func(mask):
            out[0] = 0
        else:
            out[0] = 1

        cur_a += lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        mask1 = va1 > 0
        if sdot_func(mask1):
            out[1] = 0
        else:
            out[1] = 1

        cur_a += lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        mask2 = va2 > 0
        if sdot_func(mask2):
            out[2] = 0
        else:
            out[2] = 1

        cur_a += lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        mask3 = va3 > 0
        if sdot_func(mask3):
            out[3] = 0
        else:
            out[3] = 1

        cur_a += lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        mask4 = va4 > 0
        if sdot_func(mask4):
            out[4] = 0
        else:
            out[4] = 1

    return vall_vany_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("op_name", ("vall", "vany"))
def test_vall_vany(op_name, dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)

    gt_out = []
    cur_a = 0
    for i in (hw_lanes // 2, hw_lanes - 5, hw_lanes + 3, 2 * hw_lanes + 1, 4 * hw_lanes):
        gt_out.append(0 if getattr(np, op_name[1:])(a[cur_a : cur_a + i] > 0) else 1)
        cur_a += i
    gt_out = np.array(gt_out).astype(dtype)

    py_func = gen_vall_vany_func(dtype, hw_lanes, op_name)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(5, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(5, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_scalar_flexible_width_vector("xor", "int32")
    test_vshift("int32", "shift_left")
    test_vror("uint32")
    test_vcls("int8")
    test_clz("int8")
    test_vand_vor("vand", "int32")
    test_vand_vor("vor", "int32")
    test_vinv("int8")
    test_vbrevs("int8")
    test_vpcnt("int8")
    test_vall_vany("vall", "int32")
    test_vall_vany("vany", "int32")
