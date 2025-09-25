# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import relax
from tvm.compass.relax import testing


def test_sign():
    data_shape = [1, 128, 128, 32]
    data = relax.Var("scale", relax.TensorStructInfo(data_shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.sign(data)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype("float32")
    testing.compare_relax_opt_float_result(mod, data_input)


@pytest.mark.parametrize(
    "data_shape, weight_shape",
    [
        ((10, 5), (5, 2)),
        ((5, 10), (10, 2)),
        ((10, 3), (3, 5)),
        ((3, 7), (7, 5)),
    ],
)
def test_matmul(data_shape, weight_shape):
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))
    weight = relax.const(np.random.rand(*weight_shape).astype(dtype), dtype)

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.matmul(data, weight, out_dtype=dtype)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


@pytest.mark.parametrize(
    "data_shape, axis, keepdims",
    [
        ((10, 5), 1, True),
        ((10, 5), (0, 1), True),
    ],
)
def test_mean(data_shape, axis, keepdims):
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.mean(data, axis, keepdims)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


@pytest.mark.parametrize(
    "op_name, data_shape",
    [
        ("add", (3, 5)),
        ("subtract", (4, 9)),
        ("multiply", (2, 7)),
        ("divide", (5, 6)),
    ],
)
def test_binary(op_name, data_shape):
    dtype = "float32"
    data0 = relax.Var("data0", relax.TensorStructInfo(data_shape, dtype))
    data1 = relax.Var("data1", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data0, data1]):
        with bb.dataflow() as _:
            lv0 = getattr(relax.op, op_name)(data0, data1)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = [np.random.random(data_shape).astype(dtype) for _ in range(2)]
    testing.compare_relax_opt_float_result(mod, *data_input)


def test_concat():
    data_shape = (3, 2)
    dtype = "float32"
    data0 = relax.Var("data0", relax.TensorStructInfo(data_shape, dtype))
    data1 = relax.Var("data1", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data0, data1]):
        with bb.dataflow() as _:
            lv0 = relax.op.concat((data0, data1), axis=1)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = [np.random.random(data_shape).astype(dtype) for _ in range(2)]
    testing.compare_relax_opt_float_result(mod, *data_input)


def test_clip():
    data_shape = (3, 2)
    dtype = "float32"
    data0 = relax.Var("data0", relax.TensorStructInfo(data_shape, dtype))
    min_data = relax.PrimValue(3)
    max_data = relax.PrimValue(6)

    bb = relax.BlockBuilder()
    with bb.function("main", [data0]):
        with bb.dataflow() as _:
            lv0 = relax.op.clip(data0, min_data, max_data)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_silu():
    data_shape = (3, 2)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.nn.silu(data)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_rms_norm():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))
    weight = relax.const(np.random.rand(1024).astype(dtype), dtype)

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.nn.rms_norm(data, weight)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_permute_dims():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.permute_dims(data, [1, 0, 2])
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_reshape():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.reshape(data, [3, 4, 256])
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_split():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.split(data, 2, 2)
            lv1 = relax.op.add(lv0[0], lv0[1])
            gv0 = bb.emit_output(lv1)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_take():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.take(data, relax.const(1, "int32"), 0)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


def test_softmax():
    data_shape = (3, 1, 1024)
    dtype = "float32"
    data = relax.Var("data", relax.TensorStructInfo(data_shape, dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.nn.softmax(data)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype(dtype)
    testing.compare_relax_opt_float_result(mod, data_input)


if __name__ == "__main__":
    test_sign()
    test_matmul((10, 5), (5, 2))
    test_mean((10, 5), (0, 1), True)
    test_binary("add", (3, 2))
    test_binary("multiply", (3, 2))
    test_concat()
    test_clip()
    test_silu()
    test_rms_norm()
    test_permute_dims()
    test_reshape()
    test_split()
    test_take()
    test_softmax()
