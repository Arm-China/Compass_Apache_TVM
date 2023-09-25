# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import math
import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def np_all_close(outputs, golden):
    if len(outputs) == 1:
        assert np.allclose(outputs[0].numpy(), golden.numpy())
    else:
        for i, data in enumerate(outputs):
            assert np.allclose(data.numpy(), golden[i].numpy())


def test_relay_grid_sample():
    data_shape = [1, 32, 128, 128]
    grid_shape = [1, 2, 128, 128]

    data = relay.var("data", shape=data_shape)
    grid = relay.var("grid", shape=grid_shape)
    grid_sample = relay.image.grid_sample(data, grid)
    mod = tvm.IRModule.from_expr(grid_sample)

    data_input = np.random.random(data_shape).astype(np.float32)
    grid_input = np.random.random(grid_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, grid_input)


def test_relay_layout_transform_4d_to_5d():
    data_shape = [1, 128, 128, 64]

    data = relay.var("data", shape=data_shape)
    output = relay.layout_transform(data, "NHWC", "NCHW32c")
    mod = tvm.IRModule.from_expr(output)

    data_input = np.random.random(data_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_layout_transform_5d_to_4d():
    data_shape = [1, 4, 128, 128, 16]

    data = relay.var("data", shape=data_shape)
    output = relay.layout_transform(data, "NCHW16c", "NHWC")
    mod = tvm.IRModule.from_expr(output)

    data_input = np.random.random(data_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_copy():
    data_shape = [1, 128, 128, 64]

    data = relay.var("data", shape=data_shape)
    output = relay.copy(data)
    mod = tvm.IRModule.from_expr(output)

    data_input = np.random.random(data_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_get_valid_counts():
    def verify_get_valid_counts(dshape, score_threshold, id_index, score_index):
        dtype = "float32"
        np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)

        inp = relay.var("input", relay.ty.TensorType(dshape, dtype))
        expr = relay.vision.get_valid_counts(inp, score_threshold, id_index, score_index)
        expr = expr.astuple()
        ele0 = relay.TupleGetItem(expr, 0)
        ele1 = relay.TupleGetItem(expr, 1)
        ele2 = relay.TupleGetItem(expr, 2)
        expr = relay.Tuple((ele0, ele1, ele2))

        mod = tvm.IRModule.from_expr(expr)

        def compare(outputs, golden):
            count, tensor, indices = outputs
            gt_count, gt_tensor, gt_indices = golden

            assert np.allclose(count.numpy(), gt_count.numpy())
            assert np.allclose(tensor.numpy(), gt_tensor.numpy())
            assert np.allclose(indices.numpy(), gt_indices.numpy())

        aipu_testing.compare_relay_opt_float_result(mod, np_data, compare=compare)

    verify_get_valid_counts((1, 2500, 6), 0, 0, 1)
    verify_get_valid_counts((1, 2500, 5), -1, -1, 0)
    verify_get_valid_counts((3, 1000, 6), 0.55, 1, 0)
    verify_get_valid_counts((16, 500, 5), 0.95, -1, 0)


def test_multibox_transform_loc():
    def test_default_value():
        num_anchors = 3
        num_classes = 3

        np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]]).astype(
            "float32"
        )
        np_loc_preds = np.array(
            [[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]]
        ).astype("float32")
        np_anchors = np.array(
            [[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]]
        ).astype("float32")

        cls_prob = relay.var(
            "cls_prob", relay.ty.TensorType((1, num_anchors, num_classes), "float32")
        )
        loc_pred = relay.var("loc_pred", relay.ty.TensorType((1, num_anchors * 4), "float32"))
        anchors = relay.var("anchors", relay.ty.TensorType((1, num_anchors, 4), "float32"))

        mtl = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob, loc_pred=loc_pred, anchor=anchors
        ).astuple()

        expr0 = relay.TupleGetItem(mtl, 0)
        expr1 = relay.TupleGetItem(mtl, 1)
        expr = relay.Tuple((expr0, expr1))
        mod = tvm.IRModule.from_expr(expr)

        def compare(outputs, golden):
            valid, count = outputs
            valid_gt, count_gt = golden

            assert np.allclose(count.numpy(), count_gt.numpy())
            assert np.allclose(valid.numpy(), valid_gt.numpy())

        aipu_testing.compare_relay_opt_float_result(
            mod, np_cls_prob, np_loc_preds, np_anchors, compare=compare
        )

    test_default_value()


def test_relay_non_max_suppression():
    dshape = [1, 5, 5]

    x0 = relay.var("data", relay.ty.TensorType(dshape, "float32"))
    x1 = relay.var("valid_count", relay.ty.TensorType((dshape[0],), "int32"))

    indices = np.array([-1] * (dshape[0] * dshape[1]), dtype=np.int32)
    indices = indices.reshape(dshape[0], dshape[1])
    x2 = relay.const(tvm.nd.array(indices))
    x3 = -1

    nms = relay.vision.non_max_suppression(
        x0,
        x1,
        x2,
        x3,
        iou_threshold=0.5,
        top_k=-1,
        return_indices=False,
        coord_start=1,
        score_index=0,
        id_index=-1,
    )

    mod = tvm.IRModule.from_expr(nms)

    np_data = np.array(
        [
            [
                [0.8, 1, 20, 25, 45],
                [0.7, 30, 60, 50, 80],
                [0.4, 4, 21, 19, 40],
                [0.9, 35, 61, 52, 79],
                [0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype("float32")
    np_valid_count = np.array([5]).astype("int32")

    def compare(outputs, golden):
        golden = golden.numpy()
        batch, _, element = golden.shape
        golden = golden[golden != -1].reshape(batch, -1, element)
        fetch_num = golden.shape[1]
        outputs = outputs[0].numpy()[:, :fetch_num, :]
        np.allclose(outputs, golden)

    aipu_testing.compare_relay_opt_float_result(mod, np_data, np_valid_count, compare=compare)


def test_relay_roi_align():
    data_shape = (3, 16, 16, 4)
    rois_shape = (32, 5)
    pooled_size = 7
    spatial_scale = 0.5
    sample_ratio = -1
    mode = "avg"

    data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
    rois = relay.var("rois", relay.ty.TensorType(rois_shape, "float32"))
    roi = relay.vision.roi_align(
        data,
        rois,
        pooled_size=(pooled_size, pooled_size),
        spatial_scale=spatial_scale,
        sample_ratio=sample_ratio,
        mode=mode,
        layout="NHWC",
    )
    mod = tvm.IRModule.from_expr(roi)

    num_roi = rois_shape[0]
    in_size = data_shape[1]
    np_data = np.random.uniform(size=data_shape).astype("float32")
    np_rois = np.random.uniform(size=rois_shape).astype("float32") * in_size
    np_rois[:, 0] = np.random.randint(low=0, high=data_shape[0], size=num_roi)

    aipu_testing.compare_relay_opt_float_result(mod, np_data, np_rois)


def test_relay_roi_pool():
    data_shape = (1, 4, 32, 32)
    rois_shape = (32, 5)
    pooled_size = 7
    spatial_scale = 1

    data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
    rois = relay.var("rois", relay.ty.TensorType(rois_shape, "float32"))
    roi = relay.vision.roi_pool(
        data,
        rois,
        pooled_size=(pooled_size, pooled_size),
        spatial_scale=spatial_scale,
        layout="NCHW",
    )

    mod = tvm.IRModule.from_expr(roi)

    num_roi = rois_shape[0]
    in_size = data_shape[2]
    np_data = np.random.uniform(size=data_shape).astype("float32")
    np_rois = np.random.uniform(size=rois_shape).astype("float32") * in_size
    np_rois[:, 0] = np.random.randint(low=0, high=data_shape[0], size=num_roi)

    aipu_testing.compare_relay_opt_float_result(mod, np_data, np_rois)


def test_batch_flatten():
    data_shape = (5, 10, 5)
    ty1 = relay.TensorType(data_shape)
    x_var = relay.Var("x", ty1)
    mod = tvm.IRModule.from_expr(relay.nn.batch_flatten(x_var))

    data = np.random.rand(*data_shape).astype(ty1.dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data)


def test_batch_to_space_nd():
    def verify_batch_to_space_nd(dshape, block_shape, crops):
        x_data = np.random.uniform(size=dshape).astype("float32")

        x_var = relay.var("x", relay.TensorType(dshape, "float32"))
        expr = relay.nn.batch_to_space_nd(x_var, block_shape, crops)
        mod = tvm.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        aipu_testing.compare_relay_opt_float_result(mod, x_data)

    verify_batch_to_space_nd([4, 1, 1, 3], [2, 2], [[0, 0], [0, 0]])
    verify_batch_to_space_nd([8, 1, 3, 1], [2, 2], [[0, 0], [2, 0]])


def test_relay_instance_norm():
    dtype = "float32"
    shape = (1, 56, 56, 64)
    x = relay.var("x", shape=shape)
    beta = relay.var("beta", relay.TensorType((shape[-1],), dtype))
    gamma = relay.var("gamma", relay.TensorType((shape[-1],), dtype))

    y = relay.nn.instance_norm(x, gamma, beta, axis=3)
    mod = tvm.IRModule.from_expr(y)

    x_data = np.random.random(shape).astype(dtype)
    gamma_data = np.random.random((shape[-1],)).astype(dtype)
    beta_data = np.random.random((shape[-1],)).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_data, gamma_data, beta_data)


def test_relay_layer_norm():
    dtype = "float32"
    shape = (1, 56, 56, 64)
    x = relay.var("x", shape=shape)
    beta = relay.var("beta", relay.TensorType((shape[-1],), dtype))
    gamma = relay.var("gamma", relay.TensorType((shape[-1],), dtype))

    y = relay.nn.layer_norm(x, gamma, beta, axis=3)
    mod = tvm.IRModule.from_expr(y)

    x_data = np.random.random(shape).astype(dtype)
    gamma_data = np.random.random((shape[-1],)).astype(dtype)
    beta_data = np.random.random((shape[-1],)).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_data, gamma_data, beta_data)


def test_relay_group_norm():
    dtype = "float32"
    shape = (1, 56, 56, 64)
    x = relay.var("x", shape=shape)
    beta = relay.var("beta", relay.TensorType((shape[-1],), dtype))
    gamma = relay.var("gamma", relay.TensorType((shape[-1],), dtype))

    y = relay.nn.group_norm(x, gamma, beta, 4, axis=3)
    mod = tvm.IRModule.from_expr(y)

    x_data = np.random.random(shape).astype(dtype)
    gamma_data = np.random.random((shape[-1],)).astype(dtype)
    beta_data = np.random.random((shape[-1],)).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_data, gamma_data, beta_data)


def test_relay_l2_norm():
    dtype = "float32"
    shape = (1, 56, 56, 64)
    x = relay.var("x", shape=shape)

    y = relay.nn.l2_normalize(x, 0.00001, axis=[1])
    mod = tvm.IRModule.from_expr(y)

    x_data = np.random.random(shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_data)


def test_relay_topk():
    shape = (20, 100)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))

    out = relay.topk(x, 5)
    if isinstance(out, relay.expr.TupleWrapper):
        out = list(out)
        out = out[0] if len(out) == 1 else relay.Tuple(out)
    mod = tvm.IRModule.from_expr(out)

    data = np.random.random(shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data, compare=np_all_close)


def test_relay_gather():
    def verify_gather(data_shape, indices_shape, axis):
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        indices_value = np.random.randint(low=0, high=3, size=indices_shape, dtype="int32")
        indices = relay.const(indices_value, dtype="int32")
        output = relay.gather(data, axis, indices)
        mod = tvm.IRModule.from_expr(output)

        data_input = np.random.random(data_shape).astype("float32")

        aipu_testing.compare_relay_opt_float_result(mod, data_input)

    verify_gather([4, 4], [4, 4], 0)
    verify_gather([5, 5], [5, 5], 1)


def test_relay_gather_nd():
    def verify_gather_nd(data_shape, indices_shape, batch_dims):
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        indices_value = np.random.randint(low=0, high=1, size=indices_shape, dtype="int32")
        indices = relay.const(indices_value, dtype="int32")
        output = relay.gather_nd(data, indices, batch_dims)
        mod = tvm.IRModule.from_expr(output)

        data_input = np.random.random(data_shape).astype("float32")

        aipu_testing.compare_relay_opt_float_result(mod, data_input)

    verify_gather_nd([2, 2, 2], [2, 2], 0)
    verify_gather_nd([2, 3, 2, 2], [1, 2], 1)


def test_relay_scatter_nd():
    def verify_scatter_nd(data_np, indices_np, updates_np, mode="add"):
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices = relay.const(indices_np, dtype="int64")
        updates = relay.var("updates", shape=updates_np.shape, dtype=str(updates_np.dtype))

        out = relay.op.scatter_nd(data, indices, updates, mode)
        mod = tvm.IRModule.from_expr(out)
        aipu_testing.compare_relay_opt_float_result(mod, data_np, updates_np)

    for indice_dtype in ["uint8", "uint16", "uint32"]:
        data = np.zeros((2, 2)).astype("int64")
        indices = np.array([[1, 1, 0], [0, 1, 0]]).astype(indice_dtype)
        updates = np.array([2, 3, 0])
        verify_scatter_nd(data, indices, updates)

        data = np.zeros((2, 2, 2, 2)).astype("int64")
        indices = np.array([[0, 1], [1, 1]]).astype(indice_dtype)
        updates = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        verify_scatter_nd(data, indices, updates)

        data = np.zeros((2, 1560)).astype("float32")
        indices = np.array([[1, 0, 0]]).astype(indice_dtype)
        updates = np.reshape(np.arange(1560 * 3), (3, 1560)).astype("float32")
        verify_scatter_nd(data, indices, updates)

        for mode in ["add", "update"]:
            data = np.random.random((2, 7, 3)).astype("float64")
            indices = np.stack((np.random.randint(2, size=5), np.random.randint(7, size=5))).astype(
                indice_dtype
            )
            updates = np.ones((5, 3)).astype("float64")
            verify_scatter_nd(data, indices, updates, mode)


def test_relay_expand_dims():
    def verify_expand_dims(axis):
        data_shape = [1, 128, 128, 32]
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        output = relay.expand_dims(data, axis)
        mod = tvm.IRModule.from_expr(output)

        data_input = np.random.random(data_shape).astype("float32")

        aipu_testing.compare_relay_opt_float_result(mod, data_input)

    verify_expand_dims(0)
    verify_expand_dims(2)


def test_relay_arange():
    def verify_arange(start, stop, step):
        start_expr = relay.const(start, "float32")
        stop_expr = relay.const(stop, "float32")
        step_expr = relay.const(step, "float32")
        expr = relay.arange(start_expr, stop_expr, step_expr)
        data_shape = (math.ceil((stop - start) / step), 1)
        data = relay.var("data", shape=data_shape, dtype="float32")
        expr = relay.maximum(expr, data)
        mod = tvm.IRModule.from_expr(expr)

        data_input = np.ones(data_shape).astype("float32")
        aipu_testing.compare_relay_opt_float_result(mod, data_input)

    verify_arange(1, 5, 1)
    verify_arange(2, 10, 1.5)


@pytest.mark.parametrize(
    "data_shape",
    [
        ([1, 3, 2, 2]),
        ([2, 3, 4]),
        ([4, 5]),
    ],
)
def test_relay_sqrt(data_shape):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.sqrt(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_rsqrt():
    data_shape = [1, 128, 128, 32]
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.rsqrt(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_sign():
    data_shape = [1, 128, 128, 32]
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.sign(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_sin():
    data_shape = [1, 128, 128, 32]
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.sin(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_tan():
    data_shape = [1, 128, 128, 32]
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.tan(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_erf():
    data_shape = [1, 128, 128, 32]
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    output = relay.erf(data)
    mod = tvm.IRModule.from_expr(output)
    data_input = np.random.random(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_crop_and_resize():
    def verify_crop_and_resize(interpolate_method, layout, extrapolation_value):
        if layout == "NHWC":
            img_shape = (10, 224, 224, 3)
            boxes = np.array([[0.1, 0.2, 0.8, 0.7], [0.2, 0, 1, 0.6]]).astype("float32")
            box_indices = np.array([1, 0]).astype("int32")
            crop_size = np.array([20, 30]).astype("int32")
        elif layout == "NCHW":
            img_shape = (5, 3, 255, 255)
            boxes = np.array([[0, 0, 1, 1], [0.2, 0.1, 1, 0.9]]).astype("float32")
            box_indices = np.array([0, 1]).astype("int32")
            crop_size = np.array([30, 30]).astype("int32")
        image_data = np.random.uniform(size=img_shape).astype("float32")
        img = relay.var("img", relay.TensorType(img_shape, "float32"))
        bx = relay.var("bx", relay.TensorType(boxes.shape, "float32"))
        bx_idx = relay.var("bx_idx", relay.TensorType(box_indices.shape, "int32"))
        output = relay.image.crop_and_resize(
            img, bx, bx_idx, list(crop_size), layout, interpolate_method, extrapolation_value
        )

        mod = tvm.IRModule.from_expr(output)
        aipu_testing.compare_relay_opt_float_result(mod, image_data, boxes, box_indices)

    verify_crop_and_resize("bilinear", "NHWC", 0.0)
    verify_crop_and_resize("nearest_neighbor", "NCHW", 1.0)


def test_relay_cast_like_2_var():
    data = relay.var("data", shape=(3, 4, 5))
    dtype_like = relay.var("dtype_like", shape=(2, 2, 2), dtype="int32")

    out = relay.cast_like(data, dtype_like)
    mod = tvm.IRModule.from_expr(out)

    data_input = np.random.uniform(0.0, 300.0, (3, 4, 5)).astype("float32")
    data_like = np.random.randint(0, 255, size=(2, 2, 2)).astype("int32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input, data_like)


def test_relay_cast_like_1_var():
    data = relay.var("data", shape=(3, 4, 5))
    dtype_like = relay.const(np.ones([2, 2, 2]), dtype="int32")

    out = relay.cast_like(data, dtype_like)
    mod = tvm.IRModule.from_expr(out)

    data_input = np.random.uniform(0.0, 300.0, (3, 4, 5)).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


def test_relay_full_like():
    shape_like = relay.var("shape_like", shape=(3, 4, 5))
    fill_value = relay.var("fill", relay.TensorType(()))

    out = relay.full_like(shape_like, fill_value)
    mod = tvm.IRModule.from_expr(out)

    data = np.random.randint(0, 255, size=(3, 4, 5)).astype("float32")
    fill = np.array([5], "float32").reshape([])
    aipu_testing.compare_relay_opt_float_result(mod, data, fill)


@pytest.mark.parametrize(
    "fill_value, data_shape",
    [
        (4, (2, 3, 4, 5)),
        (2, (2, 3, 4)),
        (4.0, (2, 3)),
    ],
)
def test_relay_full(fill_value, data_shape):
    dtype = "float32"
    x = relay.var("x", relay.scalar_type(dtype))
    z = relay.full(x, data_shape, dtype)
    mod = tvm.IRModule.from_expr(z)

    x_np = np.array(fill_value, dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np, compare=np_all_close)


def test_relay_ones():
    data_shape = (3, 4, 5)
    expr = relay.ones(data_shape, "float32")
    data1 = relay.var("data1", shape=data_shape)
    expr = relay.maximum(expr, data1)
    mod = tvm.IRModule.from_expr(expr)

    data1_input = np.random.randint(0, 10, size=data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data1_input)


def test_relay_ones_like():
    data_shape = [3, 4, 5]
    data = relay.var("data", shape=data_shape)
    expr = relay.ones_like(data)
    data1 = relay.var("data1", shape=data_shape)
    expr = relay.maximum(expr, data1)
    mod = tvm.IRModule.from_expr(expr)

    data_input = np.ones(data_shape).astype("float32")
    data1_input = np.ones(data_shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input, data1_input)


def test_relay_reshape_like():
    data_shape = [2, 3, 4]
    dest_shape = [6, 2, 2]

    data = relay.var("data", shape=data_shape)
    shape_like = relay.var("shape_like", shape=dest_shape)
    expr = relay.reshape_like(data, shape_like)
    mod = tvm.IRModule.from_expr(expr)

    data_input = np.random.random(data_shape).astype(np.float32)
    dest_input = np.random.random(dest_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, dest_input)


def test_relay_broadcast_to():
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape, dtype))
    expr = relay.broadcast_to(x, shape=shape_like)
    mod = tvm.IRModule.from_expr(expr)
    input_x = np.random.uniform(size=shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, input_x)


def test_relay_broadcast_to_like():
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    x = relay.var("x", shape=shape)
    y = relay.var("y", shape=shape_like)
    expr = relay.broadcast_to_like(x, y)
    mod = tvm.IRModule.from_expr(expr)

    input_x = np.random.uniform(size=shape).astype("float32")
    input_y = np.random.uniform(size=shape_like).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, input_x, input_y)


def test_relay_zeros():
    shape = (3, 4, 5)
    expr = relay.zeros(shape, "float32")
    data1 = relay.var("data1", shape=shape)
    expr = relay.maximum(expr, data1)
    mod = tvm.IRModule.from_expr(expr)

    data1_input = np.random.randint(-10, 10, size=shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data1_input, compare=np_all_close)


def test_relay_zeros_like():
    shape = [3, 4, 5]
    data = relay.var("data", shape=shape)
    expr = relay.zeros_like(data)
    data1 = relay.var("data1", shape=shape)
    expr = relay.maximum(expr, data1)
    mod = tvm.IRModule.from_expr(expr)

    data_input = np.zeros(shape).astype("float32")
    data1_input = np.ones(shape).astype("float32")
    aipu_testing.compare_relay_opt_float_result(mod, data_input, data1_input, compare=np_all_close)


@pytest.mark.parametrize("data_shape", [[2, 3, 4, 5]])
@pytest.mark.parametrize("scale_h", [2, 3])
@pytest.mark.parametrize("scale_w", [2, 3])
@pytest.mark.parametrize("layout", ["NCHW"])
@pytest.mark.parametrize("method", ["nearest_neighbor", "bilinear"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_upsampling(data_shape, scale_h, scale_w, layout, method, align_corners):
    data = relay.var("data", shape=data_shape)

    upsampling = relay.nn.upsampling(data, scale_h, scale_w, layout, method, align_corners)
    mod = tvm.IRModule.from_expr(upsampling)

    data_input = np.random.random(data_shape).astype(np.float32)

    if method in ("bilinear", "bicubic"):

        def compute_cos_distance(x, y):
            """Get cosine similarity."""
            x, y = x[0].numpy(), y.numpy()
            similarity = np.dot(x.flatten(), y.flatten()) / (
                np.linalg.norm(x) * (np.linalg.norm(y))
            )
            assert float(format(similarity, ".3f")) >= 0.95

        aipu_testing.compare_relay_opt_float_result(mod, data_input, compare=compute_cos_distance)
    else:
        aipu_testing.compare_relay_opt_float_result(mod, data_input)


@pytest.mark.parametrize("method", ["std", "variance"])
@pytest.mark.parametrize("data_shape", [[2, 3, 4, 5]])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, -1, -2, -3, -4, (0, 1), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("exclude", [False])
@pytest.mark.parametrize("unbiased", [True, False])
def test_std_variance(method, data_shape, axis, keepdims, exclude, unbiased):
    data = relay.var("data", shape=data_shape)
    funcs = {
        "std": relay.std,
        "variance": relay.variance,
    }
    out = funcs[method](data, axis, keepdims, exclude, unbiased)
    mod = tvm.IRModule.from_expr(out)
    data_input = np.random.random(data_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input)


@pytest.mark.parametrize("method", ["mean_std", "mean_variance"])
@pytest.mark.parametrize("data_shape", [[2, 3, 4, 5]])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, -1, -2, -3, -4, (0, 1), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("exclude", [False])
def test_mean_std_variance(method, data_shape, axis, keepdims, exclude):
    data = relay.var("data", shape=data_shape)

    funcs = {
        "mean_std": relay.mean_std,
        "mean_variance": relay.mean_variance,
    }
    mean = funcs[method](data, axis, keepdims, exclude)
    out = relay.Tuple(list(mean))
    mod = tvm.IRModule.from_expr(out)

    data_input = np.random.random(data_shape).astype(np.float32)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, compare=np_all_close)


def test_space_to_batch_nd():
    def verify_space_to_batch_nd(dshape, block_shape, paddings):
        x_data = np.random.uniform(size=dshape).astype("float32")

        x_var = relay.var("x", relay.TensorType(dshape, "float32"))
        expr = relay.nn.space_to_batch_nd(x_var, block_shape, paddings)
        mod = tvm.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        aipu_testing.compare_relay_opt_float_result(mod, x_data)

    verify_space_to_batch_nd([3, 3, 2, 1], [3], [[0, 0]])
    verify_space_to_batch_nd([2, 2, 4, 1], [2, 2], [[0, 0], [2, 0]])


def test_repeat():
    def verify_repeat(dshape, repeats, axis):
        x_data = np.random.uniform(size=dshape).astype("float32")

        x_var = relay.var("x", relay.TensorType(dshape, "float32"))
        expr = relay.repeat(x_var, repeats, axis)
        mod = tvm.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        aipu_testing.compare_relay_opt_float_result(mod, x_data)

    # compass tile only support 2, 3, 4 dims.
    verify_repeat((3, 10), 2, -1)
    verify_repeat((3, 2, 4), 3, 1)
    verify_repeat((3, 2, 4, 5), 2, 0)


def test_stack():
    def verify_stack(dshape, axis):
        inputs = []
        args = []
        for shape in dshape:
            inputs.append(np.random.uniform(size=shape).astype("float32"))
            args.append(relay.var("x", relay.TensorType(shape, "float32")))
        expr = relay.stack(args, axis)
        mod = tvm.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        aipu_testing.compare_relay_opt_float_result(mod, *inputs)

    verify_stack([(2,), (2,), (2,)], -1)
    verify_stack([(2,), (2,), (2,)], 0)
    verify_stack([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1)
    verify_stack([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1)
    verify_stack([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], 4)


def test_relay_sort():
    def verify_sort(data_shape, axis, is_ascend):
        data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
        output = relay.sort(data, axis, is_ascend)
        mod = tvm.IRModule.from_expr(output)
        input_data = np.random.random(data_shape).astype("float32")

        aipu_testing.compare_relay_opt_float_result(mod, input_data)

    verify_sort([1, 128, 128, 32], 3, True)
    verify_sort([1, 32, 128, 128], 1, False)


def test_reverse():
    def verify(dshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        output = relay.reverse(x, axis=axis)
        mod = tvm.IRModule.from_expr(output)
        input_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        aipu_testing.compare_relay_opt_float_result(mod, input_data)

    verify((2, 3, 4), 1)
    verify((4, 7), 0)
    verify((2, 3, 4), -1)
    verify((4,), 0)


def test_reverse_sequence():
    def verify_reverse_sequence(x_data, seq_lengths, batch_axis, seq_axis):
        seq_lengths_data = np.array(seq_lengths).astype("int32")
        x = relay.var("x", relay.TensorType(x_data.shape, str(x_data.dtype)))
        z = relay.reverse_sequence(x, relay.const(seq_lengths_data), seq_axis, batch_axis)
        func = relay.Function([x], z)
        mod = tvm.IRModule.from_expr(func)
        aipu_testing.compare_relay_opt_float_result(mod, x_data)

    indata = np.array(np.arange(0, 18)).reshape([2, 3, 3]).astype("int32")
    verify_reverse_sequence(indata, [2, 3, 2], 2, 1)


def test_where():
    def verify_where(con_shape, x_shape, y_shape):
        con = relay.var("con", relay.ty.TensorType(con_shape, "bool"))
        x = relay.var("x", relay.ty.TensorType(x_shape, "float32"))
        y = relay.var("y", relay.ty.TensorType(y_shape, "float32"))
        output = relay.where(con, x, y)
        mod = tvm.IRModule.from_expr(output)

        dtype = "float32"
        cond_np = np.random.uniform(low=-1, high=1, size=con_shape) > 0
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        y_np = np.random.uniform(size=y_shape).astype(dtype)

        aipu_testing.compare_relay_opt_float_result(mod, cond_np, x_np, y_np)

    verify_where((3, 4), (3, 4), (3, 4))
    verify_where((3, 4, 5), (3, 4, 5), (3, 4, 5))
    verify_where((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5))


def test_take():
    def verify(src_shape, indices_src, axis, mode, indices_dtype):
        src_dtype = "float32"
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x = relay.var("x", relay.TensorType(src_shape, src_dtype))
        indices = relay.var("indices", relay.TensorType(indices_src.shape, indices_dtype))
        z = relay.take(x, indices, axis=axis, mode=mode)
        mod = tvm.IRModule.from_expr(z)

        x_data = np.random.uniform(low=-1, high=1, size=src_shape).astype(src_dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_data, indices_src)

    verify((4,), [1], 0, "clip", "int32")
    verify((2, 2), [[[1, 0], [0, 1]]], 0, "clip", "int32")
    verify((2, 2), [[[1, 0], [0, 1]]], 1, "clip", "int32")
    verify((4, 3, 5, 6), [[2, 1, 0, 0]], -2, "clip", "int32")
    verify((3, 4), [-1, 2], 0, "wrap", "int32")
    verify((3, 4), [-1, 2], 1, "wrap", "int32")
    verify((3, 4), [0, 2], 0, "fast", "int32")
    verify((3, 4), [0, 2], 1, "fast", "int32")
    verify((3, 4), [1, 2], 1, "clip", "uint32")
    verify((3, 4), [1, 2], 1, "wrap", "uint16")
    verify((3, 4), [0, 2], 0, "fast", "uint8")
    verify((2, 3, 4, 5), [[2, 1], [0, 1]], None, "clip", "int32")


def test_squeeze():
    def verify(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        squeeze = relay.squeeze(x, axis=axis)
        mod = tvm.IRModule.from_expr(squeeze)
        data = np.random.random_sample(shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, data)

    verify((1, 3, 2, 5), "float32", None)
    verify((1, 3, 1), "float32", [0])
    verify((1, 2, 1, 2, 1), "float32", [0, 2])


def test_concat():
    def verify(x_shape, y_shape, axis):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        y = relay.var("y", shape=y_shape, dtype=dtype)
        output = relay.concatenate((x, y), axis=axis)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        y_np = np.random.uniform(size=y_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np, y_np)

    verify((10, 5), (10, 5), 0)
    verify((10, 5), (10, 5), 1)
    verify((10, 5, 5), (10, 5, 5), 0)
    verify((10, 5, 5), (10, 5, 5), 1)
    verify((10, 5, 5), (10, 5, 5), -3)
    verify((10, 5, 5, 7), (10, 5, 5, 7), -1)
    verify((10, 5, 5, 7), (10, 5, 5, 7), -2)
    verify((10, 5, 5, 7), (10, 5, 5, 7), 2)


@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[2, 3, 4, 5], []]),
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ]
        ),
        ([[4, 5], [2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]]),
        ([[3, 4, 5], [2, 1, 1, 1]]),
        ([[5, 10], [5, 10]]),
        ([[5, 10, 2], [5, 1, 1]]),
        ([[5, 3, 4, 6], [3, 4, 1]]),
    ],
)
def test_power(input_shapes):
    dtype = "float32"
    x_shape, y_shape = input_shapes
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.var("y", shape=y_shape, dtype=dtype)
    output = relay.power(x, y)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.uniform(size=x_shape).astype(dtype)
    y_np = np.random.uniform(size=y_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np, y_np)


def test_dense():
    dtype = "float32"
    x_shape, y_shape = (10, 5), (2, 5)
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.const(np.random.rand(*y_shape).astype(dtype), dtype)
    output = relay.nn.dense(x, y)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.rand(*x_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np)


def test_dense2matmul():
    dtype = "float32"
    x_shape, y_shape = (10, 5), (2, 5)
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.var("y", shape=y_shape, dtype=dtype)
    output = relay.nn.dense(x, y)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.rand(*x_shape).astype(dtype)
    y_np = np.random.rand(*y_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np, y_np)


@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[1, 3, 2, 2], [2]]),
        ([[2, 3, 4], [4]]),
        ([[4, 5], [5]]),
    ],
)
def test_prelu(input_shapes):
    dtype = "float32"
    x_shape, y_shape = input_shapes
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.const(np.random.uniform(size=y_shape).astype(dtype), dtype)
    axis = len(x_shape) - 1
    output = relay.nn.prelu(x, y, axis)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.uniform(size=x_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np)


@pytest.mark.parametrize(
    "input_shape, alpha",
    [
        ([1, 3, 2, 2], 0.01),
        ([2, 3, 4], 0.02),
        ([4, 5], 0.3),
    ],
)
def test_leaky_relu(input_shape, alpha):
    dtype = "float32"
    x = relay.var("x", shape=input_shape, dtype=dtype)
    output = relay.nn.leaky_relu(x, alpha)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.uniform(size=input_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np)


@pytest.mark.parametrize(
    "data_shape, pad_width, mode",
    [
        ((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), "SYMMETRIC"),
        ((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), "REFLECT"),
        ((1, 256), ((0, 0), (1, 1)), "SYMMETRIC"),
        ((1, 256, 232), ((0, 0), (2, 2), (15, 15)), "SYMMETRIC"),
        ((1, 64, 10, 20, 30), ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), "SYMMETRIC"),
    ],
)
def test_mirror_pad(data_shape, pad_width, mode):
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.mirror_pad(data, pad_width, mode)
    mod = tvm.IRModule.from_expr(y)
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data_np)


@pytest.mark.parametrize(
    "data_shape, pad_width, pad_value, mode",
    [
        ((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), 0, "constant"),
        ((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), 0, "reflect"),
        ((1, 256), ((0, 0), (1, 1)), 2, "constant"),
        ((1, 256, 232), ((0, 0), (2, 2), (15, 15)), 3, "constant"),
        ((1, 64, 10, 20, 30), ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), 3.0, "constant"),
    ],
)
def test_pad(data_shape, pad_width, pad_value, mode):
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.pad(data, pad_width, pad_value, mode)
    mod = tvm.IRModule.from_expr(y)
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data_np)


def test_slice_like():
    def verify_slice_like(data, slice_like, axes):
        dtype = "float32"
        x = relay.var("data", relay.TensorType(data, dtype))
        y = relay.var("slice_like", relay.TensorType(slice_like, dtype))
        z = relay.slice_like(x, y, axes)
        mod = tvm.IRModule.from_expr(z)

        x_data = np.random.uniform(size=data).astype(dtype)
        y_data = np.random.uniform(size=slice_like).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_data, y_data)

    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=None)
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2), axes=None)
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(1, 2))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(-1, -3))
    verify_slice_like(data=(1, 3, 224, 224), slice_like=(1, 3, 112, 112), axes=(2, 3))


@pytest.mark.parametrize("method", ["any", "all"])
@pytest.mark.parametrize("data_shape", [[2, 3, 4, 5]])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, -1, -2, -3, -4, (0, 1), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("exclude", [False])
def test_any_all(method, data_shape, axis, keepdims, exclude):
    dtype = "bool"
    data = relay.var("data", relay.TensorType(data_shape, dtype))
    funcs = {
        "any": relay.any,
        "all": relay.all,
    }
    out = funcs[method](data, axis, keepdims, exclude)
    mod = tvm.IRModule.from_expr(out)

    data_input = np.random.choice([True, False], size=data_shape)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, compare=np_all_close)


def test_adaptive_avg_pool1d():
    def verify(dshape, out_size, layout="NWC"):
        dtype = "float32"
        x = relay.var("x", shape=[tvm.tir.IntImm("int32", x) for x in dshape], dtype=dtype)
        y = relay.nn.adaptive_avg_pool1d(x, out_size, layout)
        mod = tvm.IRModule.from_expr(y)
        np_data = np.random.uniform(low=0, high=255, size=dshape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, np_data)

    verify((8, 49, 768), (1), "NWC")
    verify((8, 768, 49), (1), "NCW")
    verify((1, 5, 97), (96), "NCW")
    verify((1, 97, 5), (96), "NWC")


def test_avg_pool2d():
    def verify(x_shape, pool_size, strides, dilation, padding):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.nn.avg_pool2d(x, pool_size, strides, dilation, padding)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np)

    verify((1, 96, 111, 111), (1, 1), (1, 1), (1, 1), (-1, -1, 1, 1))
    verify((1, 96, 111, 111), (1, 1), (2, 2), (1, 1), (1, 1, -1, -1))
    verify((1, 96, 111, 111), (3, 3), (2, 2), (1, 1), (3, 3, 3, 3))
    verify((1, 96, 111, 111), (3, 3), (2, 2), (1, 1), (-1, -1, -1, -1))


def test_global_avg_pool2d():
    def verify(x_shape, layout="NCHW"):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.nn.global_avg_pool2d(x, layout=layout)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np)

    verify((1, 3, 65, 65))
    verify((1, 65, 65, 3), "NHWC")
    verify((1, 3, 224, 224))
    verify((1, 224, 224, 3), "NHWC")


def test_logsumexp():
    def verify(x_shape, axis=None, keepdims=False):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.logsumexp(x, axis, keepdims)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np)

    verify((1, 3))
    verify((1, 65, 65), 2)
    verify((1, 3, 224, 224), (2, 3))
    verify((1, 224, 224, 3), (1, 2, 3), True)


def test_trunc():
    def verify(x_shape):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.trunc(x)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(-5, 5, size=x_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np, compare=np_all_close)

    verify((1, 3))
    verify((1, 65, 65))
    verify((1, 3, 224, 224))
    verify((1, 224, 224, 3))


@pytest.mark.parametrize(
    "data_shape, weight_shape, transpose_a, transpose_b",
    [
        ((10, 5), (5, 2), False, False),
        ((5, 10), (5, 2), True, False),
        ((10, 5), (2, 5), False, True),
        ((5, 10), (2, 5), True, True),
    ],
)
def test_matmul(data_shape, weight_shape, transpose_a, transpose_b):
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.const(np.random.rand(*weight_shape).astype(dtype), dtype)
    output = relay.nn.matmul(data, weight, transpose_a=transpose_a, transpose_b=transpose_b)
    mod = tvm.IRModule.from_expr(output)
    data_np = np.random.rand(*data_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data_np)


def test_meshgrid():
    def verify(shapes, indexing):
        dtype = "float32"
        inps = []
        datas = []
        for i, shape in enumerate(shapes):
            x = relay.var("x_" + str(i), shape=shape, dtype=dtype)
            inps.append(x)
            x_np = np.random.uniform(-5, 5, size=shape).astype(dtype)
            datas.append(x_np)
        reshapes = []
        for inp in inps:
            reshapes.append(relay.reshape(inp, [-1]))
        output = relay.meshgrid(reshapes, indexing=indexing)
        output = relay.Tuple(list(output))
        mod = tvm.IRModule.from_expr(output)
        aipu_testing.compare_relay_opt_float_result(mod, *datas, compare=np_all_close)

    verify(((3, 1), (5, 1)), "ij")
    verify(((3, 1), (5, 1)), "xy")


def test_dilate():
    def verify(x_shape, strides, dilation_value):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        reshape = relay.reshape(x, x_shape[:-1])
        output = relay.nn.dilate(reshape, strides, dilation_value)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.uniform(-5, 5, size=x_shape).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np, compare=np_all_close)

    verify((1, 10, 10, 3, 1), [1, 3, 3, 1], -1.0)
    verify((1, 300, 300, 3, 1), [1, 2, 2, 1], 1.23)


@pytest.mark.parametrize("method", ["and", "xor", "or"])
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int16", "uint16", "int8", "uint8"])
@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[2, 3, 4, 5], []]),
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ]
        ),
        ([[4, 5], [2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]]),
        ([[3, 4, 5], [2, 1, 1, 1]]),
        ([[5, 10], [5, 10]]),
        ([[5, 10, 2], [5, 1, 1]]),
        ([[5, 3, 4, 6], [3, 4, 1]]),
    ],
)
def test_bitwise(method, dtype, input_shapes):
    x_shape, y_shape = input_shapes
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.var("y", shape=y_shape, dtype=dtype)
    funcs = {
        "and": relay.bitwise_and,
        "xor": relay.bitwise_xor,
        "or": relay.bitwise_or,
    }
    output = funcs[method](x, y)
    mod = tvm.IRModule.from_expr(output)
    x_np = np.random.choice([0, 1], size=tuple(x_shape)).astype(dtype)
    y_np = np.random.choice([0, 1], size=tuple(y_shape)).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, x_np, y_np, compare=np_all_close)


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int16", "uint16", "int8", "uint8"])
def test_bitwise_not(dtype):
    def verify(x_shape):
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.bitwise_not(x)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.choice([0, 1], size=tuple(x_shape)).astype(dtype)
        aipu_testing.compare_relay_opt_float_result(mod, x_np, compare=np_all_close)

    verify((1, 10))
    verify((1, 65, 65))
    verify((1, 3, 224, 224))
    verify((1, 224, 224, 3))
    verify((1, 224, 224, 3, 4))


def test_cumulate():
    def verify(
        x_shape, axis=0, exclusive=False, relay_cumulate_method=relay.cumsum, dtype="float32"
    ):
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay_cumulate_method(x, axis, dtype, exclusive)
        mod = tvm.IRModule.from_expr(output)

        num_elem = np.prod(x_shape)
        x_np = np.array(list(range(num_elem))).astype(dtype)
        x_np = x_np.reshape(x_shape)

        aipu_testing.compare_relay_opt_float_result(mod, x_np, compare_threshold=0.94)

    verify((2, 3), 0, True, relay.cumsum)
    verify((2, 3), 0, False, relay.cumprod)


def test_depth_to_space():
    def verify(x_shape, block_size, layout, mode):
        dtype = "float32"
        x = relay.var("x", shape=x_shape, dtype=dtype)
        output = relay.nn.depth_to_space(x, block_size, layout, mode)
        mod = tvm.IRModule.from_expr(output)
        x_np = np.random.randint(-128, 128, x_shape).astype(dtype)

        aipu_testing.compare_relay_opt_float_result(mod, x_np)

    verify((1, 16, 4, 4), 2, "NCHW", "DCR")
    verify((1, 3, 3, 27), 3, "NHWC", "CRD")


def test_deformable_conv2d():
    def verify(data_shape, strides, kernel_size, out_c, layout="NHWC"):
        # params
        dtype = "float32"
        if layout == "NHWC":
            n, in_h, in_w, in_c = data_shape
            out_h = (in_h - kernel_size[0]) // strides[0] + 1
            out_w = (in_w - kernel_size[1]) // strides[1] + 1
            offset_shape = (n, out_h, out_w, kernel_size[0] * kernel_size[1] * 2)
            weight_shape = (kernel_size[0], kernel_size[1], in_c, out_c)
        else:
            n, in_c, in_h, in_w = data_shape
            out_h = (in_h - kernel_size[0]) // strides[0] + 1
            out_w = (in_w - kernel_size[1]) // strides[1] + 1
            offset_shape = (n, kernel_size[0] * kernel_size[1] * 2, out_h, out_w)
            weight_shape = (out_c, in_c, kernel_size[0], kernel_size[1])
        # graph
        data = relay.var("data", shape=data_shape, dtype=dtype)
        offset = relay.var("offset", shape=offset_shape, dtype=dtype)
        weight_np = np.random.uniform(-1, 1, size=weight_shape).astype(dtype)
        weight = relay.const(weight_np, dtype=dtype)
        if layout == "NHWC":
            out = relay.nn.deformable_conv2d(
                data,
                offset,
                weight,
                strides=strides,
                padding=(0, 0),
                dilation=(1, 1),
                deformable_groups=1,
                groups=1,
                channels=out_c,
                kernel_size=kernel_size,
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
        else:
            out = relay.nn.deformable_conv2d(
                data,
                offset,
                weight,
                strides=strides,
                padding=(0, 0),
                dilation=(1, 1),
                deformable_groups=1,
                groups=1,
                channels=out_c,
                kernel_size=kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
        mod = tvm.IRModule.from_expr(out)
        # data
        np.random.seed(0)
        data_np = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
        offset_np = np.random.uniform(-1, 1, size=offset_shape).astype(dtype)
        # run
        aipu_testing.compare_relay_opt_float_result(mod, data_np, offset_np, compare_threshold=0.90)

    verify(data_shape=(1, 16, 16, 4), strides=(1, 1), kernel_size=(3, 3), out_c=4, layout="NHWC")
    verify(data_shape=(1, 4, 16, 16), strides=(1, 1), kernel_size=(3, 3), out_c=4, layout="NCHW")


def test_deformable_conv2d_v2():
    def verify(data_shape, strides, kernel_size, out_c, layout="NHWC", kernel_layout="HWIO"):
        dtype = "float32"
        if layout == "NHWC":
            n, in_h, in_w, in_c = data_shape
            OH = (in_h - kernel_size[0]) // strides[0] + 1
            OW = (in_w - kernel_size[1]) // strides[1] + 1
            offset_shape = (n, OH, OW, kernel_size[0] * kernel_size[1] * 2)
            mask_shape = (n, OH, OW, kernel_size[0] * kernel_size[1])
            weight_shape = (kernel_size[0], kernel_size[1], in_c, out_c)
        else:
            n, in_c, in_h, in_w = data_shape
            OH = (in_h - kernel_size[0]) // strides[0] + 1
            OW = (in_w - kernel_size[1]) // strides[1] + 1
            offset_shape = (n, kernel_size[0] * kernel_size[1] * 2, OH, OW)
            mask_shape = (n, kernel_size[0] * kernel_size[1], OH, OW)
            weight_shape = (out_c, in_c, kernel_size[0], kernel_size[1])

        data = relay.var("data", shape=data_shape, dtype=dtype)
        offset = relay.var("offset", shape=offset_shape, dtype=dtype)
        weight_np = np.random.uniform(size=weight_shape).astype(dtype)
        weight = relay.const(tvm.nd.array(weight_np))
        mask_np = np.random.uniform(size=mask_shape).astype(dtype)
        mask = relay.const(tvm.nd.array(mask_np))

        expr = relay.op.contrib.aipu_compass.deformable_conv2d_v2(
            data,
            offset,
            weight,
            mask,
            strides=strides,
            channels=out_c,
            kernel_size=(kernel_size[0], kernel_size[1]),
            data_layout=layout,
            kernel_layout=kernel_layout,
        )

        func = relay.Function([data, offset], expr)
        mod = tvm.IRModule.from_expr(func)
        np.random.seed(0)
        data_np = np.random.uniform(size=data_shape).astype(dtype)
        offset_np = np.random.uniform(size=offset_shape).astype(dtype)

        golden_res = None
        if layout == "NCHW":
            mod_aft = relay.transform.InferType()(mod)
            desired_layouts = {"contrib.aipu_compass.deformable_conv2d_v2": ["NHWC", "HWIO"]}
            mod_aft = relay.transform.ConvertLayout(desired_layouts)(mod_aft)
            golden_res = relay.create_executor(mod=mod_aft).evaluate()(data_np, offset_np)

        aipu_testing.compare_relay_opt_float_result(mod, data_np, offset_np, golden=golden_res)

    verify(
        data_shape=(1, 16, 16, 4),
        strides=(1, 1),
        kernel_size=(3, 3),
        out_c=4,
        layout="NHWC",
        kernel_layout="HWIO",
    )
    verify(
        data_shape=(1, 4, 16, 16),
        strides=(1, 1),
        kernel_size=(3, 3),
        out_c=4,
        layout="NCHW",
        kernel_layout="OIHW",
    )


@pytest.mark.parametrize(
    "data_shape, axes",
    [
        ((2, 3), (1, 0)),
        ((1, 2, 3), None),
        ((2, 3, 4, 5), (0, 2, 1, 3)),
        ((2, 3, 4, 5, 6), (0, 4, 3, 1, 2)),
        ((2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5)),
    ],
)
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "float32"])
def test_relay_transpose(data_shape, axes, dtype):
    data = relay.var("data", relay.TensorType(data_shape, dtype))
    x_out = relay.transpose(data, axes)
    mod = tvm.IRModule.from_expr(x_out)
    data_input = np.random.randint(0, 255, size=data_shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, compare=np_all_close)


@pytest.mark.parametrize(
    "shape, begin, end, strides",
    [
        ((1, 100, 91), (0, 0, 1), (-1, -1, -1), (1, 1, 1)),
        ((1, 100, 91), (0, 90, 50), (-1, 30, -1), (1, 3, 5)),
        ((1, 100, 91), (0, 10, 20), (-1, 50, 30), (1, 10, 20)),
    ],
)
@pytest.mark.parametrize("slice_mode", ["size"])
def test_relay_strided_slice(shape, begin, end, strides, slice_mode):
    dtype = "float32"
    data = relay.var("x0", shape=shape, dtype=dtype)
    x_out = relay.strided_slice(data, begin, end, strides, slice_mode=slice_mode)
    mod = tvm.IRModule.from_expr(x_out)
    data_input = np.random.randint(0, 255, size=shape).astype(dtype)
    aipu_testing.compare_relay_opt_float_result(mod, data_input, compare=np_all_close)


if __name__ == "__main__":
    test_relay_grid_sample()
    test_relay_layout_transform_4d_to_5d()
    test_relay_layout_transform_5d_to_4d()
    test_relay_copy()
    test_relay_non_max_suppression()
    test_relay_roi_align()
    test_relay_roi_pool()
    test_batch_flatten()
    test_batch_to_space_nd()
    test_relay_instance_norm()
    test_relay_layer_norm()
    test_relay_group_norm()
    test_relay_l2_norm()
    test_relay_topk()
    test_relay_gather()
    test_relay_gather_nd()
    test_relay_scatter_nd()
    test_relay_expand_dims()
    test_relay_arange()
    test_relay_sqrt([1, 3, 2, 2])
    test_relay_rsqrt()
    test_relay_sign()
    test_relay_sin()
    test_relay_tan()
    test_relay_erf()
    test_relay_crop_and_resize()
    test_relay_cast_like_2_var()
    test_relay_cast_like_1_var()
    test_relay_full_like()
    test_relay_full(4, (2, 3, 4, 5))
    test_relay_ones()
    test_relay_ones_like()
    test_relay_reshape_like()
    test_relay_broadcast_to()
    test_relay_broadcast_to_like()
    test_relay_zeros()
    test_relay_zeros_like()
    test_space_to_batch_nd()
    test_repeat()
    test_stack()
    test_upsampling((2, 3, 4, 5), 2, 2, "NCHW", "nearest_neighbor", False)
    test_std_variance("std", [2, 3, 4, 5], None, False, False, False)
    test_std_variance("variance", [2, 3, 4, 5], None, False, False, False)
    test_mean_std_variance("mean_std", (2, 3, 4, 5), None, False, False)
    test_mean_std_variance("mean_variance", (2, 3, 4, 5), None, False, False)
    test_relay_sort()
    test_reverse()
    test_reverse_sequence()
    test_where()
    test_take()
    test_squeeze()
    test_concat()
    test_power([[2, 3, 4, 5], []])
    test_dense()
    test_dense2matmul()
    test_prelu([[1, 3, 2, 2], [2]])
    test_leaky_relu([1, 3, 2, 2], 0.01)
    test_mirror_pad((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), "SYMMETRIC")
    test_pad((1, 256, 232, 232), ((0, 0), (0, 0), (2, 2), (16, 16)), 0, "constant")
    test_slice_like()
    test_any_all("any", [2, 3, 4, 5], None, True, False)
    test_any_all("all", [2, 3, 4, 5], None, True, False)
    test_adaptive_avg_pool1d()
    test_avg_pool2d()
    test_global_avg_pool2d()
    test_logsumexp()
    test_trunc()
    test_matmul((10, 5), (5, 2), False, False)
    test_meshgrid()
    test_dilate()
    test_bitwise("and", "uint8", [[4, 5], [2, 3, 4, 5]])
    test_bitwise_not("uint8")
    test_cumulate()
    test_depth_to_space()
    test_relay_get_valid_counts()
    test_multibox_transform_loc()
    test_deformable_conv2d()
    test_deformable_conv2d_v2()
    test_relay_transpose((1, 2, 3), None, "int8")
    test_relay_strided_slice([1, 100, 91], [0, 0, 0], [1, -1, -1], [1, 1, 1], "end")
