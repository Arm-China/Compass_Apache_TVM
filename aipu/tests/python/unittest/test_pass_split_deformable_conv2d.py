# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform
from tvm.relay import testing as relay_test
from tvm.relay.backend.contrib.aipu_compass import utils


# Pytest Specific Function
def test_split_deformable_conv2d():
    # Param
    layout = "NHWC"
    kernel_layout = "HWIO"
    dtype = "float32"
    dshape = (1, 16, 16, 4)
    kshape = (3, 3, 4, 4)  # HWIO

    strides = (1, 1)
    OH = (dshape[1] - kshape[0]) // strides[0] + 1
    OW = (dshape[2] - kshape[1]) // strides[1] + 1
    offset_shape = (1, OH, OW, kshape[0] * kshape[1] * 2)
    # Graph
    data = relay.var("data", shape=dshape, dtype=dtype)
    offset = relay.var("offset", shape=offset_shape, dtype=dtype)
    kernel_np = np.random.uniform(size=kshape).astype(dtype)
    weight = relay.const(kernel_np, dtype=dtype)

    def before():
        expr = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(kshape[0], kshape[1]),
            strides=strides,
            channels=kshape[3],
            data_layout=layout,
            kernel_layout=kernel_layout,
        )
        return relay.Function(relay.analysis.free_vars(expr), expr)

    # data
    data_np = np.random.uniform(size=dshape).astype(dtype)
    offset_np = np.random.uniform(size=offset_shape).astype(dtype)
    compass_transform.split_deformable_conv2d.gen_offset_base(
        dshape[1], dshape[2], OH, OW, kshape[0], kshape[1], strides
    )
    inputs = [data_np, offset_np]

    # build and compare
    func_bef = before()
    mod_bef = tvm.IRModule.from_expr(func_bef)
    mod_bef = tvm.relay.transform.InferType()(mod_bef)
    ret_bef = relay.create_executor(mod=mod_bef).evaluate()(*inputs).numpy()
    func_aft = relay_test.run_opt_pass(before(), compass_transform.SplitDeformableConv2d())
    func_aft = relay_test.run_opt_pass(
        func_aft, relay.transform.ConvertLayout(utils.X86_DESIRED_LAYOUTS)
    )
    mod_aft = tvm.IRModule.from_expr(func_aft)
    mod_aft = tvm.relay.transform.InferType()(mod_aft)
    ret_aft = relay.create_executor(mod=mod_aft).evaluate()(*inputs).numpy()
    tvm.testing.assert_allclose(ret_aft, ret_bef, rtol=1e-5, atol=1e-5)


# Pytest Specific Function
def test_split_deformable_conv2d_v2():
    layout = "NHWC"
    kernel_layout = "HWIO"
    dtype = "float32"
    dshape = (1, 16, 16, 4)
    kshape = (3, 3, 4, 4)  # HWIO

    strides = (1, 1)
    OH = (dshape[1] - kshape[0]) // strides[0] + 1
    OW = (dshape[2] - kshape[1]) // strides[1] + 1
    offset_shape = (1, OH, OW, kshape[0] * kshape[1] * 2)
    mask_shape = (1, OH, OW, kshape[0] * kshape[1])

    data = relay.var("data", shape=dshape, dtype=dtype)
    offset = relay.var("offset", shape=offset_shape, dtype=dtype)

    data_np = np.random.uniform(size=dshape).astype(dtype)
    offset_np = np.random.uniform(size=offset_shape).astype(dtype)
    mask_np = np.random.uniform(size=mask_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kshape).astype(dtype)

    weight = relay.const(tvm.nd.array(kernel_np))
    mask = relay.const(tvm.nd.array(mask_np))
    inputs = [data_np, offset_np]
    args = [data, offset]

    expr = relay.op.contrib.aipu_compass.deformable_conv2d_v2(
        data,
        offset,
        weight,
        mask,
        kernel_size=(kshape[0], kshape[1]),
        strides=strides,
        channels=kshape[3],
        data_layout=layout,
        kernel_layout=kernel_layout,
    )

    func = relay.Function(args, expr)

    mod_bef = tvm.IRModule.from_expr(func)
    mod_bef = tvm.relay.transform.InferType()(mod_bef)
    ret_bef = relay.create_executor(mod=mod_bef).evaluate()(*inputs).numpy()
    func_aft = relay_test.run_opt_pass(func, compass_transform.SplitDeformableConv2d())
    func_aft = relay_test.run_opt_pass(
        func_aft, relay.transform.ConvertLayout(utils.X86_DESIRED_LAYOUTS)
    )
    mod_aft = tvm.IRModule.from_expr(func_aft)
    mod_aft = tvm.relay.transform.InferType()(mod_aft)
    ret_aft = relay.create_executor(mod=mod_aft).evaluate()(*inputs).numpy()
    tvm.testing.assert_allclose(ret_aft, ret_bef, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_split_deformable_conv2d()
    test_split_deformable_conv2d_v2()
