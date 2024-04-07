# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import textwrap
from tvm import ir, relay
from tvm.relay.op import op
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass.codegen import CodeGenAipuCompass
from tvm.relay.op.contrib.aipu_compass import codegen_plugin_register, parser_plugin_register


def custom_topk_rel(arg_types, attrs):
    assert len(arg_types) == 1

    shape = arg_types[0].shape
    axis = attrs.axis if attrs.axis >= 0 else attrs.axis + len(shape)
    k = attrs.k
    out_shape = []
    for i, val in enumerate(shape):
        if i != axis:
            out_shape.append(val)
        else:
            if k < 1:
                out_shape.append(val)
            else:
                out_shape.append(k)

    values_tt = relay.TensorType(out_shape, arg_types[0].dtype)
    indices_tt = relay.TensorType(out_shape, attrs.dtype)

    if attrs.ret_type == "both":
        return relay.TupleType([values_tt, indices_tt])
    elif attrs.ret_type == "values":
        return values_tt
    elif attrs.ret_type == "indices":
        return indices_tt
    else:
        raise RuntimeError(f"Unsupported ret type:{attrs.ret_type}")


def register_relay_op_topk(op_name):
    op.register(op_name, "test custom op topk")
    op.get(op_name).set_num_inputs(1)
    op.get(op_name).set_attrs_type_key("DictAttrs")
    op.get(op_name).add_type_rel(op_name, custom_topk_rel)


op_name = "custom_topk"
register_relay_op_topk(op_name)


@parser_plugin_register("tf", "TopKV2")
def custom_topk_converter(inputs, attr, params):
    if len(inputs) != 2:
        raise ValueError("Expect 2 input only")
    axis = attr.get("axis", 1)
    k = inputs[1].data.numpy().item()
    largest = attr.get("largest", 1)
    sorted_ = attr.get("sorted", 1)
    select_index = attr.get("select_index", "random")
    attrs = ir.make_node(
        "DictAttrs",
        axis=axis,
        k=k,
        largest=largest,
        sorted=sorted_,
        select_index=select_index,
        ret_type="both",
        dtype="int64",
    )

    out = relay.Call(op.get(op_name), [inputs[0]], attrs)
    return relay.TupleWrapper(out, 2)


@ir.register_op_attr("custom_topk", "target.aipu_compass")
def _check(custom_topk):
    # Check if it is supported by AIPU Compass.
    return True


@codegen_plugin_register("custom_topk")
def _gen_topk(call):
    op_type = "TopK"
    constants = dict()
    attrs = call.attrs
    attr_text = textwrap.dedent(
        f"""
        axis={attrs.axis}
        k={attrs.k}
        largest={bool(attrs.largest)}
        sorted={bool(attrs.sorted)}
        select_index={attrs.select_index}
        """
    )
    return (op_type, call.args, constants, attr_text)


gt_relay_ir = """fn (%Placeholder: Tensor[(6, 4, 10, 3), float32] /* ty=Tensor[(6, 4, 10, 3), float32] span=Placeholder:0:0 */) -> (Tensor[(6, 2, 10, 3), float32], Tensor[(6, 2, 10, 3), int64]) {
  %0 = custom_topk(%Placeholder, __dict__={"axis"=1, "select_index"="random", "dtype"="int64", "largest"=1, "sorted"=1, "ret_type"="both", "k"=2}) /* ty=(Tensor[(6, 2, 10, 3), float32], Tensor[(6, 2, 10, 3), int64]) span=TopKV2:0:0 */;
  %1 = %0.0 /* ty=Tensor[(6, 2, 10, 3), float32] span=TopKV2:0:0 */;
  %2 = %0.1 /* ty=Tensor[(6, 2, 10, 3), int64] span=TopKV2:0:0 */;
  (%1, %2) /* ty=(Tensor[(6, 2, 10, 3), float32], Tensor[(6, 2, 10, 3), int64]) */
} /* ty=fn (Tensor[(6, 4, 10, 3), float32]) -> (Tensor[(6, 2, 10, 3), float32], Tensor[(6, 2, 10, 3), int64]) */"""

gt_topk_layer = """layer_id=1
layer_name=1_topk
layer_type=TopK
layer_bottom=[Placeholder]
layer_bottom_shape=[[6, 4, 10, 3]]
layer_bottom_type=[float32]
layer_top=[temp_var_0, temp_var_1]
layer_top_shape=[[6, 2, 10, 3], [6, 2, 10, 3]]
layer_top_type=[float32, int64]
axis=1
k=2
largest=True
sorted=True
select_index=random
"""


def gen_topk_model(path):
    import tensorflow as tf

    try:
        # Package "tf.compat.v1" is added from version "r1.13".
        tf_compat_v1 = tf.compat.v1  # pylint: disable=invalid-name
    except AttributeError:
        tf_compat_v1 = tf  # pylint: disable=invalid-name

    g = tf.Graph()
    with g.as_default():
        inp = tf_compat_v1.placeholder(tf.float32, shape=[6, 4, 10, 3])
        tf.math.top_k(inp, k=2, sorted=True)
    with tf_compat_v1.Session(graph=g) as sess:
        sess.run(tf_compat_v1.global_variables_initializer())
        output_nodes = ["TopKV2"]
        frozen_graph_def = tf_compat_v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_nodes
        )
        with open(path, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())


def test_custom_op_plugin():
    gen_topk_model("./TopK.pb")
    cfg = """
    [Parser]
    model_type = tensorflow
    model_name = topk
    input_model = ./TopK.pb
    input = Placeholder_0
    input_shape = [6,4,10,3]
    """
    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)

    # 2. Parse model and get compass ir.
    compass.parse()
    mod = relay.transform.InferType()(compass.ir_mod)
    compass_ir, _ = CodeGenAipuCompass().gen(mod["main"])

    # 3. Check relay ir and topk layer in compass ir.
    assert gt_relay_ir == str(mod["main"])
    assert gt_topk_layer == compass_ir.split("\n\n")[2]


if __name__ == "__main__":
    test_custom_op_plugin()
