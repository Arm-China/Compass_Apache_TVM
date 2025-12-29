# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm import ir, relax
from tvm.compass.relax import Compass
from tvm.compass.relax.codegen import CodeGenCompass
from tvm.compass.relax.op import codegen_plugin_register, parser_plugin_register


def custom_topk_rel(call: relax.Call, context):
    attrs = call.attrs
    assert len(call.args) == 1

    shape = call.args[0].struct_info.shape
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

    values_tt = relax.TensorStructInfo(out_shape, call.args[0].struct_info.dtype)
    indices_tt = relax.TensorStructInfo(out_shape, attrs.dtype)

    if attrs.ret_type == "both":
        return relax.TupleStructInfo([values_tt, indices_tt])
    elif attrs.ret_type == "values":
        return values_tt
    elif attrs.ret_type == "indices":
        return indices_tt
    else:
        raise RuntimeError(f"Unsupported ret type:{attrs.ret_type}")


def register_relax_op_topk(op_name):
    ir.register_op_attr(op_name, "FInferStructInfo", custom_topk_rel)
    ir.register_op_attr(op_name, "FPurity", True)
    op = ir.Op.get(op_name)
    op.set_num_inputs(1)
    op.set_attrs_type_key("DictAttrs")


OP_NAME = "custom_topk"
register_relax_op_topk(OP_NAME)
codegen_plugin_register(OP_NAME, "TopK")


@parser_plugin_register("tf", "TopKV2")
def custom_topk_converter(inputs, attr, params, bb):
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
        largest=bool(largest),
        sorted=bool(sorted_),
        select_index=select_index,
        ret_type="both",
        dtype="int64",
    )

    return relax.Call(ir.Op.get(OP_NAME), [inputs[0]], attrs)


gt_relax_ir = """# from tvm.script import relax as R

@R.function
def main(Placeholder: R.Tensor((6, 4, 10, 3), dtype="float32")) -> R.Tuple(R.Tensor((6, 2, 10, 3), dtype="float32"), R.Tensor((6, 2, 10, 3), dtype="int64")):
    R.func_attr({"num_input": 1})
    with R.dataflow():
        lv: R.Tuple(R.Tensor((6, 2, 10, 3), dtype="float32"), R.Tensor((6, 2, 10, 3), dtype="int64")) = custom_topk(Placeholder, axis=1, dtype="int64", k=2, largest=True, ret_type="both", select_index="random", sorted=True)
        lv1: R.Tensor((6, 2, 10, 3), dtype="float32") = lv[0]
        lv2: R.Tensor((6, 2, 10, 3), dtype="int64") = lv[1]
        gv3: R.Tuple(R.Tensor((6, 2, 10, 3), dtype="float32"), R.Tensor((6, 2, 10, 3), dtype="int64")) = lv1, lv2
        R.output(gv3)
    return gv3"""

gt_topk_layer = """layer_id=1
layer_name=TopK_1
layer_type=TopK
layer_bottom=[Placeholder]
layer_bottom_shape=[[6,4,10,3]]
layer_bottom_type=[float32]
layer_top=[TopK_1_out_1,TopK_1_out_2]
layer_top_shape=[[6,2,10,3],[6,2,10,3]]
layer_top_type=[float32,int64]
layer_top_scale=[1.0,1.0]
layer_top_zp=[0,0]
axis=1
dtype=int64
k=2
largest=true
ret_type=both
select_index=random
sorted=true"""


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
        frozen_graph_def = tf_compat_v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_nodes)
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
    # 1. Create Compass instance and set configurations.
    compass = Compass(cfg)

    # 2. Parse model and get compass ir.
    compass.parse()
    mod = compass.ir_mod
    compass_ir, _ = CodeGenCompass().gen(mod["main"])

    # 3. Check relax ir and topk layer in compass ir.
    assert gt_relax_ir == str(mod["main"])
    assert gt_topk_layer == compass_ir.split("\n\n")[2]


if __name__ == "__main__":
    test_custom_op_plugin()
