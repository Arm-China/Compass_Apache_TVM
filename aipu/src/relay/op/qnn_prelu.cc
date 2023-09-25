// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file src/relay/qnn/op/prelu.cc
 * \brief QNN prelu operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include "../../../../src/relay/qnn/op/op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnPReluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // Expected Types: data, alpha, input_scale, input_zero_point, alpha_scale, alpha_zero_point,
  // output_scale, output_zero_point, out_type
  ICHECK_EQ(types.size(), 9);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized prelu type(int8, uint8) for input but was " << x->dtype;
  const auto* param = attrs.as<PReluAttrs>();
  ICHECK(param != nullptr) << "PReluAttrs cannot be nullptr.";

  // Check the types of scale and zero points.
  for (size_t i = 2; i < 8; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  ICHECK(IsScalarType(types[2], DataType::Float(32)));  // input_scale
  ICHECK(IsScalarType(types[3], DataType::Int(32)));    // input_zero_point
  ICHECK(IsScalarType(types[4], DataType::Float(32)));  // alpha_scale
  ICHECK(IsScalarType(types[5], DataType::Int(32)));    // alpha_zero_point
  ICHECK(IsScalarType(types[6], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[7], DataType::Int(32)));    // output_zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[2], TensorType({}, DataType::Float(32)));  // input_scale
  reporter->Assign(types[3], TensorType({}, DataType::Int(32)));    // input_zero_point
  reporter->Assign(types[4], TensorType({}, DataType::Float(32)));  // alpha_scale
  reporter->Assign(types[5], TensorType({}, DataType::Int(32)));    // alpha_zero_point
  reporter->Assign(types[6], TensorType({}, DataType::Float(32)));  // output_scale
  reporter->Assign(types[7], TensorType({}, DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[8]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

// Positional relay function to create quantized prelu operator used by frontend FFI.
Expr MakeQuantizedPRelu(Expr x, Expr alpha, Expr input_scale, Expr input_zero_point,
                        Expr alpha_scale, Expr alpha_zero_point, Expr output_scale,
                        Expr output_zero_point, int axis) {
  auto attrs = make_object<PReluAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("qnn.prelu");
  return Call(op,
              {x, alpha, input_scale, input_zero_point, alpha_scale, alpha_zero_point, output_scale,
               output_zero_point},
              Attrs(attrs), {});
}

/*
 * \brief Canonicalizes the QNN prelu op.
 * \param attrs The empty attribute.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for pelu op.
 */
Expr QnnPReluCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                          const Array<tvm::relay::Type>& arg_types) {
  //  PReLU can be written in terms of respective quantized tensors, scales and
  //  zero points as

  //  When Q_i < zp_i:
  //     scale_o * (Q_o - zp_o) = scale_i * (Q_i - zp_i) * scale_alpha * (Q_alpha - zp_alpha) (1)
  //  When Q_i >= zp_i:
  //     scale_o * (Q_o - zp_o) = scale_i * (Q_i - zp_i) (2)

  //  when Q_i < zp_i, Consider the product (Q_i - zp_i) * (Q_alpha - zp_alpha) as a different
  //  quantized tensor of o with Q_o', then (1) becames:

  //     Q_o = (scale_i * scale_alpha) / scale_o * Q_o' + zp_o
  //         = scale' / scale_o * Q_o' + zp_o (3)

  //  when Q_i >= zp_i, Requantize the input tensor to output qnn params.After requantizing Q_i,
  //  equation (2) becames:

  //     Q_o = requantize(Q_i)  when Q_i >= zp_i (4)

  //  Finnally, Q_o could be calculated by equation (3) and equation (4).

  ICHECK_EQ(new_args.size(), 8);
  Expr data = Cast(new_args[0], DataType::Int(32));
  Expr alpha_data = Cast(new_args[1], DataType::Int(32));
  Expr input_scale = new_args[2];
  Expr input_zero_point = Cast(new_args[3], DataType::Int(32));
  Expr alpha_scale = new_args[4];
  Expr alpha_zero_point = Cast(new_args[5], DataType::Int(32));
  Expr output_scale = new_args[6];
  Expr output_zero_point = Cast(new_args[7], DataType::Int(32));

  const auto* q_attrs = attrs.as<PReluAttrs>();
  auto axis = q_attrs->axis;
  ICHECK_EQ(axis, 0);

  const auto input_shape = get_shape(arg_types[0]);
  const auto input_dtype = arg_types[0].as<TensorTypeNode>()->dtype;
  const auto int32_dtype = DataType::Int(32);
  const auto float32_dtype = DataType::Float(32);
  auto zero_scalar = MakeConstantScalar(int32_dtype, 0);
  Expr tmp_data = data;
  if (!IsEqualScalar(input_zero_point, zero_scalar)) {
    tmp_data = Subtract(data, input_zero_point);
  }

  if (!IsEqualScalar(alpha_zero_point, zero_scalar)) {
    alpha_data = Subtract(alpha_data, alpha_zero_point);
  }

  // Create a new tensor Q'
  Expr output_left;
  output_left = Multiply(tmp_data, alpha_data);

  // Get the adjusted new scale and zero points.
  float input_scale_float = GetScalarFromConstant<float>(input_scale);
  float alpha_scale_float = GetScalarFromConstant<float>(alpha_scale);
  float new_scale_float = input_scale_float * alpha_scale_float;
  auto new_input_scale = MakeConstantScalar(float32_dtype, new_scale_float);
  auto new_input_zero_point = zero_scalar;

  // Requantize to get Q_o'
  output_left = Requantize(output_left, input_shape, new_input_scale, new_input_zero_point,
                           output_scale, output_zero_point, int32_dtype);

  // requantize the input to Q_i'
  auto output_right = RequantizeOrUpcast(data, input_scale, input_zero_point, output_scale,
                                         output_zero_point, input_shape);

  auto output = Where(Less(data, input_zero_point), output_left, output_right);

  return ConvertDtype(output, input_dtype);
}

RELAY_REGISTER_OP("qnn.prelu")
    .describe("Prelu for quantized tensors.")
    .set_attrs_type<PReluAttrs>()
    .set_num_inputs(8)
    .add_argument("data", "Quantized Tensor", "The input data.")
    .add_argument("alpha", "Quantized Tensor", "The slop data.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("alpha_scale", "Tensor", "The quantization scale of the alpha tensor.")
    .add_argument("alpha_zero_point", "Tensor", "The quantization zero_point of the alpha tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QPRelu", QnnPReluRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnPReluCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.prelu").set_body_typed(MakeQuantizedPRelu);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
