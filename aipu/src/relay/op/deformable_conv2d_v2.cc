// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/matrix_band_part.cc
 * \brief Operator definitions for MatrixBandPart.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../../../../src/relay/op/op_common.h"
#include "../../../../src/relay/op/type_relations.h"

namespace tvm {
namespace relay {

// Deformable Convolution V2 shape relations.
bool DeformableConv2DV2Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                           const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[2].as<TensorTypeNode>();

  ICHECK(data);
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  auto* param = attrs.as<DeformableConv2DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  if (!trans_in_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d_v2 only support input layouts that are convertible from NCHW."
        << " The provided layout is: " << in_layout);
    return false;
  }

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  if (!trans_kernel_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d_v2 only support kernel layouts that are convertible from OIHW."
        << " The provided layout is: " << kernel_layout);
    return false;
  }

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  if (!trans_out_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d_v2 only support output layouts that are convertible from NCHW."
        << "The provided layout is: " << out_layout);
    return false;
  }

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x, ksize_y, ksize_x;

  // infer weight shape if kernel_size and channels are defiend
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 2);
    ICHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape({param->channels, indexdiv(dshape_nchw[1], param->groups),
                             param->kernel_size[0], param->kernel_size[1]});

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    ksize_y = param->kernel_size[0];
    ksize_x = param->kernel_size[1];
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    // assign result to reporter
    reporter->Assign(types[2], TensorType(wshape, data->dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);

    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
             reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "DeformableConv2DV2: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "DeformableConv2DV2: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    if (!dshape_nchw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(indexdiv(dshape_nchw[1], param->groups), wshape[1]));
    }
    channels = wshape[0];
    ksize_y = wshape[2];
    ksize_x = wshape[3];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y, param->strides[0]) + 1);
  oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x, param->strides[1]) + 1);
  DataType out_dtype = param->out_dtype;

  // infer offset shape
  Array<IndexExpr> offset_shape(
      {dshape_nchw[0], 2 * ksize_y * ksize_x * param->deformable_groups, oshape[2], oshape[3]});
  offset_shape = trans_in_layout.BackwardShape(offset_shape);
  reporter->Assign(types[1], TensorType(offset_shape, data->dtype));
  // infer mask shape
  Array<IndexExpr> mask_shape(
      {dshape_nchw[0], ksize_y * ksize_x * param->deformable_groups, oshape[2], oshape[3]});
  mask_shape = trans_in_layout.BackwardShape(mask_shape);
  reporter->Assign(types[3], TensorType(mask_shape, data->dtype));
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  oshape = trans_out_layout.BackwardShape(oshape);
  reporter->Assign(types[4], TensorType(oshape, out_dtype));
  return true;
}

InferCorrectLayoutOutput DeformableConvV2InferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* params = attrs.as<DeformableConv2DAttrs>();
  return InferCorrectLayoutOutput(
      {params->data_layout, params->data_layout, params->kernel_layout, params->data_layout},
      {params->out_layout == "" ? params->data_layout : params->out_layout}, attrs);
}

Expr MakeDeformableConvV2(Expr data, Expr offset, Expr weight, Expr mask, Array<IndexExpr> strides,
                          Array<IndexExpr> padding, Array<IndexExpr> dilation,
                          int deformable_groups, int groups, IndexExpr channels,
                          Array<IndexExpr> kernel_size, std::string data_layout,
                          std::string kernel_layout, std::string out_layout, DataType out_dtype,
                          std::string op_name) {
  auto attrs = make_object<DeformableConv2DAttrs>();
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->deformable_groups = deformable_groups;
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = kernel_size;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = kernel_layout;
  attrs->out_layout = out_layout;
  attrs->out_dtype = out_dtype;
  const Op& op = Op::Get(op_name);
  return Call(op, {data, offset, weight, mask}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.deformable_conv2d_v2")
    .set_body_typed([](Expr data, Expr offset, Expr weight, Expr mask, Array<IndexExpr> strides,
                       Array<IndexExpr> padding, Array<IndexExpr> dilation, int deformable_groups,
                       int groups, IndexExpr channels, Array<IndexExpr> kernel_size,
                       String data_layout, String kernel_layout, String out_layout,
                       DataType out_dtype) {
      return MakeDeformableConvV2(data, offset, weight, mask, strides, padding, dilation,
                                  deformable_groups, groups, channels, kernel_size, data_layout,
                                  kernel_layout, out_layout, out_dtype,
                                  "contrib.aipu_compass.deformable_conv2d_v2");
    });

RELAY_REGISTER_OP("contrib.aipu_compass.deformable_conv2d_v2")
    .set_num_inputs(4)
    .set_support_level(5)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("offset", "Tensor", "The offset tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("mask", "Tensor", "The mask tensor.")
    .set_attrs_type<DeformableConv2DAttrs>()
    .add_type_rel("DeformableConv2DV2", DeformableConv2DV2Rel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", DeformableConvV2InferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

}  // namespace relay
}  // namespace tvm
