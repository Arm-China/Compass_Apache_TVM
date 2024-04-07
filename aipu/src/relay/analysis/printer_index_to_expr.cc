// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.

/*!
 * \file aipu/src/relay/transforms/printer_index_to_expr.cc
 */
#include "../../../../src/relay/printer/text_printer.h"

namespace tvm {
namespace relay {

/*
Print IRModule would get text like:
  %0 = nn.conv2d(%data, meta[relay.Constant][0], padding=[1, 1, 1, 1], channels=64, kernel_size=[3,
3]); %1 = nn.relu(%0); %2 = nn.conv2d(%1, meta[relay.Constant][1], padding=[1, 1, 1, 1],
channels=64, kernel_size=[3, 3]); %3 = nn.relu(%2); %4 = nn.conv2d(%3, meta[relay.Constant][2],
padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);

Sometimes we'd like to get the exprssion by the printer SSA form index.
*/
Array<Expr> PrinterIndexToExpr(const BaseFunc& func) {
  TextMetaDataContext meta_;
  auto printer = RelayTextPrinter(false, &meta_, nullptr);
  printer.PrintFunc(Doc(), func);

  Array<Expr> ret;
  std::map<int, Expr> expr_mm(printer.memo_index_.begin(), printer.memo_index_.end());

  for (auto key : expr_mm) {
    ret.push_back(key.second);
  }

  return ret;
}

TVM_REGISTER_GLOBAL("relay.analysis.PrinterIndexToExpr").set_body_typed(PrinterIndexToExpr);

}  // namespace relay
}  // namespace tvm
