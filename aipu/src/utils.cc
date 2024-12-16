// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/utils.cc
 * \brief Common utilities.
 */
#include <aipu/utils.h>
#include <tvm/ir/transform.h>

namespace tvm {

std::string GenAipuC(DataType t) {
  static const std::set<std::string> supported_dtypes = {
      "int8",     "uint8",     "int8x4",   "uint8x4", "int8x8",  "uint8x8",  "int8x16",
      "uint8x16", "int8x32",   "uint8x32", "int16",   "uint16",  "int16x8",  "uint16x8",
      "int16x16", "uint16x16", "int32",    "uint32",  "int32x8", "uint32x8",
  };
  ICHECK_NE(supported_dtypes.count(DLDataType2String(t)), 0)
      << "AIPU don't support type \"" << t << "\".";

  std::ostringstream ret;
  if (t.is_uint() || t.is_int()) {
    if (t.is_fixed_length_vector()) ret << "v" << t.lanes();
    if (t.is_uint()) ret << "u";
    ret << "int" << t.bits() << "_t";
  }
  return ret.str();
}

}  // namespace tvm
