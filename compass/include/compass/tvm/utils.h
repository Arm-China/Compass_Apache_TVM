// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/tvm/utils.h
 * \brief Common utilities.
 */
#ifndef COMPASS_TVM_UTILS_H_
#define COMPASS_TVM_UTILS_H_

#include <compass/tvm/runtime/utils.h>
#include <tvm/runtime/data_type.h>

#include <string>

namespace tvm {

using runtime::StrEndsWith;
using runtime::StrReplace;
using runtime::StrStartsWith;

/*! \brief Return the maximum value of current datatype.
 *
 *  \note Now only support 8bit and 16bit int.
 */
inline int64_t GetMaxValue(const DataType& dtype) {
  ICHECK((dtype.is_int() || dtype.is_uint()));

  if (dtype.is_int()) {
    if (dtype.bits() == 8) {
      return 127;
    } else if (dtype.bits() == 16) {
      return 32767;
    } else if (dtype.bits() == 32) {
      return 2147483647;
    }
  } else if (dtype.is_uint()) {
    if (dtype.bits() == 8) {
      return 255;
    } else if (dtype.bits() == 16) {
      return 65535;
    } else if (dtype.bits() == 32) {
      return 4294967295UL;
    }
  }

  LOG(FATAL) << "Can't get max value from " << DLDataTypeToString(dtype) << ".";
  return 0;
}

/*! \brief Return the minimum value of current datatype.
 *
 *  \note Now only support 8bit and 16bit int.
 */
inline int64_t GetMinValue(const DataType& dtype) {
  ICHECK((dtype.is_int() || dtype.is_uint()));

  if (dtype.is_int()) {
    if (dtype.bits() == 8) {
      return -128;
    } else if (dtype.bits() == 16) {
      return -32768;
    } else if (dtype.bits() == 32) {
      return -2147483648;
    }
  } else if (dtype.is_uint()) {
    return 0;
  }

  LOG(FATAL) << "Can't get min value from " << DLDataTypeToString(dtype) << ".";
  return 0;
}

}  // namespace tvm

#endif  // COMPASS_TVM_UTILS_H_
