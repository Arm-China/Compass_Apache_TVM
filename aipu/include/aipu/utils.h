// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/utils.h
 * \brief Common utilities.
 */
#ifndef AIPU_UTILS_H_
#define AIPU_UTILS_H_

#include <aipu/runtime/utils.h>
#include <aipu/target/target_info.h>

#include <string>

namespace tvm {

using runtime::StrEndsWith;
using runtime::StrReplace;
using runtime::StrStartsWith;

/*! \brief Return whether a string starts with "__builtin_aipu" and has
           substring "p_", which means the builtin with pg(predicate). */
inline bool BuiltinHasP(const std::string& str) {
  bool IsBuiltin = false;
  std::string ret = str;
  IsBuiltin = StrStartsWith(ret, "__builtin_aipu");

  auto pos = ret.find("p_");
  if (!IsBuiltin || pos == std::string::npos) return false;

  return true;
}

/*! \brief Return the bits accroding to value. */
inline int GetBits(int value) {
  if (value < 256) {
    return 8;
  } else if (value < 65536) {
    return 16;
  } else {
    return 32;
  }
}

/*! \brief Return the largest factor in the range of 8bit of value. */
inline int GetBasicNum(int value) {
  // is there any better algorithm?
  for (int i = 255; i > 1; i--) {
    if (value % i == 0) {
      return value / i;
    }
  }
  return value;
}

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

  LOG(FATAL) << "Can't get max value from " << DLDataType2String(dtype) << ".";
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

  LOG(FATAL) << "Can't get min value from " << DLDataType2String(dtype) << ".";
  return 0;
}

/*! \brief Generate AIPU C source code for the given data structure. */
std::string GenAipuC(DataType t);

}  // namespace tvm

#endif  // AIPU_UTILS_H_
