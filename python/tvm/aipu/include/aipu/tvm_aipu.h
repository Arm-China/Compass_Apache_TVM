// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/tvm_aipu.h
 * \brief TVM specific AIPU header file.
 */
#ifndef TVM_AIPU_H_
#define TVM_AIPU_H_

#if defined __X2_1204__
#include <aipu/tvm_aipu_v2.h>
#elif defined __X3_1304__
#include <aipu/tvm_aipu_v3.h>
#endif

#endif  // TVM_AIPU_H_
