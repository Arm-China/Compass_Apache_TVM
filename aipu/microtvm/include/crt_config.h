// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file include/crt_config.h
 * \brief CRT configuration for demo app.
 */
#ifndef TVM_RUNTIME_CRT_CONFIG_H_
#define TVM_RUNTIME_CRT_CONFIG_H_

/*! Log level of the CRT runtime */
#define TVM_CRT_LOG_LEVEL TVM_CRT_LOG_LEVEL_DEBUG

/*! Support low-level debugging in MISRA-C runtime */
#define TVM_CRT_DEBUG 0

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6
/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10
/*! Maximum supported string length in dltype, e.g. "int8", "int16", "float32"
 */
#define TVM_CRT_MAX_STRLEN_DLTYPE 10
/*! Maximum supported string length in function names */
#define TVM_CRT_MAX_STRLEN_FUNCTION_NAME 120
/*! Maximum supported string length in parameter names */
#define TVM_CRT_MAX_STRLEN_PARAM_NAME 80

/*! Maximum number of registered modules. */
#define TVM_CRT_MAX_REGISTERED_MODULES 2

/*! Size of the global function registry, in bytes. */
#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 512

/*! Maximum packet size, in bytes, including the length header. */
#define TVM_CRT_MAX_PACKET_SIZE_BYTES 512

#endif  // TVM_RUNTIME_CRT_CONFIG_H_
