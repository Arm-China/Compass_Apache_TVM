// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/dsl.h
 * \brief Compass DSL header file.
 */
#ifndef DSL_H_
#define DSL_H_

#if defined(__AIPUX2__)
#include <compass/dsl_v2.h>
#elif defined(__AIPUX3P__) || defined(__AIPUX3S__)
#include <compass/dsl_v3.h>
#else
#error "Unsupported hardware platform !"
#endif

#endif  // DSL_H_
