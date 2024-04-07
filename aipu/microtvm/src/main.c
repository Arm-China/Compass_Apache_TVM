// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>

#include "tvmgen_default.h"

extern uint8_t* input0;
extern uint8_t* gt;

const size_t output_len = 1001;

void PrintResult(void* result, size_t len) {
  unsigned int bytes = 0;
  const unsigned int max_printf = 256;
  unsigned char* ref_buffer = (unsigned char*)result;

  unsigned int index = 0;
  while (len > 0) {
    bytes = len > max_printf ? max_printf : len;
    for (int i = 0; i < bytes; i++) {
      TVMLogf("0x%x ", *(ref_buffer + index));
      index++;
    }
    len -= bytes;
  }
  TVMLogf("\n\n");
}

void ArgSort(uint8_t* arr, int* res, size_t len) {
  uint8_t max0 = arr[0];
  res[0] = 0;
  for (int i = 0; i < len; i++) {
    if (max0 < arr[i]) {
      max0 = arr[i];
      res[0] = i;
    }
  }

  uint8_t max1 = arr[0];
  res[1] = 0;
  for (int i = 0; i < len; i++) {
    if (max1 < arr[i] && i != res[0]) {
      max1 = arr[i];
      res[1] = i;
    }
  }

  uint8_t max2 = arr[0];
  res[2] = 0;
  for (int i = 0; i < len; i++) {
    if (max2 < arr[i] && i != res[0] && i != res[1]) {
      max2 = arr[i];
      res[2] = i;
    }
  }
}

void main(void) {
  TVMPlatformInitialize();

  uint8_t* output = malloc(output_len);
  if (output == NULL) {
    TVMLogf("Malloc for output failed!!!\n");
    exit(-1);
  }
  struct tvmgen_default_inputs inputs = {.input = input0};
  struct tvmgen_default_outputs outputs = {.output = output};

  if (tvmgen_default_run(&inputs, &outputs) != 0) {
    TVMLogf("Model run failed!!!\n");
    exit(-1);
  }
  TVMLogf("AIPU run already finish!!!\n");
  int gt_arg[3];
  int aipu_arg[3];
  ArgSort(gt, gt_arg, output_len);
  ArgSort(outputs.output, aipu_arg, output_len);
  int ret = memcmp(gt_arg, aipu_arg, 12);
  if (ret != 0) {
    TVMLogf("result compare failed!!!\n");

    TVMLogf("\n\ninput:\n");
    PrintResult(inputs.input, 128);

    TVMLogf("\n\nrefr output:\n");
    PrintResult(gt, output_len);

    TVMLogf("\naipu output:\n");
    PrintResult(outputs.output, output_len);

    TVMLogf("\n\nrefr arg:\n");
    PrintResult(gt_arg, 12);

    TVMLogf("\naipu arg:\n");
    PrintResult(aipu_arg, 12);
  } else {
    TVMLogf("result compare success!!!\n");
  }

  exit(0);
}
