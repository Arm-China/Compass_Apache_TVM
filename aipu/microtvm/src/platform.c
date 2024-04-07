// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>

#ifdef __cplusplus
extern "C" {
#endif

g_crt_workspace[TVM_WORKSPACE_SIZE_BYTES];
tvm_workspace_t app_workspace;

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stdout, msg, args);
  va_end(args);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  TVMLogf("TVMPlatformAbort: %d\n", error_code);
  TVMLogf("EXITTHESIM\n");
  exit(-1);
}

tvm_crt_error_t TVMPlatformInitialize() {
  Xil_DCacheFlush();
  Xil_DCacheDisable();
  sleep(1);
  TVMLogf("disable dcache\n");
  StackMemoryManager_Init(&app_workspace, g_crt_workspace, TVM_WORKSPACE_SIZE_BYTES);
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) { return 0; }

#ifdef __cplusplus
}
#endif