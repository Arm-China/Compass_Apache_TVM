# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.

cmake_minimum_required(VERSION 3.2)
project(cpp_deploy CXX)

get_filename_component(TVM_ROOT ../../../../../../ ABSOLUTE)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

add_executable(cpp_deploy cpp_deploy.cc)
target_include_directories(cpp_deploy PUBLIC
  ${TVM_ROOT}/include
  ${TVM_ROOT}/aipu/include
  ${TVM_ROOT}/3rdparty/dlpack/include
  ${TVM_ROOT}/3rdparty/dmlc-core/include
)

set(TVM_RUNTIME_LIB_DIR ${TVM_ROOT}/build)
target_link_libraries(cpp_deploy PUBLIC
  "-ltvm_runtime"
  "-L${TVM_RUNTIME_LIB_DIR}"
  "-Wl,-rpath,${TVM_RUNTIME_LIB_DIR}"
)

add_executable(pipeline_deploy pipeline_deploy.cc)
target_include_directories(pipeline_deploy PUBLIC
  ${TVM_ROOT}/include
  ${TVM_ROOT}/aipu/include
  ${TVM_ROOT}/3rdparty/dlpack/include
  ${TVM_ROOT}/3rdparty/dmlc-core/include
)

set(TVM_RUNTIME_LIB_DIR ${TVM_ROOT}/build)
target_link_libraries(pipeline_deploy PUBLIC
  "-ltvm_runtime -lpthread -lm"
  "-L${TVM_RUNTIME_LIB_DIR}"
  "-Wl,-rpath,${TVM_RUNTIME_LIB_DIR}"
)

add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

set(ZHOUYI_LINUX_DRIVER_HOME $ENV{ZHOUYI_LINUX_DRIVER_HOME})
add_executable(dmabuf_deploy dmabuf_deploy.cc)
target_include_directories(dmabuf_deploy PUBLIC
  ${TVM_ROOT}/include
  ${TVM_ROOT}/aipu/include
  ${TVM_ROOT}/3rdparty/dlpack/include
  ${TVM_ROOT}/3rdparty/dmlc-core/include
  ${ZHOUYI_LINUX_DRIVER_HOME}/driver/umd/include
)

set(TVM_RUNTIME_LIB_DIR ${TVM_ROOT}/build)
# Here set AIPU_DRIVER_LIB_DIR

set(AIPU_DRIVER_LIB_DIR ${AIPU_DRIVER_LIB_DIR})
target_link_libraries(dmabuf_deploy PUBLIC
  "-ltvm_runtime -lpthread -lm"
  "-L${TVM_RUNTIME_LIB_DIR} -L${AIPU_DRIVER_LIB_DIR}"
  "-Wl,-rpath,${TVM_RUNTIME_LIB_DIR}"
)

add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
