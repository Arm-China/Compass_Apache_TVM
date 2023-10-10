# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.


cmake_minimum_required(VERSION 3.14.0)
project(AIPU_BareMental_demo)
add_definitions(-DTVM_WORKSPACE_SIZE_BYTES=200000)

if(NOT DEFINED ENV{AIPU_TVM_BM_DEVICE_COMPILER})
    message(FATAL_ERROR "not defined environment variable:AIPU_TVM_BM_DEVICE_COMPILER")
endif()

if(NOT DEFINED ENV{ZHOUYI_BM_DRIVER_HOME})
    message(FATAL_ERROR "not defined environment variable:ZHOUYI_BM_DRIVER_HOME")
endif()

SET(CMAKE_C_COMPILER $ENV{AIPU_TVM_BM_DEVICE_COMPILER})
SET(CMAKE_C_FLAGS "-Wall -O0 -g3 -fmessage-length=0 -MMD -MP")
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-T -Wl,${PROJECT_SOURCE_DIR}/bsp/lscript.ld")

####################################################
#libaipu_driver_wrapper.a
####################################################
add_library(aipu_driver_wrapper STATIC "${PROJECT_SOURCE_DIR}/src/aipu_driver_wrapper.c")

target_include_directories(aipu_driver_wrapper
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC $ENV{ZHOUYI_BM_DRIVER_HOME}/legacy/output/release
)

####################################################
#libtvm_runtime.a
####################################################
set(TVM_RUNTIME_SRCS
    ${PROJECT_SOURCE_DIR}/runtime/src/runtime/crt/common/crt_backend_api.c
    ${PROJECT_SOURCE_DIR}/runtime/src/runtime/crt/memory/stack_allocator.c
    ${PROJECT_SOURCE_DIR}/src/platform.c
)

add_library(tvm_runtime STATIC ${TVM_RUNTIME_SRCS})
target_include_directories(tvm_runtime
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include
)

####################################################
#libcodegen.a
####################################################
set(DATA_FILES
    ${PROJECT_SOURCE_DIR}/convert/output/gbin4bm/aipu/aipu_in0.c
    ${PROJECT_SOURCE_DIR}/convert/output/gbin4bm/aipu/aipu_gt.c
    ${PROJECT_SOURCE_DIR}/convert/output/gbin4bm/aipu/gbin4bm.c
)

add_custom_command(
    OUTPUT ${DATA_FILES}
    COMMAND ./bin2src_all.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/convert
)

file(GLOB CODEGEN_SRCS "${PROJECT_SOURCE_DIR}/codegen/host/src/*.c")
list(APPEND CODEGEN_SRCS ${DATA_FILES})

add_library(codegen STATIC ${CODEGEN_SRCS})
target_include_directories(codegen
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include
    PUBLIC ${PROJECT_SOURCE_DIR}/codegen/host/include
    PUBLIC ${PROJECT_SOURCE_DIR}/convert/output/gbin4bm
)

####################################################
#app
####################################################
add_executable(app ${PROJECT_SOURCE_DIR}/src/main.c)

target_include_directories(app
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/codegen/host/include
)

file(GLOB BSP_LIB "${PROJECT_SOURCE_DIR}/bsp/lib/lib*.a")

target_link_libraries(app
    -Wl,--start-group
    ${BSP_LIB}
    $ENV{ZHOUYI_BM_DRIVER_HOME}/legacy/output/release/libaipu.a
    aipu_driver_wrapper
    tvm_runtime
    codegen
    c
    m
    -Wl,--end-group
)
