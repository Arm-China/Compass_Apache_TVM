# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# CMake build rules for AIPU

if(NOT ${USE_AIPU} MATCHES ${IS_FALSE_PATTERN})
  include_directories("aipu/include")
  # Source files used to construct TVM's compiler part, these files will always
  # be added to the CMake library target "tvm" once AIPU support is enabled.
  file(GLOB COMPILER_AIPU_SRCS
    aipu/src/*.cc
    aipu/src/target/*.cc
    aipu/src/target/source/*.cc
    aipu/src/tir/*.cc
    aipu/src/tir/transforms/*.cc
    aipu/src/ir/*.cc
    aipu/src/relay/transforms/*.cc
    aipu/src/relay/analysis/*.cc
    aipu/src/relay/backend/compass/*.cc
    aipu/src/relay/op/*.cc
  )
  list(APPEND COMPILER_SRCS ${COMPILER_AIPU_SRCS})

  # Source files used to construct TVM's runtime part, some of these files will
  # always be added to the CMake library target "tvm_runtime" once AIPU support
  # is enabled, because they are commonly used by the different AIPU platforms.
  # The files that are specific to each AIPU platforms will only be added, when
  # "USE_AIPU" is set to the corresponding AIPU platform name.
  file(GLOB RUNTIME_AIPU_SRCS
    aipu/src/runtime/*.cc
    aipu/src/runtime/compass/*.cc
    aipu/src/runtime/pipeline/*.cc
    aipu/src/runtime/relax_vm/*.cc
  )
  list(APPEND RUNTIME_SRCS ${RUNTIME_AIPU_SRCS})

  if (${USE_AIPU} MATCHES ${IS_TRUE_PATTERN})
    # If "USE_AIPU" is not set to the name of any AIPU platforms, then use the
    # AIPU simulator as the platform.
    set(AIPU_PLATFORM "sim")
  else()
    set(AIPU_PLATFORM ${USE_AIPU})
  endif()

  message(STATUS "Build with AIPU platform \"${AIPU_PLATFORM}\".")

  # AIPU Driver relevant settings.
  include_directories("${AIPU_DRIVER_INCLUDE_DIR}")
  get_filename_component(AIPU_DRIVER_LIB_DIR ${AIPU_DRIVER_LIB} DIRECTORY)
  get_filename_component(AIPU_DRIVER_LIB_NAME ${AIPU_DRIVER_LIB} NAME)
  list(APPEND TVM_RUNTIME_LINKER_LIBS
    "-l:${AIPU_DRIVER_LIB_NAME}"
    "-L${AIPU_DRIVER_LIB_DIR}"
    "-Wl,--enable-new-dtags,-rpath,$ORIGIN"
    "-Wl,--enable-new-dtags,-rpath,${AIPU_DRIVER_LIB_DIR}"
  )

  if (${AIPU_PLATFORM} STREQUAL "sim")
    file(GLOB RUNTIME_AIPU_PLATFORM_SRCS
      aipu/src/runtime/compass/sim/*.cc
    )
    list(APPEND RUNTIME_SRCS ${RUNTIME_AIPU_PLATFORM_SRCS})
    # In code of AIPU simulator platform, the STL "Filesystem library" is used,
    # in C++14 it is inside namespace "std::experimental::filesystem" and below
    # library must be linked.
    list(APPEND TVM_RUNTIME_LINKER_LIBS "-lstdc++fs")
  else()
    file(GLOB RUNTIME_AIPU_PLATFORM_SRCS
      aipu/src/runtime/compass/device/*.cc
    )

    if (${AIPU_PLATFORM} STREQUAL "android")
      list(APPEND RUNTIME_AIPU_PLATFORM_SRCS
        jvm/native/src/main/native/org_apache_tvm_native_c_api.cc
      )
    endif()

    list(APPEND RUNTIME_SRCS ${RUNTIME_AIPU_PLATFORM_SRCS})
  endif()

  if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    
  endif()
endif()
