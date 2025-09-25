# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# CMake build rules for Zhouyi Compass

if(NOT ${USE_COMPASS} MATCHES ${IS_FALSE_PATTERN})
  include_directories("compass/include")
  # Source files used to construct TVM's compiler part, these files will always be added to the
  # CMake library target "tvm" once Zhouyi Compass support is enabled.
  file(GLOB COMPILER_COMPASS_SRCS
    compass/src/target/*.cc
    compass/src/tir/transforms/*.cc
    compass/src/relax/*.cc
    compass/src/relax/op/*.cc
    compass/src/relax/transforms/*.cc
  )
  list(APPEND COMPILER_SRCS ${COMPILER_COMPASS_SRCS})

  # Source files used to construct TVM's runtime part, some of these files will always be added to
  # the CMake library target "tvm_runtime" once Zhouyi Compass support is enabled, because they are
  # commonly used by the different Zhouyi Compass platforms. The files that are specific to each
  # Zhouyi Compass platforms will only be added, when "USE_COMPASS" is set to the corresponding
  # Zhouyi Compass platform name.
  file(GLOB RUNTIME_COMPASS_SRCS
    compass/src/runtime/*.cc
  )
  list(APPEND RUNTIME_SRCS ${RUNTIME_COMPASS_SRCS})

  if (${USE_COMPASS} MATCHES ${IS_TRUE_PATTERN})
    set(COMPASS_PLATFORM "sim")  # If not set to specific name, then use simulator.
  else()
    set(COMPASS_PLATFORM ${USE_COMPASS})
  endif()

  message(STATUS "Build with Zhouyi Compass platform \"${COMPASS_PLATFORM}\".")

  # Driver relevant settings.
  include_directories("${COMPASS_DRIVER_INCLUDE_DIR}")
  get_filename_component(COMPASS_DRIVER_LIB_DIR ${COMPASS_DRIVER_LIB} DIRECTORY)
  get_filename_component(COMPASS_DRIVER_LIB_NAME ${COMPASS_DRIVER_LIB} NAME)
  list(APPEND TVM_RUNTIME_LINKER_LIBS
    "-l:${COMPASS_DRIVER_LIB_NAME}"
    "-L${COMPASS_DRIVER_LIB_DIR}"
    "-Wl,--enable-new-dtags,-rpath,$ORIGIN"
    "-Wl,--enable-new-dtags,-rpath,${COMPASS_DRIVER_LIB_DIR}"
  )

  if (${COMPASS_PLATFORM} STREQUAL "sim")
    file(GLOB RUNTIME_COMPASS_PLATFORM_SRCS
      compass/src/runtime/sim/*.cc
    )
    list(APPEND RUNTIME_SRCS ${RUNTIME_COMPASS_PLATFORM_SRCS})
  else()
    file(GLOB RUNTIME_COMPASS_PLATFORM_SRCS
      compass/src/runtime/device/*.cc
    )

    if (${COMPASS_PLATFORM} STREQUAL "android")
      list(APPEND RUNTIME_COMPASS_PLATFORM_SRCS
        jvm/native/src/main/native/org_apache_tvm_native_c_api.cc
      )
    endif()

    list(APPEND RUNTIME_SRCS ${RUNTIME_COMPASS_PLATFORM_SRCS})
  endif()

  if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
    
  endif()
endif()
