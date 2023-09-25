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
