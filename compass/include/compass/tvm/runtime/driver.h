// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/include/compass/tvm/runtime/driver.h
 */
#ifndef COMPASS_TVM_RUNTIME_DRIVER_H_
#define COMPASS_TVM_RUNTIME_DRIVER_H_

#ifndef __QNX__
#include <internal/internal_api.h>
#else
#include <standard_api.h>
#endif

#include <tvm/ffi/container/array.h>
#include <tvm/runtime/ndarray.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {

#define COMPASS_DRIVER_HANDLE_ERROR(status)                         \
  do {                                                              \
    if (status != AIPU_STATUS_SUCCESS) {                            \
      const char* error_message = nullptr;                          \
      aipu_get_error_message(ctx_, status, &error_message);         \
      LOG(FATAL) << error_message << " at function " << func_name_; \
    }                                                               \
  } while (false)

struct ParamInfoObj : public Object {
  // Things that will interface with user directly.
  DataType dtype;
  size_t size;

  // Internal supporting.
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "compass.runtime.ParamInfo";
  static constexpr const bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_BASE_OBJECT_INFO(ParamInfoObj, Object);
};

struct ParamInfo : public ObjectRef {
  // Things that will interface with user directly.
  ParamInfo(DataType dtype, size_t size);

  // Internal supporting.
  // TVM C++ object protocol relevant.
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ParamInfo, ObjectRef, ParamInfoObj);
};

struct CompassDriverObj : public Object {
  // Things that will interface with user directly.
  CompassDriverObj(const std::string& cps_bin_path, const std::string& func_name, bool with_profile,
                   const std::string& target, const std::string& umd_dtcm_sz);
  ~CompassDriverObj();
  void SetInputs(Array<NDArray> inputs);
  void SetOutputs(Array<NDArray> outputs);
  void SetInputsWithDynamicShape(Array<NDArray> inputs);
  void Run();
  void GetOutputs(Array<NDArray> outputs);
  void DumpProfileData();
  Array<ParamInfo> GetParamInfo(bool is_input);
  void DeinitGraphJob();

  // As to dynamic shapes, this interface get output shape.
  std::vector<int64_t> GetOutputShape(uint32_t idx);

  // Internal supporting.
  void LoadGraph(const std::string& cps_bin_path, const std::string& target,
                 const std::string& umd_dtcm_sz);
  void ConfigJobItems();

  aipu_ctx_handle_t* ctx_ = nullptr;
  aipu_status_t status_ = AIPU_STATUS_SUCCESS;
  uint64_t job_id_ = 0;
  uint64_t graph_id_ = 0;
  std::string work_dir_;
  std::string func_name_;
  bool with_profile_ = false;

  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "compass.runtime.CompassDriver";
  static constexpr const bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(CompassDriverObj, Object);
};

struct CompassDriver : public ObjectRef {
  // Things that will interface with user directly.
  CompassDriver(const std::string& cps_bin_path, const std::string& func_name, bool with_profile,
                const std::string& target, const std::string& umd_dtcm_sz);

  // Internal supporting.
  // TVM C++ object protocol relevant.
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CompassDriver, ObjectRef, CompassDriverObj);
};

}  // namespace runtime
}  // namespace tvm
#endif  // COMPASS_TVM_RUNTIME_DRIVER_H_
