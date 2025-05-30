// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/include/aipu/runtime/compass/driver.h
 */
#ifndef AIPU_RUNTIME_COMPASS_DRIVER_H_
#define AIPU_RUNTIME_COMPASS_DRIVER_H_

// AIPU driver C API header file.
#ifndef __QNX__
#include <internal/internal_api.h>
#else
#include <standard_api.h>
#endif

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

#define AIPU_DRIVER_HANDLE_ERROR(status)                            \
  do {                                                              \
    if (status != AIPU_STATUS_SUCCESS) {                            \
      const char* error_message = nullptr;                          \
      aipu_get_error_message(ctx_, status, &error_message);         \
      LOG(FATAL) << error_message << " at function " << func_name_; \
    }                                                               \
  } while (false)

class ParamInfo : public Object {
  // Things that will interface with user directly.
 public:
  explicit ParamInfo(DataType dtype, size_t size) : dtype(dtype), size(size) {}

  DataType dtype;
  size_t size;
};

class ParamInfoRef : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ParamInfoRef, ObjectRef, ParamInfo);
};

class AipuDriver {
  // Things that will interface with user directly.
 public:
  AipuDriver();
  ~AipuDriver();
  void Init(const std::string& aipu_bin, const std::string& work_dir, const std::string& target,
            const std::string& umd_dtcm_sz, const std::string& func_name,
            const std::string& extra_path = "");
  void SetInputs(const std::vector<DLTensor*>& inputs);
  void SetOutputs(const std::vector<DLTensor*>& outputs);
  void SetInputsWithDynamicShape(const std::vector<DLTensor*>& inputs);
  void Run();
  void GetOutputs(const std::vector<DLTensor*>& outputs);
  void DumpProfileData();
  std::vector<ParamInfo> GetParamInfo(bool is_input);
  void DeinitGraphJob();

  void MarkOutputShared(int* outputs_fds);
  void SetInputShared(int* inputs_fds);
  void MarkOutputShared(uint64_t* outputs_pa);
  void SetInputShared(uint64_t* inputs_pa);

  // As to dynamic shapes, this interface get output shape.
  std::vector<int64_t> GetOutputShape(uint32_t idx);
  // Internal supporting.
 private:
  void ConfigEnvItems(const std::string& aipu_bin, const std::string& extra_path = "");
  void ConfigGlobal(bool is_profile);
  void ConfigGraphItems();
  void ConfigJobItems();
  aipu_ctx_handle_t* ctx_ = nullptr;
  aipu_create_job_cfg_t job_cfg_ = {0};

  aipu_status_t status_ = AIPU_STATUS_SUCCESS;
  uint64_t job_id_ = 0;
  uint64_t graph_id_ = 0;
  std::string work_dir_;
  std::string target_;
  std::set<uint32_t> shared_inputs_idx;
  std::set<uint32_t> shared_outputs_idx;
  std::vector<aipu_share_buf_t> shared_outputs_buf;
  // The size of the Data Tightly Coupled Memory, used by AIPU simulator.
  std::string umd_dtcm_sz_;
  std::string func_name_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // AIPU_RUNTIME_COMPASS_DRIVER_H_
