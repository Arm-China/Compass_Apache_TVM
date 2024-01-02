// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/include/aipu/runtime/compass/driver.h
 */
#ifndef AIPU_RUNTIME_COMPASS_DRIVER_H_
#define AIPU_RUNTIME_COMPASS_DRIVER_H_

// AIPU driver C API header file.
#include <standard_api.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>

#include <map>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

#define AIPU_DRIVER_HANDLE_ERROR(status)                    \
  do {                                                      \
    if (status != AIPU_STATUS_SUCCESS) {                    \
      const char* error_message = nullptr;                  \
      aipu_get_error_message(ctx_, status, &error_message); \
      LOG(FATAL) << error_message;                          \
    }                                                       \
  } while (false)

class ParamInfo {
  // Things that will interface with user directly.
 public:
  explicit ParamInfo(DataType dtype, size_t size) : dtype(dtype), size(size) {}

  DataType dtype;
  size_t size;
};

class AipuDriver {
  // Things that will interface with user directly.
 public:
  AipuDriver();
  ~AipuDriver();
  void Init(const std::string& aipu_bin, const std::string& work_dir, const std::string& target,
            uint64_t* shared_inputs_pa, uint64_t* shared_outputs_pa,
            const std::string& umd_dtcm_sz);
  void SetInputs(const std::vector<DLTensor*>& inputs);
  void Run();
  void GetOutputs(const std::vector<DLTensor*>& outputs);
  void DumpProfileData();
  std::vector<ParamInfo> GetParamInfo(bool is_input);
  void DeinitGraphJob();

  // Internal supporting.
 private:
  void MarkOutputShared(uint64_t* outputs_pa);
  void SetInputShared(uint64_t* inputs_pa);
  void ConfigEnvItems();
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
  std::map<uint32_t, uint64_t> shared_inputs_pa;
  std::map<uint32_t, uint64_t> shared_outputs_pa;
  std::vector<aipu_share_buf_t> shared_outputs_buf;
  // The size of the Data Tightly Coupled Memory, used by AIPU simulator.
  std::string umd_dtcm_sz_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // AIPU_RUNTIME_COMPASS_DRIVER_H_
