// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file aipu/src/runtime/compass/device/driver.cc
 */
#include <aipu/runtime/compass/driver.h>
#include <aipu/runtime/utils.h>

namespace tvm {
namespace runtime {

void AipuDriver::ConfigGlobal(bool is_profile) { return; }
void AipuDriver::ConfigEnvItems() { return; }
void AipuDriver::ConfigGraphItems() { return; }

void AipuDriver::ConfigJobItems() {
  uint32_t buffer_cnt = 0;
  status_ = aipu_get_tensor_count(ctx_, graph_id_, AIPU_TENSOR_TYPE_PROFILER, &buffer_cnt);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  if (buffer_cnt != 0) {
    aipu_job_config_dump_t cfg = {0};
    cfg.dump_dir = work_dir_.c_str();
    // name prefix of dump files
    cfg.prefix = "prefix";
    // name prefix of output dump files
    cfg.output_prefix = "output_prefix";
    // name prefix of profile/printf data files, UMD would add _PerfData.bin for v3
    cfg.misc_prefix = "profile_data";
    status_ = aipu_config_job(ctx_, job_id_, AIPU_JOB_CONFIG_TYPE_DUMP_PROFILE, &cfg);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }
  return;
}

void AipuDriver::Run() {
  status_ = aipu_finish_job(ctx_, job_id_, -1);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  return;
}

void AipuDriver::DumpProfileData() {
  uint32_t buffer_cnt = 0;
  status_ = aipu_get_tensor_count(ctx_, graph_id_, AIPU_TENSOR_TYPE_PROFILER, &buffer_cnt);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  if (buffer_cnt == 0) return;
  // There only must be one profile buffer for each job.
  aipu_tensor_desc_t desc = {0};
  status_ = aipu_get_tensor_descriptor(ctx_, graph_id_, AIPU_TENSOR_TYPE_PROFILER, 0, &desc);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  std::string profile_data;
  profile_data.resize(desc.size);
  status_ = aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_PROFILER, 0, &(profile_data[0]));
  AIPU_DRIVER_HANDLE_ERROR(status_);

  // Create directories recursively and ignore the directories exist error.
  CreateDirectories(work_dir_);

  SaveBinary2File((work_dir_ + "/profile_data.bin"), profile_data.c_str(), profile_data.size());
  return;
}

}  // namespace runtime
}  // namespace tvm
