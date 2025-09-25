// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/device/driver.cc
 */
#include <compass/tvm/runtime/driver.h>
#include <compass/tvm/runtime/utils.h>

namespace tvm {
namespace runtime {

void CompassDriverObj::LoadGraph(const std::string& cps_bin_path, const std::string& target,
                                 const std::string& umd_dtcm_sz) {
  std::string cps_bin_dir = Dirname(cps_bin_path);
#ifndef __QNX__
  aipu_load_graph_cfg_t cfg = {0};
  cfg.extra_weight_path = cps_bin_dir.c_str();
  status_ = aipu_load_graph(ctx_, cps_bin_path.c_str(), &graph_id_, &cfg);
#else
  status_ = aipu_load_graph(ctx_, cps_bin_path.c_str(), &graph_id_);
#endif
  COMPASS_DRIVER_HANDLE_ERROR(status_);
  return;
}

void CompassDriverObj::ConfigJobItems() {
  if (with_profile_ == true) {
    aipu_job_config_dump_t cfg = {0};
    cfg.dump_dir = work_dir_.c_str();
    // name prefix of dump files
    cfg.prefix = "prefix";
    // name prefix of output dump files
    cfg.output_prefix = "output_prefix";
    // name prefix of profile/printf data files, UMD would add _PerfData.bin for v3
    cfg.misc_prefix = "profile_data";

    status_ = aipu_config_job(ctx_, job_id_, AIPU_JOB_CONFIG_TYPE_DUMP_PROFILE, &cfg);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
  }
  return;
}

void CompassDriverObj::Run() {
  status_ = aipu_finish_job(ctx_, job_id_, -1);
  COMPASS_DRIVER_HANDLE_ERROR(status_);
  return;
}

void CompassDriverObj::DumpProfileData() {
  if (with_profile_ == false) return;
  // There only must be one profile buffer for each job.
  aipu_tensor_desc_t desc = {0};

  status_ = aipu_get_tensor_descriptor(ctx_, graph_id_, AIPU_TENSOR_TYPE_PROFILER, 0, &desc);
  COMPASS_DRIVER_HANDLE_ERROR(status_);

  std::string profile_data;
  profile_data.resize(desc.size);

  status_ = aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_PROFILER, 0, &(profile_data[0]));
  COMPASS_DRIVER_HANDLE_ERROR(status_);

  CreateDirectories(work_dir_);  // Create recursively and ignore the directories exist error.
  SaveBinary2File((work_dir_ + "/profile_data.bin"), profile_data.c_str(), profile_data.size());
  return;
}

}  // namespace runtime
}  // namespace tvm
