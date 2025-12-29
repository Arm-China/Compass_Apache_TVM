// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/driver.cc
 */
#include <compass/tvm/runtime/basic_config.h>
#include <compass/tvm/runtime/driver.h>
#include <compass/tvm/runtime/utils.h>

#include <mutex>

namespace tvm {
namespace runtime {

#ifndef __QNX__
static aipu_ctx_handle_t* process_ctx = nullptr;
static std::mutex inst_lock;
static std::atomic<int32_t> inst_counter(0);
#else
thread_local aipu_ctx_handle_t* process_ctx = nullptr;
thread_local std::mutex inst_lock;
thread_local std::atomic<int32_t> inst_counter(0);
#endif

aipu_ctx_handle_t* ContextAlloc() {
  inst_counter.fetch_add(1);
  if (process_ctx == nullptr) {
    std::lock_guard<std::mutex> lock(inst_lock);
    if (process_ctx == nullptr) {
      aipu_status_t status = aipu_init_context(&process_ctx);
      if (status != AIPU_STATUS_SUCCESS) {
        const char* error_message = nullptr;
        aipu_get_error_message(process_ctx, status, &error_message);
        LOG(FATAL) << error_message;
      }
    }
  }
  return process_ctx;
}

void ContextReset() {
  if (inst_counter.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    if (process_ctx != nullptr) {
      std::lock_guard<std::mutex> lock(inst_lock);
      if (process_ctx != nullptr) {
        aipu_status_t status = aipu_deinit_context(process_ctx);
        if (status != AIPU_STATUS_SUCCESS) {
          const char* error_message = nullptr;
          aipu_get_error_message(process_ctx, status, &error_message);
          LOG(FATAL) << error_message;
        }
        process_ctx = nullptr;
      }
    }
  }
}

ParamInfo::ParamInfo(DataType dtype, size_t size) {
  ObjectPtr<ParamInfoObj> n = make_object<ParamInfoObj>();
  n->dtype = dtype;
  n->size = size;
  data_ = std::move(n);
}

CompassDriverObj::CompassDriverObj(const std::string& cps_bin_path, const std::string& func_name,
                                   bool with_profile, const std::string& target,
                                   const std::string& umd_dtcm_sz) {
  func_name_ = func_name;
  work_dir_ = CompassBasicConfig::Global().GetRuntimeWorkDir(func_name);
  with_profile_ = with_profile;
  ctx_ = ContextAlloc();

  CreateDirectories(work_dir_);  // Create recursively and ignore the directories exist error.

  LoadGraph(cps_bin_path, target, umd_dtcm_sz);

  // Create job with the loaded Compass binary program.
  aipu_create_job_cfg_t job_cfg = {0};
  status_ = aipu_create_job(ctx_, graph_id_, &job_id_, &job_cfg);
  COMPASS_DRIVER_HANDLE_ERROR(status_);

  ConfigJobItems();
  return;
}

void CompassDriverObj::SetInputs(Array<NDArray> inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    NDArray input = inputs[i];
    if (input->device.device_type != kDLCPU) {
      input = input.CopyTo(Device{kDLCPU, 0});
    }

    status_ = aipu_load_tensor(ctx_, job_id_, i, input->data);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
  }
  return;
}

void CompassDriverObj::SetOutputs(Array<NDArray> outputs) {
#ifndef __QNX__
  for (size_t i = 0; i < outputs.size(); ++i) {
    NDArray output = outputs[i];
    if (output->device.device_type != kDLCPU) {
      output = output.CopyTo(Device{kDLCPU, 0});
    }

    status_ = aipu_load_output_tensor(ctx_, job_id_, i, output->data);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
  }
  return;
#endif
}

void CompassDriverObj::SetInputsWithDynamicShape(Array<NDArray> inputs) {
#ifndef __QNX__
  size_t input_cnt = inputs.size();
  aipu_dynshape_param_t dyn_param = {0};
  dyn_param.input_shape_cnt = input_cnt;
  std::vector<std::vector<uint32_t>> dyn_shapes(input_cnt);
  std::vector<aipu_dynshape_item_t> dyn_items(input_cnt);
  dyn_param.shape_items = &(dyn_items[0]);

  for (size_t i = 0; i < input_cnt; ++i) {
    dyn_items[i].ds_idx = i;
    for (int32_t j = 0; j < inputs[i]->ndim; ++j) {
      dyn_shapes[i].push_back(inputs[i]->shape[j]);
    }
    dyn_items[i].ds_data = &(dyn_shapes[i][0]);
  }

  // need recreate job to flush the io size
  if (job_id_ != 0) {
    status_ = aipu_clean_job(ctx_, job_id_);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
    job_id_ = 0;
  }
  aipu_create_job_cfg_t job_cfg = {0};
  job_cfg.dynshape_params = dyn_param;
  status_ = aipu_create_job(ctx_, graph_id_, &job_id_, &job_cfg);
  COMPASS_DRIVER_HANDLE_ERROR(status_);
  ConfigJobItems();

  SetInputs(inputs);
  return;
#endif
}

void CompassDriverObj::GetOutputs(Array<NDArray> outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    NDArray output = outputs[i];
    NDArray output_on_cpu = output;
    if (output->device.device_type != kDLCPU) {
      output_on_cpu = output.CopyTo(Device{kDLCPU, 0});
    }

    status_ = aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_OUTPUT, i, output_on_cpu->data);
    COMPASS_DRIVER_HANDLE_ERROR(status_);

    if (output->device.device_type != kDLCPU) {
      output_on_cpu.CopyTo(output);
    }
  }
  return;
}

static inline DataType CreateDataType(aipu_data_type_t type_code) {
  switch (type_code) {
    case AIPU_DATA_TYPE_S8:
      return DataType::Int(8);
    case AIPU_DATA_TYPE_U8:
      return DataType::UInt(8);
    case AIPU_DATA_TYPE_S16:
      return DataType::Int(16);
    case AIPU_DATA_TYPE_U16:
      return DataType::UInt(16);
    case AIPU_DATA_TYPE_S32:
      return DataType::Int(32);
    case AIPU_DATA_TYPE_U32:
      return DataType::UInt(32);
    case AIPU_DATA_TYPE_S64:
      return DataType::Int(64);
    case AIPU_DATA_TYPE_U64:
      return DataType::UInt(64);
    case AIPU_DATA_TYPE_F16:
      return DataType::Float(16);
    case AIPU_DATA_TYPE_F32:
      return DataType::Float(32);
    case AIPU_DATA_TYPE_F64:
      return DataType::Float(64);
    case AIPU_DATA_TYPE_BF16:
      return DataType::BFloat(16);
    default:
      LOG(FATAL) << "Unsupported Compass driver data type: " << type_code;
      throw;  // unreachable, written to stop compiler warning
  }
}

Array<ParamInfo> CompassDriverObj::GetParamInfo(bool is_input) {
  Array<ParamInfo> ret;
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = is_input ? AIPU_TENSOR_TYPE_INPUT : AIPU_TENSOR_TYPE_OUTPUT;
  status_ = aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);
  COMPASS_DRIVER_HANDLE_ERROR(status_);
  for (uint32_t i = 0; i < count; ++i) {
    aipu_tensor_desc_t desc = {0};
    status_ = aipu_get_tensor_descriptor(ctx_, graph_id_, tensor_type, i, &desc);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
    ret.push_back(ParamInfo(CreateDataType(desc.data_type), desc.size));
  }
  return ret;
}

std::vector<int64_t> CompassDriverObj::GetOutputShape(uint32_t idx) {
  std::vector<int64_t> ret;
#ifndef __QNX__
  aipu_tensor_desc_t desc = {0};
  status_ =
      aipu_get_tensor_descriptor(ctx_, graph_id_, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, idx, &desc);
  COMPASS_DRIVER_HANDLE_ERROR(status_);
  uint32_t dim = desc.size / sizeof(uint32_t);
  std::vector<uint32_t> shape(dim, 0);

  aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, idx, &shape[0]);
  for (auto dim_val : shape) {
    ret.push_back(dim_val);
  }
#endif
  return ret;
}

void CompassDriverObj::DeinitGraphJob() {
  // The valid job id isn't possible to be 0 and will be more greater than 0.
  if (job_id_ != 0) {
    status_ = aipu_clean_job(ctx_, job_id_);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
    job_id_ = 0;
  }

  // The valid graph id isn't possible to be 0 and will be more greater than 0.
  if (graph_id_ != 0) {
    status_ = aipu_unload_graph(ctx_, graph_id_);
    COMPASS_DRIVER_HANDLE_ERROR(status_);
    graph_id_ = 0;
  }
}

CompassDriverObj::~CompassDriverObj() {
  if (ctx_ == nullptr) return;
  DeinitGraphJob();
  ContextReset();
  ctx_ = nullptr;
}

CompassDriver::CompassDriver(const std::string& cps_bin_path, const std::string& func_name,
                             bool with_profile, const std::string& target,
                             const std::string& umd_dtcm_sz) {
  ObjectPtr<CompassDriverObj> n =
      make_object<CompassDriverObj>(cps_bin_path, func_name, with_profile, target, umd_dtcm_sz);
  data_ = std::move(n);
}

}  // namespace runtime
}  // namespace tvm
