// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/runtime/compass/driver.cc
 */
#include <aipu/runtime/compass/basic_config.h>
#include <aipu/runtime/compass/driver.h>
#include <aipu/runtime/utils.h>
#include <tvm/runtime/ndarray.h>

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

AipuDriver::AipuDriver() {
  ctx_ = ContextAlloc();
  return;
}

void AipuDriver::Init(const std::string& aipu_bin, const std::string& work_dir,
                      const std::string& target, const std::string& umd_dtcm_sz) {
  work_dir_ = work_dir;
  target_ = target;
  umd_dtcm_sz_ = umd_dtcm_sz;

  // Create directories recursively and ignore the directories exist error.
  CreateDirectories(work_dir_);

  // For X2 the output directory of driver will be created inside "aipu_config_global" or
  // "aipu_load_graph_helper", and it will use the current directory, so change the current
  // directory to the working directory temporarily.
  std::string old_cur_dir = GetCwd();
  ChDir(work_dir_);

  // These items are independent with graph ID and job ID, they must be
  // configured firstly, because they are used during loading the graph.
  ConfigEnvItems();

  status_ = aipu_load_graph_helper(ctx_, aipu_bin.c_str(), aipu_bin.size(), &graph_id_);
  AIPU_DRIVER_HANDLE_ERROR(status_);

  ChDir(old_cur_dir);

  // These items must be configured first, otherwise the invocation of
  // "aipu_create_job" will failed for simulaltor.
  ConfigGraphItems();

  // Create job with the loaded AIPU binary program.
  status_ = aipu_create_job(ctx_, graph_id_, &job_id_, &job_cfg_);
  AIPU_DRIVER_HANDLE_ERROR(status_);

  ConfigJobItems();
  return;
}

void AipuDriver::SetInputs(const std::vector<DLTensor*>& inputs) {
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    if (shared_inputs_idx.find(i) != shared_inputs_idx.end()) {
      continue;
    }
    ICHECK(inputs[i]->data != nullptr);

    DLTensor* in_tensor = inputs[i];
    NDArray nd_arr = NDArray::FromExternalDLTensor(*in_tensor);
    if (in_tensor->device.device_type != kDLCPU) {
      nd_arr = NDArray::NewFromDLTensor(in_tensor, Device{kDLCPU, 0});
    }

    status_ = aipu_load_tensor(ctx_, job_id_, i, nd_arr->data);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }
  return;
}

void AipuDriver::GetOutputs(const std::vector<DLTensor*>& outputs) {
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    if (shared_outputs_idx.find(i) != shared_outputs_idx.end()) {
      continue;
    }

    DLTensor* out_tensor = outputs[i];
    NDArray nd_arr = NDArray::FromExternalDLTensor(*out_tensor);
    if (out_tensor->device.device_type != kDLCPU) {
      nd_arr = NDArray::NewFromDLTensor(out_tensor, Device{kDLCPU, 0});
    }

    status_ = aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_OUTPUT, i, nd_arr->data);
    AIPU_DRIVER_HANDLE_ERROR(status_);

    if (out_tensor->device.device_type != kDLCPU) {
      nd_arr.CopyTo(out_tensor);
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
    case AIPU_DATA_TYPE_f16:
      return DataType::Float(16);
    case AIPU_DATA_TYPE_f32:
      return DataType::Float(32);
    case AIPU_DATA_TYPE_f64:
      return DataType::Float(64);
    default:
      LOG(FATAL) << "Unsupported AIPU driver data type: " << type_code;
      throw;  // unreachable, written to stop compiler warning
  }
}

std::vector<ParamInfo> AipuDriver::GetParamInfo(bool is_input) {
  std::vector<ParamInfo> ret;
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = is_input ? AIPU_TENSOR_TYPE_INPUT : AIPU_TENSOR_TYPE_OUTPUT;
  status_ = aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  for (uint32_t i = 0; i < count; ++i) {
    aipu_tensor_desc_t desc = {0};
    status_ = aipu_get_tensor_descriptor(ctx_, graph_id_, tensor_type, i, &desc);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    ret.push_back(ParamInfo(CreateDataType(desc.data_type), desc.size));
  }
  return ret;
}

void AipuDriver::SetInputShared(uint64_t* inputs_pa) {
#ifndef __QNX__
  if (inputs_pa == nullptr) {
    return;
  }
  shared_inputs_idx.clear();
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = AIPU_TENSOR_TYPE_INPUT;
  aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);

  for (uint32_t idx = 0; idx < count; idx++) {
    if (inputs_pa[idx] != 0) {
      aipu_shared_tensor_info_t shared_tensor_info;
      memset(&shared_tensor_info, 0, sizeof(shared_tensor_info));
      shared_tensor_info.id = job_id_;
      shared_tensor_info.type = AIPU_TENSOR_TYPE_INPUT;
      shared_tensor_info.tensor_idx = idx;
      shared_tensor_info.pa = inputs_pa[idx];
      shared_tensor_info.shared_case_type = AIPU_SHARE_BUF_IN_ONE_PROCESS;
      AIPU_DRIVER_HANDLE_ERROR(aipu_specify_iobuf(ctx_, job_id_, &shared_tensor_info));
      shared_inputs_idx.insert(idx);
    }
  }
#endif
}

void AipuDriver::SetInputShared(int* inputs_fds) {
#ifndef __QNX__
  if (inputs_fds == nullptr) {
    return;
  }
  shared_inputs_idx.clear();
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = AIPU_TENSOR_TYPE_INPUT;
  aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);
  // if fd which means it is dma buf
  for (uint32_t idx = 0; idx < count; idx++) {
    if (inputs_fds[idx] > 0) {
      aipu_shared_tensor_info_t shared_tensor_info;
      memset(&shared_tensor_info, 0, sizeof(shared_tensor_info));

      shared_tensor_info.type = AIPU_TENSOR_TYPE_INPUT;
      shared_tensor_info.tensor_idx = idx;
      shared_tensor_info.dmabuf_fd = inputs_fds[idx];
      shared_tensor_info.shared_case_type = AIPU_SHARE_BUF_DMABUF;
      shared_tensor_info.offset_in_dmabuf = 0;
      AIPU_DRIVER_HANDLE_ERROR(aipu_specify_iobuf(ctx_, job_id_, &shared_tensor_info));
      shared_inputs_idx.insert(idx);
    }
  }
#endif
}

void AipuDriver::MarkOutputShared(uint64_t* outputs_pa) {
#ifndef __QNX__
  // shared_outputs_idx
  if (outputs_pa == nullptr) {
    return;
  }
  shared_outputs_idx.clear();
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = AIPU_TENSOR_TYPE_OUTPUT;
  aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);

  for (uint32_t idx = 0; idx < count; idx++) {
    if (outputs_pa[idx] != 0xFFFFFFFFFFFFFFFF) {
      aipu_tensor_desc_t desc;
      AIPU_DRIVER_HANDLE_ERROR(
          aipu_get_tensor_descriptor(ctx_, graph_id_, AIPU_TENSOR_TYPE_OUTPUT, idx, &desc));
      uint32_t buf_size = desc.size;
      aipu_share_buf_t shared_buf = {0};
      shared_buf.size = buf_size;
      shared_buf.mem_type = AIPU_MEM_REGION_SRAM;
      AIPU_DRIVER_HANDLE_ERROR(aipu_ioctl(ctx_, AIPU_IOCTL_ALLOC_SHARE_BUF, &shared_buf));
      shared_outputs_buf.push_back(shared_buf);

      aipu_shared_tensor_info_t shared_tensor_info;
      memset(&shared_tensor_info, 0, sizeof(shared_tensor_info));
      shared_tensor_info.id = job_id_;
      shared_tensor_info.type = AIPU_TENSOR_TYPE_OUTPUT;
      shared_tensor_info.tensor_idx = idx;
      shared_tensor_info.pa = shared_buf.pa;
      shared_tensor_info.shared_case_type = AIPU_SHARE_BUF_IN_ONE_PROCESS;
      AIPU_DRIVER_HANDLE_ERROR(aipu_specify_iobuf(ctx_, job_id_, &shared_tensor_info));
      outputs_pa[idx] = shared_buf.pa;
      shared_outputs_idx.insert(idx);
    }
  }
#endif
}

void AipuDriver::MarkOutputShared(int* outputs_fds) {
#ifndef __QNX__
  // shared_outputs_idx
  if (outputs_fds == nullptr) {
    return;
  }
  shared_outputs_idx.clear();
  uint32_t count = 0;
  aipu_tensor_type_t tensor_type = AIPU_TENSOR_TYPE_OUTPUT;
  aipu_get_tensor_count(ctx_, graph_id_, tensor_type, &count);

  for (uint32_t idx = 0; idx < count; idx++) {
    if (outputs_fds[idx] > 0) {
      aipu_shared_tensor_info_t shared_tensor_info;
      memset(&shared_tensor_info, 0, sizeof(shared_tensor_info));

      shared_tensor_info.type = AIPU_TENSOR_TYPE_OUTPUT;
      shared_tensor_info.tensor_idx = idx;
      shared_tensor_info.dmabuf_fd = outputs_fds[idx];
      shared_tensor_info.shared_case_type = AIPU_SHARE_BUF_DMABUF;
      shared_tensor_info.offset_in_dmabuf = 0;
      AIPU_DRIVER_HANDLE_ERROR(aipu_specify_iobuf(ctx_, job_id_, &shared_tensor_info));
      shared_outputs_idx.insert(idx);
    }
  }
#endif
}

void AipuDriver::DeinitGraphJob() {
  // The valid job id isn't possible to be 0 and will be more greater than 0.
  if (job_id_ != 0) {
    status_ = aipu_clean_job(ctx_, job_id_);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    job_id_ = 0;
  }

  // The valid graph id isn't possible to be 0 and will be more greater than 0.
  if (graph_id_ != 0) {
    status_ = aipu_unload_graph(ctx_, graph_id_);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    graph_id_ = 0;
  }

#ifndef __QNX__
  for (uint32_t idx = 0; idx < shared_outputs_buf.size(); idx++) {
    AIPU_DRIVER_HANDLE_ERROR(aipu_ioctl(ctx_, AIPU_IOCTL_FREE_SHARE_BUF, &shared_outputs_buf[idx]));
  }
  shared_outputs_buf.clear();
#endif
}

AipuDriver::~AipuDriver() {
  DeinitGraphJob();
  ContextReset();
}

}  // namespace runtime
}  // namespace tvm
