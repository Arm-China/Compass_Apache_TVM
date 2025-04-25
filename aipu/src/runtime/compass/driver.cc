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
                      const std::string& target, const std::string& umd_dtcm_sz,
                      const std::string& func_name, const std::string& extra_path) {
  work_dir_ = work_dir;
  target_ = target;
  umd_dtcm_sz_ = umd_dtcm_sz;
  func_name_ = func_name;

  // Create directories recursively and ignore the directories exist error.
  CreateDirectories(work_dir_);

  // These items are independent with graph ID and job ID, they must be
  // configured firstly, because they are used during loading the graph.
  ConfigEnvItems(aipu_bin, extra_path);

  // These items must be configured first, otherwise the invocation of
  // "aipu_create_job" will failed for simulaltor.
  ConfigGraphItems();

  uint32_t ds_num = 0;
#ifndef __QNX__
  aipu_dynshape_num_t dynshape_num = {0};
  dynshape_num.graph_id = graph_id_;
  dynshape_num.ds_num = &ds_num;
  status_ = aipu_ioctl(ctx_, AIPU_IOCTL_GET_DS_NUM, &dynshape_num);
#endif
  // dynamic graph create job would fail if no input shape info.
  if (ds_num == 0) {
    // Create job with the loaded AIPU binary program.
    status_ = aipu_create_job(ctx_, graph_id_, &job_id_, &job_cfg_);
    AIPU_DRIVER_HANDLE_ERROR(status_);

    ConfigJobItems();
  }
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

void AipuDriver::SetOutputs(const std::vector<DLTensor*>& outputs) {
#ifndef __QNX__
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ICHECK(outputs[i]->data != nullptr);

    DLTensor* out_tensor = outputs[i];
    NDArray nd_arr = NDArray::FromExternalDLTensor(*out_tensor);
    if (out_tensor->device.device_type != kDLCPU) {
      nd_arr = NDArray::NewFromDLTensor(out_tensor, Device{kDLCPU, 0});
    }

    status_ = aipu_load_output_tensor(ctx_, job_id_, i, nd_arr->data);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }
  return;
#endif
}

void AipuDriver::SetInputsWithDynamicShape(const std::vector<DLTensor*>& inputs) {
#ifndef __QNX__
  aipu_dynshape_num_t dynshape_num = {0};
  dynshape_num.graph_id = graph_id_;
  uint32_t ds_num = 0;
  dynshape_num.ds_num = &ds_num;
  status_ = aipu_ioctl(ctx_, AIPU_IOCTL_GET_DS_NUM, &dynshape_num);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  ICHECK(ds_num == inputs.size());
  aipu_dynshape_param_t dynshape_param = {0};
  dynshape_param.input_shape_cnt = ds_num;
  std::vector<std::vector<uint32_t>> dyn_shapes;
  dyn_shapes.resize(inputs.size());
  std::vector<aipu_dynshape_item_t> config_items;
  config_items.resize(inputs.size());
  dynshape_param.shape_items = &config_items[0];
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    aipu_dynshape_dim_num_t dynshape_dim_num = {0};
    dynshape_dim_num.graph_id = graph_id_;
    dynshape_dim_num.ds_idx = i;
    dynshape_dim_num.max_threshhold = true;
    uint32_t ds_dim_num = 0;
    dynshape_dim_num.ds_dim_num = &ds_dim_num;
    status_ = aipu_ioctl(ctx_, AIPU_IOCTL_GET_DS_DIM_NUM, &dynshape_dim_num);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    ICHECK(static_cast<int32_t>(ds_dim_num) == inputs[i]->ndim);

    std::vector<uint32_t> ds_dim_data;
    ds_dim_data.resize(ds_dim_num);

    aipu_dynshape_info_t dynshape_info = {0};
    dynshape_info.graph_id = graph_id_;
    dynshape_info.ds_idx = i;
    dynshape_info.max_threshhold = true;
    dynshape_info.ds_data = &ds_dim_data[0];
    status_ = aipu_ioctl(ctx_, AIPU_IOCTL_GET_DS_INFO, &dynshape_info);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    config_items[i].ds_idx = i;
    for (uint32_t dim_id = 0; dim_id < ds_dim_num; dim_id++) {
      ICHECK_GE(ds_dim_data[dim_id], inputs[i]->shape[dim_id]);
      dyn_shapes[i].push_back(inputs[i]->shape[dim_id]);
    }
    config_items[i].ds_data = &dyn_shapes[i][0];
  }

  // need recreate job to flush the io size
  if (job_id_ != 0) {
    status_ = aipu_clean_job(ctx_, job_id_);
    AIPU_DRIVER_HANDLE_ERROR(status_);
    job_id_ = 0;
  }
  job_cfg_.dynshape = &dynshape_param;
  status_ = aipu_create_job(ctx_, graph_id_, &job_id_, &job_cfg_);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  ConfigJobItems();

  SetInputs(inputs);
  job_cfg_.dynshape = nullptr;
  return;
#endif
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
    case AIPU_DATA_TYPE_F16:
      return DataType::Float(16);
    case AIPU_DATA_TYPE_F32:
      return DataType::Float(32);
    case AIPU_DATA_TYPE_F64:
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

std::vector<int64_t> AipuDriver::GetOutputShape(uint32_t idx) {
  std::vector<int64_t> ret;
#ifndef __QNX__
  aipu_tensor_desc_t desc = {0};
  status_ =
      aipu_get_tensor_descriptor(ctx_, graph_id_, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, idx, &desc);
  AIPU_DRIVER_HANDLE_ERROR(status_);
  uint32_t dim = desc.size / sizeof(uint32_t);
  std::vector<uint32_t> shape(dim, 0);

  aipu_get_tensor(ctx_, job_id_, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, idx, &shape[0]);
  for (auto dim_val : shape) {
    ret.push_back(dim_val);
  }
#endif
  return ret;
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
