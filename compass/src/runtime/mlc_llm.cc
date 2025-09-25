// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/mlc_llm.cc
 */

#include <compass/tvm/runtime/driver.h>
#include <picojson.h>
#include <tvm/runtime/module.h>

#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "../../../../src/runtime/vm/kv_state.h"

namespace tvm {
namespace runtime {
namespace vm {

static Module cps_module;
static Array<NDArray> kvcaches;

TVM_FFI_REGISTER_GLOBAL("compass.runtime.mlc_llm.init").set_body_typed([](String model_path) {
  std::ifstream ifs(model_path + "/aipullm.json");
  if (!ifs) {
    // LOG(WARN) << "failed to find compass config file " << model_path + "/aipullm.json";
    return false;
  }
  std::string json_string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();
  picojson::value value;
  picojson::parse(value, json_string);

  const picojson::object& obj = value.get<picojson::object>();
  std::string target = obj.at("target").get<std::string>();
  std::string cps_bin_path = model_path + "/" + obj.at("aipu.bin").get<std::string>();

  static auto create_fn = ffi::Function::GetGlobalRequired("compass.runtime.CompassModule");
  cps_module = create_fn(cps_bin_path, "unknown", false, target, "").cast<Module>();
  static auto init_fn = ffi::Function::GetGlobalRequired("compass.runtime.init");
  init_fn(cps_module);

  int64_t layer_num = obj.at("layer_num").get<int64_t>();
  int64_t head_dim = obj.at("head_dim").get<int64_t>();
  int64_t kv_head_num = obj.at("kv_head_num").get<int64_t>();

  kvcaches.clear();
  Device cpu = {kDLCPU, 0};
  for (int i = 0; i < layer_num * 2; i++) {
    kvcaches.push_back(
        NDArray::Empty({1, kv_head_num, 0, head_dim}, DLDataType{kDLInt, 16, 1}, cpu));
  }

  return true;
});

TVM_FFI_REGISTER_GLOBAL("compass.runtime.mlc_llm.fused_layers")
    .set_body_typed([](NDArray hidden_status, AttentionKVCache kv_cache, NDArray o_data) {
      NDArray pos_dev = kv_cache->GetQueryPositions();

      Device cpu = {kDLCPU, 0};
      NDArray pos = NDArray::Empty({1, pos_dev->shape[0]}, DLDataType{kDLInt, 32, 1}, cpu);
      pos.CopyFrom(pos_dev);
      int64_t n_token = pos_dev->shape[0];
      int64_t hidden_dim = hidden_status->shape[2];

      DLDataType fp16 = DLDataType{kDLFloat, 16, 1};
      NDArray cpu_hidden_status = NDArray::Empty({1, n_token, hidden_dim}, fp16, cpu);
      cpu_hidden_status.CopyFrom(hidden_status);

      ffi::Function get_param_info = cps_module->GetFunction("compass_get_param_info");
      auto param_info = get_param_info(1, true).cast<ParamInfo>();

      DataType dtype = param_info->dtype;
      ffi::Function compass_run = cps_module->GetFunction("compass_dynamic_run");

      int32_t start_pos = *static_cast<int32_t*>(pos->data);
      int64_t past_context_size = kvcaches[0]->shape[2];

      if (start_pos < past_context_size) {
        Array<NDArray> newcaches;
        for (uint32_t idx = 0; idx < kvcaches.size(); idx++) {
          NDArray origin = kvcaches[idx];
          int64_t kv_head_num = origin->shape[1];
          int64_t head_dim = origin->shape[3];
          NDArray new_cache =
              NDArray::Empty({1, kv_head_num, start_pos, head_dim}, origin->dtype, cpu);
          int16_t* origin_ptr = static_cast<int16_t*>(origin->data);
          int16_t* new_ptr = static_cast<int16_t*>(new_cache->data);
          for (int idx_token = 0; idx_token < kv_head_num; idx_token++) {
            memcpy(new_ptr, origin_ptr, sizeof(int16_t) * start_pos * head_dim);
            origin_ptr += past_context_size * head_dim;
            new_ptr += start_pos * head_dim;
          }
          newcaches.push_back(new_cache);
        }
        kvcaches = newcaches;
        past_context_size = start_pos;
      }

      NDArray mask = NDArray::Empty({1, 1, n_token, past_context_size + n_token}, dtype, cpu);

      int16_t* workspace = static_cast<int16_t*>(mask->data);
      for (int i = 0; i < n_token; i++) {
        int16_t* data = workspace + (past_context_size + n_token) * i;
        memset(data, 0, (past_context_size + n_token) * sizeof(int16_t));
        for (int j = past_context_size + i + 1; j < past_context_size + n_token; j++) {
          data[j] = std::numeric_limits<int16_t>::min();
        }
      }

      std::vector<AnyView> inputs;
      inputs.push_back(cpu_hidden_status);
      inputs.push_back(mask);
      inputs.push_back(pos);
      for (auto arr : kvcaches) {
        inputs.push_back(arr);
      }
      Any rv;
      compass_run.CallPacked(inputs.data(), inputs.size(), &rv);
      auto outs = rv.cast<Array<NDArray>>();
      kvcaches.clear();
      for (auto it = outs.begin() + 1; it != outs.end(); it++) {
        kvcaches.push_back(*it);
      }
      o_data.CopyFrom(outs[0]);
    });
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
