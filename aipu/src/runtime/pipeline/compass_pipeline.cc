// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/runtime/compass_pipeline.cc
 */
#include <tvm/runtime/container/array.h>

#include "../../../../src/runtime/pipeline/pipeline_executor.h"

namespace tvm {
namespace runtime {

using module_shared_info = std::pair<std::vector<std::string>, std::vector<std::string>>;

class TVM_DLL CompassPipelineExecutor : public PipelineExecutor {
 public:
  std::vector<Module> CreateGraphModulesWithSharedInfo(ModuleConfig&, const std::string&);
};

std::vector<module_shared_info> GetSharedInfo(const std::string& shared_json) {
  dmlc::JSONObjectReadHelper helper;
  std::vector<std::map<std::string, std::vector<std::string>>> modules;

  std::istringstream is(shared_json);
  dmlc::JSONReader reader(&is);
  helper.DeclareField("modules", &modules);
  helper.ReadAllFields(&reader);

  std::vector<module_shared_info> ret;
  for (uint32_t i = 0; i < modules.size(); i++) {
    module_shared_info val;
    val.first = std::move(modules[i]["inputs"]);
    val.second = std::move(modules[i]["outputs"]);
    ret.push_back(val);
  }
  return ret;
}

Array<Module> ReconfigCompassModuleWithSharedInfo(const Array<Module>& modules,
                                                  const std::string& shared_json) {
  Array<Module> ret;
  std::vector<module_shared_info> shared_infos = GetSharedInfo(shared_json);
  for (auto& m : modules) {
    ret.push_back(m);
  }

  ssize_t size = ret.size();
  ICHECK(ret.size() == shared_infos.size())
      << "The module config number should the same with modules number";
  std::map<std::string, uint64_t> record_address;
  for (auto& info : shared_infos) {
    for (auto& name : info.first) {
      if (name != "not_shared") {
        record_address[name] = 0;
      }
    }
    for (auto& name : info.second) {
      if (name != "not_shared") {
        record_address[name] = 0;
      }
    }
  }

  for (uint32_t idx = 0; idx < size; idx++) {
    const auto& shared_info = shared_infos[idx];
    auto lib = ret[idx];
    auto fp = lib.GetFunction("compass_drvier_reinit_with_shared", true);
    if (fp != nullptr) {
      std::vector<uint64_t> inputs, outputs;
      inputs.resize(shared_info.first.size());
      outputs.resize(shared_info.second.size());
      for (uint32_t id = 0; id < shared_info.first.size(); id++) {
        const std::string& inp_name = shared_info.first[id];
        if (inp_name == "not_shared") {
          inputs[id] = 0;
        } else {
          inputs[id] = record_address[inp_name];
        }
      }
      for (uint32_t id = 0; id < shared_info.second.size(); id++) {
        const std::string& out_name = shared_info.second[id];
        if (out_name == "not_shared") {
          outputs[id] = 0xFFFFFFFFFFFFFFFF;
        } else {
          outputs[id] = 0;
        }
      }
      Device cpu_dev;
      cpu_dev.device_type = kDLCPU;
      cpu_dev.device_id = 0;
      int64_t inp_size = inputs.size();
      int64_t out_size = outputs.size();
      NDArray input_arr = NDArray::Empty({inp_size}, DLDataType{kDLUInt, 64, 1}, cpu_dev);
      NDArray output_arr = NDArray::Empty({out_size}, DLDataType{kDLUInt, 64, 1}, cpu_dev);

      input_arr.CopyFromBytes(inputs.data(), inputs.size() * sizeof(uint64_t));
      output_arr.CopyFromBytes(outputs.data(), outputs.size() * sizeof(uint64_t));

      fp(input_arr, output_arr);
      for (uint32_t id = 0; id < shared_info.second.size(); id++) {
        const std::string& out_name = shared_info.second[id];
        if (out_name != "not_shared") {
          uint64_t addr = (reinterpret_cast<uint64_t*>(output_arr->data))[id];
          record_address[out_name] = addr;
        }
      }
    }
  }
  return ret;
}

std::vector<Module> CompassPipelineExecutor::CreateGraphModulesWithSharedInfo(
    ModuleConfig& mod_config, const std::string& shared_json) {
  const PackedFunc* graph_executor_create = Registry::Get("tvm.graph_executor.create");
  std::vector<Module> ret;
  ret.resize(mod_config.size());
  uint32_t size = mod_config.size();
  Array<Module> ret_reconf;
  for (uint32_t idx = 0; idx < size; idx++) {
    const auto& config = mod_config[idx];
    auto lib = Module::LoadFromFile(config.lib_name.c_str());
    ret_reconf.push_back(lib);
  }

  ret_reconf = ReconfigCompassModuleWithSharedInfo(ret_reconf, shared_json);

  for (uint32_t idx = 0; idx < size; idx++) {
    const auto& config = mod_config[idx];
    auto lib = ret_reconf[idx];

    std::ifstream ifJson(config.json_name.c_str());
    if (ifJson.fail()) {
      LOG(FATAL) << "json file not found: " << config.json_name;
    }
    const std::string json((std::istreambuf_iterator<char>(ifJson)),
                           std::istreambuf_iterator<char>());

    // Create a graph executor.
    std::istringstream istr(config.dev);
    std::string str;
    int device_type = 1, device_id = 0;
    while (getline(istr, str, ';')) {
      std::istringstream istr_dev(str);
      std::string str_temp;
      if (getline(istr_dev, str_temp)) {
        device_type = stoi(str_temp);
      }
      if (getline(istr_dev, str_temp)) {
        device_id = stoi(str_temp);
      }
    }
    Module graph_module = (*graph_executor_create)(json, lib, device_type, device_id);
    // Load parameters.
    TVMByteArray params_arr;
    const char* params_file_name = config.params_name.c_str();
    std::ifstream if_param(params_file_name);
    if (if_param.fail()) {
      LOG(FATAL) << "params file not found: " << params_file_name;
    }
    const std::string params((std::istreambuf_iterator<char>(if_param)),
                             std::istreambuf_iterator<char>());
    params_arr.data = params.c_str();
    params_arr.size = params.length();
    auto load_params = graph_module.GetFunction("load_params");
    load_params(params_arr);

    ret[idx] = graph_module;
  }
  return ret;
}

Module CompassPipelineExecutorLoad(const std::string& load_json, const std::string& pipeline_json,
                                   const std::string& shared_json) {
  auto exec = make_object<CompassPipelineExecutor>();
  std::istringstream is(load_json);
  dmlc::JSONReader reader(&is);
  ModuleConfig& mod_config = exec->LoadModuleConfig(&reader);
  ICHECK(!mod_config.empty()) << "The module config is empty.";

  std::vector<Module> modules = exec->CreateGraphModulesWithSharedInfo(mod_config, shared_json);

  exec->Init(modules, pipeline_json);
  return Module(exec);
}

std::string file_to_string(const std::string& filename) {
  std::ifstream ifs(filename);

  ICHECK(ifs) << "Failed to open file " << filename;
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

Module CompassPipelineExecutorLoadFromConfig(const std::string& config_file) {
  std::ifstream is(config_file);
  ICHECK(is) << "Failed to open file " << config_file;
  dmlc::JSONReader reader(&is);

  dmlc::JSONObjectReadHelper helper;
  std::string load_config, pipeline_config, shared_config;
  helper.DeclareField("load_config", &load_config);
  helper.DeclareField("pipeline_config", &pipeline_config);
  helper.DeclareField("shared_config", &shared_config);
  helper.ReadAllFields(&reader);

  load_config = file_to_string(load_config);
  pipeline_config = file_to_string(pipeline_config);
  shared_config = file_to_string(shared_config);
  return CompassPipelineExecutorLoad(load_config, pipeline_config, shared_config);
}

TVM_REGISTER_GLOBAL("compass_pipeline.load").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CompassPipelineExecutorLoadFromConfig(args[0]);
});
TVM_REGISTER_GLOBAL("compass_pipeline.reconfig_with_shared_info")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = ReconfigCompassModuleWithSharedInfo(args[0], args[1]);
    });
}  // namespace runtime
}  // namespace tvm
