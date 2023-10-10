// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/runtime/execution_engine.cc
 */
#include <aipu/runtime/execution_engine.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>

namespace tvm {
namespace runtime {

ExecutionEngineObj::~ExecutionEngineObj() {}

Array<NDArray> ExecutionEngine::GetOnDeviceArgs(Array<NDArray> args) const {
  Array<NDArray> ret;
  for (int i = 0; i < static_cast<int>(args.size()); ++i) {
    Device param_dev = this->get()->GetInputDevice(i);
    const NDArray& arg = args[i];
    const Device& arg_dev = arg->device;
    if (arg_dev.device_type == param_dev.device_type) {
      if ((arg_dev.device_id == param_dev.device_id) ||
          (RemoveRPCSessionMask(arg_dev).device_type == kDLCPU)) {
        // The data can be accessed by the device.
        ret.push_back(arg);
        continue;
      }
    }
    // If can't ensure whether the data can be accessed by the device, then just
    // copy to the device memory space.
    ret.push_back(arg.CopyTo(param_dev));
  }
  return ret;
}

ExecutionEngine::ExecutionEngine(Module compiled_model, Device device)
    : ExecutionEngine(compiled_model, std::vector<Device>{device}) {}

ExecutionEngine::ExecutionEngine(Module compiled_model, const std::vector<Device>& devices) {
  if (compiled_model.GetFunction("vm_load_executable") != nullptr) {
    data_ = make_object<VmExecutionEngineObj>(compiled_model, devices);
    return;
  }
  data_ = make_object<GraphExecutionEngineObj>(compiled_model, devices);
  return;
}

ExecutionEngine::ExecutionEngine(const std::string& compiled_model_path, Device device)
    : ExecutionEngine(compiled_model_path, std::vector<Device>{device}) {}

ExecutionEngine::ExecutionEngine(const std::string& compiled_model_path,
                                 const std::vector<Device>& devices)
    : ExecutionEngine(Module::LoadFromFile(compiled_model_path), devices) {}

void ExecutionEngine::SetInputs(Array<NDArray> args) {
  if (args.empty()) return;
  auto* obj = const_cast<ExecutionEngineObj*>(this->get());
  obj->on_device_args = this->GetOnDeviceArgs(args);
  // "GetOnDeviceArgs" ensures the device can access the data of all arguments,
  // so shouldn't do data movement anymore from here.
  obj->SetInputs();
  return;
}

Array<NDArray> ExecutionEngine::Run(Array<NDArray> args) {
  if (args.empty() == false) this->SetInputs(args);
  auto* obj = const_cast<ExecutionEngineObj*>(this->get());
  Array<NDArray> outputs = obj->Execute();
  obj->on_device_args.clear();
  return outputs;
}

void ExecutionEngine::SetInputs(NDArray arg) { return this->SetInputs(Array<NDArray>{arg}); }

Array<NDArray> ExecutionEngine::Run(NDArray arg) { return this->Run(Array<NDArray>{arg}); }

TVM_REGISTER_GLOBAL("runtime.ExecutionEngine").set_body([](TVMArgs args, TVMRetValue* rv) {
  TVMArgValue arg0 = args[0];
  // Structure "Device" isn't derived from "ObjectRef", i.e., multiple "Device"
  // can't be represented by "Array<Device>", so it can't be passed by a single
  // "TVMArgValue".
  std::vector<Device> devices;
  for (int i = 1; i < args.num_args; ++i) {
    devices.emplace_back(args[i]);
  }

  if (String::CanConvertFrom(arg0)) {
    *rv = ExecutionEngine(arg0.operator std::string(), devices);
  } else {
    *rv = ExecutionEngine(arg0.operator Module(), devices);
  }
  return;
});

TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_GetInputDevice")
    .set_body_typed([](ExecutionEngine ee, int idx) { return ee->GetInputDevice(idx); });
TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_GetEntryParamDataType")
    .set_body_typed([](ExecutionEngine ee, int idx) { return ee->get_entry_param_dtype(idx); });
TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_SetInputs")
    .set_body_typed([](ExecutionEngine ee, Array<NDArray> args) { return ee.SetInputs(args); });
TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_Execute").set_body_typed([](ExecutionEngine ee) {
  return ee.Run();
});
TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_GetTypeKey").set_body_typed([](ExecutionEngine ee) {
  return ee->GetTypeKey();
});
TVM_REGISTER_GLOBAL("runtime.ExecutionEngine_GetExecutor").set_body_typed([](ExecutionEngine ee) {
  return ee->executor;
});

// This statement is necessary, otherwise front-end languages can't associate
// their corresponding classes with it through "tvm.register_object".
TVM_REGISTER_OBJECT_TYPE(VmExecutionEngineObj);

VmExecutionEngineObj::VmExecutionEngineObj(Module executable, const std::vector<Device>& devices) {
  std::vector<Device> updated_devices = devices;
  if (std::none_of(devices.begin(), devices.end(), [](const Device& device) {
        return RemoveRPCSessionMask(device).device_type == kDLCPU;
      })) {
    // CPU is required for executing shape functions.
    updated_devices.push_back({kDLCPU, 0});
  }
  const size_t arg_cnt = updated_devices.size() * 3;
  auto values = std::vector<TVMValue>(arg_cnt);
  auto type_codes = std::vector<int>(arg_cnt);
  auto arg_setter = TVMArgsSetter(values.data(), type_codes.data());
  for (size_t i = 0; i < updated_devices.size(); ++i) {
    Device dev = RemoveRPCSessionMask(updated_devices[i]);
    arg_setter(i, static_cast<int>(dev.device_type));
    arg_setter(i + 1, dev.device_id);
    arg_setter(i + 2, static_cast<int>(vm::AllocatorType::kPooled));
  }
  auto args = TVMArgs(values.data(), type_codes.data(), arg_cnt);
  TVMRetValue ret_value;
  Module executor = executable.GetFunction("vm_load_executable")();
  executor.GetFunction("init").CallPacked(args, &ret_value);
  this->executor = executor;
  this->get_input_device_ = executor.GetFunction("get_input_device");
  this->get_entry_param_dtype = executor.GetFunction("get_entry_param_dtype");
  this->set_input_ = executor.GetFunction("set_input");
  this->get_output_ = executor.GetFunction("get_output");
  this->get_num_outputs_ = executor.GetFunction("get_num_outputs");
  this->invoke_stateful_ = executor.GetFunction("invoke_stateful");
  return;
}

Device VmExecutionEngineObj::GetInputDevice(int idx) const {
  return this->get_input_device_("main", idx);
}

void VmExecutionEngineObj::SetInputs() {
  auto& args = this->on_device_args;
  const size_t arg_cnt = args.size() + 1;
  auto values = std::vector<TVMValue>(arg_cnt);
  auto type_codes = std::vector<int>(arg_cnt);
  auto arg_setter = TVMArgsSetter(values.data(), type_codes.data());
  arg_setter(0, "main");
  for (size_t i = 1; i < arg_cnt; ++i) {
    arg_setter(i, args[i - 1]);
  }
  TVMRetValue ret_value;
  this->set_input_.CallPacked(TVMArgs(values.data(), type_codes.data(), arg_cnt), &ret_value);
  return;
}

Array<NDArray> VmExecutionEngineObj::Execute() {
  // Currently RPC can't pass through the instance of "runtime::ADT", so here
  // can't call the function "invoke".
  this->invoke_stateful_("main");

  Array<NDArray> outputs;
  // The output count must be got here each time, it can't be stored ahead like
  // graph executor, because vm can support dynamic function which output count
  // can be different between different invocations.
  for (int64_t i = 0; i < this->get_num_outputs_().operator int64_t(); ++i) {
    outputs.push_back(this->get_output_(i));
  }
  return outputs;
}

// This statement is necessary, otherwise front-end languages can't associate
// their corresponding classes with it through "tvm.register_object".
TVM_REGISTER_OBJECT_TYPE(GraphExecutionEngineObj);

GraphExecutionEngineObj::GraphExecutionEngineObj(Module graph_ex_factory,
                                                 const std::vector<Device>& devices) {
  // Structure "Device" isn't derived from "ObjectRef", i.e., multiple "Device"
  // can't be represented by "Array<Device>", so it can't be passed by a single
  // "TVMArgValue".
  const size_t arg_cnt = devices.size();
  auto values = std::vector<TVMValue>(arg_cnt);
  auto type_codes = std::vector<int>(arg_cnt);
  auto arg_setter = TVMArgsSetter(values.data(), type_codes.data());
  for (size_t i = 0; i < arg_cnt; ++i) {
    arg_setter(i, devices[i]);
  }
  TVMRetValue ret_value;
  auto args = TVMArgs(values.data(), type_codes.data(), arg_cnt);
  graph_ex_factory.GetFunction("default").CallPacked(args, &ret_value);
  Module executor = ret_value;
  this->executor = executor;
  this->get_input_device_ = executor.GetFunction("get_input_device");
  this->get_entry_param_dtype = executor.GetFunction("get_entry_param_dtype");
  this->set_input_zero_copy_ = executor.GetFunction("set_input_zero_copy");
  this->get_output_ = executor.GetFunction("get_output");
  this->out_cnt_ = executor.GetFunction("get_num_outputs")();
  this->run_ = executor.GetFunction("run");
  return;
}

Device GraphExecutionEngineObj::GetInputDevice(int idx) const {
  return this->get_input_device_(idx);
}

void GraphExecutionEngineObj::SetInputs() {
  auto& args = this->on_device_args;
  for (int i = 0; i < static_cast<int>(args.size()); ++i) {
    this->set_input_zero_copy_(i, args[i]);
  }
  return;
}

Array<NDArray> GraphExecutionEngineObj::Execute() {
  this->run_();

  Array<NDArray> outputs;
  for (int i = 0; i < this->out_cnt_; ++i) {
    outputs.push_back(this->get_output_(i));
  }
  return outputs;
}

}  // namespace runtime
}  // namespace tvm
