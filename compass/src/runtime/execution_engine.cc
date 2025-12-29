// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/execution_engine.cc
 */
#include <compass/tvm/runtime/execution_engine.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

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

ObjectPtr<ExecutionEngineObj> CreateExecutionEngineObj(Module compiled_model,
                                                       const std::vector<Device>& devices,
                                                       bool with_profile) {
  return make_object<VmExecutionEngineObj>(compiled_model, devices, with_profile);
}

ExecutionEngine::ExecutionEngine(Module compiled_model, const std::vector<Device>& devices) {
  ICHECK_NE(devices.size(), 0) << "The device list can't be empty.";
  data_ = CreateExecutionEngineObj(compiled_model, devices, false);
}

ExecutionEngine::ExecutionEngine(const std::string& compiled_model_dir, Device device)
    : ExecutionEngine(compiled_model_dir, std::vector<Device>{device}) {}

ExecutionEngine::ExecutionEngine(const std::string& compiled_model_dir,
                                 const std::vector<Device>& devices)
    : ExecutionEngine(Module::LoadFromFile(compiled_model_dir), devices) {}

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

TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      Module compiled_model;
      if (std::optional<Module> opt_mod = args[0].as<Module>()) {
        compiled_model = opt_mod.value();
      } else {
        compiled_model = Module::LoadFromFile(args[0].cast<std::string>());
      }

      bool with_profile = args[1].cast<bool>();
      // Structure "Device" isn't derived from "ObjectRef", i.e., multiple "Device" can't be
      // represented by "Array<Device>", so it can't be passed by a single "AnyView".
      std::vector<Device> devices;
      for (int i = 2; i < args.size(); ++i) {
        devices.emplace_back(args[i].cast<Device>());
      }

      *rv = ExecutionEngine(CreateExecutionEngineObj(compiled_model, devices, with_profile));
      return;
    });

TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_GetInputDevice")
    .set_body_typed([](ExecutionEngine ee, int idx) { return ee->GetInputDevice(idx); });
TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_GetEntryParamDataType")
    .set_body_typed([](ExecutionEngine ee, int idx) { return ee->get_entry_param_dtype(idx); });
TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_SetInputs")
    .set_body_typed([](ExecutionEngine ee, Array<NDArray> args) { return ee.SetInputs(args); });
TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_Execute")
    .set_body_typed([](ExecutionEngine ee) { return ee.Run(); });
TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_GetTypeKey")
    .set_body_typed([](ExecutionEngine ee) { return ee->GetTypeKey(); });
TVM_FFI_REGISTER_GLOBAL("compass.runtime.ExecutionEngine_GetExecutor")
    .set_body_typed([](ExecutionEngine ee) { return ee->executor; });

// This statement is necessary, otherwise front-end languages can't associate
// their corresponding classes with it through "tvm.register_object".
TVM_REGISTER_OBJECT_TYPE(VmExecutionEngineObj);

VmExecutionEngineObj::VmExecutionEngineObj(Module executable, const std::vector<Device>& devices,
                                           bool with_profile) {
  std::vector<Device> updated_devices = devices;
  if (RemoveRPCSessionMask(devices.back()).device_type != kDLCPU) {
    // CPU is required for executing shape functions.
    updated_devices.push_back({kDLCPU, 0});
  }
  std::vector<AnyView> packed_args;
  for (size_t i = 0; i < updated_devices.size(); ++i) {
    Device dev = RemoveRPCSessionMask(updated_devices[i]);
    packed_args.push_back(static_cast<int>(dev.device_type));
    packed_args.push_back(dev.device_id);
    packed_args.push_back(static_cast<int>(AllocatorType::kPooled));
  }
  String func_name = with_profile == true ? "vm_profiler_load_executable" : "vm_load_executable";
  auto executor = executable.GetFunction(func_name)().cast<Module>();
  Any rv;
  executor.GetFunction("vm_initialization").CallPacked(packed_args.data(), packed_args.size(), &rv);
  this->executor = executor;
  this->devices_ = devices;
  this->set_input_ = executor.GetFunction("set_input");
  this->get_output_ = executor.GetFunction("get_output");
  this->get_output_arity_ = executor.GetFunction("get_output_arity");
  this->invoke_stateful_ = executor.GetFunction("invoke_stateful");
  return;
}

Device VmExecutionEngineObj::GetInputDevice(int idx) const {
  return this->devices_[0];  // Now Relax only support all arguments on the same single device.
}

void VmExecutionEngineObj::SetInputs() {
  auto& args = this->on_device_args;
  std::vector<AnyView> packed_args{"main"};
  packed_args.insert(packed_args.end(), args.begin(), args.end());
  Any rv;
  this->set_input_.CallPacked(packed_args.data(), packed_args.size(), &rv);
  return;
}

Array<NDArray> VmExecutionEngineObj::Execute() {
  // Currently RPC can't pass through a tuple, so here can't call the function "invoke_closure".
  this->invoke_stateful_("main");

  Array<NDArray> outputs;
  // The output count must be got here each time, it can't be stored ahead like
  // graph executor, because vm can support dynamic function which output count
  // can be different between different invocations.
  // TODO(compass_tvm): Support the arbitrary nested situation later.
  auto out_arity = this->get_output_arity_("main").cast<int>();
  if (out_arity == -1) {
    outputs.push_back(this->get_output_("main").cast<NDArray>());
  } else {
    for (int i = 0; i < out_arity; ++i) {
      outputs.push_back(this->get_output_("main", i).cast<NDArray>());
    }
  }
  return outputs;
}

}  // namespace runtime
}  // namespace tvm
