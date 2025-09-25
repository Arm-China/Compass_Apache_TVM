// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/include/compass/tvm/runtime/module.h
 */
#ifndef COMPASS_TVM_RUNTIME_MODULE_H_
#define COMPASS_TVM_RUNTIME_MODULE_H_

#include <compass/tvm/runtime/driver.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>

#include <string>

namespace tvm {
namespace runtime {

struct CompassModule : public ModuleNode {
  // Things that will interface with user directly.
  void Init();
  void ReviseCpsBinPath(const std::string& base_dir);

  // Member variables need to be serialized.
  std::string cps_bin_path;  // The path of the Compass executable.
  std::string func_name;     // The name of original function which generate this module.
  bool with_profile;   // Whether the Compass executable is built with profiling enabled or not.
  std::string target;  // The target that the Compass executable is built to, used by simulator.
  std::string umd_dtcm_sz;  // The size of the Data Tightly Coupled Memory, used by simulator.

  // Internal supporting.
  // TVM framework relevant begin.
  const char* type_key() const final { return "compass.runtime.CompassModule"; }
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }
  void SaveToBinary(dmlc::Stream* stream) final;
  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;
  // TVM framework relevant end.

  void GetOutputs(Array<NDArray> out_tensors);  // Get outputs args from Compass runtime.

  CompassDriver cps_driver_;
  // Meta data of input and output parameters, they are the quantized inputs and outputs generated
  // by Compass Optimizer.
  Array<ParamInfo> in_params_;
  Array<ParamInfo> out_params_;
  // Function implemented in the Python side, it will be called to dump inputs and outputs if "dump"
  // in configuration file or "CPS_TVM_RUNTIME_DUMP" in environment variable is set to True.
  std::optional<ffi::Function> dump_func_;
};

Array<Module> GetAllCompassModule(Module module);

}  // namespace runtime
}  // namespace tvm
#endif  // COMPASS_TVM_RUNTIME_MODULE_H_
