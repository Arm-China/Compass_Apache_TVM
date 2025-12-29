// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/include/compass/tvm/runtime/execution_engine.h
 */
#ifndef COMPASS_TVM_RUNTIME_EXECUTION_ENGINE_H_
#define COMPASS_TVM_RUNTIME_EXECUTION_ENGINE_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {

class ExecutionEngineObj : public Object {
  // Things that will interface with user directly.
 public:
  Module executor;

  virtual ~ExecutionEngineObj();
  virtual Device GetInputDevice(int idx) const = 0;
  virtual void SetInputs() = 0;
  virtual Array<NDArray> Execute() = 0;

  // Internal supporting.
 public:
  // Hold the reference of the arguments prevent them being released.
  Array<NDArray> on_device_args;
  // The packed function used to get the data type of specific parameter of the entry function.
  ffi::Function get_entry_param_dtype;

 protected:
  ffi::Function get_output_;
};

class ExecutionEngine : public ObjectRef {
  // Things that will interface with user directly.
 public:
  /*!
   * \brief The constructor of this class.
   *
   * \param compiled_model_dir It is the deployed directory (i.e., the exported compiled NN model)
   * path.
   * \param device The device on which to execute the compiled NN model.
   */
  ExecutionEngine(const std::string& compiled_model_dir, Device device = {kDLCPU, 0});

  /*!
   * \brief The constructor of this class.
   *
   * \param compiled_model_dir It is the deployed directory path.
   * \param devices The devices on which to execute the compiled NN model.
   */
  ExecutionEngine(const std::string& compiled_model_dir, const std::vector<Device>& devices);

  /*!
   * \brief The constructor of this class, used when the deployed directory already be deserialized.
   *
   * \param compiled_model It need to be the equivalent in memory object of the deployed directory.
   * \param device The device on which to execute the compiled NN model.
   */
  ExecutionEngine(Module compiled_model, Device device = {kDLCPU, 0});

  /*!
   * \brief The constructor of this class, used when the deployed directory already be deserialized.
   *
   * \param compiled_model It need to be the equivalent in memory object of the deployed directory.
   * \param devices The devices on which to execute the compiled NN model.
   */
  ExecutionEngine(Module compiled_model, const std::vector<Device>& devices);

  /*!
   * \brief The API that is used to set the real input data to the compiled NN model.
   *
   * \param args The array of all real input data. If it is empty, nothing will be done.
   */
  void SetInputs(Array<NDArray> args);

  /*!
   * \brief The API that is used to set the real input data to the compiled NN model, used when
   *        there only is 1 input.
   *
   * \param arg The real input data.
   */
  void SetInputs(NDArray arg);

  /*!
   * \brief The API that is used to run the compiled NN model.
   *
   * \param args The array of all real input data. If it is not empty, the API SetInputs will be
   *        invocated with its value.
   *
   * \return The outputs of the compiled NN model for this execution. Its type is always list even
   *         though there is only one output.
   */
  Array<NDArray> Run(Array<NDArray> args = {});

  /*!
   * \brief The API that is used to run the compiled NN model.
   *
   * \param arg The real input data.
   *
   * \return The outputs of the compiled NN model for this execution. Its type is always list even
   *         though there is only one output.
   */
  Array<NDArray> Run(NDArray arg);

  // Internal supporting.
  // Override things that inherited from ObjectRef.
 public:
  // TVM C++ object protocol relevant.
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExecutionEngine, ObjectRef, ExecutionEngineObj);

 private:
  Array<NDArray> GetOnDeviceArgs(Array<NDArray> args) const;
};

class VmExecutionEngineObj final : public ExecutionEngineObj {
  // Things that will interface with user directly.
 public:
  VmExecutionEngineObj(Module executable, const std::vector<Device>& devices,
                       bool with_profile = false);
  Device GetInputDevice(int idx) const final;
  void SetInputs() final;
  Array<NDArray> Execute() final;

  // Internal supporting.
  // Override things that inherited from ExecutionEngineObj.
 public:
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "compass.runtime.VmExecutionEngine";
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(VmExecutionEngineObj, ExecutionEngineObj);

  // Things of current class.
 private:
  ffi::Function set_input_;
  ffi::Function get_output_arity_;
  ffi::Function invoke_stateful_;
  std::vector<Device> devices_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // COMPASS_TVM_RUNTIME_EXECUTION_ENGINE_H_
