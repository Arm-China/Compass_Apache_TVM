// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/module.cc
 */
#include <compass/tvm/runtime/basic_config.h>
#include <compass/tvm/runtime/module.h>

namespace tvm {
namespace runtime {

static inline Array<NDArray> ConvertToNDArrayWithCheck(ffi::PackedArgs args,
                                                       Array<ParamInfo> params,
                                                       bool check_size = true) {
  // Ensure the count of arguments match those of parameters.
  ICHECK_EQ(args.size(), params.size()) << "The count of parameter and arguments mismatched.";

  Array<NDArray> ret;
  for (size_t i = 0; i < params.size(); ++i) {
    ParamInfo param = params[i];

    NDArray arg;
    if (auto opt_nd = args[i].as<NDArray>()) {
      arg = opt_nd.value();
    } else {
      // "NDArray" will be passed as "DLTensor*" through RPC.
      auto* managed_tensor = new DLManagedTensorVersioned();
      managed_tensor->version.major = DLPACK_MAJOR_VERSION;
      managed_tensor->version.minor = DLPACK_MINOR_VERSION;
      managed_tensor->dl_tensor = *(args[i].cast<DLTensor*>());
      managed_tensor->manager_ctx = nullptr;
      managed_tensor->deleter = [](DLManagedTensorVersioned* tensor) { delete tensor; };
      managed_tensor->flags = 0;
      arg = NDArray::FromDLPackVersioned(managed_tensor);
    }

    // Ensure the data is a simple contiguous value array.
    ICHECK(arg->data != nullptr);
    ICHECK(arg.IsContiguous());
    ICHECK(arg->byte_offset == 0);

    // Ensure the type and size of arguments match those of parameters.
    ICHECK_EQ(arg.DataType(), param->dtype);
    if (check_size) {
      ICHECK_EQ(GetDataSize(*arg.get()), param->size);
    }

    ret.push_back(arg);
  }
  return ret;
}

void CompassModule::Init() {
  cps_driver_ = CompassDriver(cps_bin_path, func_name, with_profile, target, umd_dtcm_sz);

  in_params_ = cps_driver_->GetParamInfo(true);
  out_params_ = cps_driver_->GetParamInfo(false);

  dump_func_ = ffi::Function::GetGlobal("compass.runtime.dump_tensors");
}

void CompassModule::ReviseCpsBinPath(const std::string& base_dir) {
  cps_bin_path = base_dir + "/" + cps_bin_path;
}

void CompassModule::GetOutputs(Array<NDArray> out_tensors) {
  cps_driver_->GetOutputs(out_tensors);

  if (dump_func_ != std::nullopt) {
    (*dump_func_)(func_name, false, out_tensors);  // Dump output tensors as binary files.
  }

  cps_driver_->DumpProfileData();  // Dump the profile data if it exist.
}

ffi::Function CompassModule::GetFunction(const String& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);

  // The "sptr_to_self" must be captured, because if current instance is destroyed before this
  // closure, then this closure will crash when it is called.
  if (name == "compass_set_inputs") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, Any* rv) {
      // Because the container "Array" can't be passed through RPC, so here must write as
      // "packed format", can't write like "ffi::Function::FromTyped([xxx](Array<xxx>) {".
      cps_driver_->SetInputs(ConvertToNDArrayWithCheck(args, in_params_));
    });
  } else if (name == "compass_set_outputs") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, Any* rv) {
      cps_driver_->SetOutputs(ConvertToNDArrayWithCheck(args, out_params_));
    });
  } else if (name == "compass_execute") {
    return ffi::Function::FromTyped([sptr_to_self, this]() { cps_driver_->Run(); });
  } else if (name == "compass_get_param_info") {
    return ffi::Function::FromTyped([sptr_to_self, this](int idx, bool is_input) -> ParamInfo {
      ICHECK_GE(idx, 0) << "The index mismatched.";
      if (is_input) {
        ICHECK_LT(idx, in_params_.size()) << "The index mismatched.";
        return in_params_[idx];
      } else {
        ICHECK_LT(idx, out_params_.size()) << "The index mismatched.";
        return out_params_[idx];
      }
    });
  } else if (name == "compass_get_outputs") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, Any* rv) {
      GetOutputs(ConvertToNDArrayWithCheck(args, out_params_));
    });
  } else if (name == "compass_run" || name == func_name) {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, Any* rv) {
      Array<NDArray> nd_arrs = ConvertToNDArrayWithCheck(args, Concat(in_params_, out_params_));
      const size_t& in_cnt = in_params_.size();
      // Split the input and output arguments away.
      Array<NDArray> in_args(nd_arrs.begin(), nd_arrs.begin() + in_cnt);
      Array<NDArray> out_args(nd_arrs.begin() + in_cnt, nd_arrs.end());
      // Dump input tensors as a binary file.
      if (dump_func_ != std::nullopt) {
        (*dump_func_)(func_name, true, in_args);
      }
      cps_driver_->SetInputs(in_args);
      cps_driver_->Run();
      GetOutputs(out_args);
    });
  } else if (name == "compass_dynamic_run") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, Any* rv) {
      cps_driver_->SetInputsWithDynamicShape(ConvertToNDArrayWithCheck(args, in_params_, false));
      cps_driver_->Run();

      // Update parameter information as shape changed.
      out_params_ = cps_driver_->GetParamInfo(false);
      Array<NDArray> ret;
      for (size_t i = 0; i < out_params_.size(); ++i) {
        std::vector<int64_t> shape = cps_driver_->GetOutputShape(i);
        ret.push_back(NDArray::Empty(shape, out_params_[i]->dtype, {kDLCPU, 0}));
      }
      GetOutputs(ret);
      *rv = ret;
    });
  }
  return nullptr;
}

void CompassModule::SaveToBinary(dmlc::Stream* stream) {
  size_t output_dir_length = CompassBasicConfig::Global()->common["output_dir"].length();
  stream->Write(cps_bin_path.substr(output_dir_length + 1));
  stream->Write(func_name);
  stream->Write(with_profile);
  stream->Write(target);
  stream->Write(umd_dtcm_sz);
}

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_compass.runtime.CompassModule")
    .set_body_typed([](void* stream) -> Module {
      dmlc::Stream* strm = static_cast<dmlc::Stream*>(stream);
      ObjectPtr<CompassModule> obj = make_object<CompassModule>();

      if ((strm->Read(&(obj->cps_bin_path)) == false) || (strm->Read(&(obj->func_name)) == false) ||
          (strm->Read(&(obj->with_profile)) == false) || (strm->Read(&(obj->target)) == false) ||
          (strm->Read(&(obj->umd_dtcm_sz)) == false)) {
        LOG(FATAL) << "Load compass.runtime.CompassModule from binary failed!";
      }
      return Module(obj);
    });

TVM_FFI_REGISTER_GLOBAL("compass.runtime.CompassModule")
    .set_body_typed([](std::string cps_bin_path, std::string func_name, bool with_profile,
                       std::string target, std::string umd_dtcm_sz) -> Module {
      ObjectPtr<CompassModule> obj = make_object<CompassModule>();
      obj->cps_bin_path = cps_bin_path;
      obj->func_name = func_name;
      obj->with_profile = with_profile;
      obj->target = target;
      obj->umd_dtcm_sz = umd_dtcm_sz;
      return Module(obj);
    });

Array<Module> GetAllCompassModule(Module module) {
  Array<Module> ret;
  Array<Module> visited;
  Array<Module> stack{module};

  while (!stack.empty()) {
    Module cur_mod = stack.back();
    stack.pop_back();
    if (!cur_mod.defined()) continue;
    if (cur_mod->type_key() == std::string("compass.runtime.CompassModule")) {
      ret.push_back(cur_mod);
    }
    for (auto mod : cur_mod->imports()) {
      if (!visited.Contains(mod)) {
        visited.push_back(mod);
        stack.push_back(mod);
      }
    }
  }

  return ret;
}

TVM_FFI_REGISTER_GLOBAL("compass.runtime.init").set_body_typed([](Module module) {
  for (auto cps_mod : GetAllCompassModule(module)) {
    static_cast<CompassModule*>(cps_mod.operator->())->Init();
  }
});

}  // namespace runtime
}  // namespace tvm
