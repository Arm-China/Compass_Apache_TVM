// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/relax/config.cc
 */
#include <compass/tvm/runtime/basic_config.h>
#include <tvm/ffi/function.h>
#include <tvm/node/reflection.h>

namespace tvm {
namespace relax {
namespace contrib {

class CompassConfigObj final : public runtime::CompassBasicConfigObj {
  // Things that will interface with user directly.
 public:
  Map<String, String> parser;
  Map<String, Map<String, String>> optimizers;
  Map<String, String> gbuilder;

  // Internal supporting.
 public:
  // TVM C++ reflection relevant.
  void VisitAttrs(AttrVisitor* v);

  // Override things that inherited from runtime::CompassBasicConfigObj.
 public:
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "compass.relax.CompassConfig";
  static constexpr bool _type_has_method_visit_attrs = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(CompassConfigObj, runtime::CompassBasicConfigObj);
};

void CompassConfigObj::VisitAttrs(AttrVisitor* v) {
  v->Visit("common", &common);
  v->Visit("parser", &parser);
  v->Visit("optimizers", &optimizers);
  v->Visit("gbuilder", &gbuilder);
  v->Visit("runtime", &runtime);
  return;
}

// This statement is necessary, otherwise front-end languages not only can't associate their
// corresponding classes with it through "tvm.register_object", but also can't access the data
// member of C++ side class through TVM C++ reflection mechanism.
TVM_REGISTER_NODE_TYPE(CompassConfigObj);

class CompassConfig final : public ObjectRef {
  // Things that will interface with user directly.
 public:
  static void InitSingleton(Map<String, String> common, Map<String, String> parser,
                            Map<String, Map<String, String>> optimizers,
                            Map<String, String> gbuilder, Map<String, String> runtime);
  static CompassConfig Global();

  // Internal supporting.
  // Override things that inherited from ObjectRef.
 public:
  // TVM C++ object protocol relevant.
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CompassConfig, ObjectRef, CompassConfigObj);
};

void CompassConfig::InitSingleton(Map<String, String> common, Map<String, String> parser,
                                  Map<String, Map<String, String>> optimizers,
                                  Map<String, String> gbuilder, Map<String, String> runtime) {
  ObjectPtr<CompassConfigObj> obj = make_object<CompassConfigObj>();
  obj->common = common;
  obj->parser = parser;
  obj->optimizers = optimizers;
  obj->gbuilder = gbuilder;
  obj->runtime = runtime;
  // Store the singleton of derived class to the singleton of base class, so we can access it either
  // as the derived class or as the base class.
  runtime::CompassBasicConfig::SetSingleton(obj);
  return;
}

CompassConfig CompassConfig::Global() {
  return Downcast<CompassConfig>(runtime::CompassBasicConfig::Global());
}

TVM_FFI_REGISTER_GLOBAL("compass.relax.CompassConfig_InitSingleton")
    .set_body_typed(CompassConfig::InitSingleton);

TVM_FFI_REGISTER_GLOBAL("compass.relax.CompassConfig_Global").set_body_typed(CompassConfig::Global);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
