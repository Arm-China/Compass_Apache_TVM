// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/backend/compass/config.cc
 */
#include <aipu/runtime/compass/basic_config.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relay {
namespace contrib {

class AipuCompassConfigObj final : public runtime::AipuCompassBasicConfigObj {
  // Things that will interface with user directly.
 public:
  Map<String, String> parser;
  Map<String, String> optimizer;
  Map<String, String> gbuilder;

  // Internal supporting.
 public:
  // TVM C++ reflection relevant.
  void VisitAttrs(AttrVisitor* v);

  // Override things that inherited from runtime::AipuCompassBasicConfigObj.
 public:
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "aipu_compass.AipuCompassConfig";
  static constexpr bool _type_has_method_visit_attrs = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(AipuCompassConfigObj, runtime::AipuCompassBasicConfigObj);
};

void AipuCompassConfigObj::VisitAttrs(AttrVisitor* v) {
  v->Visit("common", &common);
  v->Visit("parser", &parser);
  v->Visit("optimizer", &optimizer);
  v->Visit("gbuilder", &gbuilder);
  v->Visit("runtime", &runtime);
  return;
}

// This statement is necessary, otherwise front-end languages not only can't
// associate their corresponding classes with it through "tvm.register_object",
// but also can't access the data member of C++ side class through TVM C++
// reflection mechanism.
TVM_REGISTER_NODE_TYPE(AipuCompassConfigObj);

class AipuCompassConfig final : public ObjectRef {
  // Things that will interface with user directly.
 public:
  static void InitSingleton(Map<String, String> common, Map<String, String> parser,
                            Map<String, String> optimizer, Map<String, String> gbuilder,
                            Map<String, String> runtime);
  static AipuCompassConfig Global();

  // Internal supporting.
  // Override things that inherited from ObjectRef.
 public:
  // TVM C++ object protocol relevant.
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AipuCompassConfig, ObjectRef, AipuCompassConfigObj);
};

void AipuCompassConfig::InitSingleton(Map<String, String> common, Map<String, String> parser,
                                      Map<String, String> optimizer, Map<String, String> gbuilder,
                                      Map<String, String> runtime) {
  ObjectPtr<AipuCompassConfigObj> obj = make_object<AipuCompassConfigObj>();
  obj->common = common;
  obj->parser = parser;
  obj->optimizer = optimizer;
  obj->gbuilder = gbuilder;
  obj->runtime = runtime;
  // Store the singleton of derived class to the singleton of base class, so we
  // can access it either as the derived class or as the base class.
  runtime::AipuCompassBasicConfig::SetSingleton(obj);
  return;
}

AipuCompassConfig AipuCompassConfig::Global() {
  return Downcast<AipuCompassConfig>(runtime::AipuCompassBasicConfig::Global());
}

TVM_REGISTER_GLOBAL("aipu_compass.AipuCompassConfig_InitSingleton")
    .set_body_typed(AipuCompassConfig::InitSingleton);

TVM_REGISTER_GLOBAL("aipu_compass.AipuCompassConfig_Global")
    .set_body_typed(AipuCompassConfig::Global);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
