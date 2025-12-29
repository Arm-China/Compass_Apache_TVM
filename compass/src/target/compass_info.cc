// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/target/compass_info.cc
 * \brief Various information about Zhouyi NPU configuration.
 */
#include <compass/tvm/runtime/utils.h>
#include <compass/tvm/target/compass_info.h>
#include <tvm/ffi/function.h>

namespace tvm {

TVM_REGISTER_NODE_TYPE(CompassInfoObj);

// name core_count tec_count lsram_size_per_piece lsram_piece_count gsram_size_per_piece
// gsram_piece_count
const std::unordered_map<String, CompassInfo> CompassInfo::kValidConfigs = {
    // clang-format off
    {"X1_1204",     {"X1_1204",     1, 4, 32 * 1024, 2, 512 * 1024, 2}},
    {"X2_1204",     {"X2_1204",     1, 4, 32 * 1024, 1, 256 * 1024, 1}},
    {"X2_1204MP3",  {"X2_1204MP3",  3, 4, 32 * 1024, 1, 256 * 1024, 1}},
    {"X3P_1304",    {"X3P_1304",    1, 4, 32 * 1024, 1, 128 * 1024, 1}},
    {"X3P_1304MP4", {"X3P_1304MP4", 4, 4, 32 * 1024, 1, 128 * 1024, 1}},
    {"X3P_1304MP2", {"X3P_1304MP2", 2, 4, 32 * 1024, 1, 128 * 1024, 1}},
    {"X3S_1304",    {"X3S_1304",    1, 4, 32 * 1024, 1, 128 * 1024, 1}},
    {"X3S_1304MP4", {"X3S_1304MP4", 4, 4, 32 * 1024, 1, 128 * 1024, 1}},
    {"X3S_1304MP2", {"X3S_1304MP2", 2, 4, 32 * 1024, 1, 128 * 1024, 1}},
    // clang-format on
};

CompassInfo::CompassInfo(String name, int core_count, int tec_count, int lsram_size_per_piece,
                         int lsram_piece_count, int gsram_size_per_piece, int gsram_piece_count) {
  auto obj = make_object<CompassInfoObj>();
  obj->name = name;
  obj->core_count = core_count;
  obj->tec_count = tec_count;
  obj->lsram_size_per_piece = lsram_size_per_piece;
  obj->lsram_piece_count = lsram_piece_count;
  obj->gsram_size_per_piece = gsram_size_per_piece;
  obj->gsram_piece_count = gsram_piece_count;
  obj->version = runtime::StrSplit(name, "_")[0];
  data_ = std::move(obj);
  return;
}

int CompassInfo::LsramSize(int piece_idx) const {
  ICHECK((piece_idx >= 0) && (piece_idx < (*this)->lsram_piece_count))
      << "Invalid local SRAM piece index \"" << piece_idx << "\", valid range: "
      << "[0, " << ((*this)->lsram_piece_count - 1) << "].";

  if (piece_idx == 1) {
    // Currently local SRAM 1 reserves 4KB to handle T register spilling.
    return ((*this)->lsram_size_per_piece - (4 * 1024));
  }
  return (*this)->lsram_size_per_piece;
}

int CompassInfo::GsramSize(int piece_idx) const {
  ICHECK((piece_idx >= 0) && (piece_idx < (*this)->gsram_piece_count))
      << "Invalid global SRAM piece index \"" << piece_idx << "\", valid range: "
      << "[0, " << ((*this)->gsram_piece_count - 1) << "].";

  return (*this)->gsram_size_per_piece;
}

CompassInfo CompassInfo::Get(Target target) {
  ICHECK(target.defined() && target->kind->name == "compass");

  // Get the Zhouyi NPU configuration name from class "Target" and then use it to find the
  // corresponding instance of class "CompassInfo".
  auto name = target->GetAttr<String>("mcpu").value();
  auto it = kValidConfigs.find(name);
  ICHECK(it != kValidConfigs.end()) << "Invalid Zhouyi NPU configuration \"" << name << "\".";
  return it->second;
}

Array<String> CompassInfo::GetValidConfigNames() {
  static Array<String> names;
  if (names.empty() == false) return names;

  for (const auto& kv : kValidConfigs) {
    names.push_back(kv.first);
  }
  return names;
}

TVM_FFI_REGISTER_GLOBAL("compass.CompassInfo_VectorWidth")
    .set_body_method(&CompassInfo::VectorWidth);
TVM_FFI_REGISTER_GLOBAL("compass.CompassInfo_LsramSize").set_body_method(&CompassInfo::LsramSize);
TVM_FFI_REGISTER_GLOBAL("compass.CompassInfo_GsramSize").set_body_method(&CompassInfo::GsramSize);
TVM_FFI_REGISTER_GLOBAL("compass.CompassInfo_Get").set_body_typed(CompassInfo::Get);

}  // namespace tvm
