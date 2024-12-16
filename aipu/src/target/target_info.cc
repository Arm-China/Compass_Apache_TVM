// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/target/target_info.cc
 * \brief Various information about AIPU configuration.
 */
#include <aipu/target/target_info.h>
#include <tvm/runtime/registry.h>

namespace tvm {

TVM_REGISTER_NODE_TYPE(AipuInfoObj);

// name core_count tec_count lsram_size_per_piece lsram_piece_count gsram_size_per_piece
// gsram_piece_count
const std::unordered_map<String, AipuInfo> AipuInfo::kValidConfigs = {
    {"X1_1204", {"X1_1204", 1, 4, 32 * 1024, 2, 512 * 1024, 2}},
    {"X2_1204", {"X2_1204", 1, 4, 32 * 1024, 1, 256 * 1024, 1}},
    {"X2_1204MP3", {"X2_1204MP3", 3, 4, 32 * 1024, 1, 256 * 1024, 1}},
    {"X3_1304", {"X3_1304", 1, 4, 32 * 1024, 1, 128 * 1024, 3}},
    {"X3_1304MP2", {"X3_1304MP2", 2, 4, 32 * 1024, 1, 128 * 1024, 3}},
};

AipuInfo::AipuInfo(String name, int core_count, int tec_count, int lsram_size_per_piece,
                   int lsram_piece_count, int gsram_size_per_piece, int gsram_piece_count) {
  auto obj = make_object<AipuInfoObj>();
  obj->name = name;
  obj->core_count = core_count;
  obj->tec_count = tec_count;
  obj->lsram_size_per_piece = lsram_size_per_piece;
  obj->lsram_piece_count = lsram_piece_count;
  obj->gsram_size_per_piece = gsram_size_per_piece;
  obj->gsram_piece_count = gsram_piece_count;
  data_ = std::move(obj);
  return;
}

int AipuInfo::LsramSize(int piece_idx) const {
  ICHECK((piece_idx >= 0) && (piece_idx < (*this)->lsram_piece_count))
      << "Invalid local SRAM piece index \"" << piece_idx << "\", valid range: "
      << "[0, " << ((*this)->lsram_piece_count - 1) << "].";

  if (piece_idx == 1) {
    // Currently local SRAM 1 reserves 4KB to handle T register spilling.
    return ((*this)->lsram_size_per_piece - (4 * 1024));
  }
  return (*this)->lsram_size_per_piece;
}

int AipuInfo::GsramSize(int piece_idx) const {
  ICHECK((piece_idx >= 0) && (piece_idx < (*this)->gsram_piece_count))
      << "Invalid global SRAM piece index \"" << piece_idx << "\", valid range: "
      << "[0, " << ((*this)->gsram_piece_count - 1) << "].";

  return (*this)->gsram_size_per_piece;
}

AipuInfo AipuInfo::Get(Target target) {
  ICHECK(target.defined() && target->kind->name == "aipu");

  // Get the AIPU configuration name from class "Target" and then use it to find
  // the corresponding instance of class "AipuInfo".
  auto name = target->GetAttr<String>("mcpu").value();
  auto it = kValidConfigs.find(name);
  ICHECK(it != kValidConfigs.end()) << "Invalid AIPU configuration \"" << name << "\".";
  return it->second;
}

Array<String> AipuInfo::GetValidConfigNames() {
  static Array<String> names;
  if (names.empty() == false) return names;

  for (const auto& kv : kValidConfigs) {
    names.push_back(kv.first);
  }
  return names;
}

TVM_REGISTER_GLOBAL("target.AipuInfo_VectorWidth").set_body_method(&AipuInfo::VectorWidth);
TVM_REGISTER_GLOBAL("target.AipuInfo_LsramSize").set_body_method(&AipuInfo::LsramSize);
TVM_REGISTER_GLOBAL("target.AipuInfo_GsramSize").set_body_method(&AipuInfo::GsramSize);
TVM_REGISTER_GLOBAL("target.AipuInfo_Get").set_body_typed(AipuInfo::Get);

}  // namespace tvm
