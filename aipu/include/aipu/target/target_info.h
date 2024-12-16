// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/target/target_info.h
 * \brief Various information about AIPU configuration.
 */
#ifndef AIPU_TARGET_TARGET_INFO_H_
#define AIPU_TARGET_TARGET_INFO_H_

#include <aipu/runtime/utils.h>
#include <tvm/target/target.h>

#include <unordered_map>

namespace tvm {

class AipuInfoObj final : public Object {
 public:
  // The AIPU configuration name.
  String name;
  int core_count = -1;
  int tec_count = -1;
  // The size of a single piece local SRAM in bytes.
  int lsram_size_per_piece = -1;
  // The piece count of local SRAM for each TEC.
  int lsram_piece_count = -1;
  // The size of a single piece global SRAM in bytes.
  int gsram_size_per_piece = -1;
  // The piece count of global SRAM for each core.
  int gsram_piece_count = -1;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("core_count", &core_count);
    v->Visit("tec_count", &tec_count);
    v->Visit("lsram_size_per_piece", &lsram_size_per_piece);
    v->Visit("lsram_piece_count", &lsram_piece_count);
    v->Visit("gsram_size_per_piece", &gsram_size_per_piece);
    v->Visit("gsram_piece_count", &gsram_piece_count);
  }

  // Object protocol relevant.
  static constexpr const char* _type_key = "AipuInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(AipuInfoObj, Object);
};

/*!
 * \brief Managed reference class to AipuInfoObj.
 * \sa AipuInfoObj
 */
class AipuInfo final : public ObjectRef {
  // Methods of current class.
 public:
  bool IsX1() const { return StrStartsWith((*this)->name, "X1"); }
  bool IsX2() const { return StrStartsWith((*this)->name, "X2"); }
  bool IsX3() const { return StrStartsWith((*this)->name, "X3"); }
  bool IsV1() const { return IsX1(); }
  bool IsV2() const { return IsX2(); }
  bool IsV3() const { return IsX3(); }

  /*!
   * \brief Hardware vector width in bits.
   *
   * Each TEC contains one vector processor unit, and the width of them are all
   * same.
   *
   * \return The vector width of corresponding AIPU configuration.
   */
  int VectorWidth() const { return 256; }

  /*!
   * \brief The size of the given piece local SRAM in bytes for each TEC.
   *
   * Each TEC contains one or more pieces of local SRAM, size and piece count of
   * the local SRAM of each TEC are same.
   *
   * \param piece_idx The local SRAM piece index.
   * \return The size of the specified piece of local SRAM.
   */
  int LsramSize(int piece_idx = 0) const;

  /*!
   * \brief The size of the given piece global SRAM in bytes for each core.
   *
   * Each core contains one or more pieces of global SRAM, they are shared by
   * all of the TECs in the same core.
   *
   * \param piece_idx The global SRAM piece index.
   * \return The size of the specified piece of global SRAM.
   */
  int GsramSize(int piece_idx = 0) const;

  /*!
   * \brief Get the predefined instance that corresponding to the given target.
   *
   * All of the instances corresponding to the supported AIPU configurations are
   * created in advance, the "mcpu" attribute of the given target will be used
   * to find the corresponding instance.
   *
   * \param target The AIPU target whose attribute "mcpu" record its AIPU
   *        configuration name.
   * \return The found instance.
   */
  static AipuInfo Get(Target target);

  /*!
   * \brief Get all of supported AIPU configuration names.
   * \return The list of valid AIPU configuration names.
   */
  static Array<String> GetValidConfigNames();

  // Internal supporting.
 public:
  // Object protocol relevant.
  TVM_DEFINE_OBJECT_REF_METHODS(AipuInfo, ObjectRef, AipuInfoObj);

 private:
  AipuInfo(String name, int core_count, int tec_count, int lsram_size_per_piece,
           int lsram_piece_count, int gsram_size_per_piece, int gsram_piece_count);

  static const std::unordered_map<String, AipuInfo> kValidConfigs;
};

}  // namespace tvm

#endif  // AIPU_TARGET_TARGET_INFO_H_
