// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/tvm/target/compass_info.h
 * \brief Various information about Zhouyi NPU configuration.
 */
#ifndef COMPASS_TVM_TARGET_COMPASS_INFO_H_
#define COMPASS_TVM_TARGET_COMPASS_INFO_H_

#include <tvm/target/target.h>

#include <unordered_map>

namespace tvm {

class CompassInfoObj final : public Object {
 public:
  // The Zhouyi NPU configuration name.
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
  // The unique string represents the generation of Zhouyi NPU chip.
  String version;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("core_count", &core_count);
    v->Visit("tec_count", &tec_count);
    v->Visit("lsram_size_per_piece", &lsram_size_per_piece);
    v->Visit("lsram_piece_count", &lsram_piece_count);
    v->Visit("gsram_size_per_piece", &gsram_size_per_piece);
    v->Visit("gsram_piece_count", &gsram_piece_count);
    v->Visit("version", &version);
  }

  // Object protocol relevant.
  static constexpr const char* _type_key = "compass.CompassInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompassInfoObj, Object);
};

/*!
 * \brief Managed reference class to CompassInfoObj.
 * \sa CompassInfoObj
 */
class CompassInfo final : public ObjectRef {
  // Methods of current class.
 public:
  /*!
   * \brief Hardware vector width in bits.
   *
   * Each TEC contains one vector processor unit, and the width of them are all same.
   *
   * \return The vector width of corresponding Zhouyi NPU configuration.
   */
  int VectorWidth() const { return 256; }

  /*!
   * \brief The size of the given piece local SRAM in bytes for each TEC.
   *
   * Each TEC contains one or more pieces of local SRAM, size and piece count of the local SRAM of
   * each TEC are same.
   *
   * \param piece_idx The local SRAM piece index.
   * \return The size of the specified piece of local SRAM.
   */
  int LsramSize(int piece_idx = 0) const;

  /*!
   * \brief The size of the given piece global SRAM in bytes for each core.
   *
   * Each core contains one or more pieces of global SRAM, they are shared by all of the TECs in the
   * same core.
   *
   * \param piece_idx The global SRAM piece index.
   * \return The size of the specified piece of global SRAM.
   */
  int GsramSize(int piece_idx = 0) const;

  /*!
   * \brief Get the predefined instance that corresponding to the given target.
   *
   * All of the instances corresponding to the supported Zhouyi NPU configurations are created in
   * advance, the "mcpu" attribute of the given target will be used to find the corresponding
   * instance.
   *
   * \param target The Zhouyi NPU target whose attribute "mcpu" record its configuration name.
   * \return The found instance.
   */
  static CompassInfo Get(Target target);

  /*!
   * \brief Get all of supported Zhouyi NPU configuration names.
   * \return The list of valid Zhouyi NPU configuration names.
   */
  static Array<String> GetValidConfigNames();

  // Internal supporting.
 public:
  // Object protocol relevant.
  TVM_DEFINE_OBJECT_REF_METHODS(CompassInfo, ObjectRef, CompassInfoObj);

 private:
  CompassInfo(String name, int core_count, int tec_count, int lsram_size_per_piece,
              int lsram_piece_count, int gsram_size_per_piece, int gsram_piece_count);

  static const std::unordered_map<String, CompassInfo> kValidConfigs;
};

}  // namespace tvm

#endif  // COMPASS_TVM_TARGET_COMPASS_INFO_H_
