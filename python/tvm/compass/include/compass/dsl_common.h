// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/dsl_common.h
 * \brief The common part of the Compass DSL header file for V2 and V3.
 */
#ifndef DSL_COMMON_H_
#define DSL_COMMON_H_

#include <clc/clc.h>

#define ALL_TRUE_b -1
#define ALL_TRUE_h ALL_TRUE_b
#define ALL_TRUE_w ALL_TRUE_b

#define _CLC_NOINLINE __attribute__((noinline))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void __attribute__((weak)) __perf_record_tick_begin(uint eid) { return; }
void __attribute__((weak)) __perf_record_tick_end(uint eid) { return; }

// The event state is a software counterpart of the hardware synchronization event flags, it also is
// a 32-bit integer value, and each TEC have its own one. If the value of some bits is 0, it means
// that the corresponding event is free to be allocated.
_CLC_INLINE int* addr_of_event_state() {
  return (int*)(__builtin_aipu_get_descriptor_base() + 512);
}

#define GEN_ALLOC_EVENT(max_event_count)                   \
  _CLC_INLINE int alloc_event() {                          \
    int event_state = *addr_of_event_state();              \
    int idx = 0;                                           \
    while (idx < max_event_count) {                        \
      if (((event_state >> idx) & 1) == 0) {               \
        *addr_of_event_state() = (1 << idx) | event_state; \
        break;                                             \
      }                                                    \
      ++idx;                                               \
    }                                                      \
    return idx;                                            \
  }

_CLC_INLINE void wait_events(int bit_mask) { __wfe(~bit_mask, 1); }
_CLC_INLINE void free_events(int bit_mask) {
  wait_events(bit_mask);
  *addr_of_event_state() &= ~bit_mask;
}

enum ASID { REGION0 = 0, REGION1 = 1 };
enum DMADirection { kInt2Ext = 0, kExt2Int = 1 };
enum DmaMode { kDirect = 0, kTranspose = 1, kUpsample = 2, kMemset = 3 };
typedef enum { kUseless = 0, kByte = 0, kHalf = 1, kWord = 2 } DmaDataUnit;

enum DmaAddrBase {
  kGlobal = 0,
  kGlobal1 = 1,
  kGlobal2 = 2,
  kGlobal3 = 3,
  kLsram = 4,
  kShared = 8,
};

#define DMA_BUFFER 32u         // The bus width is 256 bits or 32 bytes.
#define DMA_BURST_BUFFER 512u  // The max burst length is 16, 16 * 32 = 512.
// As a member of DMA descriptor, width occupy 16 bits, the max value is 0xFFFF,
// but the optimal value is a multiple of "DMA_BURST_BUFFER", so the best max value is 0xFE00.
#define DMA_MAX_ALIGNED_WIDTH 0xFE00u

#define DMA_GET_BEST_WIDTH(size)                                                            \
  (((size) < DMA_BUFFER                                                                     \
        ? (size)                                                                            \
        : ((size) < DMA_BURST_BUFFER                                                        \
               ? DMA_BUFFER                                                                 \
               : ((size) < DMA_MAX_ALIGNED_WIDTH                                            \
                      ? ((size)&0xFFFFFE00) /* Make it a multiple of "DMA_BURST_BUFFER". */ \
                      : DMA_MAX_ALIGNED_WIDTH))))

#define DMA_MAX_STRIDE 0xFFFFFFu      // 24bit
#define DMA_MAX_WIDTH 0xFFFFu         // 16bit
#define DMA_MAX_TRANS_NUM 0xFFu       // 8bit
#define DMA_MAX_TRANS_SIZE 0xFFFFFFu  // 24bit
#define DMA_MAX_GAP 0xFFFFFFu         // 24bit

// Note: ext_width_ext_stride is the lowest word, and ext_stride in low position.
typedef struct DmaDescriptor {
  uint ext_width_ext_stride;
  uint ext_num_of_trans_ext_trans_size;
  uint ext_stride_h_ext_gap;
  uint int_stride_h_int_gap;
  uint int_width_int_stride;
  uint int_num_of_trans_int_trans_size;
} DmaDesc;

_CLC_OVERLOAD _CLC_INLINE static void init_dma_desc(DmaDesc* dma_desc, uint ext_width,
                                                    uint int_width, uint ext_stride,
                                                    uint int_stride, uint ext_gap, uint int_gap,
                                                    uint ext_num_of_trans, uint int_num_of_trans,
                                                    uint ext_trans_size, uint int_trans_size) {
  dma_desc->ext_width_ext_stride = (ext_width << 16) | (ext_stride & 0x0000FFFF);
  dma_desc->ext_num_of_trans_ext_trans_size = (ext_num_of_trans << 24) | ext_trans_size;
  dma_desc->ext_stride_h_ext_gap = ((ext_stride & 0xFFFF0000) << 8) | ext_gap;
  dma_desc->int_stride_h_int_gap = ((int_stride & 0xFFFF0000) << 8) | int_gap;
  dma_desc->int_width_int_stride = (int_width << 16) | (int_stride & 0x0000FFFF);
  dma_desc->int_num_of_trans_int_trans_size = (int_num_of_trans << 24) | int_trans_size;
}

_CLC_OVERLOAD _CLC_INLINE static void init_dma_desc(DmaDesc* dma_desc, uint width, uint ext_stride,
                                                    uint int_stride, uint trans_size) {
  dma_desc->ext_width_ext_stride = (width << 16) | (ext_stride & 0x0000FFFF);
  uint num_of_trans_and_trans_size = (1 << 24) | trans_size;
  dma_desc->ext_num_of_trans_ext_trans_size = num_of_trans_and_trans_size;
  dma_desc->ext_stride_h_ext_gap = (ext_stride & 0xFFFF0000) << 8;
  dma_desc->int_stride_h_int_gap = (int_stride & 0xFFFF0000) << 8;
  dma_desc->int_width_int_stride = (width << 16) | (int_stride & 0x0000FFFF);
  dma_desc->int_num_of_trans_int_trans_size = num_of_trans_and_trans_size;
}

_CLC_OVERLOAD _CLC_INLINE static void init_dma_desc(DmaDesc* dma_desc, uint width,
                                                    uint trans_size) {
  uint width_and_stride = (width << 16) | width;
  dma_desc->ext_width_ext_stride = width_and_stride;
  uint num_of_trans_and_trans_size = (1 << 24) | trans_size;
  dma_desc->ext_num_of_trans_ext_trans_size = num_of_trans_and_trans_size;
  dma_desc->ext_stride_h_ext_gap = 0;
  dma_desc->int_width_int_stride = width_and_stride;
  dma_desc->int_num_of_trans_int_trans_size = num_of_trans_and_trans_size;
  dma_desc->int_stride_h_int_gap = 0;
}

// just 3 sync flag per TEC currently.
#define GET_IDLE_FLAG()                                    \
  ({                                                       \
    int idle_flag = 0;                                     \
    while (1) {                                            \
      uint sync_flag = __builtin_aipu_mfctrl0(0x30) & 0x7; \
      if (sync_flag < 0x7) {                               \
        idle_flag = ((~sync_flag) & (sync_flag + 1)) >> 1; \
        break;                                             \
      }                                                    \
    }                                                      \
    (idle_flag);                                           \
  })

#define GEN_ASYNC_DMA_DIRECT_INT2EXT(src_base, dst_base)                                          \
  _CLC_INLINE static void AsyncDmaDirect_##src_base##_to_##dst_base(                              \
      int dst_addr, int src_addr, uint width, uint src_stride, uint size, uint dst_stride,        \
      int sync_flag) {                                                                            \
    if (width == 0) return;                                                                       \
    __wfe(~(1 << sync_flag), WFE_MODE_IN_ASYNC_DMA);                                              \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                                \
    init_dma_desc(dma_desc, width, dst_stride, src_stride, size);                                 \
    __dma(/*mode*/ kDirect, /*rsync*/ (sync_flag << 8) | sync_flag, /*desp_ctrl*/ 0,              \
          /*int_base*/ src_base, /*ext_base*/ dst_base, /*dir*/ kInt2Ext, /*data_unit*/ kUseless, \
          /*dma_desp*/ (int)dma_desc, /*int_addr*/ src_addr, /*ext_addr*/ dst_addr);              \
    return;                                                                                       \
  }

#define GEN_ASYNC_DMA_DIRECT_EXT2INT(src_base, dst_base)                                    \
  _CLC_INLINE static void AsyncDmaDirect_##src_base##_to_##dst_base(                        \
      int dst_addr, int src_addr, uint width, uint src_stride, uint size, uint dst_stride,  \
      int sync_flag) {                                                                      \
    if (width == 0) return;                                                                 \
    __wfe(~(1 << sync_flag), WFE_MODE_IN_ASYNC_DMA);                                        \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                          \
    init_dma_desc(dma_desc, width, src_stride, dst_stride, size);                           \
    __dma(kDirect, (sync_flag << 8) | sync_flag, 0, dst_base, src_base, kExt2Int, kUseless, \
          (int)dma_desc, dst_addr, src_addr);                                               \
    return;                                                                                 \
  }

#define DMA_CONTINUOUS_TRANS_LOOP(trans_size, src_base, dst_base)                               \
  {                                                                                             \
    uint remain_size = trans_size;                                                              \
    while (remain_size > 0) {                                                                   \
      uint cur_size = MIN(remain_size, DMA_MAX_TRANS_SIZE);                                     \
      uint cur_width = DMA_GET_BEST_WIDTH(cur_size);                                            \
      init_dma_desc(dma_desc, cur_width, cur_size);                                             \
      __dma(kDirect, rsync, 0, src_base, dst_base, kInt2Ext, kUseless, (int)dma_desc, src_addr, \
            dst_addr);                                                                          \
      __wfe(~(1 << sync_flag), 1);                                                              \
      dst_addr += cur_size;                                                                     \
      src_addr += cur_size;                                                                     \
      remain_size -= cur_size;                                                                  \
    }                                                                                           \
  }

#define GEN_DMA_DIRECT_INT2EXT(src_base, dst_base)                                                \
  _CLC_INLINE static void DmaDirect_##src_base##_to_##dst_base(                                   \
      int dst_addr, int src_addr, uint width, uint src_stride, uint size, uint dst_stride) {      \
    if (width == 0) return;                                                                       \
    int sync_flag = GET_IDLE_FLAG();                                                              \
    int rsync = (sync_flag << 8) | sync_flag;                                                     \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                                \
    if (width <= DMA_MAX_WIDTH && src_stride <= DMA_MAX_STRIDE && dst_stride <= DMA_MAX_STRIDE && \
        size <= DMA_MAX_TRANS_SIZE) {                                                             \
      init_dma_desc(dma_desc, width, dst_stride, src_stride, size);                               \
      __dma(/*mode*/ kDirect, /*rsync*/ rsync, /*desp_ctrl*/ 0, /*int_base*/ src_base,            \
            /*ext_base*/ dst_base, /*dir*/ kInt2Ext, /*data_unit*/ kUseless,                      \
            /*dma_desp*/ (int)dma_desc, /*int_addr*/ src_addr, /*ext_addr*/ dst_addr);            \
      __wfe(~(1 << sync_flag), 1);                                                                \
      return;                                                                                     \
    }                                                                                             \
                                                                                                  \
    if (width == src_stride && width == dst_stride) {                                             \
      DMA_CONTINUOUS_TRANS_LOOP(size, src_base, dst_base);                                        \
      return;                                                                                     \
    }                                                                                             \
                                                                                                  \
    if (src_stride > DMA_MAX_STRIDE || dst_stride > DMA_MAX_STRIDE || width > DMA_MAX_WIDTH) {    \
      for (int i = 0; i < size / width; ++i) {                                                    \
        DMA_CONTINUOUS_TRANS_LOOP(width, src_base, dst_base);                                     \
        dst_addr += dst_stride - width;                                                           \
        src_addr += src_stride - width;                                                           \
      }                                                                                           \
      return;                                                                                     \
    }                                                                                             \
                                                                                                  \
    /*Only size > DMA_MAX_TRANS_SIZE and not a continuous transport.*/                            \
    uint cur_times = DMA_MAX_TRANS_SIZE / width;                                                  \
    uint remain_times = size / width;                                                             \
    while (remain_times > 0) {                                                                    \
      cur_times = MIN(cur_times, remain_times);                                                   \
      init_dma_desc(dma_desc, width, dst_stride, src_stride, width* cur_times);                   \
      __dma(kDirect, rsync, 0, src_base, dst_base, kInt2Ext, kUseless, (int)dma_desc, src_addr,   \
            dst_addr);                                                                            \
      __wfe(~(1 << sync_flag), 1);                                                                \
      dst_addr += cur_times * dst_stride;                                                         \
      src_addr += cur_times * src_stride;                                                         \
      remain_times -= cur_times;                                                                  \
    }                                                                                             \
    return;                                                                                       \
  }

#define GEN_DMA_DIRECT_EXT2INT(src_base, dst_base)                                              \
  _CLC_INLINE static void DmaDirect_##src_base##_to_##dst_base(                                 \
      int dst_addr, int src_addr, uint width, uint src_stride, uint size, uint dst_stride) {    \
    if (width == 0) return;                                                                     \
    int sync_flag = GET_IDLE_FLAG();                                                            \
    int rsync = (sync_flag << 8) | sync_flag;                                                   \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                              \
    if (width <= DMA_MAX_WIDTH && src_stride <= DMA_MAX_STRIDE) {                               \
      init_dma_desc(dma_desc, width, src_stride, dst_stride, size);                             \
      __dma(/*mode*/ kDirect, /*rsync*/ rsync, /*desp_ctrl*/ 0, /*int_base*/ dst_base,          \
            /*ext_base*/ src_base, /*dir*/ kExt2Int, /*data_unit*/ kUseless,                    \
            /*dma_desp*/ (int)dma_desc, /*int_addr*/ dst_addr, /*ext_addr*/ src_addr);          \
      __wfe(~(1 << sync_flag), 1);                                                              \
      return;                                                                                   \
    }                                                                                           \
    uint cur_width = DMA_GET_BEST_WIDTH(width);                                                 \
    init_dma_desc(dma_desc, cur_width, width);                                                  \
    for (int i = 0; i < size / width; ++i) {                                                    \
      __dma(kDirect, rsync, 0, dst_base, src_base, kExt2Int, kUseless, (int)dma_desc, dst_addr, \
            src_addr);                                                                          \
      __wfe(~(1 << sync_flag), 1);                                                              \
      dst_addr += dst_stride;                                                                   \
      src_addr += src_stride;                                                                   \
    }                                                                                           \
    return;                                                                                     \
  }

/**
 * \brief Moving data from source to destination with upsample mode.
 *
 * \param dst_addr The destination address.
 * \param src_addr The source address.
 * \param h_scale The scale on w direction.
 * \param w_scale The scale on h direction.
 * \param c The c of each move on source.
 * \param w The w of each move on source.
 * \param src_c_stride The c stride of each move on source.
 * \param dst_c_stride The c stride of each move on destination.
 * \param dst_w_stride The w stride of each move on destination.
 */
#define GEN_DMA_UPSAMPLE_INT2EXT(src_base, dst_base)                                             \
  _CLC_INLINE static void DmaUpsample_##src_base##_to_##dst_base(                                \
      int dst_addr, int src_addr, uint h_scale, uint w_scale, uint c, uint w, uint src_c_stride, \
      uint dst_c_stride, uint dst_w_stride) {                                                    \
    if (w_scale == 0 || h_scale == 0) {                                                          \
      return;                                                                                    \
    }                                                                                            \
    int sync_flag = GET_IDLE_FLAG();                                                             \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                               \
    /*Don't support "w_scale > DMA_MAX_TRANS_NUM", it causes low perf with more loops*/          \
    if (h_scale > DMA_MAX_TRANS_NUM) {                                                           \
      uint loop = h_scale / DMA_MAX_TRANS_NUM;                                                   \
      uint dst_ofs = DMA_MAX_TRANS_NUM * dst_w_stride * dst_c_stride;                            \
      for (uint i = 0; i < loop; ++i) {                                                          \
        init_dma_desc(dma_desc, 0, c, dst_c_stride, src_c_stride, dst_w_stride* dst_c_stride, 0, \
                      DMA_MAX_TRANS_NUM, w_scale, 0, c* w);                                      \
        __dma(/*mode*/ kUpsample, /*rsync*/ (sync_flag << 8) | sync_flag, /*desp_ctrl*/ 0,       \
              /*int_base*/ src_base, /*ext_base*/ dst_base, /*dir*/ kInt2Ext,                    \
              /*data_unit*/ kUseless, /*dma_desp*/ (int)dma_desc, /*int_addr*/ src_addr,         \
              /*ext_addr*/ dst_addr);                                                            \
        __wfe(~(1 << sync_flag), 1);                                                             \
        dst_addr += dst_ofs;                                                                     \
      }                                                                                          \
      h_scale = h_scale % DMA_MAX_TRANS_NUM;                                                     \
    }                                                                                            \
    if (h_scale > 0) {                                                                           \
      init_dma_desc(dma_desc, 0, c, dst_c_stride, src_c_stride, dst_w_stride* dst_c_stride, 0,   \
                    h_scale, w_scale, 0, c* w);                                                  \
      __dma(kUpsample, (sync_flag << 8) | sync_flag, 0, src_base, dst_base, kInt2Ext, kUseless,  \
            (int)dma_desc, src_addr, dst_addr);                                                  \
      __wfe(~(1 << sync_flag), 1);                                                               \
    }                                                                                            \
  }

/**
 * \brief Moving data from source to destination with upsample mode.
 *
 * \param dst_addr The destination address.
 * \param src_addr The source address.
 * \param h_scale The scale on w direction.
 * \param w_scale The scale on h direction.
 * \param c The c of each move on source.
 * \param w The w of each move on source.
 * \param src_c_stride The c stride of each move on source.
 * \param dst_c_stride The c stride of each move on destination.
 * \param dst_w_stride The w stride of each move on destination.
 */
#define GEN_DMA_UPSAMPLE_EXT2INT(src_base, dst_base)                                             \
  _CLC_INLINE static void DmaUpsample_##src_base##_to_##dst_base(                                \
      int dst_addr, int src_addr, uint h_scale, uint w_scale, uint c, uint w, uint src_c_stride, \
      uint dst_c_stride, uint dst_w_stride) {                                                    \
    if (w_scale == 0 || h_scale == 0) {                                                          \
      return;                                                                                    \
    }                                                                                            \
    int sync_flag = GET_IDLE_FLAG();                                                             \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                               \
    /*Don't support "w_scale > DMA_MAX_TRANS_NUM", it causes low perf with more loops*/          \
    if (h_scale > DMA_MAX_TRANS_NUM) {                                                           \
      uint loop = h_scale / DMA_MAX_TRANS_NUM;                                                   \
      uint dst_ofs = DMA_MAX_TRANS_NUM * dst_w_stride * dst_c_stride;                            \
      for (uint i = 0; i < loop; ++i) {                                                          \
        init_dma_desc(dma_desc, c, 0, src_c_stride, dst_c_stride, 0, dst_w_stride* dst_c_stride, \
                      w_scale, DMA_MAX_TRANS_NUM, c* w, 0);                                      \
        __dma(/*mode*/ kUpsample, /*rsync*/ (sync_flag << 8) | sync_flag, /*desp_ctrl*/ 0,       \
              /*int_base*/ dst_base, /*ext_base*/ src_base, /*dir*/ kExt2Int,                    \
              /*data_unit*/ kUseless, /*dma_desp*/ (int)dma_desc, /*int_addr*/ dst_addr,         \
              /*ext_addr*/ src_addr);                                                            \
        __wfe(~(1 << sync_flag), 1);                                                             \
        dst_addr += dst_ofs;                                                                     \
      }                                                                                          \
      h_scale = h_scale % DMA_MAX_TRANS_NUM;                                                     \
    }                                                                                            \
    if (h_scale > 0) {                                                                           \
      init_dma_desc(dma_desc, c, 0, src_c_stride, dst_c_stride, 0, dst_w_stride* dst_c_stride,   \
                    w_scale, h_scale, c* w, 0);                                                  \
      __dma(kUpsample, (sync_flag << 8) | sync_flag, 0, dst_base, src_base, kExt2Int, kUseless,  \
            (int)dma_desc, dst_addr, src_addr);                                                  \
      __wfe(~(1 << sync_flag), 1);                                                               \
    }                                                                                            \
  }

#define AIFF(ctrl_addr, param_addr, act_addr, mode)                                                \
  {                                                                                                \
    int sync_flag = GET_IDLE_FLAG();                                                               \
    __aiff(mode, (sync_flag << 8) | sync_flag, 0, (int)ctrl_addr, (int)param_addr, (int)act_addr); \
    __wfe(~(1 << sync_flag), 1);                                                                   \
  }

#define ASYNC_AIFF(ctrl_addr, param_addr, act_addr, sync_flag, mode) \
  { __aiff(mode, (sync_flag << 8) | sync_flag, 0, (int)ctrl_addr, (int)param_addr, (int)act_addr); }

/**
 * \brief memset addr with specific value
 *
 * \param ddr The ddr data address.
 * \param trans_size The transfer size in byte.
 * \param value The value will be set.
 * \param du Data unit.
 * \param addr_base The addr_base flag: 0 for DDR, 4 for kLsram, 8 for kShared.
 */
#define MemsetDDR(ddr, trans_size, value, du, addr_base)                                         \
  {                                                                                              \
    if (trans_size <= 0) return;                                                                 \
    int sync_flag = GET_IDLE_FLAG();                                                             \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                               \
                                                                                                 \
    uint width = DMA_GET_BEST_WIDTH(trans_size);                                                 \
    uint stride = width;                                                                         \
                                                                                                 \
    uint loop_num = trans_size / 0xfffffe;                                                       \
    uint remain_size = trans_size % 0xfffffe;                                                    \
                                                                                                 \
    if (loop_num > 0) {                                                                          \
      init_dma_desc(dma_desc, width, 0, stride, 0, 0xfffffe, 0, loop_num, 1, 0xfffffe, 0);       \
      __dma(kMemset, (sync_flag << 8) | sync_flag, 0, 0, addr_base, kInt2Ext, du, (int)dma_desc, \
            value, (int)ddr);                                                                    \
      __wfe(~(1 << sync_flag), 1);                                                               \
    }                                                                                            \
    if (remain_size > 0) {                                                                       \
      init_dma_desc(dma_desc, width, 0, stride, 0, 0, 0, 1, 1, remain_size, 0);                  \
      __dma(kMemset, (sync_flag << 8) | sync_flag, 0, 0, addr_base, kInt2Ext, du, (int)dma_desc, \
            value, (int)ddr + loop_num * 0xfffffe);                                              \
      __wfe(~(1 << sync_flag), 1);                                                               \
    }                                                                                            \
  }
/**
 * \brief memset sram with specific value
 *
 * \param sram The sram data address.
 * \param trans_size The transfer size in byte.
 * \param value The value will be set.
 * \param du Data unit.
 * \param addr_base The addr_base flag: 0 for DDR, 4 for kLsram, 8 for kShared.
 */
#define MemsetSRAM(sram, trans_size, value, du, addr_base)                                     \
  {                                                                                            \
    if (trans_size <= 0) return;                                                               \
    int sync_flag = GET_IDLE_FLAG();                                                           \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                             \
                                                                                               \
    uint width = DMA_GET_BEST_WIDTH(trans_size);                                               \
    uint stride = width;                                                                       \
                                                                                               \
    init_dma_desc(dma_desc, 0, width, 0, stride, 0, 0, 1, 1, 0, trans_size);                   \
    __dma(kMemset, (sync_flag << 8) | sync_flag, 0, addr_base, 0, kExt2Int, du, (int)dma_desc, \
          (int)sram, value);                                                                   \
    __wfe(~(1 << sync_flag), 1);                                                               \
  }
/**
 * \brief moving data from src to dst with DMA transpose mode.
 *
 * \param src The input ddr address.
 * \param dst The output ddr address.
 * \param src_region Src addr base.
 * \param dst_region Dst addr base.
 * \param col The col of transpose.
 * \param row The row of transpose.
 * \param col_stride The col_stride of transpose.
 * \param row_stride The row_stride of transpose.
 * \param src_gap The src gap of each move.
 * \param dst_gap The dst gap of each move.
 * \param trans_num The trans_num of each move.
 * \param du Data unit.
 */
#define DMATranspose(src, dst, src_region, dst_region, col, row, col_stride, row_stride, src_gap, \
                     dst_gap, trans_num, du)                                                      \
  {                                                                                               \
    int sync_flag = GET_IDLE_FLAG();                                                              \
    DmaDesc* dma_desc = (DmaDesc*)__get_dma_descriptor(sync_flag);                                \
    init_dma_desc(dma_desc, col, row, col_stride, row_stride, src_gap, dst_gap, trans_num,        \
                  trans_num, 0, 0);                                                               \
    __dma(kTranspose, (sync_flag << 8) | sync_flag, 0, dst_region, src_region, kExt2Int, du,      \
          (int)dma_desc, (int)dst, (int)src);                                                     \
    __wfe(~(1 << sync_flag), 1);                                                                  \
  }
/**
 * \brief moving data from src to dst with DMA transpose mode.
 *
 * \param src The input ddr address.
 * \param dst The output ddr address.
 * \param col The width of src.
 * \param row The width of dst.
 * \param col_stride The stride of src.
 * \param row_stride The stride of dst.
 * \param du Data unit.
 * \param data_width Data width in byte.
 */
#define GEN_DMA_Transpose2D(src_addr_space, src_region, dst_addr_space, dst_region)                \
  _CLC_OVERLOAD _CLC_INLINE static void DMA_Transpose2D(                                           \
      src_addr_space void* src, dst_addr_space void* dst, int col, int row, int col_stride,        \
      int row_stride, DmaDataUnit du, int data_width) {                                            \
    if (col * data_width > DMA_MAX_STRIDE || row * data_width > DMA_MAX_STRIDE) {                  \
      return;                                                                                      \
    }                                                                                              \
    if (col * data_width <= DMA_MAX_WIDTH && row * data_width <= DMA_MAX_WIDTH) {                  \
      DMATranspose(src, dst, src_region, dst_region, col* data_width, row* data_width,             \
                   col_stride* data_width, row_stride* data_width, 0, 0, 1, du);                   \
      return;                                                                                      \
    }                                                                                              \
    if (row >= col) {                                                                              \
      uint each_row = DMA_MAX_WIDTH / data_width;                                                  \
      uint loop_num = row / each_row;                                                              \
      uint tail_row = row % each_row;                                                              \
      uint each_tran_num = MIN(loop_num, DMA_MAX_TRANS_NUM);                                       \
      uint out_loop = loop_num / each_tran_num;                                                    \
      uint tail_loop = loop_num % each_tran_num;                                                   \
      for (int i = 0; i < out_loop; ++i) {                                                         \
        DMATranspose(src, dst, src_region, dst_region, col* data_width, each_row* data_width,      \
                     col* data_width, row* data_width, each_row* col* data_width,                  \
                     each_row* data_width, each_tran_num, du);                                     \
        src += each_row * col * each_tran_num * data_width;                                        \
        dst += each_row * each_tran_num * data_width;                                              \
      }                                                                                            \
      if (tail_loop) {                                                                             \
        DMATranspose(src, dst, src_region, dst_region, col, each_row* data_width, col* data_width, \
                     row* data_width, each_row* col* data_width, each_row* data_width, tail_loop,  \
                     du);                                                                          \
        src += each_row * col * tail_loop * data_width;                                            \
        dst += each_row * tail_loop * data_width;                                                  \
      }                                                                                            \
      if (tail_row > 0) {                                                                          \
        DMATranspose(src, dst, src_region, dst_region, col* data_width, tail_row* data_width,      \
                     col* data_width, row* data_width, 0, 0, 1, du);                               \
      }                                                                                            \
      return;                                                                                      \
    }                                                                                              \
    uint each_col = DMA_MAX_WIDTH / data_width;                                                    \
    uint loop_num = col / each_col;                                                                \
    uint tail_col = col % each_col;                                                                \
    uint each_tran_num = MIN(loop_num, DMA_MAX_TRANS_NUM);                                         \
    uint out_loop = loop_num / each_tran_num;                                                      \
    uint tail_loop = loop_num % each_tran_num;                                                     \
    for (int i = 0; i < out_loop; ++i) {                                                           \
      DMATranspose(src, dst, src_region, dst_region, each_col* data_width, row* data_width,        \
                   col* data_width, row* data_width, each_col* data_width,                         \
                   each_col* row* data_width, each_tran_num, du);                                  \
      src += each_col * each_tran_num * data_width;                                                \
      dst += each_col * row * each_tran_num * data_width;                                          \
    }                                                                                              \
    if (tail_loop) {                                                                               \
      DMATranspose(src, dst, src_region, dst_region, each_col* data_width, row* data_width,        \
                   col* data_width, row* data_width, each_col* data_width,                         \
                   each_col* row* data_width, tail_loop, du);                                      \
      src += each_col * tail_loop * data_width;                                                    \
      dst += each_col * row * tail_loop * data_width;                                              \
    }                                                                                              \
    if (tail_col > 0) {                                                                            \
      DMATranspose(src, dst, src_region, dst_region, tail_col* data_width, row* data_width,        \
                   col* data_width, row* data_width, 0, 0, 1, du);                                 \
    }                                                                                              \
    return;                                                                                        \
  }

GEN_DMA_Transpose2D(__lsram, kLsram, global, kGlobal);
GEN_DMA_Transpose2D(__local, kShared, global, kGlobal);
GEN_DMA_Transpose2D(global, kGlobal, global, kGlobal);
GEN_DMA_Transpose2D(global, kGlobal, __lsram, kLsram);
GEN_DMA_Transpose2D(global, kGlobal, __local, kShared);

#define GEN_CEIL_DIV(dtype)                                   \
  _CLC_OVERLOAD _CLC_INLINE dtype ceildiv(dtype x, dtype y) { \
    return (dtype)((dtype)((dtype)(x + y) - 1) / y);          \
  }

GEN_CEIL_DIV(char);
GEN_CEIL_DIV(uchar);
GEN_CEIL_DIV(short);
GEN_CEIL_DIV(ushort);
GEN_CEIL_DIV(int);
GEN_CEIL_DIV(uint);

#define vexp(x) exp(x)
#define vexp2(x) exp2(x)
#define verf(x) erf(x)
#define vlog(x) log(x)
#define vtanh(x) tanh(x)
#define vsin(x) sin(x)
#define vcos(x) cos(x)
#define vrsqrt(x) rsqrt(x)
#define vsqrt(x) sqrt(x)
#define vfloor(x) floor(x)
#define vceil(x) ceil(x)
#define vpow(x, exponent) pow(x, exponent)
#define vmul_hi(x, y) mul_hi(x, y)

#endif  // DSL_COMMON_H_
