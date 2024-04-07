// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file dmabuf_deploy.cc
 * \brief Example code on load and run TVM module
 */
#include <aipu/runtime/execution_engine.h>
#include <dlpack/dlpack.h>
#include <fcntl.h>
#include <kmd/armchina_aipu.h>
#include <standard_api.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>

#define DEV_AIPU "/dev/aipu"

int main(int argc, char* argv[]) {
  // test model: tf_mobilenet_v2
  // input_shape: (1, 224, 224, 3) uint8
  // output_shape: (1, 1001) uint8
  const char* model_file = argv[1];
  const char* input_file = argv[2];

  const int run_frame = 5;

  const int total_size = 224 * 224 * 3;
  const int out_size = 1001;
  uint8_t val;
  std::ifstream fin(input_file, std::ios::binary);
  std::vector<uint8_t> image;
  image.resize(total_size);
  while (fin.read(reinterpret_cast<char*>(&val), sizeof(uint8_t))) image.push_back(val);

  // Prepare inputs, input should from dmabuf exporter
  int aipu_fd = open(DEV_AIPU, O_RDWR);
  if (aipu_fd < 0) {
    LOG(ERROR) << " failed to open " << DEV_AIPU;
    exit(-1);
  }
  struct aipu_dma_buf_request dma_buf_req = {0};
  dma_buf_req.bytes = total_size;
  ioctl(aipu_fd, AIPU_IOCTL_ALLOC_DMA_BUF, &dma_buf_req);

  int inp_dmabuf_fd = dma_buf_req.fd;

  // If npu perform as an exporter, the output fd should be exported.
  // The input and output should not be on the same fd
  dma_buf_req = {0};
  dma_buf_req.bytes = out_size;
  ioctl(aipu_fd, AIPU_IOCTL_ALLOC_DMA_BUF, &dma_buf_req);
  int out_dmabuf_fd = dma_buf_req.fd;

  void* va = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, inp_dmabuf_fd, 0);
  memcpy(va, &image[0], total_size);
  munmap(va, total_size);

  // The model should be quant model
  // otherwise the input/output may have quant/dequant cpu layer
  // cpu load/store dmabuf is uncacheable, whose performance is really bad.
  // also remember to add the 2 option in buildtool config file [GBuilfer] section
  // disable_input_buffer_reuse = True
  // disable_output_buffer_reuse = True
  auto mod = tvm::runtime::Module::LoadFromFile(model_file);
  auto shared_inp_fp = mod.GetFunction("compass_set_input_shared", true);
  auto shared_out_fp = mod.GetFunction("compass_mark_output_shared", true);
  auto run = mod.GetFunction("compass_run", true);

  // test 1: normal mode, not use dmabuf as input/output
  DLDevice dev{kDLCPU, 0};

  tvm::runtime::NDArray input =
      tvm::runtime::NDArray::Empty({total_size}, DLDataType{kDLUInt, 8, 1}, dev);
  tvm::runtime::NDArray output =
      tvm::runtime::NDArray::Empty({out_size}, DLDataType{kDLUInt, 8, 1}, dev);
  input.CopyFromBytes(image.data(), total_size);

  auto start = std::chrono::steady_clock::now();
  for (int frame = 0; frame < run_frame; frame++) {
    run(input, output);
  }
  auto end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG(INFO) << "not use dmabuf total cost " << dur.count() << " us" << std::endl;
  std::ofstream fout("no_dmabuf.bin", std::ios::out | std::ios::binary);
  fout.write(static_cast<char*>(output->data), out_size);

  // test 2: assume input from dmabuf, npu output not export to fd
  tvm::runtime::NDArray fake_input =
      tvm::runtime::NDArray::Empty({total_size}, DLDataType{kDLUInt, 8, 1}, dev);
  output = tvm::runtime::NDArray::Empty({out_size}, DLDataType{kDLUInt, 8, 1}, dev);

  // only 1 input, so the input_handle only 1 value
  tvm::runtime::NDArray dma_handle =
      tvm::runtime::NDArray::Empty({1}, DLDataType{kDLInt, 32, 1}, dev);
  // set the input handle, use cpu output, the output is in NDArray
  dma_handle.CopyFromBytes(&inp_dmabuf_fd, sizeof(int));
  shared_inp_fp(dma_handle);

  start = std::chrono::steady_clock::now();
  for (int frame = 0; frame < run_frame; frame++) {
    run(fake_input, output);
  }
  end = std::chrono::steady_clock::now();
  dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG(INFO) << "use dmabuf as input total cost " << dur.count() << " us" << std::endl;
  std::ofstream inpdma("input_dmabuf.bin", std::ios::out | std::ios::binary);
  inpdma.write(static_cast<char*>(output->data), out_size);

  // test 3: assume input from dmabuf, npu output export to fd
  // the output is on the dmabuf
  tvm::runtime::NDArray fake_output =
      tvm::runtime::NDArray::Empty({out_size}, DLDataType{kDLUInt, 8, 1}, dev);

  // input/output all shared to dmabuf fd.
  dma_handle.CopyFromBytes(&inp_dmabuf_fd, sizeof(int));
  shared_inp_fp(dma_handle);
  dma_handle.CopyFromBytes(&out_dmabuf_fd, sizeof(int));
  shared_out_fp(dma_handle);

  start = std::chrono::steady_clock::now();
  for (int frame = 0; frame < run_frame; frame++) {
    run(fake_input, fake_output);
  }
  end = std::chrono::steady_clock::now();
  dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG(INFO) << "use dmabuf as input, export output to dmabuf total cost " << dur.count() << " us"
            << std::endl;

  va = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_dmabuf_fd, 0);
  std::ofstream iodma("inpout_dmabuf.bin", std::ios::out | std::ios::binary);
  iodma.write(static_cast<char*>(va), out_size);
  munmap(va, out_size);

  ioctl(aipu_fd, AIPU_IOCTL_FREE_DMA_BUF, &out_dmabuf_fd);
  ioctl(aipu_fd, AIPU_IOCTL_FREE_DMA_BUF, &inp_dmabuf_fd);
  close(aipu_fd);
  return 0;
}