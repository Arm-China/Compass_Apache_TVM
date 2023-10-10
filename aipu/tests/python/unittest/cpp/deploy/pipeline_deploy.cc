// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file pipeline_deploy.cc
 * \brief Example code on load and run TVM Compass Pipeline module.s
 */
#include <aipu/runtime/utils.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

std::string file_to_string(const std::string& filename, bool is_binary = false) {
  std::ifstream ifs;
  if (is_binary) {
    ifs = std::ifstream(filename, std::ios::binary);
  } else {
    ifs = std::ifstream(filename);
  }
  if (!ifs) {
    LOG(ERROR) << "failed to open file " << filename << ", exit!" << std::endl;
    exit(-1);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

void save_binary(const std::string& filename, tvm::runtime::NDArray& data) {
  const DLTensor tensor = *data.operator->();
  size_t data_size = tvm::runtime::GetDataSize(tensor);

  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(tensor.data), data_size);
}

int main(int argc, char* argv[]) {
  // example for onnx yolop
  // 1 backbone with 3 head.
  // input_shape: (1, 3, 320, 320)
  // model_dir is the saved dir name
  // model_dir + "/config" is the pipeline config file
  // the files is arranged automatically if export_library saved.
  const char* model_dir = argv[1];
  const char* input_file = argv[2];

  // load the image data
  const int total_size = 320 * 320 * 3;
  std::string cur_path = tvm::runtime::GetCwd();

  std::string inp = file_to_string(input_file, true);

  static auto load_lib_fn = tvm::runtime::Registry::Get("compass_pipeline.load");

  tvm::runtime::Module mod = (*load_lib_fn)(std::string(model_dir) + "/config");
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output_by_index");
  tvm::runtime::PackedFunc run = mod.GetFunction("run");

  // Use the C++ API
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::NDArray x =
      tvm::runtime::NDArray::Empty({1, 3, 320, 320}, DLDataType{kDLFloat, 32, 1}, dev);
  x.CopyFromBytes(inp.c_str(), total_size * sizeof(float));

  tvm::runtime::NDArray det_out, drv_out, line_out;

  auto det_fn = [&get_output, &det_out]() {
    auto start = std::chrono::steady_clock::now();
    det_out = get_output(0, 1000);
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Detect out cost " << dur.count() << "us" << std::endl;
  };

  auto drv_fn = [&get_output, &drv_out]() {
    auto start = std::chrono::steady_clock::now();
    drv_out = get_output(1, 1000);
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Drive area out cost " << dur.count() << "us" << std::endl;
  };

  auto line_fn = [&get_output, &line_out]() {
    auto start = std::chrono::steady_clock::now();
    line_out = get_output(2, 1000);
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Lane Line out cost " << dur.count() << "us" << std::endl;
  };

  for (int frame = 0; frame < 3; frame++) {
    set_input("images", x);
    auto det_thread = std::thread(det_fn);
    auto drv_thread = std::thread(drv_fn);
    auto line_thread = std::thread(line_fn);
    auto start = std::chrono::steady_clock::now();
    run();
    auto backbone = std::chrono::steady_clock::now();
    det_thread.join();
    drv_thread.join();
    line_thread.join();
    auto end = std::chrono::steady_clock::now();
    auto dur_bb = std::chrono::duration_cast<std::chrono::microseconds>(backbone - start);
    LOG(INFO) << "BackBone cost " << dur_bb.count() << "us" << std::endl;
    auto dur_total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Total cost " << dur_total.count() << "us" << std::endl;
  }
  // save output
  save_binary(tvm::runtime::AbsPath(cur_path) + "/det_out.bin", det_out);
  save_binary(tvm::runtime::AbsPath(cur_path) + "/drv_out.bin", drv_out);
  save_binary(tvm::runtime::AbsPath(cur_path) + "/line_out.bin", line_out);
  return 0;
}
