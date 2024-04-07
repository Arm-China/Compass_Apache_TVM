// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file cpp_deploy.cc
 * \brief Example code on load and run TVM module.s
 */
#include <aipu/runtime/execution_engine.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
  // test model: tf_mobilenet_v1
  // input_shape: (1, 224, 224, 3)
  // output_shape: (1, 56, 56, 256)
  const char* model_file = argv[1];
  const char* input_file = argv[2];

  // load in the library
  DLDevice dev{kDLCPU, 0};
  auto ee = tvm::runtime::ExecutionEngine(model_file);

  // load the image data
  const int total_size = 224 * 224 * 3;
  std::vector<float> image;
  float f = 0.f;
  std::ifstream fin(input_file, std::ios::binary);
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) image.push_back(f);

  // Use the C++ API
  tvm::runtime::NDArray x =
      tvm::runtime::NDArray::Empty({1, 224, 224, 3}, DLDataType{kDLFloat, 32, 1}, dev);
  x.CopyFromBytes(image.data(), total_size * sizeof(float));
  ee.SetInputs(x);
  tvm::runtime::Array<tvm::runtime::NDArray> outputs = ee.Run();

  // Save the result
  std::ofstream fout("output.bin", std::ios::out | std::ios::binary);
  fout.write(static_cast<char*>(outputs[0]->data), sizeof(float) * 56 * 56 * 256);

  return 0;
}
