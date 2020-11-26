/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/include/paddle_inference_api.h"
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

namespace paddle {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

Record ProcessALine(const std::string &line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  NativeConfig config;
  config.model_dir = model_dirname;
  config.use_gpu = false;
  config.device = 0;

  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = ::paddle::CreatePaddlePredictor<NativeConfig>(config);

  // Just a single batch of data.
  std::string line = "0.1 0.2 0.23 0.33 0.13 0.56 0.78 0.88 0.99 0.11 0.22 0.33 0.44	1 13";
  auto record = ProcessALine(line);
  
  // Inference.
  PaddleTensor input;
  input.shape = record.shape;
  input.data =
          PaddleBuf(record.data.data(), record.data.size() * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> output, analysis_output;
  predictor->Run({input}, &output, 1);

  auto& tensor = output.front();
  size_t numel = tensor.data.length() / PaddleDtypeSize(tensor.dtype);
  VLOG(1) << "predict numel: " << numel;
  for (size_t i = 0; i < numel; ++i) {
    VLOG(0) << "predict val: " << (static_cast<int64_t*>(tensor.data.data())[i]);
  }

}
} // namespace paddle

int main() {
  paddle::RunAnalysis(1, "../demo");
  return 0;
}

