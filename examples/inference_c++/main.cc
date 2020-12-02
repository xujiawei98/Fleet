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
  std::vector<float> dense_data;
  std::vector<std::vector<float>> sparse_data;
};


void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  NativeConfig config;
  config.model_dir = model_dirname;
  config.use_gpu = false;
  config.device = 0;

  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = ::paddle::CreatePaddlePredictor<NativeConfig>(config);

  // Just a single batch of data.
  auto record = Record();
  for (size_t i = 0; i < 13; i++) {
    // Genearte dense data here
    record.dense_data.push_back(1.0);
  }

  std::vector<int64_t> key_vec = {861009690, 836552220, 778450980, 58140440, 91813380, 117262650, 397836660, 627934030, 946120000, 11234850, 149644800, 300559680, 335589480, 556192380, 785140490, 896284770, 954638860, 979116410, 79506000, 222847040, 296136530, 458813520, 699324440, 784662080, 815949660, 869950730};
  record.sparse_data.resize(key_vec.size());

  for(auto i=0; i< key_vec.size(); i++) {
    // Do lookup table here
    record.sparse_data.push_back(std::vector<float>(10));
  }

  // Inference.
  for (size_t try_cnt =0; try_cnt < 1000; try_cnt++) {
    std::vector<PaddleTensor> p_tensor_vec;

    PaddleTensor dense_input;
    dense_input.shape = {1,13};
    dense_input.data =
            PaddleBuf(record.dense_data.data(), record.dense_data.size() * sizeof(float));
    dense_input.dtype = PaddleDType::FLOAT32;

    p_tensor_vec.push_back(dense_input);
    for(size_t i =0; i < 26; i++) {
      PaddleTensor cate_input;
      cate_input.shape = {1,10};
      cate_input.data = PaddleBuf(&(record.sparse_data[i]), record.sparse_data[i].size()*sizeof(float));
      cate_input.dtype = PaddleDType::FLOAT32;
      p_tensor_vec.push_back(cate_input);
    }

    std::vector<PaddleTensor> output, analysis_output;
    predictor->Run(p_tensor_vec, &output, 1);

    auto& tensor = output.front();
    size_t numel = tensor.data.length() / PaddleDtypeSize(tensor.dtype);

    std::cout << "Try cnt: "<< try_cnt<<" predict numel: " << numel << "\n";
    for (size_t i = 0; i < numel; ++i) {
      std::cout <<"No." << i << " predict val: " << (static_cast<float*>(tensor.data.data())[i]) << "\n";
    }

  }
} // namespace paddle
}

int main() {
  paddle::RunAnalysis(1, "../demo/");
  return 0;
}