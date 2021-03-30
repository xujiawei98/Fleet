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


void GetInputTensors(int batch_size, std::vector<PaddleTensor>* inputs) {
  int dense_dim = 13;
  int sparse_slots = 27;

  std::default_random_engine generator;  
  std::uniform_int_distribution<int> sparse_gen(5, 20);  
  std::uniform_int_distribution<int64_t> feasign_gen(0, 100000);  
  std::uniform_real_distribution<float> dense_gen(-1, 1);


  std::vector<float> denses;
  for(auto x=0; x< batch_size * dense_dim; x++) {
      denses.push_back(dense_gen(generator));
  }

  PaddleTensor dense_input;
  dense_input.shape = {batch_size, dense_dim};
  dense_input.data = PaddleBuf(denses.data(), denses.size() * sizeof(float));
  dense_input.dtype = PaddleDType::FLOAT32;
  inputs->push_back(dense_input);

  for(auto y=1; y<sparse_slots; y++) {
      std::vector<int64_t> sparse;
      std::vector<uint64_t> lod_1;

      lod_1.push_back(0);
      for(auto x=0; x<batch_size; x++) {
         int num = sparse_gen(generator);
         lod_1.push_back(num);

         for(auto m=0; m<num; m++) {
           sparse.push_back(feasign_gen(generator));
         }
      }

      PaddleTensor input;
      //input.shape = {batch_size, 1};
      input.shape = {(int)sparse.size(), 1};
      input.lod = {lod_1};
      input.data  = PaddleBuf(sparse.data(), sparse.size() * sizeof(int64_t));
      input.dtype = PaddleDType::INT64;
      inputs->push_back(input);
  }
}


void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  NativeConfig config;
  config.model_dir = model_dirname;
  config.use_gpu = false;
  config.device = 0;

  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = ::paddle::CreatePaddlePredictor<NativeConfig>(config);

  // Inference.
  for (size_t try_cnt =0; try_cnt < 1000; try_cnt++) {
    std::vector<PaddleTensor> inputs;
    GetInputTensors(100, &inputs);

    std::vector<PaddleTensor> outputs;
    predictor->Run(inputs, &outputs);
 
    std::cout << "outputs: " << outputs.size() << std::endl;

    //auto& tensor = output.front();
    //size_t numel = tensor.data.length() / PaddleDtypeSize(tensor.dtype);

    //std::cout << "Try cnt: "<< try_cnt<<" predict numel: " << numel << "\n";
    //for (size_t i = 0; i < numel; ++i) {
    //  std::cout <<"No." << i << " predict val: " << (static_cast<float*>(tensor.data.data())[i]) << "\n";
    //}

  }
} // namespace paddle
}

int main() {
  paddle::RunAnalysis(1, "../demo/");
  return 0;
}
