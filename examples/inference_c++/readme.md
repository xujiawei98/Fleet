
wget inference C++ lib from: https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/index_cn.html
such as: wget https://paddle-inference-lib.bj.bcebos.com/1.8.4-cpu-noavx-openblas/fluid_inference.tgz

command:
```
# uncompreszz inference lib.
wget https://paddle-inference-lib.bj.bcebos.com/1.8.4-cpu-noavx-openblas/fluid_inference.tgz
tar -zxvf fluid_inference.tgz

mkdir build 
cd build
cmake .. -DPADDLE_LIB=../fluid_inference -DINFER_NAME=main
make

#execute
./main

```


introduce covert.py:
1. convert sparse variable's type from LOD_TENSOR to SELECTED_ROWS in __model__
2. merge shard sparse params into one with SELECTED_ROWS.

```
specify model dirname in .py
python it.
tips: still have many hard codes in it.
```

