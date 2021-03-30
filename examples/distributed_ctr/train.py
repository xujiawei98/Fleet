#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import time
import logging

import numpy as np

import paddle
from paddle.distributed.fleet.utils.ps_util import DistributedInfer
paddle.enable_static()

from paddle.distributed import fleet
import paddle.static as fluid

from network import CTR
from argument import params_args
from py_reader_generator import CriteoDataset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("paddle.static")
logger.setLevel(logging.INFO)


def get_pyreader(inputs, params):
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
    #if not int(params.cloud):
    #    file_list = fleet.utils().get_file_shard(file_list)
    logger.info("file list: {}".format(file_list))

    py_reader = paddle.fluid.io.DataLoader.from_generator(capacity=64,
                                                       feed_list=inputs,
                                                       iterable=False,
                                                       use_double_buffer=False)

    train_generator = CriteoDataset(params.sparse_feature_dim)
    py_reader.set_sample_generator(train_generator.train(
                    file_list, fleet.worker_num(), fleet.worker_index()), params.batch_size)

    return inputs, py_reader


def get_dataset(inputs, params):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(params.batch_size)
    thread_num = int(params.cpu_num)
    dataset.set_thread(thread_num)
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
    if not int(params.cloud):
        file_list = fleet.split_files(file_list)
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    return dataset


def distributed_predict():
    place = paddle.fluid.CPUPlace()
    train_generator = CriteoDataset(params.sparse_feature_dim)

    file_list = [
        str(params.test_files_path) + "/%s" % x
        for x in os.listdir(params.test_files_path)
    ]

    test_reader = paddle.batch(train_generator.test(file_list),
                               batch_size=params.batch_size)

    startup_program = paddle.fluid.framework.Program()
    test_program = paddle.fluid.framework.Program()

    with paddle.fluid.framework.program_guard(test_program, startup_program):
        with paddle.fluid.unique_name.guard():
            ctr_model = CTR()
            inputs = ctr_model.input_data(params)
            _, _, _, words, predict = ctr_model.net(inputs, params, is_test=True, is_inference=True)

    dist_infer = DistributedInfer(
          main_program=test_program, startup_program=startup_program)

    eval_dist_infer = dist_infer.get_dist_infer_program()

    exe = paddle.fluid.Executor(place)
    feeder = paddle.fluid.DataFeeder(feed_list=inputs, place=place)

    label = words[-1]

    labels = []
    predicts = []
    for batch_id, data in enumerate(test_reader()):
        l_val, pre_val = exe.run(eval_dist_infer,
                                 feed=feeder.feed(data),
                                 fetch_list=[label, predict])
        labels.extend(l_val)
        predicts.extend(pre_val)

    with open("infer.predict", "w") as wb:
        for i in range(len(labels)):
            wb.write("{}\t{}\n".format(labels[i], predicts[i]))
    logger.info("Inference complete")



def train(params):
    # 根据环境变量确定当前机器/进程在分布式训练中扮演的角色
    # 然后使用 fleet api的 init()方法初始化这个节点
    fleet.init()

    # 我们还可以进一步指定分布式的运行模式，通过 DistributedStrategy进行配置
    # 如下，我们设置分布式运行模式为同步(async)
    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True

    ctr_model = CTR()
    inputs = ctr_model.input_data(params)
    inputs, reader = get_pyreader(inputs, params)
    avg_cost, auc_var, batch_auc_var, inputs, predict = ctr_model.net(inputs, params)
    optimizer = paddle.optimizer.Adam(params.learning_rate)

    # 配置分布式的optimizer，传入我们指定的strategy，构建program
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    # 根据节点角色，分别运行不同的逻辑
    if fleet.is_server():
        # 初始化及运行参数服务器节点
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        # 初始化工作节点
        fleet.init_worker()

        for epoch in range(params.epochs):
            start_time = time.time()

            reader.start()
            batch_id = 0
            try:
                while True:
                    loss_val, auc_val, batch_auc_val = exe.run(
                        program=paddle.static.default_main_program(),
                        fetch_list=[
                            avg_cost.name, auc_var.name, batch_auc_var.name
                        ])
                    loss_val = np.mean(loss_val)
                    auc_val = np.mean(auc_val)
                    batch_auc_val = np.mean(batch_auc_val)
                    if batch_id % 10 == 0 and batch_id != 0:
                        logger.info(
                            "TRAIN --> pass: {} batch: {} loss: {} auc: {}, batch_auc: {}"
                            .format(epoch, batch_id,
                                    loss_val / params.batch_size, auc_val,
                                    batch_auc_val))
                    batch_id += 1
            except paddle.fluid.core.EOFException:
                reader.reset()

            end_time = time.time()
            logger.info("epoch %d finished, use time=%d\n" %
                        ((epoch), end_time - start_time))

            #fleet.barrier_worker()
            #distributed_predict()
            #fleet.barrier_worker()

            # 默认使用0号节点保存模型
            if params.test and fleet.is_first_worker():
                model_path = (str(params.model_path) + "/" + "epoch_" +
                              str(epoch))

                input_varnames = [var.name for var in inputs[:-1]]
                fleet.save_persistables(executor=exe, dirname=model_path)
                fleet.save_inference_model(dirname=model_path,
                                          feeded_var_names=input_varnames,
                                          #feeded_var_names=["dense_input"]+["sparse_embedding_{}.tmp_0".format(i) for i in range(26) ],
                                           target_vars=[predict], executor=exe)

        fleet.stop_worker()
        logger.info("Distribute Train Success!")


if __name__ == "__main__":
    params = params_args()
    train(params)
