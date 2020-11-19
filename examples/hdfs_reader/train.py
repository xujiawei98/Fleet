# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import time
import paddle.fluid as fluid
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

class Model(object):

    def input_data(self):
        dense_input = fluid.layers.data(name="dense_input",
                                        shape=[13],
                                        dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(name="C" + str(i),
                              shape=[1],
                              lod_level=1,
                              dtype="int64") for i in range(1, 27)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs):
        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[1000000, 10]
            )
            return emb
        sparse_embed_seq = list(map(embedding_layer, inputs[1:-1]))

        concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)

        predict = fluid.layers.fc(
            input=concated,
            size=2,
            act="softmax",
        )
        label = fluid.layers.cast(inputs[-1], dtype="int64")
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        cost = fluid.layers.reduce_sum(loss)
        return cost


def create_dataset(model, hdfs_name, hdfs_ugi, file_list):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(model.input_data())
    dataset.set_batch_size(5)
    dataset.set_thread(2)
    dataset.set_hdfs_config(hdfs_name, hdfs_ugi)
    dataset.set_filelist(file_list)
    dataset.set_pipe_command("python dataset_generator.py")
    return dataset

def train():
    model = Model()
    inputs = model.input_data()
    loss = model.net(inputs)

    optimizer = fluid.optimizer.Adam(1e-4)
    optimizer.minimize(loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    hdfs_name="hdfs://xxx.com:8888"
    hdfs_ugi="usr,pwd"
    file_list=["hdfs:/user/paddle/train_data/part_999"]
    dataset = create_dataset(model, hdfs_name, hdfs_ugi, file_list)

    logger.info("Training Begin!")
    for epoch in range(2):
        start_time = time.time()
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[loss],
                               fetch_info=["Epoch {} loss ".format(epoch)],
                               print_period=1,
                               debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time=%d\n" %
                    ((epoch), end_time - start_time))

    logger.info("Training Success!")

if __name__ == "__main__":
    train()
