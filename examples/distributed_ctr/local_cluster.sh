#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle Fluid on one node"


if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

# environment variables for fleet distribute training
unset http_proxy
unset https_proxy

export PADDLE_TRAINER_ID=0
export CPU_NUM=4

export FLAGS_communicator_thread_pool_size=5
export FLAGS_rpc_retry_times=3
export FLAGS_communicator_independent_recv_thread=0

export PADDLE_PSERVER_NUMS=4
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29031,127.0.0.1:29132,127.0.0.1:29233,127.0.0.1:29234"
export PADDLE_PSERVER_PORT_ARRAY=(29031 29132 29233 29234)

export PADDLE_TRAINERS=4
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}

export TRAINING_ROLE=PSERVER
export GLOG_v=0

SC="train.py --test=True"

for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    export PADDLE_PORT=${cur_port}
    export POD_IP=127.0.0.1
    python -u $SC &> ./log/pserver.$i.log &
done

export TRAINING_ROLE=TRAINER
export GLOG_v=0


for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    PADDLE_TRAINER_ID=$i
    python -u $SC &> ./log/trainer.$i.log &
done

echo "Training log stored in ./log/"
