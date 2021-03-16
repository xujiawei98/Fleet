#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle on one node"


if [ ! -d "./log" ]; then
  mkdir ./log
fi

# environment variables for fleet distribute training

export PADDLE_WITH_GLOO=1
export PADDLE_GLOO_RENDEZVOUS=3
export PADDLE_GLOO_HTTP_ENDPOINT=127.0.0.1:30019

export PADDLE_PSERVER_NUMS=2
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29031,127.0.0.1:29132"
export PADDLE_PSERVER_PORT_ARRAY=(29031 29132)

export PADDLE_TRAINERS_NUM=2
export CPU_NUM=2

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

for((i=0;i<$PADDLE_TRAINERS_NUM;i++))
do
    echo "PADDLE WILL START Trainer "$i
    export PADDLE_TRAINER_ID=$i
    python -u $SC &> ./log/trainer.$i.log &
done

echo "Training log stored in ./log/"
