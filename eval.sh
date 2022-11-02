#!/bin/sh
cur_time=$(date "+%Y-%m-%d-%H-%M-%S")
python ./eval.py \
    --load_config ./config/eval.yaml >& evalLog-$cur_time.log
echo "eval over."