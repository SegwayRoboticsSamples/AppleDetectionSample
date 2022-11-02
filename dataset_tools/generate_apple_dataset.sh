#!/bin/sh
workspace=$(cd "$(dirname "$0")";pwd)
python $workspace/create_apple_tf_record.py \
    --data_dir '/raid/data/object_detect/Apple_221019/train' \
    --from_database 0 \
    --data_family 'original' \
    --output_path './tf_record/Apple_221019/20221019' \
    --visual_dir './visualization' \
    --tfrecord_width 512 \
    --tfrecord_height 512

