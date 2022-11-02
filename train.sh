cur_time=$(date +%Y%m%d)
python ./train.py \
    --load_config ./config/train.yaml >& trainLog-apple-$cur_time.log