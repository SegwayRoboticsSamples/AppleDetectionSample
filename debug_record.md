1.mobilenetv1_0.25_v3_1.0匹配颜旭的model, --> .pb为21M
2.mobilenetv1_0.25_v3_0.25 --> .pb  为800K
3.mobilenetv1_0.25_v3_0.5  --> .pb  为1M
4.mobilenetv1_0.5_v3_1.0   --> .pb  为3.8M
5.mobilenetv1_1.0_v3_1.0   --> .pb  为10M

2022.01.14：目前正在跑颜旭的model，跑完当做teacher，之后训练student。


mobilenetv1_0.5_v3_1.0   3.8M模型效果不错，且速度也不慢。
frozen_eval_graph-80599-20000.pb    2M大小，效果最好，可在海外采集的图片中测试行人效果。

# val
## 1.val .checkpoint:
The type of data is ".tfrecord". Run
``python eval.py --load_config config/eval.yaml``,(note change the lines of eval.yaml file which has "# change" notes!)

## 2.val .pb:
The type of data is the data bag of collection. Run
``python deploy_aibox.py --load_config config/deploy.yaml``,(note change the lines of deploy.yaml file which has "# change" notes!)

## 3.val the .tflite:
The type of data is the data bag of collection. Run
``./inference.sh``,(note change the lines of inference.yaml file which has "# change" notes!)


# convert model
## 1.convert .checkpoint --> .pb
``bash ./sh generate-tflite-float-model.sh``(NOTE: the graph.pbtxt use the "graph_eval.pbtxt" where the first step of val __1.val .checkpoint__ generate)

## 2.convert .pb --> .lite
``bash ./sh generate-tflite-float-model.sh``  -->  .lite == 1.96M
or
### quantize(this way does not work):
``python script/quantize/quantize.py``        -->  .lite == 0.53M