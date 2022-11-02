export CUDA_VISIBLE_DEVICES='2'

freeze_graph \
  --input_graph=./models/experiment-AiBox-Apple-model-mbv1-0.25-20221017-num1/graph_eval.pbtxt \
  --input_checkpoint=./models/experiment-AiBox-Apple-model-mbv1-0.25-20221019/model.ckpt-571130 \
  --output_graph=./best_distillation_float_model_folder/frozen_eval_apple_graph-571130.pb \
  --output_node_names=head/reg13x13_output/BiasAdd,head/reg26x26/BiasAdd,head/reg52x52/BiasAdd
echo "freeze graph done."

tflite_convert \
  --output_file=./best_distillation_float_model_folder/frozen_eval_apple_graph-571130.tflite \
  --graph_def_file=./best_distillation_float_model_folder/frozen_eval_apple_graph-571130.pb \
  --input_arrays=Placeholder \
  --output_arrays=head/reg13x13_output/BiasAdd,head/reg26x26/BiasAdd,head/reg52x52/BiasAdd
echo "tflite convertion done."
