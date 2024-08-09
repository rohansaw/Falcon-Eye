#!/bin/bash

source "configs/sds.sh"

# replace dry with saved_model


python3 model_inspect.py --runmode=dry \
  --model_name=efficientdet-lite0 --ckpt_path=$MODEL_OUT_DIR \
  --hparams=$HPARAMS_FILE \
  --saved_model_dir=$MODEL_OUT_DIR/saved_model

pip install git+https://github.com/onnx/tensorflow-onnx

python3 -m tf2onnx.convert --saved-model $MODEL_OUT_DIR/saved_model --output $MODEL_OUT_DIR/model.onnx --opset 12 --dequantize