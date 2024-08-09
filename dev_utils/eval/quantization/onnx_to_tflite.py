import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from PIL import Image
import numpy as np
import json

input_path = "/data/results/faster_rcnn_resnet101/sw/end2end.onnx"
output_path_tf ="/data/results/faster_rcnn_resnet101/sw/model.tf"

onnx_model = onnx.load(input_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(output_path_tf)

img_root = "/data/sw/coco/val/"
val_set_json = "/data/sw/coco/annotations/instances_val.json"

img_size = (2688,1512)
coco = None
with open(val_set_json, "r") as f:
    coco = json.load(f)

def representative_dataset():
    for img in coco["images"]:
        img_path = img_root + img["file_name"]
        img = Image.open(img_path)
        img = img.resize(img_size)
        img = np.array(img)
        img = np.transpose(img)
        img = np.expand_dims(img, 0)
        print(img.shape)
        yield [tf.dtypes.cast(img, tf.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(output_path_tf)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()