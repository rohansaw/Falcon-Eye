import onnx
import onnx_tf
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from onnx import helper


model_path = "/scratch1/rsawahn/exported_models/mydet_sw_base_new_trasposed.onnx"
out_root = os.path.split(model_path)[0]
representative_dataset_path = "/scratch1/rsawahn/data/sw_sliced/coco/val"
# TODO ensure we have mostly positive samples in here

model_name = os.path.basename(model_path)
print("loading onnx model")
onnx_model = onnx.load(model_path)

name_map = {"input.1": "input_1"}

# Initialize a list to hold the new inputs
new_inputs = []

# Iterate over the inputs and change their names if needed
for inp in onnx_model.graph.input:
    if inp.name in name_map:
        # Create a new ValueInfoProto with the new name
        new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                inp.type.tensor_type.elem_type,
                                                [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
        new_inputs.append(new_inp)
    else:
        new_inputs.append(inp)

# Clear the old inputs and add the new ones
onnx_model.graph.ClearField("input")
onnx_model.graph.input.extend(new_inputs)

# Go through all nodes in the model and replace the old input name with the new one
for node in onnx_model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in name_map:
            node.input[i] = name_map[input_name]

# Save the renamed ONNX model
model_path = os.path.join(out_root, model_name + "-renamed_input.onnx")
onnx.save(onnx_model, model_path)


print("converting to tf")
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model_path = out_root +"/"+ model_name + ".tf"
tf_model.export_graph(tf_model_path)
print("tf model saved to", tf_model_path)


def preprocess_image(image_path, height, width):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    image = image.resize((width, height))
    image_data = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    mean = np.array([0.4263, 0.4856, 0.4507], dtype=np.float32)
    std = np.array([0.1638, 0.1515, 0.1755], dtype=np.float32)
    image_data = (image_data - mean) / std  # Normalize
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    # change shape to NHWC
    #image_data = np.transpose(image_data, (0, 3, 1, 2))
    return image_data

def representative_data_gen():
    # Load first 100 images from the representative_dataset_path, preprocess, normalize by mean and std
    files = os.listdir(representative_dataset_path)
    # shuffle files
    np.random.shuffle(files) 
    for f in files[:100]:
        img_path = os.path.join(representative_dataset_path, f)
        img = preprocess_image(img_path, 512, 512)
        print(img.shape, "IMG")
        yield [img.astype(np.float32)]  # Yield the image as a list

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert the model to TensorFlow Lite format
try:
    tflite_model = converter.convert()
    # Save the TensorFlow Lite model
    model_name = os.path.basename(model_path)
    tflite_model_path = out_root + "/" + model_name + "_INT8.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {tflite_model_path}")
except Exception as e:
    print("Error during conversion:", str(e))
