import tensorflow as tf
import os
from PIL import Image
import numpy as np
import json
from torchvision.transforms import v2

# Define the paths
model_path = "/scratch1/rsawahn/data/results/quantization/saved_model_mobOne"

coco_json_path = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val.json"
base_path = "/scratch1/rsawahn/data/sw_sliced/coco/val"
files = []
with open(coco_json_path, 'r') as f:
    coco = json.load(f)
    
    img_with_anns = []
    img_without_anns = []
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_with_anns.append(img_id)
    
    imgs_with_anns = list(set(img_with_anns))
    imgs_without_anns = [img["id"] for img in coco['images'] if img["id"] not in imgs_with_anns]
    
    # randomly sample 100 images with annotations
    np.random.seed(42)
    sampled_imgs_with_anns = np.random.choice(imgs_with_anns, 100, replace=False)
    sampled_imgs_without_anns = np.random.choice(imgs_without_anns, 200, replace=False)
    files.extend([img["file_name"] for img in coco['images'] if img["id"] in sampled_imgs_with_anns])
    files.extend([img["file_name"] for img in coco['images'] if img["id"] in sampled_imgs_without_anns])
    files_new = [os.path.join(base_path, file) for file in files]
    files = files_new

def preprocess_image(image_path, height, width):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    transform = v2.Compose([
        v2.Resize((512, 512)), 
        v2.ToTensor(),
        v2.Normalize(mean=[0.2761, 0.4251, 0.5644], std=[0.2060, 0.1864, 0.2218])  # Normalize
    ])
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    input_data_original = tf.convert_to_tensor(tensor.numpy(), dtype=tf.float32)
    input_data_original = tf.transpose(input_data_original, [0, 2, 3, 1])
    
    # image = image.resize((width, height))
    # image_data = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    # print(image_data.shape, "IM")
    # mean = np.array([0.2761, 0.4251, 0.5644], dtype=np.float32)
    # std = np.array([0.2060, 0.1864, 0.2218], dtype=np.float32)
    # image_data = (image_data - mean) / std  # Normalize
    # image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    # print(image_data.shape)
    # change shape to NHWC
    #image_data = np.transpose(image_data, (0, 1, 2))
    return input_data_original

def representative_data_gen():
    # Load first 100 images from the representative_dataset_path, preprocess, normalize by mean and std
    # shuffle files
    np.random.shuffle(files) 
    for f in files:
        #img_path = os.path.join(representative_dataset_path, f)
        img_path = f
        img = preprocess_image(img_path, 512, 512)
        print(img.shape, "IMG")
        yield [img]  # Yield the image as a list

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# debugger = tf.lite.experimental.QuantizationDebugger(
#     converter=converter, debug_dataset=representative_dataset(ds))

# Convert the model to TensorFlow Lite format
try:
    tflite_model = converter.convert()
    # Save the TensorFlow Lite model
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    tflite_model_path = os.path.join(model_dir, model_name + "_100bg100fg_INT8.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {tflite_model_path}")
except Exception as e:
    print("Error during conversion:", str(e))
