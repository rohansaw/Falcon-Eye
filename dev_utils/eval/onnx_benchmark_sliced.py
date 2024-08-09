import os
import time
import numpy as np
import onnxruntime
from PIL import Image
import onnx
from onnx import numpy_helper

import onnxruntime as ort
print(ort.get_device())

def preprocess_image(image_path, width, height):
    image = Image.open(image_path)
    image = image.resize((width, height))
    image_data = np.array(image)
    #.astype(np.float32) / 255.0  # Normalize the image
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # image_data = (image_data - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    image_data = np.transpose(image_data, (2, 0, 1))  # Reorder dimensions to CxHxW
    return image_data

def print_model_info(model):
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.HasField('dim_value')) else 'None' for dim in input.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)
        print(f"Type: {elem_type}, Shape: {shape}")

    print("\nModel Outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.HasField('dim_value')) else 'None' for dim in output.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)
        print(f"Type: {elem_type}, Shape: {shape}")

def count_parameters(model):
    total_params = 0
    for initializer in model.graph.initializer:
        param = numpy_helper.to_array(initializer)
        total_params += param.size
    return total_params

def slice(img, stride, tile_sz):
    # img is numpy array in BxCxHxW
    height, width = img.shape[2], img.shape[3]
    print(width, height, stride, tile_sz)
    tiles = []
    
    for x in range(0, width, stride):
        for y in range(0, height, stride):
            if x + tile_sz > width:
                x = width - tile_sz
            
            if y + tile_sz > height:
                y = height - tile_sz
                
            tile = img[:, :, y:y+tile_sz, x:x+tile_sz]
            
            tiles.append(tile)
    
    return tiles


times = []
times_no_slice = []
iterations = 1

if __name__ == "__main__":
    model_path = "/models/results/all_best/model_deploy/yolo/sds_sliced.onnx"
    img_root = "/data/sds/coco/val"
    
    model = onnx.load(model_path)
    print_model_info(model)
    num_params = count_parameters(model)
    print(f"Total Number of Parameters: {num_params}")
    
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    
    key = "image_arrays_0" if "efficientdet" in model_path else "input"

    def run_inference():
        idx = 0
        for _ in range(iterations):
            img_list = os.listdir(img_root)[:110]
            for img_path in img_list:
                idx += 1
                
                tile = np.random.rand(63, 3, 512, 512).astype(np.float32)
                
                start_time = time.time()
                outputs = session.run([], {"images": tile})
                t = time.time() - start_time
                
                if idx > 1:
                    times.append(t)
                
                # full_img_path = os.path.join(img_root, img_path)
                # print("Starting inference on:", full_img_path)
                # preprocessed_image = preprocess_image(full_img_path, 1536, 2688)
                # preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                # print(preprocessed_image.shape)
                
                # start_time = time.time()
                # tiles = slice(preprocessed_image, 384, 512)
                # print(len(tiles))
                # slicing_dur = time.time() - start_time
                
                # for tile in tiles:
                #     tile = np.transpose(tile, (0, 2, 3, 1))
                #     try:
                #         outputs = session.run([], {"image_arrays_0": tile})
                #     except Exception as e:
                #         print(f"Failed to process image due to {str(e)}")
                # if idx > 1:
                #     t = time.time() - start_time
                #     print("Slicing completed in", slicing_dur, "seconds")
                #     print("Inference completed in", t, "seconds")
                #     times.append(t)

    run_inference()
    print("Average time: ", sum(times) / len(times)) 
