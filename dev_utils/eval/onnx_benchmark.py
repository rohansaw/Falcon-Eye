import os
import time
import numpy as np
from PIL import Image
import onnx
from onnx import numpy_helper
import torch

#from memory_profiler import profile

import onnxruntime as ort
#print(ort.get_device())

def preprocess_image(image_path, height, width):
    image = Image.open(image_path)
    image = image.resize((width, height))
    image_data = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_data = (image_data - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    image_data = np.transpose(image_data, (2, 0, 1))  # Reorder dimensions to CxHxW
    return image_data

def get_model_info(model):
    # Print information about the inputs
    input_name = model.graph.input[0].name
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        # The shape can sometimes include None or symbolic names if the model supports variable batch sizes
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.HasField('dim_value')) else 'None' for dim in input.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)
        print(f"Type: {elem_type}, Shape: {shape}")

    # Print information about the outputs
    print("\nModel Outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        shape = [dim.dim_value if (dim.dim_value > 0 and dim.HasField('dim_value')) else 'None' for dim in output.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)
        print(f"Type: {elem_type}, Shape: {shape}")
        
    return input_name, input_shape

def count_parameters(model):
    total_params = 0
    # Iterate over all initializers (parameters of the model)
    for initializer in model.graph.initializer:
        # Convert initializer to numpy array
        param = numpy_helper.to_array(initializer)
        total_params += param.size
    return total_params


times = []
iterations = 1

if __name__ == "__main__":
    model_path = "/models/results/all_best/model_deploy/yolo/sw_full.onnx"
    img_root = "/data/sds/coco/val"
    device = "cuda:0"
    
    model = onnx.load(model_path)
    model_info = get_model_info(model)
    input_name, input_shape = model_info
    img_sz = (input_shape[2].dim_value, input_shape[3].dim_value)
    print(img_sz)
    
    num_params = count_parameters(model)
    print(f"Total Number of Parameters: {num_params}")
    
    provider = "CUDAExecutionProvider" if device == "cuda:0" else "CPUExecutionProvider"
    session = ort.InferenceSession(model_path, providers=[provider])
    

    #@profile
    def run_inference():
        idx = 0
        for _ in range(iterations):
            img_list = os.listdir(img_root)[:100]
            for img_path in img_list:
                idx += 1
                full_img_path = os.path.join(img_root, img_path)
                print("Starting inference on:", full_img_path)
                preprocessed_image = preprocess_image(full_img_path, img_sz[0], img_sz[1])
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

                # Move data to GPU explicitly if needed (uncomment the line below if manual control is necessary)
                # preprocessed_image = ort.OrtValue.ortvalue_from_numpy(preprocessed_image, 'cuda', 0)
                
                # Measure only the inference time
                try:
                    # Note: 'input' should be changed to match the actual input name in the ONNX model
                    torch.cuda.synchronize(device)
                    start_time = time.time()
                    outputs = session.run([], {input_name: preprocessed_image})
                    print(len(outputs))
                    torch.cuda.synchronize(device)
                    t = time.time() - start_time
                    if idx > 10:
                        times.append(t)
                    print("Inference completed in", t, "seconds")
                except Exception as e:
                    print(f"Failed to process image due to {str(e)}")

    run_inference()
    print("Average time: ", sum(times) / len(times))  # Changed to calculate average over all images processed
