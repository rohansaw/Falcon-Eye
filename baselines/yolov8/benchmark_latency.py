import torch
import numpy as np
import time

from ultralytics import YOLO

# Load a model
model_path = "/models/results_delab/results/yolo/20240406_083837/sw/weights/best.pt"
mode = "sliced" if "sliced" in model_path else "full_img"

model = YOLO(model_path)  # pretrained YOLOv8n model

# # Check if CUDA is available and use it if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# # Create dummy input
if mode == "sliced":
    batch_size = 28  # Define the batch size => 28 equals the tiled per benchmark image size
    channels = 3
    height = 512
    width = 512
else:
    batch_size = 1
    channels = 3
    height = 1504 # stride compatibility
    width = 2688
    
input_size = (batch_size, channels, height, width)
inputs = torch.randn(input_size, device=device)
model(inputs)

# # Function to measure inference time
def measure_inference_time(model, inputs, num_runs=110):
    durations = []
    with torch.no_grad():
        for idx in range(num_runs):
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()  # Wait for CUDA to finish (only if CUDA is used)
            end_time = time.time()
            if idx >= 10:
                durations.append(end_time - start_time)
    
    return np.mean(durations), np.std(durations)

# Measure the inference time
mean_latency, std_deviation = measure_inference_time(model, inputs)
print(f'Mean Inference Latency: {mean_latency:.6f} seconds')
print(f'Standard Deviation of Latency: {std_deviation:.6f} seconds')
