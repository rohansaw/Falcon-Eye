from mmdet.apis import init_detector, DetInferencer
import time
import numpy as np
# Paths to your config and checkpoint
config_file = '/models/results_delab/results/detecto_rs_cascade_rcnn/sds/detecto_rs_cascade_rcnn_sds.py'
checkpoint_file = '/models/results_delab/results/detecto_rs_cascade_rcnn/sds/best_coco_bbox_mAP_epoch_9.pth'
mode = "sliced" if "sliced" in config_file else "full_img"

# Initialize the model using init_detector for simplicity
inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda:0', )

import torch

def create_batch(batch_size, channels, height, width):
    # Generate batch data as a list of tensors
    batch_data = [np.random.random((height, width, channels)) for _ in range(batch_size)]
    return batch_data

if mode == "sliced":
    batch_size = 28  # Define the batch size => 28 equals the tiled per benchmark image size
    channels = 3
    height = 512
    width = 512
else:
    batch_size = 1
    channels = 3
    height = 1512
    width = 2688

# Create batch data
times = []
for i in range(0, 110):
    batch_data = create_batch(batch_size, channels, height, width)
    # Run inference
    t = time.time()
    results = inferencer(batch_data, out_dir="out", no_save_vis=True, no_save_pred=True, batch_size=batch_size)
    torch.cuda.synchronize()
    if i >= 10:
        times.append(time.time() - t)
    
# We actually didnt measure the times here, but modified the inferencer code to return the time taken for inference without preprocessing
print(f"Mean inference time: {sum(times) / len(times)} seconds")