import os
import time
import numpy as np
from PIL import Image

from memory_profiler import profile
from mmdet.apis import DetInferencer, init_detector, inference_detector

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image_size(config):
    # Load the first transform in the test pipeline that changes image size
    # This assumes that the configuration defines the input size in a Resize or similar transform.
    if 'img_scale' in config.data.test.pipeline[1]:  # Assuming 'Resize' is usually the second step
        img_scale = config.data.test.pipeline[1]['img_scale']
        return (img_scale[1], img_scale[0])  # Return size as (width, height)
    else:
        raise ValueError("No img_scale found in test pipeline of the config")

def preprocess_image(image_path, size):
    image = Image.open(image_path)
    start_time = time.time()
    image = image.resize(size)  # Resize according to the extracted config size
    t1 = time.time() - start_time
    print("Resized in", t1, "seconds")
    image = image.convert('RGB')  # Ensure image is in the correct format
    image = np.array(image, dtype=np.float32)
    start_time = time.time()
    image /= 255.0
    image = (image - mean) / std
    t = time.time() - start_time
    print("Normalized in", t, "seconds")
    return np.array(image), t+t1

if __name__ == "__main__":
    times = []
    preproc_times = []
    iterations = 1
    device = "cuda:0"
    img_root = "/data/sds/coco/val"
    config_file = '/models/ssd_lite/sds/ssd_lite_sds.py'
    checkpoint_file = '/models/ssd_lite/sds/best_coco_bbox_mAP_epoch_93.pth'

    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device=device)
    model = init_detector(config_file, checkpoint_file, device=device)
    img_size = (2688, 1536)


    @profile
    def run_inference(type):
        for _ in range(iterations):
            for idx, img in enumerate(os.listdir(img_root)[:110]):
                img_path = os.path.join(img_root, img)
                #print("Starting inference")
                #print("Processing file:", img_path)
                try:
                    #img = preprocess_image(img_path, 2272, 1536)
                    if type == "e2e" and not "sliced" in config_file:
                        start_time = time.time()
                        result = inferencer(img_path)
                        t = time.time() - start_time
                    else:
                        image, preproc_time = preprocess_image(img_path, img_size)
                        if idx > 10:
                            preproc_times.append(preproc_time)
                        
                        start_time = time.time()
                        result = inference_detector(model, image)
                        t = time.time() - start_time
                   
                    if idx > 10:
                        times.append(t)
                    print("Inference completed in", t, "seconds")
                except Exception as e:
                    print(f"Failed to process {img_path} due to {str(e)}")

    run_inference("pred")
    print("Average time: ", sum(times) / len(times))
    print("Average postproc time: ", sum(preproc_times) / len(preproc_times))
    print("Average e2e-pred", sum(times) / len(times) + sum(preproc_times) / len(preproc_times))
    times = []
    run_inference("e2e")
    print("Average time: ", sum(times) / len(times))
