import os
from PIL import Image
import numpy as np

def compute_mean_std(folder_path):
    sum_ = np.array([0.0, 0.0, 0.0])
    sum_squared = np.array([0.0, 0.0, 0.0])
    n = 0

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_arr = np.array(img, dtype=np.float32) / 255.0

                sum_ += img_arr.sum(axis=(0, 1))
                sum_squared += np.sum(np.square(img_arr), axis=(0, 1))
                n += img_arr.shape[0] * img_arr.shape[1]
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    mean = sum_ / n
    std = np.sqrt((sum_squared / n) - np.square(mean))

    return mean, std

folder_path = '/scratch1/rsawahn/data/sw_sliced_centered/coco/train'
mean, std = compute_mean_std(folder_path)
print(f"Mean: {mean}, Std: {std}")