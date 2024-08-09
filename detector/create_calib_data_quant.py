import cv2
import glob
import numpy as np
import json
import os
from PIL import Image
from torchvision.transforms import v2

# Not used during data generation ################################
# You will need to do the calculations yourself using the test data
MEAN = np.asarray([[[[0.2761, 0.4251, 0.5644]]]], dtype=np.float32) # [1,1,1,3]
STD = np.asarray([[[[0.2060, 0.1864, 0.2218]]]], dtype=np.float32) # [1,1,1,3]
# Not used during data generation ################################


coco_json_path = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val.json"
base_path = "/scratch1/rsawahn/data/sw_sliced/coco/val"
files = []
with open(coco_json_path, 'r') as f:
    coco = json.load(f)
    # sample 900 images with and 200 without annotations
    
    img_with_anns = []
    img_without_anns = []
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_with_anns.append(img_id)
    
    imgs_with_anns = list(set(img_with_anns))
    imgs_without_anns = [img["id"] for img in coco['images'] if img["id"] not in imgs_with_anns]
    
    # randomly sample 900 images with annotations
    np.random.seed(42)
    sampled_imgs_with_anns = np.random.choice(imgs_with_anns, 900, replace=False)
    sampled_imgs_without_anns = np.random.choice(imgs_without_anns, 200, replace=False)
    files.extend([img["file_name"] for img in coco['images'] if img["id"] in sampled_imgs_with_anns])
    files.extend([img["file_name"] for img in coco['images'] if img["id"] in sampled_imgs_without_anns])
    files_new = [os.path.join(base_path, file) for file in files]
    files = files_new

img_datas = []
for idx, file in enumerate(files):
    # Possibly use PIL
    rgb_img = Image.open(file).convert("RGB")
    #print("---------")
    #print(rgb_img.size)
    resized_img = v2.ToTensor()(rgb_img)
    #print(resized_img.shape)
    # bgr_img = cv2.imread(file)
    # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # resized_img = cv2.resize(rgb_img, dsize=(512,512))
    # pil image to numpy
    resized_img = resized_img.numpy()
    #print(resized_img.shape)
    extend_batch_size_img = resized_img[np.newaxis, :]
    #print(extend_batch_size_img.shape)
    extend_batch_size_img = extend_batch_size_img.transpose(0, 2, 3, 1)
    #normalized_img = extend_batch_size_img / 255.0 # 0.0 - 1.0
    print(
        f'{str(idx+1).zfill(2)}. extend_batch_size_img.shape: {extend_batch_size_img.shape}'
    ) # [1,112,200,3]
    img_datas.append(extend_batch_size_img)
calib_datas = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_datas.shape}') # [10,112,200,3]
np.save(file='/scratch1/rsawahn/data/results/calibdata.npy', arr=calib_datas)

loaded_data = np.load('/scratch1/rsawahn/data/results/calibdata.npy')
print(f'loaded_data.shape: {loaded_data.shape}') # [10,112,200,3]

"""
-cind INPUT_NAME NUMPY_FILE_PATH MEAN STD
int8_calib_datas = (loaded_data - MEAN) / STD # -1.0 - 1.0

e.g. How to specify calibration data in CLI or Script respectively.
1. CLI
  -cind "pc_dep" "data/calibdata.npy" "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"
  -cind "feat" "data/calibdata2.npy" "[[[[0.123,...,0.321]]]]" "[[[[0.112,...,0.451]]]]"

2. Script
  custom_input_op_name_np_data_path=[
      ["pc_dep", "data/calibdata.npy", [[[[0.485,0.456,0.406]]]], [[[[0.229,0.224,0.225]]]]],
      ["feat", "data/calibdata2.npy", [[[[0.123,...,0.321]]]], [[[[0.112,...,0.451]]]],
  ]
"""