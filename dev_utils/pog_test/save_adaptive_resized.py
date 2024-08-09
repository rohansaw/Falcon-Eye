import os
import cv2
import torch
from resize import CocoDataset, AdaptiveResizer

def process_and_save_images(coco_root_dir, output_dir, ann_file, img_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = CocoDataset(root_dir=coco_root_dir, transform=None, ann_file=ann_file, img_dir=img_dir)
    
    class Opt:
        resize_mode = 'linear'
        resize_parameters = None
    opt = Opt()
    resizer = AdaptiveResizer(use_adaptive=True, opt=opt)
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(sample["img"])
        resized_sample = resizer(sample)
        
        img_data = resized_sample['img'][0].numpy()
        img_data = (img_data * 255).astype('uint8')
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        image_info = dataset.coco.loadImgs(dataset.image_ids[idx])[0]
        output_path = os.path.join(output_dir, image_info['file_name'])
        
        cv2.imwrite(output_path, img_data)
        print(f'Saved: {output_path}')

coco_root_dir = '/scratch1/rsawahn/PeopleOnGrass'
output_dir = '/scratch1/rsawahn/PeopleOnGrass/train_normalized'
process_and_save_images(coco_root_dir, output_dir, ann_file='instances_train_bev.json', img_dir='train')
