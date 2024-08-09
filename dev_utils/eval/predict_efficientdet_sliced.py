import os
import json
from pathlib import Path
import time
from numpy import ndarray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.models.base import DetectionModel

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import yaml

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from typing import List, Optional

import numpy as np


# !! Sahi file "predict.py" needs to be replaced by "sahi_ed_predict.py" in the sahi package

img_sz = 512
out_base = "/data/results_coco/slicedTrain_fullEval"
treshold = 0.05

configs = [
    # {
    #     "name": "efficientdet_sds-sliced-new_new_fullImgEval",
    #     "img_root": "/data/sds/coco/val",
    #     "coco_json": "/data/sds/coco/annotations/instances_val.json",
    #     "sahi": {
    #         "model_path": "/data/results/efficientdet-lite0_sds_sliced_new/sds_sliced_new/saved_model",
    #         "config_path": "/data/results/efficientdet-lite0_sds_sliced_new/sds_sliced_new/config.yaml",
    #     }
    # },
     {
        "name": "efficientdet_sw-sliced-new_new_fullImgEval",
        "img_root": "/data/sw/coco/val",
        "coco_json": "/data/sw/coco/annotations/instances_val.json",
        "sahi": {
            "model_path": "/data/results/efficientdet_lite0_nobn/sw_sliced/saved_model",
            "config_path": "/data/results/efficientdet_lite0_nobn/sw_sliced/config.yaml",
        }
    },
]
        
# This code is used to collect the results from model trained with slicing
def predict_all(configs):
    for idx,config in enumerate(configs):
        coco = None
        img_root = config["img_root"]
        with open(config["coco_json"], "r") as f:
            coco = json.load(f)
        out_path = os.path.join(out_base, config["name"])
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        sliced_prediction(coco, img_root, config["sahi"]["model_path"], config["sahi"]["config_path"], full_image=False, overlap=0.25, out_path=out_path, out_name="sahi_sf_overlap_full.json")

def save_json(res_path, preds):
    with open(res_path, "w") as f:
        json.dump(preds, f)

# class EfficientDetModel(DetectionModel):
#     def load_model(self):
#         try:
#             with open(self.config_path) as f:
#                 config = yaml.unsafe_load(f)
                
#             #self.img_sz =  params['image_size'], Rather use actual input in call
#             self.mean_rgb = np.array(config['mean_rgb']).flatten()
#             self.stddev_rgb = np.array(config['stddev_rgb'])

#             # Load the latest checkpoint
#             print("Loading model from", self.model_path)
#             #latest_checkpoint = tf.train.latest_checkpoint(self.model_path)

#             # Load the model using the checkpoint
#             self.sess = tf.Session()
#             tf.saved_model.load(self.sess, ['serve'], self.model_path)
#             # model = tf.saved_model.load(self.model_path)
#             # #model.to(self.device)
#             # self.model = model
#         except Exception as e:
#             raise TypeError("an error loading the model occured", e)
        
#     # def preprocess_image(self, image: ndarray, img_sz):
#     #     # ensure image size is tuple, if not convert
#     #     if not isinstance(img_sz, tuple):
#     #         img_sz = (img_sz, img_sz)
        
#     #     input_processor = dataloader.DetectionInputProcessor(image, img_sz)
#     #     input_processor.normalize_image(self.mean_rgb, self.stddev_rgb)
#     #     input_processor.set_scale_factors_to_output_size()
#     #     image = input_processor.resize_and_crop_image()
#     #     image_scale = input_processor.image_scale_to_original
#     #     return image, image_scale
    
#     def preprocess_image(self, image: ndarray, img_sz):
#         # ensure image size is tuple, if not convert
#         if not isinstance(img_sz, tuple):
#             img_sz = (img_sz, img_sz)
            
#         image = image.astype(np.float32)

#         # Normalize the image
#         image -= self.mean_rgb
#         image /= self.stddev_rgb
#         return image
        
        
        
#     def perform_inference(self, image: ndarray):
#         img_sz = self.image_size
#         confidence_treshold = self.confidence_threshold
        
#         # ToDo use treshold
    
#         # ensure channels last
#         if image.shape[-1] != 3:
#             print(image.shape)
#             raise ValueError("Image must have 3 channels and channels last format")
            
#         #cv2.imwrite("test.jpg", image)
#         #image = self.preprocess_image(image, img_sz)
        
#         # image is np array, save it to disk
#         # cv2.imwrite("test.jpg", image)
#         # exit()
        
#         #detections = self.model.signatures['serving_default'](tf.constant(image))
        
#         # save image to disk
#         #cv2.imwrite("temp.jpg", image)
        
#         detections = self.sess.run('detections:0', {'image_arrays:0': [image]})
#         # for predictions in detections:
#         #     image_with_bboxes = image.copy()
#         #     has_pred = False
#         #     for prediction in predictions:
#         #         if float(prediction[5]) > 0.3:
#         #             print("has pred")
#         #             has_pred = True
#         #             x1, y1, x2, y2 = prediction[1:5]
#         #             cv2.rectangle(image_with_bboxes, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
            
#         #     if has_pred:    
#         #         t = time.time()
#         #         filename = f"{t}.jpg"
#         #         # Construct the full path for the output image
#         #         output_path = os.path.join("/home/thesis/tiny-od-on-edge/eval/preds_vis", filename)
#         #         # Save the image with bounding boxes
#         #         print("saving")
#         #         cv2.imwrite(output_path, image_with_bboxes)
        
#         # delete image
#         #os.remove("temp.jpg")

#         #prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLOv8 expects numpy arrays to have BGR

#         # We do not filter results again as confidence threshold is already applied above
#         prediction_result = detections

#         self._original_predictions = prediction_result
    
#     def _create_object_prediction_list_from_original_predictions(
#         self,
#         shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
#         full_shape_list: Optional[List[List[int]]] = None,
#     ):
#         original_predictions = self._original_predictions

#         # compatilibty for sahi v0.8.15
#         shift_amount_list = fix_shift_amount_list(shift_amount_list)
#         full_shape_list = fix_full_shape_list(full_shape_list)

#         # print("a")
#         # print(original_predictions)
#         # print("-------------")
#         # handle all predictions
#         object_prediction_list_per_image = []
#         for image_ind, predictions in enumerate(original_predictions):
#             shift_amount = shift_amount_list[image_ind]
            
#             full_shape = None if full_shape_list is None else full_shape_list[image_ind]
#             object_prediction_list = []
#             # process predictions
           
#             for prediction in predictions:
#                 x1 = prediction[1]
#                 y1 = prediction[2]
#                 x2 = prediction[3]
#                 y2 = prediction[4]
#                 bbox = [x1, y1, x2, y2]
                
#                 # if float(prediction[5]) > 0.3:
#                 #         x1, y1, x2, y2 = bbox
#                 #         cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                
#                 # bbox = [
#                 #     bbox[0] * self.image_size,  # x1
#                 #     bbox[1] * self.image_size, # y1
#                 #     bbox[2] * self.image_size,  # x2
#                 #     bbox[3] * self.image_size  # y2
#                 # ]
#                 score = float(prediction[5])
#                 category_id = int(prediction[6])
#                 #category_name = self.category_mapping[str(category_id)]
#                 category_name = str(category_id)

#                 # fix negative box coords
#                 bbox[0] = max(0, bbox[0])
#                 bbox[1] = max(0, bbox[1])
#                 bbox[2] = max(0, bbox[2])
#                 bbox[3] = max(0, bbox[3])

#                 # fix out of image box coords
#                 if full_shape is not None:
#                     bbox[0] = min(full_shape[1], bbox[0])
#                     bbox[1] = min(full_shape[0], bbox[1])
#                     bbox[2] = min(full_shape[1], bbox[2])
#                     bbox[3] = min(full_shape[0], bbox[3])
                    
#                 if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
#                     print(f"ignoring invalid prediction with bbox: {bbox}")
#                     continue


#                 object_prediction = ObjectPrediction(
#                     bbox=bbox,
#                     category_id=category_id,
#                     score=score,
#                     bool_mask=None,
#                     category_name=category_name,
#                     shift_amount=shift_amount,
#                     full_shape=full_shape,
#                 )
#                 object_prediction_list.append(object_prediction)
#             object_prediction_list_per_image.append(object_prediction_list)

#         self._object_prediction_list_per_image = object_prediction_list_per_image
    

def sliced_prediction(coco, img_root, model_path, config_path, full_image, overlap, out_path, out_name):
    print(f"Processing {out_name}")
    # detection_model = EfficientDetModel(
    #     model_path=model_path,
    #     config_path=config_path,
    #     confidence_threshold=treshold,
    #     image_size=img_sz,
    #     device="cuda:0",
    # )

    preds = []

    for img in tqdm(coco["images"]):
        img_path = os.path.join(img_root, img["file_name"])
        
        result = get_sliced_prediction(
            img_path,
            None,
            slice_height = img_sz,
            slice_width = img_sz,   
            overlap_height_ratio = overlap,
            overlap_width_ratio = overlap,
            postprocess_type = "NMS",
            postprocess_match_metric = "IOU",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
            perform_standard_pred=full_image
        )
        
        #results = 
        # original_preds = detection_model._original_predictions
        # for preds in original_preds:
        #     for pred in preds:
        #         print(pred)
        #         if pred[5] > 0.3:
        #             print("orig ", pred[1:5])
        
        res_coco = result.to_coco_annotations()
        # for res in res_coco:
        #     if res["bbox"][2] < res["bbox"][0] or res["bbox"][3] < res["bbox"][1]:
        #         res["bbox"][2] = res["bbox"][0] + 
        #print(res_coco)
        if not res_coco:
            continue
        for r in res_coco:
            r["image_id"] = img["id"]
        # for res in res_coco:
        #     if res["score"] > 0.3:
        #         print(res)
        preds.extend(res_coco)
        
        # draw anns on image and save
        # fn = img["file_name"]
        # img = cv2.imread(os.path.join(img_root, img["file_name"]))
        
        # for ann in res_coco:
        #     if ann["score"] > 0.3:
        #         x, y, w, h = [int(v) for v in ann["bbox"]]
        #         img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # cv2.imwrite(f"/home/thesis/tiny-od-on-edge/eval/preds_vis/{Path(fn).stem}.jpg", img)
        
    save_json(os.path.join(out_path, out_name), preds)


predict_all(configs)
