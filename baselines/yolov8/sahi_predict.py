from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import json
import os

yolov8_model_path = "/scratch1/rsawahn/results_delab/results/yolo/20240406_083837/sw/weights/best.pt"
coco_ann_file = "/scratch1/rsawahn/data/sw/coco/annotations/instances_val.json"
coco_data_dir = "/scratch1/rsawahn/data/sw/coco/val/"

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cuda:1",
)

with open(coco_ann_file, "r") as f:
    coco_ann = json.load(f)

coco_res = []
for idx, img in enumerate(coco_ann["images"]):  
    if len([ann for ann in coco_ann["annotations"] if ann["image_id"] == img["id"]]) == 0:
        continue
    img_path = os.path.join(coco_data_dir, img["file_name"])
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        postprocess_type="NMS",
        postprocess_class_agnostic=True,
        postprocess_match_threshold=0.99,
    )
    #result = get_prediction(image=img_path, detection_model=detection_model)
    coco_res.extend(result.to_coco_predictions(image_id=img["id"]))
    print(idx)
    
with open("coco_res_sw_sliced_nms_yolo.json", "w") as f:
    json.dump(coco_res, f)
