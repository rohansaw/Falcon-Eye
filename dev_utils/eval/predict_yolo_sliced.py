import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict

img_sz = 512
out_base = "/data/results_coco/slicedTrain_fullEval"
treshold = 0.05

configs = [
    # {
    #     "name": "yolov8_sw_05pms",
    #     "img_root": "/data/data/sw/coco/val",
    #     "coco_json": "/data/data/sw/coco/annotations/instances_val.json",
    #     "no_sahi": {
    #         "model_path": "/data/results_delab/results/yolo/20240406_083837/sw/weights/best.pt",
    #     },
    #     "sahi": {
    #         "model_path": "/data/results_delab/results/yolo/20240408_170433/sw_sliced/weights/best.pt",
    #     }
    # },
    {
        "name": "yolov8_sds_noclass_ag2",
        "img_root": "/data/sds/coco/val",
        "coco_json": "/data/sds/coco/annotations/instances_val.json",
        "sahi": {
            "model_path": "/yolov8/artifacts/run_oszsb61j_model:v0/best.pt",
        }
    }
]


# This Code is used for the SAHI Framework Experiments
def predict_all_paralell(configs):
   with ThreadPoolExecutor() as executor:
        for config in config:
            coco = None
            img_root = config["img_root"]
            with open(config["coco_json"], "r") as f:
                coco = json.load(f)
            out_path = os.path.join(out_base, config["name"])
            futures = []
            print("Processing config: ", config["name"])
            print("Running no sahi")
            futures.append(executor.submit(single_prediction, coco, img_root, config["no_sahi"]["model_path"], config["no_sahi"]["config_path"], out_path, "no_sahi.json"))
            print("Running sahi")
            futures.append(executor.submit(sliced_prediction, coco, img_root, config["no_sahi"]["model_path"], config["no_sahi"]["config_path"], full_image=False, overlap=None, out_path=out_path, out_name="sahi.json"))
            print("Running sahi + SF")
            futures.append(executor.submit(sliced_prediction, coco, img_root, config["sahi"]["model_path"], config["sahi"]["config_path"], full_image=False, overlap=0, out_path=out_path, out_name="sahi_sf.json"))
            print("Running sahi + SF + overlap")
            futures.append(executor.submit(sliced_prediction, coco, img_root, config["sahi"]["model_path"], config["sahi"]["config_path"], full_image=False, overlap=0.25, out_path=out_path, out_name="sahi_sf_overlap.json"))
            print("Running sahi + SF + overlap full image")
            futures.append(executor.submit(sliced_prediction, coco, img_root, config["sahi"]["model_path"], config["sahi"]["config_path"], full_image=True, overlap=0.25, out_path=out_path, out_name="sahi_sf_overlap_full.json"))
            
            for future in as_completed(futures):
                future.result() 

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
        
        sliced_prediction(coco, img_root, config["sahi"]["model_path"], full_image=True, overlap=0.25, out_path=out_path, out_name="sahi_sf_overlap_full.json")

def save_json(res_path, preds):
    with open(res_path, "w") as f:
        print("saving to ", res_path)
        json.dump(preds, f)


def single_prediction(coco, img_root, model_path, out_path, out_name):
    print(f"Processing {out_name}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=treshold,
        device="cuda:0", # or 'cuda:0'
    )

    preds = []
    for img in tqdm(coco["images"]):
        img_path = os.path.join(img_root, img["file_name"])
        
        result = get_prediction(
            img_path,
            detection_model,
        )
        
        res_coco = result.to_coco_annotations()
        if not res_coco:
            continue
        for r in res_coco:
            r["image_id"] = img["id"]
        preds.extend(res_coco)
    
    save_json(os.path.join(out_path, out_name), preds)

def sliced_prediction(coco, img_root, model_path, full_image, overlap, out_path, out_name):
    print(f"Processing {out_name}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=treshold,
        image_size=img_sz,
        device="cuda:0", # or 'cuda:0'
    )

    preds = []

    for img in tqdm(coco["images"]):
        img_path = os.path.join(img_root, img["file_name"])
        
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height = img_sz,
            slice_width = img_sz,
            overlap_height_ratio = overlap,
            overlap_width_ratio = overlap,
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False, # True if not SDS
            postprocess_type = "NMS",
            postprocess_match_metric = "IOU",
            perform_standard_pred=full_image
        )
        
        res_coco = result.to_coco_annotations()
        if not res_coco:
            continue
        for r in res_coco:
            r["image_id"] = img["id"]
        preds.extend(res_coco)
        save_json(os.path.join(out_path, out_name), preds)
        
    save_json(os.path.join(out_path, out_name), preds)
    
predict_all(configs)
