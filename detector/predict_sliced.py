# import cv2
import numpy as np
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection
from dataset.v2_wrapper import wrap_dataset_for_transforms_v2
from torchvision.utils import save_image
import torch
import random
from dataset.transforms import ImageTransformWrapper, ToGrid, collate_fn
from backends.mobileone import mobileone, reparameterize_model
from backends.mobilenet_v2 import MobileNetV2
from utils.visualization import draw_prediction, unnormalize
from heads.fcn_head import FCNHead
from model import Detector
from utils.metrics import Metrics

from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
from lightning.pytorch.utilities.model_summary import ModelSummary
import matplotlib.pyplot as plt
from torch.nn import MultiheadAttention
from attention.sim_am import SimAM
import json
import wandb


from dataset.gsd_coco_detection import GSDCOCODetection

seed = 2203
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# Usage Example for ONNX, TF, and PyTorch-uncompiled
#model_path = "/data/results/quantization/saved_model/mobOne_notReparamterized_full_integer_quant.tflite"
# model_path = "/models/exported_models/mydet_sw_mobOne.onnx"
model_path = "searchwing/my-detector/model-btqk55mr:v32"
# model_path = "searchwing/my-detector/model-5nunrux7:v20"

run_tf = False
is_quantized = False
run_onnx = False
benchmark_fps_pi = False
gsd = None


data_dir = "/data/sw/coco/val"
ann_file = "/data/sw/coco/annotations/instances_val.json"

# data_dir = "/data/PeopleOnGrass/val"
# ann_file = "/data/PeopleOnGrass/annotations/instances_val_bev.json"

# data_dir = "/data/sds/coco/val"
# ann_file = "/data/sds/coco/annotations/instances_val.json"

input_shape = (3, 1512, 2688)
detetor_input_shape = (3, 512, 512)
classes = [("boat", 0)]
#classes = [("ignored", 0), ("swimmer", 1), ("boat", 2), ("jetski", 3), ("life_saving_appliances", 4), ("buoy", 5)]

class_sizes = [10]

#class_sizes = [0, 2, 10, 3, 2, 2]

# SW
mean = [0.2761, 0.4251, 0.5644]
std = [0.2060, 0.1864, 0.2218]

# PeopleOnGrass
# mean = [0.43717304, 0.44310874, 0.33362516]
# std = [0.23407398, 0.21981522, 0.2018422 ]

# SDS
# mean = [0.4263, 0.4856, 0.4507]
# std = [0.1638, 0.1515, 0.1755]

sensor_height_mm = 2.76
focal_len_mm = 3.04

bbox_out = False
bbox_treshold = 0.1
binary_eval = False
eval_tresholds = np.linspace(0, 1, num=100, endpoint=False)


if run_tf:
    import tensorflow as tf
    artifact_dir = model_path  
    checkpoint = model_path
elif run_onnx:
    import onnxruntime
    artifact_dir = model_path
    checkpoint = model_path
else:
    run = wandb.init()
    artifact = run.use_artifact(model_path, type='model')
    artifact_dir = artifact.download()
    checkpoint = os.path.join(artifact_dir, "model.ckpt")


num_classes = len(classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def slice_image_only(image, scale_factor, tile_size, overlap_factor):
    image_tiles = []

    _, image_width, image_height = image.shape
    stride_x = int(tile_size[0] * (1 - overlap_factor))
    stride_y = int(tile_size[1] * (1 - overlap_factor))
    tile_coordinates = []


    t = time.perf_counter()
    for y in range(0, image_height, stride_y):
        for x in range(0, image_width, stride_x):
            if x + tile_size[0] > image_width:
                x = image_width - tile_size[0]
            if y + tile_size[1] > image_height:
                y = image_height - tile_size[1]
            
            image_tile = image[:, x : x + tile_size[0], y : y + tile_size[1]]
            image_tiles.append(image_tile)
            tile_coordinates.append((x, y))

    print(f"Image slicing time: {time.perf_counter() - t}")
    return image_tiles, tile_coordinates


def slice_image_and_map(image, seg_map, scale_factor, tile_size, overlap_factor):
    if isinstance(image, tuple):
        image, altitude = image
    else:
        altitude = None
    image_tiles = []
    map_tiles = []
    map_coordinates = []
    _, image_width, image_height = image.shape
    _, map_width, map_height = seg_map.shape
    stride_x = int(tile_size[0] * (1 - overlap_factor))
    stride_y = int(tile_size[1] * (1 - overlap_factor))
    tile_coordinates = []


    t = time.perf_counter()
    for y in range(0, image_height, stride_y):
        for x in range(0, image_width, stride_x):
            if x + tile_size[0] > image_width:
                x = image_width - tile_size[0]
            if y + tile_size[1] > image_height:
                y = image_height - tile_size[1]
            
            image_tile = image[:, x : x + tile_size[0], y : y + tile_size[1]]
            image_tiles.append(image_tile)
            tile_coordinates.append((x, y))
            
            map_x1, map_y1 = int(x * scale_factor), int(y * scale_factor)
            map_x2, map_y2 = int((x + tile_size[0]) * scale_factor), int((y + tile_size[1]) * scale_factor)
            map_tile = seg_map[:, map_x1:map_x2, map_y1:map_y2]
            map_tiles.append(map_tile)
            map_coordinates.append((map_x1, map_y1))
    
    print(f"Image slicing time: {time.perf_counter() - t}")
    if altitude is not None:
        altitudes = torch.tensor([altitude] * len(image_tiles)).to(device)
    else:
        altitudes = None
    return image_tiles, map_tiles, tile_coordinates, map_coordinates, altitudes

def predict(image_tiles, full_img, detector, batch_size=None, altitudes=None):
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = len(image_tiles)

    t0 = time.perf_counter()
    for i in range(0, len(image_tiles), batch_size):
        batch = torch.stack(image_tiles[i:i+batch_size]).to(device)
        t = time.perf_counter()
        outputs = detector(batch, altitudes)
        #outputs = detector(batch)
        inf_time = time.perf_counter() - t
        print(f"Prediction time: {inf_time}")
        results.extend(outputs.detach().cpu().numpy())
    print(f"Total prediction time: {time.perf_counter() - t0}")

    full_img_res = None
    if full_img is not None:
        print("Using full image")
        full_img = full_img.unsqueeze(0).to(device)
        full_img_res = detector(full_img)
        full_img_res = full_img_res[0].detach().cpu().numpy()

    return results, full_img_res, inf_time

def merge_predictions(predictions, coordinates, segmap_shape):
    merged_map = torch.zeros(segmap_shape, dtype=torch.float32)
    for prediction, (x, y) in zip(predictions, coordinates):
        for channel in range(1, segmap_shape[0]):
            pred_h = prediction.shape[1]
            pred_w = prediction.shape[2]
            old_t = merged_map[:, x : x + pred_w, y : y + pred_h]
            new_t_softmax = torch.softmax(torch.tensor(prediction), dim=0)
            merged_map[channel, x : x + pred_w, y : y + pred_h] = torch.maximum(old_t[channel, :, :], new_t_softmax[channel])

    return merged_map

def merge_full_img_pred(predictions, full_img_pred, original_shape):
    if full_img_pred is not None:
        full_img_pred = v2.Resize((original_shape[1], original_shape[2]))(torch.tensor(full_img_pred))
        for channel in range(1, full_img_pred.shape[0]):
            old_t = predictions[channel, :, :]
            new_t_softmax = torch.softmax(full_img_pred, dim=0)
            predictions[channel, :, :] = torch.maximum(old_t, new_t_softmax[channel])

    return predictions

def process_image(image, seg_map, scale_factor, tile_size, overlap_factor, detector, use_full_img=False):
    image_tiles, map_tiles, tile_coordinates, map_coordinates, altitudes = slice_image_and_map(image, seg_map, scale_factor, tile_size, overlap_factor)
    print(f"Procceing {len(image_tiles)} tiles")

    full_img = None
    if use_full_img:
        full_img = v2.Resize(tile_size)(image)
        
    predictions, full_img_pred, inf_time = predict(image_tiles, full_img, detector, altitudes=altitudes)
    merged_predictions = merge_predictions(predictions, map_coordinates, seg_map.shape)
    merged_predictions = merge_full_img_pred(merged_predictions, full_img_pred, seg_map.shape)

    return merged_predictions, inf_time


def create_detector(input_shape, num_classes, device, checkpoint):
    #backend = MobileNetV2(bn_momentum=0.9)
    backend = mobileone(variant="s0")
    backend.truncate(2)

    feat_map_shape = backend.get_feature_map_shape(input_shape)
    attention = None
    #attention = MultiheadAttention(feat_map_shape[0], 8, batch_first=True)
    #attention = SimAM()
    
    head = FCNHead(
        num_classes=num_classes, in_channels=feat_map_shape[0]+1, middle_channels=32
    )
    detector = (
        Detector.load_from_checkpoint(checkpoint, head=head, attention=attention, backend=backend)
        .type(torch.FloatTensor)
        .to(device)
    )
    
    detector.eval()
    if attention:
        detector.attention.eval()
    detector.head.eval()
    detector.backend.eval()
    detector.backend = reparameterize_model(detector.backend)

    return detector

def create_bboxes(y, altitude, img_h, img_w, scale_factor, treshold):
    gsd = (altitude * sensor_height_mm) / (focal_len_mm * img_h)
    
    indices = torch.nonzero(y > treshold)
    
    #create bboxes from indices based on class sizes and scale factor (class equals to 0 is ignored )
    bboxes = []

    for idx in indices:
        object_cls = idx[1] -1
        center_x, center_y = idx[2], idx[3]
        
        width = (class_sizes[object_cls] * gsd * 100) / 2 # / 2  only because we currently resize the full images
        height = (class_sizes[object_cls] * gsd * 100) / 2 # / 2  only because we currently resize the full images
        
        center_x = center_x / scale_factor
        center_y = center_y / scale_factor
        
        top_left_x = max(0, center_x - (width / 2))
        top_left_y = max(0, center_y - (height / 2))
        bottom_right_x = min(img_h, center_x + (width / 2))
        bottom_right_y = min(img_w, center_y + (height / 2))
        
        bbox = [object_cls, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        bboxes.append(bbox)
    
    return bboxes
    
    
    

class TFLiteWrapper():
    def __init__(self, model_path):
    
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
    
    def __call__(self, img, altitudes):
        input_data_original = tf.convert_to_tensor(img.cpu().numpy(), dtype=tf.float32)
        input_data_original = tf.transpose(input_data_original, [0, 2, 3, 1])
        results = []
        for i in range(input_data_original.shape[0]):
            input_data = input_data_original[i]
            if is_quantized:
                input_scale, input_zero_point = self.input_details['quantization']
                input_data = np.clip((input_data / input_scale + input_zero_point), -128, 127).astype(np.int8)
            input_data = tf.expand_dims(input_data, 0)
            self.interpreter.set_tensor(self.input_details['index'], input_data)
            self.interpreter.invoke()
            res = self.interpreter.get_tensor(self.output_details['index'])
            if is_quantized:
                output_scale, output_zero_point = self.output_details['quantization']
                res = output_scale * (res - output_zero_point)
            
            res = torch.tensor(res)
            res = res.permute(0, 3, 1, 2)
            res = res.squeeze(0)
            results.append(res)
        
        results = torch.stack(results)
        return results


class ONNXWrapper():
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        
    def __call__(self, img, altitudes):
        input_data_original = img.cpu().numpy()
        results = []
        for i in range(input_data_original.shape[0]):
            input_data = input_data_original[i, : , :, :]
            input_data = np.expand_dims(input_data, 0)
            res = self.model.run(None, {"input.1": input_data})
            res = torch.tensor(res)
            res = res.squeeze(0)
            res = res.squeeze(0)
            results.append(res)
        
        results = torch.stack(results)
        return results

if run_tf:
    detector = TFLiteWrapper(checkpoint)
    scale_factor = 0.125
elif run_onnx:
    detector = ONNXWrapper(checkpoint)
    scale_factor = 0.125
else:
    detector = create_detector(detetor_input_shape, num_classes, device, checkpoint)
    print(ModelSummary(detector))
    scale_factor = detector.metrics.scale_factor
    print("Scale factor: ", scale_factor)
    

class PadToSize():
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, labels):
        if img.size[0] < self.size[0] or img.size[1] < self.size[1]:
            pad_x0 = (self.size[0] - img.size[0]) // 2
            pad_y0 = (self.size[1] - img.size[1]) // 2
            pad_x1 = self.size[0] - img.size[0] - pad_x0
            pad_y1 = self.size[1] - img.size[1] - pad_y0

            img_new, labels_new =  v2.Pad((pad_x0, pad_y0, pad_x1, pad_y1))(img, labels)
            print(img_new.size)
            return img_new, labels_new
        return img, labels
    


if gsd != None:
    print("Scale factor: ", scale_factor)
    transforms = v2.Compose(
        [
            PadToSize((512, 512)),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),
            ToGrid(len(classes), scale_factor),
        ]
    )
else:
    transforms = ImageTransformWrapper(v2.Compose(
        [
            v2.Resize(input_shape[1:]),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),
            ToGrid(len(classes), scale_factor),
        ]
    ), include_metadata=True)


def collate_fn(batch):
    return tuple(zip(*batch))

if benchmark_fps_pi:
    for i in range(100):
        image = torch.rand(3, 2464, 3280)
        tiles, coords = slice_image_only(image, scale_factor, (512, 512), 0.25)
        predict(tiles, None, detector)
    
else:
    coco = GSDCOCODetection(
        root=data_dir,
        annFile=ann_file,
        transforms=transforms,
        target_gsd=gsd,
    )
    
    
    metrics = {}
    for treshold in eval_tresholds:
        metrics[treshold] = Metrics(scale_factor=scale_factor, treshold=treshold, tolerance=4, classes=[], use_softmax=False, binary=binary_eval)

    coco = wrap_dataset_for_transforms_v2(coco)
    dataloader = DataLoader(coco, batch_size=5, shuffle=False, collate_fn=collate_fn)

    fully_correct_images = 0
    inf_times = []
    for idx, batch in enumerate(tqdm(dataloader, desc="Batches")):
        for x, y in zip(batch[0], batch[1]):
            if "boxes" not in y["original_annotations"]:
                continue
            else:
                x[0].to(device)
                boxes = y["original_annotations"]["boxes"]
                prediction, inf_time = process_image(
                    x,
                    y["train_segmentation_grid"],
                    scale_factor=scale_factor,
                    tile_size=(512, 512),
                    overlap_factor=0.25,
                    detector=detector,
                    use_full_img=False
                )
                inf_times.append(inf_time)
            
                prediction = torch.tensor(prediction).to(device)
                outputs = torch.unsqueeze(prediction, 0)
                
                
                original_annotations = [y["original_annotations"]]
                for m in metrics:
                    errs = metrics[m].process_batch(outputs, original_annotations)
                    if m == 0.1:
                        prediction_errors = errs
                        print(errs)
                    # if m == 0.5:
                    #     print(errs)
                        
                t = time.time()
                if not os.path.exists("/home/thesis/tiny-od-on-edge/detector/pred_images/fp"):
                    os.makedirs("/home/thesis/tiny-od-on-edge/detector/pred_images/fp")
                if not os.path.exists("/home/thesis/tiny-od-on-edge/detector/pred_images/fn"):
                    os.makedirs("/home/thesis/tiny-od-on-edge/detector/pred_images/fn")
                save_path_fp = f"/home/thesis/tiny-od-on-edge/detector/pred_images/fp/img-{t}.png"
                save_path_fn = f"/home/thesis/tiny-od-on-edge/detector/pred_images/fn/img-{t}.png"
                    
                if bbox_out:
                    print("saving bboxes")
                    if not isinstance(x, tuple):
                        raise ValueError("x must be tuple")

                    altitude = x[1]
                    bboxes = create_bboxes(outputs, altitude, x[0].shape[1], x[0].shape[2], scale_factor, bbox_treshold)
                    
                    # save bbox crops
                    x_unnormalized = unnormalize(x[0], mean, std)
                    for i, bbox in enumerate(bboxes):
                        bbox = [int(val) for val in bbox]
                        obj_cls, x1, y1, x2, y2 = bbox
                        crop = x_unnormalized[:, x1:x2, y1:y2]
                        
                        if not os.path.exists(f"/home/thesis/tiny-od-on-edge/detector/pred_images/crops/{t}"):
                            os.makedirs(f"/home/thesis/tiny-od-on-edge/detector/pred_images/crops/{t}")
                        
                        save_image(crop, f"/home/thesis/tiny-od-on-edge/detector/pred_images/crops/{t}/img-{obj_cls}_{i}.png")
                        
                if isinstance(x, tuple):
                    x = x[0]
                
                # save_path = f"/home/thesis/tiny-od-on-edge/detector/pred_images/all/img-{t}.png"
                # if not os.path.exists("/home/thesis/tiny-od-on-edge/detector/pred_images/all"):
                #     os.makedirs("/home/thesis/tiny-od-on-edge/detector/pred_images/all")
                # draw_prediction(prediction, x, save_path, 0.1, mean, std, use_softmax=False, original_annotations=original_annotations)
                # print(save_path)
                # print("--------------------")
                
                # draw_prediction(prediction, x, save_path_fp, 0.5, mean, std, use_softmax=False)
                # exit()
                # error = False
                # if x is tuple
                
                # if isinstance(x, tuple):
                #     x = x[0]
                # if prediction_errors[0]["has_FP"]:
                #     draw_prediction(prediction, x, save_path_fp, 0.1, mean, std, use_softmax=False)
                #     error = True
                # if prediction_errors[0]["has_FN"]:
                #     draw_prediction(prediction, x, save_path_fn, 0.1, mean, std, use_softmax=False)
                #     error = True
                
                # if not error:
                #     fully_correct_images += 1
                # print("Fully correct images: ", fully_correct_images)

    precisions = []
    recalls = []
    recalls_by_size = {}

    for m in metrics:
        res = metrics[m].compute_metrics()
        print("Confidence treshold: ", m)
        print(res)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        tp_fn_by_size = metrics[m].compute_sized_metrics()
        for size in tp_fn_by_size:
            tp = tp_fn_by_size[size]["TP"]
            fn = tp_fn_by_size[size]["FN"]
            
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            print("Recall ", size, ": ", recall)
            if not size in recalls_by_size:
                recalls_by_size[size] = []
            recalls_by_size[size].append(recall)
        
        
        
    json_res = {
        "model_name": artifact_dir,
        "precisions": precisions,
        "recalls": recalls,
        "mean_inf_time": np.mean(inf_times),
    }

    fn = f"predictions_{len(os.listdir('predictions_store'))+5}.json"
    with open(f"predictions_store/{fn}.json", "w") as f:
        json.dump(json_res, f)
    print("Saved under ", fn)
    print("Mean inference time: ", np.mean(inf_times))