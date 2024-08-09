from model import Detector
import lightning as L
import json
import os
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from PIL import Image, ImageDraw
from model import Detector
from heads.fcn_head import FCNHead
from backends.mobilenet_v2 import MobileNetV2
from utils.metrics import Metrics
from torchvision.datasets import wrap_dataset_for_transforms_v2
from dataset.transforms import ToGrid, collate_fn
from torchvision.transforms import v2
from backends.mobileone import mobileone, reparameterize_model
import time

from utils.visualization import draw_prediction

checkpoint = "/home/tiny-od-on-edge/detector/artifacts/model-596kp70e:v32/model.ckpt"
img_root = "/data/data/sw_sliced_centered/coco/val"
labels_file = "/data/data/sw_sliced_centered/coco/annotations/instances_val-0percent_bg.json"
save_dir = "results"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

draw = True
compute_metrics = False
# classes = [("ignored", 0), ("swimmer", 1), ("boat", 2), ("jetski", 3), ("life_saving_appliances", 4), ("buoy", 5)]
classes = [("boat", 0)]
num_classes = len(classes)
treshold = 0.1
input_shape = (3, 512, 512)
mean = [0.2761, 0.4251, 0.5644]
std = [0.2060, 0.1864, 0.2218]

# backend = MobileNetV2(bn_momentum=0.9)

backend = mobileone(variant="s0")
# checkpoint_backend = torch.load(
#     "pretrained_models/mobileone_s0_unfused.pth.tar"
# )
# backend.load_state_dict(checkpoint_backend)


feat_map_shape = backend.get_feature_map_shape(input_shape)
head = FCNHead(
    num_classes=num_classes, in_channels=feat_map_shape[0], middle_channels=32
)
detector = (
    Detector.load_from_checkpoint(checkpoint, head=head, backend=backend)
    .type(torch.FloatTensor)
    .to(device)
)


grid_shape = (
    num_classes + 1,
    feat_map_shape[1],
    feat_map_shape[2],
)  # (num_classes + 1, height, width)
transforms_custom = v2.Compose(
    [
        v2.Resize(input_shape[1:]),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std),
        ToGrid(grid_shape, 0.125),
    ]
)


dataset = CocoDetection(
    root=img_root, annFile=labels_file, transforms=transforms_custom
)
dataset = wrap_dataset_for_transforms_v2(dataset)


dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
detector.eval()

detector.backend.eval()
detector.backend = reparameterize_model(detector.backend)

scale_factor = feat_map_shape[1] / input_shape[1]
metrics = Metrics(
    scale_factor=scale_factor, treshold=treshold, tolerance=2, classes=classes
)


def bbox_area(bbox):
    # format x0,y0,x1,y1
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


import time

times = []
for idx, batch in enumerate(dataloader):
    images, labels = batch
    images = [img.to(device) for img in images]
    labels = [el["original_annotations"] for el in labels]

    with torch.no_grad():
        if len(images) != 1:
            raise Exception("Currently cant handle a batch size > 1")
        start_time = time.time()
        output = detector(images[0])
        outputs = [output]
        for i in range(len(outputs)):
            if torch.min(outputs[i][0]) < torch.max(outputs[i][1]):
                print(torch.min(outputs[i][0]))
                print(torch.max(outputs[i][1]))

        end_time = time.time()
        times.append(end_time - start_time)
        if compute_metrics:
            metrics.process_batch(outputs, labels)

        if draw:
            for img_num, (img, output) in enumerate(zip(images, outputs)):
                save_path = os.path.join(save_dir, f"image_{idx}_{img_num}.png")
                draw_prediction(output, img, save_path, treshold, mean, std)


print("Mean time (ms): ", (sum(times) / len(times)) * 100)

print(metrics.compute_sized_metrics())
print(metrics.compute_classwise_metrics())
print(metrics.compute_metrics())
