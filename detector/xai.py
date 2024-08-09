from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    SemanticSegmentationTarget,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import Detector
import torch
from backends.mobileone import mobileone, reparameterize_model
from heads.fcn_head import FCNHead
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import os
import json

num_classes = 1
device = "cpu"
input_shape = (3, 512, 512)  # Desired shape for the model input
images_path = "/data/sw_sliced_centered/coco/val"
#img_path = "/home/rohan/Documents/Uni/MA/raspi/images/1230.png"
checkpoint = "artifacts/model-596kp70e:v32/model.ckpt"
mean = [0.2761, 0.4251, 0.5644]
std = [0.2060, 0.1864, 0.2218]
scale_factor = 0.125


coco_ann_path = "/data/sw_sliced_centered/coco/annotations/instances_val.json"
with open(coco_ann_path, "r") as f:
    coco = json.load(f)
    ann = coco["annotations"][120]
    image_id = ann["image_id"]
    fp = [img["file_name"] for img in coco["images"] if img["id"] == image_id][0]
    
    img_path = os.path.join(images_path, fp)

#img_path = os.path.join(images_path, os.listdir(images_path)[0])

backend = mobileone(variant="s0")
backend.truncate(2)

feat_map_shape = backend.get_feature_map_shape(input_shape)
head = FCNHead(
    num_classes=num_classes, in_channels=feat_map_shape[0], middle_channels=32
)

model = (
    Detector.load_from_checkpoint(checkpoint, head=head, backend=backend)
    .type(torch.FloatTensor)
    .to(device)
)

transforms = v2.Compose(
    [
        v2.Resize(
            input_shape[1:]
        ),  # Ensuring input image is resized to the same size as input_shape
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std),
    ]
)

target_layers = [model.backend]
img = Image.open(img_path)
# save image for visualizatio
img.save("input_image.png")
input_tensor = torch.unsqueeze(transforms(img).to(device), 0)

cam = GradCAM(model=model, target_layers=target_layers)
#cam = ScoreCAM(model=model, target_layers=target_layers)

seg_mask = np.ones((1, 1, int(512 * scale_factor), int(512 * scale_factor)))

targets = [SemanticSegmentationTarget(0, seg_mask)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#grayscale_cam = 1 - grayscale_cam

# Load and prepare the RGB image for visualization
rgb_img = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB format
rgb_img = rgb_img.resize(
    (512, 512)
)  # Resize the RGB image to match the model input dimensions
rgb_img = np.float32(rgb_img) / 255  # Normalize the RGB image
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
img = Image.fromarray(visualization, "RGB")

# Save the image
img.save("output_image.png")


