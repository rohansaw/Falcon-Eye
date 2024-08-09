from pytorch_grad_cam import GradCAM, ScoreCAM, HiResCAM
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
checkpoint = "artifacts/model-9bb7qmn9:v40/model.ckpt"
mean = [0.2761, 0.4251, 0.5644]
std = [0.2060, 0.1864, 0.2218]
scale_factor = 0.125



# 120, 140, 240
coco_ann_path = "/data/sw_sliced_centered/coco/annotations/instances_val.json"
with open(coco_ann_path, "r") as f:
    coco = json.load(f)
    ann = coco["annotations"][240]
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

model.eval()
model.head.eval()
model.backend.eval()

transforms = v2.Compose(
    [
        v2.Resize(
            input_shape[1:]
        ),  # Ensuring input image is resized to the same size as input_shape
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std),
    ]
)

img = Image.open(img_path).convert("RGB")
img.save("input_image.png")
print(model)

#target_layers = [model.backend.stage2[15]]
target_layers = [model.attention]
category_idx = 1

input_tensor = torch.unsqueeze(transforms(img).to(device), 0)

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    
class ModelWrapperAttention(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapperAttention, self).__init__()
        self.model = model
    def __call__(self, x):
        res = self.model(x)
        if isinstance(res, tuple):
            # Select the relevant output from the tuple
            res = res[0]
        return res

prediction = model(input_tensor)
print(prediction.shape)
prediction = prediction.squeeze(0)
normalized_masks = torch.softmax(torch.tensor(prediction), dim=0).detach().cpu()

boat_mask = normalized_masks[category_idx, :, :].numpy()
boat_mask = boat_mask > 0.5
boat_mask_float = np.float32(boat_mask)
rgb_img = np.float32(img) / 255  # Normalize the RGB image

# visualize boat mask on image and save
masked_img = np.copy(rgb_img)
scale_factor = 0.125
boat_mask_resized = v2.Resize((512,512))(Image.fromarray(boat_mask.astype(np.uint8)))
boat_mask_resized = np.array(boat_mask_resized)
mask_img = 255 * np.uint8(boat_mask_resized)
Image.fromarray(mask_img).save("masked_image.png")  # Save masked image

# masked_img[boat_mask_resized] = [0, 255, 0]  # Set boat mask pixels to green
# masked_img = np.uint8(masked_img * 255)  # Convert back to uint8
# masked_img = Image.fromarray(masked_img)  # Convert to PIL Image
# masked_img.save("masked_image.png")  # Save masked image

    
targets = [SemanticSegmentationTarget(category_idx, boat_mask)]
with ScoreCAM(model=ModelWrapperAttention(model),
             target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

img_out = Image.fromarray(cam_image)
# save img_out
img_out.save("output_image.png")