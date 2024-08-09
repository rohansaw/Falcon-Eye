from torchvision.datasets import CocoDetection
from typing import Any, Tuple
from torchvision.tv_tensors._dataset_wrapper import WRAPPER_FACTORIES, parse_target_keys, list_of_dicts_to_dict_of_lists
from torchvision.transforms.v2 import functional as F
import torch
from torchvision import tv_tensors

class CustomCocoDataset(CocoDetection):
    def __init__(self, meta_keys,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = meta_keys
        
    def _load_meta(self, id: int):
        meta = {}
        for meta_key in self.meta_keys:
            meta[meta_key] = self.coco.loadImgs(id)[0].get(meta_key)
        return meta

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        meta = self._load_meta(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, meta, target

    def __len__(self):
        return super().__len__()
    
# @WRAPPER_FACTORIES.register(CocoDetection)
# def coco_dectection_wrapper_factory(dataset, target_keys):
#     target_keys = parse_target_keys(
#         target_keys,
#         available={
#             "segmentation",
#             "area",
#             "iscrowd",
#             "image_id",
#             "bbox",
#             "category_id",
#             "boxes",
#             "masks",
#             "labels",
#         },
#         default={"image_id", "boxes", "labels"},
#     )
    
#     def segmentation_to_mask(segmentation, *, canvas_size):
#         from pycocotools import mask

#         segmentation = (
#             mask.frPyObjects(segmentation, *canvas_size)
#             if isinstance(segmentation, dict)
#             else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
#         )
#         return torch.from_numpy(mask.decode(segmentation))

#     def wrapper(idx, sample):
#         if len(sample) == 3:
#             image, meta, target = sample
#         else:
#             raise ValueError("The dataset should return a tuple of (image, meta, target)")

#         image_id = dataset.ids[idx]

#         if not target:
#             return image, meta, dict(image_id=image_id)

#         canvas_size = tuple(F.get_size(image))

#         batched_target = list_of_dicts_to_dict_of_lists(target)
#         target = {}

#         if "image_id" in target_keys:
#             target["image_id"] = image_id

#         if "boxes" in target_keys:
#             target["boxes"] = F.convert_bounding_box_format(
#                 tv_tensors.BoundingBoxes(
#                     batched_target["bbox"],
#                     format=tv_tensors.BoundingBoxFormat.XYWH,
#                     canvas_size=canvas_size,
#                 ),
#                 new_format=tv_tensors.BoundingBoxFormat.XYXY,
#             )

#         if "masks" in target_keys:
#             target["masks"] = tv_tensors.Mask(
#                 torch.stack(
#                     [
#                         segmentation_to_mask(segmentation, canvas_size=canvas_size)
#                         for segmentation in batched_target["segmentation"]
#                     ]
#                 ),
#             )

#         if "labels" in target_keys:
#             target["labels"] = torch.tensor(batched_target["category_id"])

#         for target_key in target_keys - {"image_id", "boxes", "masks", "labels"}:
#             target[target_key] = batched_target[target_key]

#         return image, meta, target

#     return wrapper
