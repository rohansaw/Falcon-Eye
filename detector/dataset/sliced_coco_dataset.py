from PIL import Image
from PIL.Image import Image as PILImage
from torchvision.datasets import CocoDetection
from typing import Any, Tuple, List
import os
import random
from dataset.mosaic_augmentation import mosaic_augmentation
import wandb

class SlicedCocoDataset(CocoDetection):
    def __init__(self, meta_keys, tile_sz, overlap, min_area_ratio, use_cache, target_gsd=None, full_image=False, amount_bg=0, cache_disk=True, use_mosaic=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = meta_keys
        self.min_area_ratio = min_area_ratio
        self.overlap = overlap
        self.target_gsd = target_gsd
        self.tile_sz = tile_sz
        self.use_cache = use_cache
        self.cache_disk = cache_disk
        self.full_image = full_image
        self.num_images = len(self.ids)
        self.amount_bg = amount_bg
        self.use_mosaic = use_mosaic
        
        if "root" in kwargs:
            self.cache_dir_disk = os.path.join(kwargs["root"], "cache")
            if not os.path.exists(self.cache_dir_disk):
                os.makedirs(self.cache_dir_disk)
        else:
            raise ValueError("data_dir must be provided as a keyword argument.")
        
        self.img_cache = {}
        self.tiles = []
        self.bg_tiles = []
        
        # PiCamV2 specifics
        self.picamv2 = {
            "focal_len_mm": 3.04, 
            "sensor_width_mm": 3.68,
            "sensor_height_mm": 2.76,
            "sensor_width_px": 3280,
            "sensor_height_px": 2464,
        }
        
        self.scaling_params_default = {
            "focal_len_mm": 15, 
            "sensor_width_mm": 17.3,
            "sensor_height_mm": 13,
            "sensor_width_px": 3840,
            "sensor_height_px": 2160,
        }
        
        self._init_tiles()
        print(len(self.ids))
        print(len(self.tiles))
        
    def _init_tiles(self):
        print("Starting to initialize tiles. Altitude based scaling activated: {}".format(self.target_gsd is not None))
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            width = img_info["width"]
            height = img_info["height"]
            altitude = img_info.get("altitude")
            anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            
            if self.target_gsd is not None and altitude:
                scaled_w, scaled_h = self._compute_scaled_img_size(width, height, altitude, **self.scaling_params_default)
                self.img_cache[img_id] = {
                    "scaled_w": scaled_w,
                    "scaled_h": scaled_h,
                    "altitude": altitude,
                    #"img": self._load_scaled_image(img_id, scaled_w, scaled_h) if self.use_cache else None
                    "img": None
                }
                scaled_anns = self._scale_annotations(anns, scaled_w / width, scaled_h / height)
                tiles = self._create_img_tiles(img_id, scaled_w, scaled_h, scaled_anns)
            else:
                self.img_cache[img_id] = {
                    #"img": self._load_image(img_id) if self.use_cache else None
                    "img": None,
                    "altitude": altitude if altitude else None,
                }
                tiles = self._create_img_tiles(img_id, width, height, anns)             
            
            if self.full_image:
                if len(anns) > 0 or self.amount_bg > 0:
                    full_img_tile = self._get_full_img_tile(img_id, anns, width, height)
                    tiles.append(full_img_tile)
            
            self.img_cache[img_id]["tiles"] = tiles
            self.tiles.extend(tiles)
            
        # choose amount_bg random tiles
        if self.amount_bg > 1:
            num_bg_tiles = len(self.bg_tiles) - 1
        else:
            num_bg_tiles = min(len(self.bg_tiles) - 1, int(len(self.tiles) * self.amount_bg))
        random.shuffle(self.bg_tiles)
        self.tiles.extend(self.bg_tiles[:num_bg_tiles])
        random.shuffle(self.tiles)
        
        self.ids = list(range(len(self.tiles)))
        print("Finished initializing tiles")
    
    def _load_scaled_image(self, id: int, scaled_w, scaled_h) -> PILImage:
        return self._load_image(id).resize((scaled_w, scaled_h))
    
    def _compute_scaled_img_size(self, img_w, img_h, altitude, focal_len_mm, sensor_width_mm, sensor_height_mm, sensor_width_px, sensor_height_px):
        focal_len_px = focal_len_mm * (sensor_width_px / sensor_width_mm)
        
        current_gsd = (altitude * sensor_height_mm) / (focal_len_mm * img_h)
        scale_factor = current_gsd / self.target_gsd
        
        # print()
        # print("-----------------")
        # print("Current GSD: ", current_gsd)
        # print("Curretn Altitude", altitude)
        # print("target GSD: ", self.target_gsd)
        # print("Scale factor: ", scale_factor)
        
        # print("Image size before scaling: ", img_w, img_h)
        img_size_w = int(img_w * scale_factor)
        img_size_h = int(img_h * scale_factor)
        #print("Image size after scaling: ", img_size_w, img_size_h)
        
        
        # img_size_w = int((altitude / focal_len_px * img_w) / self.target_gsd)
        # img_size_h = int((altitude / focal_len_px * img_h) / self.target_gsd)
        
        # print("Altitude: ", altitude)
        # print("w, h before: ", img_w, img_h)
        # print("w, h after: ", img_size_w, img_size_h)
        
        return (img_size_w, img_size_h)
    
    def _scale_annotations(self, anns, scale_factor_w, scale_factor_h):
        for ann in anns:
            ann["bbox"][0] = int(ann["bbox"][0] * scale_factor_w)
            ann["bbox"][1] = int(ann["bbox"][1] * scale_factor_h)
            ann["bbox"][2] = int(ann["bbox"][2] * scale_factor_w)
            ann["bbox"][3] = int(ann["bbox"][3] * scale_factor_h)
        return anns
    
    def _ann_has_min_area_in_tile(self, x, y, ann, min_area_ratio=None):
        x0, y0, w, h = [int(i) for i in ann["bbox"]]
        x1 = x0 + w
        y1 = y0 + h
        
        tile_x1 = x + self.tile_sz
        tile_y1 = y + self.tile_sz
        
        # if x1 <= x or x0 >= tile_x1 or y1 <= y or y0 >= tile_y1:
        #     return False
       
        ix0 = max(x0, x)
        iy0 = max(y0, y)
        ix1 = min(x1, tile_x1)
        iy1 = min(y1, tile_y1)

        # Calculate intersection width and height
        intersection_width = max(0, ix1 - ix0)
        intersection_height = max(0, iy1 - iy0)

        # Calculate intersection area
        intersection_area = intersection_width * intersection_height

        # Calculate original area
        original_area = w * h
        if original_area == 0:
            return False
        
        min_area_ratio = min_area_ratio if min_area_ratio is not None else self.min_area_ratio

        # Check if the intersection area is at least the minimum area ratio of the original area
        return (intersection_area / original_area) >= min_area_ratio
    
    def _ann_to_ann_in_tile(self, ann, x, y):
        x0, y0, w, h = ann["bbox"]
        new_x0 = x0 - x
        new_y0 = y0 - y
        new_ann = ann.copy()
        if new_x0 < 0:
            new_x0 = 0
        if new_y0 < 0:
            new_y0 = 0
        if new_x0 + w > self.tile_sz:
            w = self.tile_sz - new_x0
        if new_y0 + h > self.tile_sz:
            h = self.tile_sz - new_y0
        new_ann["bbox"] = [new_x0, new_y0, w, h]
        return new_ann
    
    def _get_full_img_tile(self, img_id, anns, img_w, img_h):
        scale_factor_h = self.tile_sz / img_h
        scale_factor_w = self.tile_sz / img_w
        anns = self._scale_annotations(anns, scale_factor_h=scale_factor_h, scale_factor_w=scale_factor_w)
        return {
            "img_id": img_id,
            "x": 0,
            "y": 0,
            "anns": anns,
            "full_image": True,
        }
        
    def get_anns_partly_in_tile(self, x, y, anns):
        filtered_anns = []
        for ann in anns:
            x0, y0, w, h = ann["bbox"]
            x1 = x0 + w
            y1 = y0 + h
            
            tile_x1 = x + self.tile_sz
            tile_y1 = y + self.tile_sz
            
            if x1 <= x or x0 >= tile_x1 or y1 <= y or y0 >= tile_y1:
                continue
            filtered_anns.append(ann)
        return filtered_anns
    
    def _get_anns_in_tile(self, x, y, anns):
        anns_in_tile = []
        filtered_anns = self.get_anns_partly_in_tile(x, y, anns)
        if any([self._ann_has_min_area_in_tile(x, y, ann) for ann in filtered_anns]):
            for ann in filtered_anns:
                anns_in_tile.append(self._ann_to_ann_in_tile(ann, x, y))
        # for ann in anns:
        #     if self._ann_has_min_area_in_tile(x, y, ann):
        #         anns_in_tile.append(self._ann_to_ann_in_tile(ann, x, y))
        return anns_in_tile
    
    def is_bg_tile(self, x, y, anns):
        filtered_anns = self.get_anns_partly_in_tile(x, y, anns)
        if len(filtered_anns) == 0:
            return True
        return False
        # for ann in filtered_anns:
        #     if self._ann_has_min_area_in_tile(x, y, ann, min_area_ratio=0.001):
        #         return False
        # return True
        
    def contains_partly_annotation(self, x, y, anns):
        for ann in anns:
            x0, y0, w, h = ann["bbox"]
            x1 = x0 + w
            y1 = y0 + h
            tile_x1 = x + self.tile_sz
            tile_y1 = y + self.tile_sz
            if x1 <= x or x0 >= tile_x1 or y1 <= y or y0 >= tile_y1:
                continue
            if not self._ann_has_min_area_in_tile(x, y, ann):
                return True
        return False
    
    def _create_img_tiles(self, img_id, width, height, anns):
        stride = int(self.tile_sz * (1 - self.overlap))
        tiles = []
        bg_tiles = []
        
        for x in range(0, width , stride):
            for y in range(0, height, stride):
                if self.contains_partly_annotation(x, y, anns):
                    continue
                anns_in_tile = self._get_anns_in_tile(x, y, anns)
                if not anns_in_tile and self.amount_bg == 0:
                    continue
                
                if x + self.tile_sz > width:
                    x = width - self.tile_sz
                
                if y + self.tile_sz > height:
                    y = height - self.tile_sz
                
                tile_info = {
                    "img_id": img_id,
                    "x": x,
                    "y": y,
                    "anns": anns_in_tile,
                    "full_image": False,
                }
                
                # do not include if last row is same as previous
                
                if len(anns_in_tile) == 0:
                    if self.is_bg_tile(x, y, anns) and tile_info not in bg_tiles:
                        bg_tiles.append(tile_info)
                else:
                    if tile_info not in tiles:  
                        tiles.append(tile_info)

        self.bg_tiles.extend(bg_tiles)
        return tiles
        
    def _load_meta(self, id: int):
        meta = {}
        for meta_key in self.meta_keys:
            meta[meta_key] = self.coco.loadImgs(id)[0].get(meta_key)
        return meta
    
    def get_full_img(self, img_id: int) -> PILImage:
        if self.img_cache[img_id]["img"] is not None:
            img = self.img_cache[img_id]["img"]
        else:
            #print("FUll image cache miss")
            if self.target_gsd:
                img = self._load_scaled_image(img_id, self.img_cache[img_id]["scaled_w"], self.img_cache[img_id]["scaled_h"])
                #print(img.size)
            else:
                img = self._load_image(img_id)
        
        return img
    
    def get_img_tile(self, img_id: int, x: int, y: int, is_full_image_tile: bool) -> PILImage:
        full_img = self.get_full_img(img_id)
        if is_full_image_tile:
            return full_img.resize((self.tile_sz, self.tile_sz))
        return full_img.crop((x, y, x + self.tile_sz, y + self.tile_sz))
    
    def check_cache_hit(self, index):
        if not self.use_cache:
            return False
        if self.cache_disk:
            tile = self.tiles[index]
            img_id = tile["img_id"]
            file_name = f"{img_id}_{tile['x']}_{tile['y']}.jpg"
            cached_file_path = os.path.join(self.cache_dir_disk, file_name)
            return os.path.exists(cached_file_path)
        else:
            tile = self.tiles[index]
            if "cached_tile" in tile and tile["cached_tile"] is not None:
                return True
            return False
        
    def save_tile_to_disk(self, index, tile_img):
        tile = self.tiles[index]
        img_id = tile["img_id"]
        file_name = f"{img_id}_{tile['x']}_{tile['y']}.jpg"
        out_path = os.path.join(self.cache_dir_disk, file_name)
        tile_img.save(out_path)
        
        
    def load_cached_tile_from_disk(self, index):
        tile = self.tiles[index]
        img_id = tile["img_id"]
        file_name = f"{img_id}_{tile['x']}_{tile['y']}.jpg"
        cached_file_path = os.path.join(self.cache_dir_disk, file_name)
        if not os.path.exists(cached_file_path):
            raise ValueError(f"File {cached_file_path} does not exist.")
        return Image.open(cached_file_path).convert("RGB")
        
        
    def set_cache(self, index, tile_img):
        if self.cache_disk:
            self.save_tile_to_disk(index, tile_img)
        else:
            self.tiles[index]["cached_tile"] = tile_img
        
    def load_cached_tile(self, index):
        if self.cache_disk:
            return self.load_cached_tile_from_disk(index)
        else:
            return self.tiles[index]["cached_tile"]
    
    def load_tile_at_index(self, index):
        tile = self.tiles[index]
        img_id = tile["img_id"]
        return self.get_img_tile(img_id=img_id, x=tile["x"], y=tile["y"], is_full_image_tile=tile["full_image"])

    
    def get_img_tiles(self, img_id) -> List[PILImage]:
        tiles_info =  self.img_cache[img_id]["tiles"]
        return [self.get_img_tile(tile_info["img_id"], tile_info["x"], tile_info["y"], tile_info["full_image"]) for tile_info in tiles_info]
    
    def log_original_image(self, img_id):
        img = self.get_full_img(img_id)
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        self.log(img, anns, "full_original")
        
    def log(self, img, anns, log_name):
        all_boxes = []
        for ann in anns:
            img_h, img_w = img.size
            x0, y0, w, h = ann["bbox"]
            x1 = x0 + w
            y1 = y0 + h
            # get coordinates and labels
            box_data = {"position" : {
                "minX" : int(x0) / img_h,
                "maxX" : int(x1) / img_h,
                "minY" : int(y0) / img_w,
                "maxY" : int(y1) / img_w,},
                "class_id" : int(ann["category_id"]),
            }
            all_boxes.append(box_data)
        box_image = wandb.Image(img, boxes = {"predictions": {"box_data": all_boxes}})
        # log all images and labels from batch to wandb to be visualized there
        wandb.log({log_name: box_image})
        
    
    def get_item_standard(self, index: int) -> Tuple[PILImage, Any]:
        target  = self.tiles[index]["anns"]
        img_id = self.tiles[index]["img_id"]
        
        #self.log_original_image(img_id)
        
        # cache hit
        if self.use_cache:
            if self.check_cache_hit(index):
                tile_img = self.load_cached_tile(index)
            else:
                tile_img = self.load_tile_at_index(index)
                self.set_cache(index, tile_img)
        else:
            tile_img = self.load_tile_at_index(index)

        #self.log(tile_img, target, "tiled_original")
        
        img_id = self.tiles[index]["img_id"]
        tile_altitude = self.img_cache[img_id]["altitude"]
        
        return (tile_img, tile_altitude), target
    
    def get_item_moasic(self, index: int) -> Tuple[PILImage, Any]:
        indices = [index] + [random.randint(0, len(self.tiles) - 1) for _ in range(3)]
        images = [self.get_item_standard(i) for i in indices]
        return mosaic_augmentation(images, self.tile_sz)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.use_mosaic:
            return self.get_item_moasic(index)
        return self.get_item_standard(index)
      

    def __len__(self):
        return len(self.tiles)