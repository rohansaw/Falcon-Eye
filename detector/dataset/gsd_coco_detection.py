from torchvision.datasets import CocoDetection

class GSDCOCODetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, target_gsd = None):
        super(GSDCOCODetection, self).__init__(root, annFile, transform, target_transform, transforms)
        
        self.scaling_params_default = {
            "focal_len_mm": 15, 
            "sensor_width_mm": 17.3,
            "sensor_height_mm": 13,
            "sensor_width_px": 3840,
            "sensor_height_px": 2160,
        }
        
        self.target_gsd = target_gsd
        
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

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        if self.target_gsd is not None:
            img_info = self.coco.loadImgs(img_id)[0]
            anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            width = img_info["width"]
            height = img_info["height"]
            altitude = img_info.get("altitude")
            
            scaled_w, scaled_h = self._compute_scaled_img_size(width, height, altitude, **self.scaling_params_default)
            target = self._scale_annotations(anns, scaled_w / width, scaled_h / height)
            image = self._load_image(img_id).resize((scaled_w, scaled_h))
        else:
            target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            image = self._load_image(img_id)
            img_info = self.coco.loadImgs(img_id)[0]
            altitude = img_info.get("altitude")
        
        return (image, altitude), target

    def __len__(self):
        return super(GSDCOCODetection, self).__len__()