import random
import numpy as np
import torchvision
from PIL import Image

def mosaic_augmentation(images, img_size):
    """Create a mosaic image from four images with random cropping."""
    s = img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    mosaic_img = Image.new('RGB', (2 * s, 2 * s), (114, 114, 114))  # base image filled with gray (114)
    mosaic_labels = []

    for i, (img, labels) in enumerate(images):
        w, h = img.size
        # Random crop parameters
        crop_x1 = random.randint(0, max(0, w - s))
        crop_y1 = random.randint(0, max(0, h - s))
        crop_x2 = crop_x1 + s
        crop_y2 = crop_y1 + s

        img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        img_np = np.array(img)

        new_labels = []
        for label in labels:
            bbox = label['bbox']
            bbox[0] = np.clip(bbox[0] - crop_x1, 0, s)  # Adjust x coordinate
            bbox[1] = np.clip(bbox[1] - crop_y1, 0, s)  # Adjust y coordinate
            bbox[2] = np.clip(bbox[2], 0, s)  # Width remains the same within the bounds
            bbox[3] = np.clip(bbox[3], 0, s)  # Height remains the same within the bounds
            new_label = label.copy()
            new_label['bbox'] = bbox
            new_labels.append(new_label)

        if i == 0:  # top left
            x1, y1, x2, y2 = max(xc - s, 0), max(yc - s, 0), xc, yc
        elif i == 1:  # top right
            x1, y1, x2, y2 = xc, max(yc - s, 0), min(xc + s, 2 * s), yc
        elif i == 2:  # bottom left
            x1, y1, x2, y2 = max(xc - s, 0), yc, xc, min(2 * s, yc + s)
        elif i == 3:  # bottom right
            x1, y1, x2, y2 = xc, yc, min(xc + s, 2 * s), min(2 * s, yc + s)

        mosaic_img.paste(Image.fromarray(img_np[:y2 - y1, :x2 - x1]), (x1, y1))

        for label in new_labels:
            label['bbox'][0] += x1
            label['bbox'][1] += y1

        mosaic_labels.extend(new_labels)

    return mosaic_img, mosaic_labels


class MosaicDataset(torchvision.datasets.CocoDetection):
    def __init__(self, dataset, img_size, *args, **kwargs):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if random.random() < 0.5:
            indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
            images = [self.dataset[i] for i in indices]
            img, labels = mosaic_augmentation(images, self.img_size)
        else:
            img, labels = self.dataset[index]
        return img, labels