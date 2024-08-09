from PIL import Image, ImageDraw
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time


def draw_prediction_on_image(image, prediction, grid_size, cell_size, num_classes):
    draw = ImageDraw.Draw(image)
    for i in range(grid_size):
        for j in range(grid_size):
            cell_pred = prediction[:, i, j]
            class_pred = torch.argmax(cell_pred[1:]) + 1  # Skip the background class
            if class_pred > 0:
                color = tuple(torch.randint(0, 256, (3,)).tolist())
                label = f"Class {class_pred}"
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = (j + 1) * cell_size, (i + 1) * cell_size
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1), label, fill=color)


def draw_predictions(images, predictions, grid_size, num_classes):
    for img, prediction in zip(images, predictions):
        draw_prediction_on_image(img, prediction, grid_size=3, num_classes=1)


def unnormalize(tensor, mean, std):
    """
    Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
        mean (list): The mean used for normalization (for each channel).
        std (list): The standard deviation used for normalization (for each channel).

    Returns:
        Tensor: Unnormalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # The inverse of normalization: multiply by std and add mean
    return tensor


def draw_prediction(output, img, out_path, treshold, mean, std, use_softmax=True, original_annotations=None):
    print(output.shape)
    if use_softmax:
        output = torch.softmax(output, dim=0)
    grid_w = output.shape[-1]
    grid_h = output.shape[-2]
    scale_factor = img.shape[-1] // grid_w
    img = unnormalize(img.clone().detach(), mean, std)
    img = v2.ToPILImage()(img.squeeze(0))
    draw = ImageDraw.Draw(img)

    indexes = torch.nonzero(output[1:, :, :] > treshold)
    for index in indexes:
        c, i, j = index
        x1, y1 = j * scale_factor, i * scale_factor
        x2, y2 = (j + 1) * scale_factor, (i + 1) * scale_factor
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width -1, x2)
        y2 = min(img.height -1, y2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        #draw.text((x1, y1), f"Class {c}: {output[c, i, j]:.2f}")
        draw.text((x1, y1), f"{c}")

    # for i in range(grid_h):
    #     for j in range(grid_w):
    #         cell_pred = output[:, i, j]
    #         for c in range(1, cell_pred.shape[0]):
    #             if output[c, i, j] >= treshold:
    #                 x1, y1 = j * scale_factor, i * scale_factor
    #                 x2, y2 = (j + 1) * scale_factor, (i + 1) * scale_factor
    #                 # clip the box to the image
    #                 x1 = max(0, x1)
    #                 y1 = max(0, y1)
    #                 x2 = min(img.width -1, x2)
    #                 y2 = min(img.height -1, y2)
    #                 draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    #                 #draw.text((x1, y1), f"Class {c}: {output[c, i, j]:.2f}")
    #                 draw.text((x1, y1), str(c))

    if original_annotations is not None:
        for entry in original_annotations:
            bboxes = entry["boxes"]
            for bbox in bboxes:
                tl = (int(bbox[0]), int(bbox[1]))
                br = (int(bbox[2]), int(bbox[3]))
                draw.rectangle(
                    [tl[0], tl[1], br[0], br[1]],
                    outline=(57, 255, 20),
                    width=2,
                )

    #img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))
    img.save(out_path)

def draw_boxes(image, boxes, labels=None, color='red'):
    """
    Draws bounding boxes on the image.
    Args:
        image: PIL Image or NumPy array
        boxes: List or array of bounding boxes in [x1, y1, x2, y2] format
        labels: List of labels for each bounding box (optional)
        color: Color of the bounding boxes
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i, box in enumerate(boxes):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # if labels is not None:
        #     plt.text(box[0], box[1], labels[i], color='white', fontsize=12, backgroundcolor="red")
    
    # save with timestamp
    plt.savefig(f"vis_aug/output_{str(time.time())}.png")

def visualize_augmentation(dataset, transform, num_images=64):
    """
    Visualizes images before and after augmentation.
    Args:
        dataset: Dataset object
        transform: Transformation function
        num_images: Number of images to visualize
    """
    for i in range(num_images):
        image, target = dataset[i]
        if "boxes" in target:
            draw_boxes(image, target['boxes'], target.get('labels'))
            transformed_image, transformed_target = transform(image, target)
            transformed_target = transformed_target["original_annotations"]
            draw_boxes(transformed_image, transformed_target['boxes'], transformed_target.get('labels'), color='blue')