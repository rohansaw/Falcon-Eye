import os
import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import torchvision

def draw_boxes(image, boxes, color, label=None):
    """
    Draw bounding boxes on an image.
    
    Parameters:
    image (np.ndarray): Image on which to draw boxes.
    boxes (list): List of bounding boxes, each defined as [x1, y1, x2, y2].
    color (tuple): Color for the boxes in BGR format.
    label (str): Label to add to each box (optional).
    
    Returns:
    np.ndarray: Image with drawn boxes.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def save_false_predictions(model, data_yaml, output_dir='/data/results/yolo_false_predictions'):
    # Load dataset yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    val_path = os.path.join(data['path'], data['val'])

    # Load model
    model = YOLO(model)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run inference and evaluation
    results = model(val_path, save=True, show_labels=False, show_conf=False)
    #eval_results = model.val(data=data_yaml)

    # Process each image
    # for idx, result in enumerate(results):
    #     image_path = result['image_path']
    #     image = cv2.imread(image_path)
        
    #     pred_boxes = result['boxes'].cpu().numpy()  # Get predicted boxes
    #     pred_scores = result['scores'].cpu().numpy()  # Get predicted scores
    #     pred_classes = result['classes'].cpu().numpy()  # Get predicted classes

    #     # Ground truth boxes and labels from eval_results
    #     # gt_boxes = eval_results[idx]['boxes'].cpu().numpy()
    #     # gt_classes = eval_results[idx]['classes'].cpu().numpy()

    #     # Define false positive and false negative
    #     # false_positives = []  # Predicted boxes that do not match any GT boxes
    #     # false_negatives = []  # GT boxes that do not match any predicted boxes

    #     # # Calculate overlaps (using IoU)
    #     # overlaps = torch.tensor([
    #     #     [torchvision.ops.box_iou(torch.tensor(box).unsqueeze(0), torch.tensor(gt_boxes)) for box in pred_boxes]
    #     # ])

    #     # # Determine false positives and false negatives based on a threshold
    #     # iou_threshold = 0.5
    #     # for i, overlap in enumerate(overlaps):
    #     #     if overlap.max() < iou_threshold:
    #     #         false_positives.append(pred_boxes[i])
        
    #     # for j, overlap in enumerate(overlaps.T):
    #     #     if overlap.max() < iou_threshold:
    #     #         false_negatives.append(gt_boxes[j])

    #     # Draw false positives (red) and false negatives (blue)
    #     image = draw_boxes(image, pred_boxes, (0, 0, 255), label='FP')
    #     #image = draw_boxes(image, false_negatives, (255, 0, 0), label='FN')

    #     # Save the result image
    #     image_name = os.path.basename(image_path)
    #     save_path = os.path.join(output_dir, image_name)
    #     cv2.imwrite(save_path, image)

    print(f"False predictions saved in {output_dir}")


save_false_predictions('/models/results_delab/results/yolo/20240407_012210/sds/weights/best.pt', '/home/thesis/tiny-od-on-edge/baselines/yolov8/dataset_spec/sds.yaml')
