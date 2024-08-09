from collections import defaultdict
import cv2
import json
import os

# Load annotations from a COCO-style JSON file
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        data = json.load(file)
    image_data = {item['id']: item['file_name'] for item in data['images']}
    return image_data

# Load predictions from a COCO-style JSON results file
def load_predictions(prediction_file):
    with open(prediction_file, 'r') as file:
        predictions = json.load(file)
        
    res = defaultdict(list)
    for pred in predictions:
        res[pred["image_id"]].append(pred)
    
    return res

# Draw predictions on an image
def draw_predictions(image_path, predictions):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return None

    for pred in predictions:
        x, y, w, h = pred['bbox']
        class_id = pred['category_id']
        score = pred['score']
        
        # Draw rectangle
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # Put class ID and score
        cv2.putText(image, f'{class_id} {score:.2f}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return image

# Setup paths and load data
images_root = '/scratch1/rsawahn/data/sw_sliced/coco/val'
annotation_file = '/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val.json'
prediction_file = "/home/rsawahn/thesis/tiny-od-on-edge/baselines/efficientdet/testdev_output_sw_sliced/detections_test-dev2017_test_results.json"
#prediction_file = "/scratch1/rsawahn/baseline_results2/efficientdet/sw_sliced/merged.json"

image_data = load_annotations(annotation_file)
predictions = load_predictions(prediction_file)
image_ids = list(image_data.keys())

# Save the first 200 images with predictions drawn
output_directory = 'img_out'  # Define your output directory
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it does not exist

for idx, image_id in enumerate(image_ids[:200]):  # Limit to the first 200 images
    image_path = os.path.join(images_root, image_data[image_id])
    if "2021-12-16T12-21-36.574137" not in image_path:
        continue
    else:
        print("ok")
    image_with_predictions = draw_predictions(image_path, predictions[image_id])
    if image_with_predictions is not None:
        save_path = os.path.join(output_directory, f"predicted_{idx+1}.jpg")
        cv2.imwrite(save_path, image_with_predictions)
        print(f"Saved: {save_path}")
        print(image_path)