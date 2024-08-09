import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

def load_tf_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def preprocess_image(image_path, input_shape, mean, std):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[2], input_shape[3])) 
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    # currently is NHWC, change to NCHW
    image = np.transpose(image, (0, 3, 1, 2))
    return image

def run_inference(model, image):
    infer = model.signatures['serving_default']
    output_data = infer(tf.convert_to_tensor(image))
    return output_data

def postprocess_results(output_data, image_id, score_threshold=0.1):
    dets = output_data['dets'][0].numpy()  # Assuming 'dets' is a key in the output dictionary
    labels = output_data['labels'][0].numpy().astype(int)  # Assuming 'labels' is a key in the output dictionary

    scores = dets[:, -1]
    boxes = dets[:, :-1]

    # Filter boxes by score threshold
    selected_indices = scores > score_threshold
    selected_boxes = boxes[selected_indices]
    selected_scores = scores[selected_indices]
    selected_labels = labels[selected_indices]

    results = []
    for i in range(len(selected_scores)):
        ymin, xmin, ymax, xmax = selected_boxes[i]
        result = {
            "image_id": int(image_id),
            "category_id": int(selected_labels[i]),
            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "score": float(selected_scores[i])
        }
        results.append(result)
    return results

def main(model_path, image_dir, coco_json_path, output_json_path, mean, std):
    model = load_tf_model(model_path)
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    imgs_with_anns = [ann["image_id"] for ann in coco_data['annotations']]

    results = []
    for image_info in tqdm(coco_data['images']):
        # only run inference if image has annotations
        image_id = image_info['id']
        if image_id not in imgs_with_anns:
            continue
        image_path = os.path.join(image_dir, image_info['file_name'])
        input_shape = [1, 3, 512, 512]  # Update this based on your model's expected input shape
        image = preprocess_image(image_path, input_shape, mean, std)
        output_data = run_inference(model, image)
        results.append(postprocess_results(output_data, image_id))

    with open(output_json_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    model_path = "/home/rsawahn/thesis/tiny-od-on-edge/dev_utils/model_export/ssd_sds_sliced.onnx.tf"
    image_dir = "/scratch1/rsawahn/data/sds_sliced/coco/val"
    coco_json_path = "/scratch1/rsawahn/data/sds_sliced/coco/annotations/instances_val.json"
    output_json_path = "results.json"
    
    mean = np.array([0.4263, 0.4856, 0.4507], dtype=np.float32)
    std = np.array([0.1638, 0.1515, 0.1755], dtype=np.float32)
    
    main(model_path, image_dir, coco_json_path, output_json_path, mean, std)
