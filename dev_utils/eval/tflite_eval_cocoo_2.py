import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_image(image_path, input_shape, mean, std):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[2], input_shape[3]))
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    # currently is NHWC, change to NCHW
    image = np.transpose(image, (0, 3, 1, 2))
    print(image.shape, "IM")
    
    return image

def quantize_image(image, scale, zero_point):
    image = image / scale + zero_point
    image = image.astype(np.uint8)
    return image

def run_inference(interpreter, input_details, output_details, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]
    return output_data

def postprocess_results(output_data, image_id, category_id=1, score_threshold=0.1, iou_threshold=0.5):
    cat_ids = output_data[0][0]
    boxes_scores = output_data[1][0]
    
    boxes = boxes_scores[:, :4]
    scores = boxes_scores[:, 4]

    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=100, iou_threshold=iou_threshold, score_threshold=score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    selected_labels = tf.gather(cat_ids, selected_indices).numpy()

    results = []
    for i in range(len(selected_scores)):
        ymin, xmin, ymax, xmax = selected_boxes[i]
        result = {
            "image_id": int(image_id),
            "category_id": int(selected_labels[i]),
            "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
            "score": float(selected_scores[i])
        }
        
        print(result)
        results.append(result)
    return results

def main(model_path, image_dir, coco_json_path, output_json_path, mean, std):
    interpreter, input_details, output_details = load_tflite_model(model_path)
    print(input_details)
    input_shape = input_details[0]['shape']
    
    global input_scale, input_zero_point
    input_scale, input_zero_point = input_details[0]['quantization']

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
        image = preprocess_image(image_path, input_shape, mean, std)
        image = quantize_image(image, input_scale, input_zero_point)
        output_data = run_inference(interpreter, input_details, output_details, image)
        print(output_data)
        results += postprocess_results(output_data, image_id)

    with open(output_json_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    model_path = "/home/rsawahn/thesis/tiny-od-on-edge/dev_utils/model_export/modified_model.onnx2.tf2-13_INT8.tflite"
    image_dir = "/scratch1/rsawahn/data/sw_sliced/coco/val"
    coco_json_path = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val.json"
    output_json_path = "results2.json"
    
    mean = np.array([0.2761, 0.4251, 0.5644], dtype=np.float32)
    std = np.array([0.2060, 0.1864, 0.2218], dtype=np.float32)
    
    main(model_path, image_dir, coco_json_path, output_json_path, mean, std)
