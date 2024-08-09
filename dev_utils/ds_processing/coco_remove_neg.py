import json

def filter_coco_json(input_json_path, output_json_path):
    # Load the COCO JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    
    # Initialize containers for filtered data
    filtered_images = []
    filtered_annotations = []
    
    # Keep track of image IDs that have bounding boxes
    image_ids_with_bbox = set()

    # Filter annotations to keep those with bounding boxes
    for annotation in data['annotations']:
        # Check if the bounding box area is greater than 0 (has area)
        if annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:  # bbox format is [x,y,width,height]
            filtered_annotations.append(annotation)
            image_ids_with_bbox.add(annotation['image_id'])
    
    # Filter images to keep those that are referenced by the filtered annotations
    for image in data['images']:
        if image['id'] in image_ids_with_bbox:
            filtered_images.append(image)
    
    # Create a new dictionary with the filtered data
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': data['categories']  # Assuming categories do not need to be filtered
    }
    
    # Write the filtered data to a new JSON file
    with open(output_json_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)


input_json_path = '/scratch1/rsawahn/sw_new/coco/annotations/instances_val.json'
output_json_path = '/scratch1/rsawahn/sw_new/coco/annotations/instances_val_no_bg.json'

filter_coco_json(input_json_path, output_json_path)
