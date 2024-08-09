import json

json_path = "/scratch1/rsawahn/data/slicing_test_res/sahi"
with open(json_path, 'r') as f:
    coco = json.load(f)
    # print(coco.keys())
    # #imgs_without_anns = [img for img in coco['images'] if not img['id'] in [ann['image_id'] for ann in coco['annotations']]]
    # print(len(coco["annotations"]))
    # print(len(coco["images"]))
    
    print(set([ann["category_id"] for ann in coco]))