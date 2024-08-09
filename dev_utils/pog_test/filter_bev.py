import json
import pandas as pd

coco_json_path = "/scratch1/rsawahn/PeopleOnGrass/instances_val.json"
coco_json_path_out = "/scratch1/rsawahn/PeopleOnGrass/instances_val_bev.json"

csv_file_path = '/scratch1/rsawahn/PeopleOnGrass/meta.csv'
pitch_column_name = 'gimbal_pitch(degrees)'

df = pd.read_csv(csv_file_path)
pitch_90 = df[(df[pitch_column_name] <= -85) & (df[pitch_column_name] >= -90)]
bev_files = pitch_90["image_name"].tolist()
bev_ids = [int(x.split(".")[0]) for x in bev_files]

with open(coco_json_path, 'r') as f:
    coco = json.load(f)

coco["images"] = [img for img in coco["images"] if img["id"] in bev_ids]
coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in bev_ids]
print(len(coco["images"]))

with open(coco_json_path_out, 'w') as f:
    json.dump(coco, f)