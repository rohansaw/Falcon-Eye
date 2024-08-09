import json
import os

in_files = [
    "/scratch1/rsawahn/sds/annotations/instances_train.json",
    "/scratch1/rsawahn/sds/annotations/instances_val.json",
]


for in_file in in_files:
    with open(in_file, 'r') as json_file:
        data = json.load(json_file)

    for annotation in data['annotations']:
        annotation['iscrowd'] = 0

    out_file = os.path.join(os.path.split(in_file)[0], os.path.splitext(os.path.split(in_file)[1])[0] + "2.json")
    with open(out_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)