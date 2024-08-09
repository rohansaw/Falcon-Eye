configs = [
         {
        "data_path": "dataset_spec/sw.yaml",
        "image_size": (2688,1512),
        "batch": 8,
        "epochs": 200,
        "single_cls": True,
        "ds_name": "sw",
        "model": "yolov8n.pt",
        "workers": 8,
        "cache": True
    },{
        "data_path": "dataset_spec/sds.yaml",
        "image_size": (2688,1512),
        "batch": 8,
        "epochs": 200,
        "single_cls": False,
        "ds_name": "sds",
        "model": "yolov8n.pt",
        "cache": True
    },
    {
        "data_path": "dataset_spec/sds_sliced.yaml",
        "image_size": (512,512),
        "batch": 256,
        "epochs": 400,
        "single_cls": False,
        "ds_name": "sds_sliced",
        "model": "yolov8n.pt",
    },
    {
        "data_path": "dataset_spec/sw_sliced.yaml",
        "image_size": (512,512),
        "batch": 256,
        "epochs": 400,
        "single_cls": True,
        "ds_name": "sw_sliced",
        "model": "yolov8n.pt"
    },
    {
        "data_path": "dataset_spec/sw.yaml",
        "image_size": (2688,1512),
        "batch": 8,
        "epochs": 200,
        "single_cls": True,
        "ds_name": "sw",
        "model": "yolov8n-p2.yaml",
        "pretrained": True,
        "cache": True
    },{
        "data_path": "dataset_spec/sds.yaml",
        "image_size": (2688,1512),
        "batch": 8,
        "epochs": 200,
        "single_cls": False,
        "ds_name": "sds",
        "model": "yolov8n-p2.yaml",
        "pretrained": True,
        "cache": True
    },
    {
        "data_path": "dataset_spec/sds_sliced.yaml",
        "image_size": (512,512),
        "batch": 256,
        "epochs": 400,
        "single_cls": False,
        "ds_name": "sds_sliced",
        "model": "yolov8n-p2.yaml",
        "pretrained": True
    },
    {
        "data_path": "dataset_spec/sw_sliced.yaml",
        "image_size": (512,512),
        "batch": 256,
        "epochs": 400,
        "single_cls": True,
        "ds_name": "sw_sliced",
        "model": "yolov8n-p2.yaml",
        "pretrained": True
    }  
]

import json
# Function to split array of dictionaries into multiple JSON files
def split_dicts_to_json_files(dicts_array, output_directory):
    for i, d in enumerate(dicts_array):
        # Constructing filename for each dictionary
        filename = f"{output_directory}/dconfig_{i+1}.json"
        # Writing each dictionary to a separate JSON file
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)
        print(f"Created: {filename}")

# Assuming an output directory named 'output_jsons'
# Make sure this directory exists or adjust the script to create it if necessary
output_directory = "run_configs"
split_dicts_to_json_files(configs, output_directory)