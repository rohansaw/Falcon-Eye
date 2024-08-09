#!/bin/bash

# Define the array of parameter combinations
params=(
  #"/models/results_delab/results/ssd_lite/sw_sliced/best_coco_bbox_mAP_epoch_99.pth /models/results_delab/results/ssd_lite/sw_sliced/ssd_lite_sw_sliced.py"
  #"/models/results_delab/results/ssd_lite/sds_sliced/best_coco_bbox_mAP_epoch_18.pth /models/results_delab/results/ssd_lite/sds_sliced/ssd_lite_sds_sliced.py"
  "/models/results_delab/results/faster_rcnn_resnet101/sw_sliced/best_coco_bbox_mAP_epoch_21.pth /models/results_delab/results/faster_rcnn_resnet101/sw_sliced/faster_rcnn_resnet101_sw_sliced.py"
  "/models/results_delab/results/faster_rcnn_resnet101/sds_sliced/best_coco_bbox_mAP_epoch_12.pth /models/results_delab/results/faster_rcnn_resnet101/sds_sliced/faster_rcnn_resnet101_sds_sliced.py"
  "/models/results_delab/results/detecto_rs_cascade_rcnn/sw_sliced/best_coco_bbox_mAP_epoch_49.pth /models/results_delab/results/detecto_rs_cascade_rcnn/sw_sliced/detecto_rs_cascade_rcnn_sw_sliced.py"
  "/models/results_delab/results/detecto_rs_cascade_rcnn/sds_sliced/best_coco_bbox_mAP_epoch_21.pth /models/results_delab/results/detecto_rs_cascade_rcnn/sds_sliced/detecto_rs_cascade_rcnn_sds_sliced.py"
)

# Directory where the Python script is located
script_directory="/mmdetection"
python_script="tools/test.py"

# Loop through each parameter combination
for param in "${params[@]}"; do
  # Extract the parameters
  IFS=' ' read -r param1 param2 <<< "$param"
  
  # Navigate to the script directory
  cd "$script_directory" || exit

  # Define the dataset name based on the second parameter (path filename)
  dataset_name=$(basename "$param2" .py)  # Remove the .py extension for clarity

  # get all outfile_prefix values from $param2 and create directories for them
    outfile_prefixes=$(grep -oP 'outfile_prefix=".*?"' "$param2" | cut -d'"' -f2)
    for outfile_prefix in $outfile_prefixes; do
      mkdir -p $outfile_prefix
    done

#   # Debug: Show current parameters
#   echo "Processing with parameters:"
#   echo "Checkpoint: $param1"
#   echo "Config: $param2"
#   echo "Dataset name: $dataset_name"

#   # Update or add outfile_prefix in val_evaluator
#   if grep -q "val_evaluator = dict(" "$param2"; then
#     sed -i "/val_evaluator = dict(/,/}/ {
#       /outfile_prefix/ c\    outfile_prefix=\"/data/results_coco/$dataset_name\",
#       t
#       /}/ i\    outfile_prefix=\"/data/results_coco/$dataset_name\",
#     }" "$param2"
#   fi

#   # Update or add outfile_prefix in test_evaluator
#   if grep -q "test_evaluator = dict(" "$param2"; then
#     sed -i "/test_evaluator = dict(/,/}/ {
#       /outfile_prefix/ c\    outfile_prefix=\"/data/results_coco/$dataset_name\",
#       t
#       /}/ i\    outfile_prefix=\"/data/results_coco/$dataset_name\",
#     }" "$param2"
#   fi

#   # Debug: Display modified config
#   echo "Modified config for val_evaluator:"
#   grep -A 10 "val_evaluator = dict(" "$param2"
#   echo "Modified config for test_evaluator:"
#   grep -A 10 "test_evaluator = dict(" "$param2"

  # Run the Python script with the parameters
  python3 "$python_script" "$param2" "$param1"

  # Debug: Check for the expected output
  output_path="/data/results_coco/$dataset_name"
  echo "Checking for output in $output_path"
  ls -l "$output_path"
done
