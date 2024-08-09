import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the converted TensorFlow model
model = tf.saved_model.load('ssd_sds_sliced.onnx.tf')

representative_dataset_path = "/scratch1/rsawahn/data/sds_sliced/coco/val"
image_path = os.path.join(representative_dataset_path, os.listdir(representative_dataset_path)[10])

image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
image = image.resize((512, 512))
image_data = np.array(image).astype(np.float32) / 255.0  # Normalize the image
print(image_data.shape, "IM")
mean = np.array([0.4263, 0.4856, 0.4507], dtype=np.float32)
std = np.array([0.1638, 0.1515, 0.1755], dtype=np.float32)
image_data = (image_data - mean) / std  # Normalize
image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
# change shape to NHWC
image_data = np.transpose(image_data, (0, 3, 1, 2))
image_data = image_data.astype(np.float32)

dummy_input = image_data
# Get the concrete function from the model
concrete_func = model.signatures['serving_default']

# Make a prediction using the model
output = concrete_func(tf.constant(dummy_input))

# Extract the 'dets' and 'labels' tensors
dets = output['dets'].numpy()[0]
labels = output['labels'].numpy()[0]

# Convert dummy input to image format for visualization (dummy input is in the range [0, 1])
image = np.transpose(dummy_input[0], (1, 2, 0))

# Create a plot to visualize the detections
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Loop through each detection and draw the bounding box if the confidence is above a threshold
confidence_threshold = 0.15

for i in range(dets.shape[0]):
    ymin, xmin, ymax, xmax, confidence = dets[i]
    if confidence > confidence_threshold:
        # Scale the coordinates to the image size
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        
        # Draw the bounding box
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                          edgecolor='red', facecolor='none', linewidth=2))
        
        # Draw the confidence score
        plt.text(xmin, ymin, f'{confidence:.2f}', color='red', fontsize=12, 
                 bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis('off')
plt.show()

# save the image
plt.savefig('output.png')
