import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

def check_dynamic_inputs(tflite_model_path):
    # Load the TFLite model
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()

    model = schema_fb.Model.GetRootAsModel(model_data, 0)

    # Function to check if a tensor shape is dynamic
    def is_dynamic_shape(shape):
        return any(dim == -1 for dim in shape)

    # Check all tensors for dynamic shapes
    dynamic_tensors = []
    for subgraph_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_idx)
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            shape = tensor.ShapeAsNumpy()
            if is_dynamic_shape(shape):
                dynamic_tensors.append(tensor.Name().decode('utf-8'))

    # Output the results
    if dynamic_tensors:
        print("The model has dynamic shapes for the following tensors:")
        for tensor_name in dynamic_tensors:
            print(tensor_name)
    else:
        print("The model does not have any dynamic tensors.")

# Path to your TFLite model
tflite_model_path = '/home/rsawahn/thesis/tiny-od-on-edge/dev_utils/model_export/modified_model.onnx2.tf2-13_INT8.tflite'

# Check for dynamic inputs
check_dynamic_inputs(tflite_model_path)
