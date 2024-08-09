# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/resnet50_data_reader.py

import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image

def _preprocess_images(images_folder: str, height: int, width: int, mean, std, size_limit=0):
    """
    Loads a batch of images including those in subdirectories and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    batch_filenames = []
    for root, dirs, files in os.walk(images_folder):
        for name in files:
            if name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                batch_filenames.append(os.path.join(root, name))
                if size_limit > 0 and len(batch_filenames) >= size_limit:
                    break
        if size_limit > 0 and len(batch_filenames) >= size_limit:
            break
        
    print(batch_filenames)

    unconcatenated_batch_data = []

    for image_filepath in batch_filenames:
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = numpy.float32(pillow_img)
        input_data = (input_data - numpy.array(mean, dtype=numpy.float32)) / numpy.array(std, dtype=numpy.float32)
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, mean, std):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, mean, std, size_limit=100,
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None