import tensorflow as tf
import ai_edge_torch

torch_model = 


# Pass TfLite Converter quantization flags to _ai_edge_converter_flags parameter.
tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}

tfl_drq_model = ai_edge_torch.convert(
    torch_model.eval(), sample_args, _ai_edge_converter_flags=tfl_converter_flags
)