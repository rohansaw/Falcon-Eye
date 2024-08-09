_base_ = ['/root/workspace/mmdeploy/configs/mmdet/_base_/base_static.py', '/root/workspace/mmdeploy/configs/_base_/backends/ncnn.py']

partition_config = dict(type='two_stage', apply_marks=True)
#backend_config = dict(precision='INT8')
#codebase_config = dict(model_type='ncnn_end2end')
onnx_config = dict(output_names=['detection_output'], input_shape=[2688,1512])