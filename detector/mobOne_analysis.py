import torch
from backends.mobileone import mobileone, reparameterize_model
from torchsummary import summary
from heads.fcn_head import FCNHead
from model import Detector

# import onnx
import time
import onnxruntime as ort
from torch.profiler import profile, record_function, ProfilerActivity

backend = mobileone(variant="s0")
checkpoint_backend = torch.load(
    "pretrained_models/mobileone_s0_unfused.pth.tar", map_location="cpu"
)
backend.load_state_dict(checkpoint_backend)
backend.eval()

# print number of params in backend
print("Number of params in backend: ", sum(p.numel() for p in backend.parameters()))
backend_reparam = reparameterize_model(backend)

# print number of params in backend
print(
    "Number of params in backend rep: ",
    sum(p.numel() for p in backend_reparam.parameters()),
)


input_shape = (3, 2688, 1536)
# summary(backend_reparam, input_shape)


feat_map_shape = backend.get_feature_map_shape(input_shape)
head = FCNHead(num_classes=1, in_channels=feat_map_shape[0], middle_channels=32)
scale_factor = feat_map_shape[1] / input_shape[1]

model = Detector(
    head=head,
    backend=backend_reparam,
    pos_weight=10,
    learning_rate=0.001,
    scale_factor=scale_factor,
    input_size=input_shape[1:],
)
head.eval()
model.eval()

input_sample = torch.randn(input_shape).unsqueeze(0)
onnx_program = torch.onnx.export(model, input_sample, "model_full.onnx")
# onnx_program.save("model_rep.onnx")
# onnx_model = onnx.load("model_rep.onnx")
# onnx.checker.check_model(onnx_model)


with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("model_inference"):
        model(input_sample)
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         times = []
#         for i in range(100):
#             start = time.perf_counter()
#             backend_reparam(input_sample)
#             end = time.perf_counter()
#             if i > 5:
#                 times.append(end - start)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print("Pytorch avg inference time: ", sum(times) / len(times))


# @profile
def onnx_inference():
    res_onnx = ort_session.run(None, {input_name: input_sample.cpu().numpy()})


ort_session = ort.InferenceSession("model_full.onnx", provider="CPU")
input_name = ort_session.get_inputs()[0].name
start = time.perf_counter()
onnx_inference()
end = time.perf_counter()
print("Onnx inference time: ", end - start)
