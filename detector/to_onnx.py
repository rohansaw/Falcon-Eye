from model import Detector
import lightning as L
import torch
from model import Detector
from heads.fcn_head import FCNHead
from backends.mobileone import mobileone, reparameterize_model
from backends.mobilenet_v2 import MobileNetV2
from attention.sim_am import SimAM
import time
import wandb
import os

# Checkpoints to export:
# SIMAM-SW: searchwing/my-detector/model-wuq1ialu:v41
# MobNet-SW: searchwing/my-detector/model-t4507ntk:v32
# MobileOne-SW: searchwing/my-detector/model-djlq0yut:v30
# MobileOne-SW no Rep: searchwing/my-detector/model-djlq0yut:v30
# Concat Alt: searchwing/my-detector/model-btqk55mr:v32
# Modulate Alt: searchwing/my-detector/model-p7ns67a1:v16

# Final SW
# Final SDS

run = wandb.init()

model_name = "searchwing/my-detector/model-djlq0yut:v30"
artifact = run.use_artifact(model_name, type='model')
artifact_dir = artifact.download()
checkpoint = os.path.join(artifact_dir, "model.ckpt")


input_shape = (3, 512, 512)
mean = [0.2761, 0.4251, 0.5644]
std = [0.2060, 0.1864, 0.2218]
num_classes = 1


#backend = MobileNetV2(bn_momentum=0.9)

backend = mobileone(variant="s0")
backend.truncate(2)
# checkpoint_backend = torch.load("pretrained_models/mobileone_s0_unfused.pth.tar",  map_location=torch.device('cpu'))
# backend.load_state_dict(checkpoint_backend)
# backend.truncate(2)


feat_map_shape = backend.get_feature_map_shape(input_shape)
head = FCNHead(
    num_classes=num_classes, in_channels=feat_map_shape[0], middle_channels=32
)

attention = None
#attention = torch.nn.MultiheadAttention(feat_map_shape[0], 4, batch_first=True)
#attention = SimAM()



detector = (
    Detector.load_from_checkpoint(checkpoint, head=head, backend=backend, attention=attention, map_location=torch.device('cpu'))
    .type(torch.FloatTensor)
)

# detector = Detector(
#     head=head,
#     backend=backend,
#     pos_weight=10,
#     learning_rate=0.001,
#     scale_factor=feat_map_shape[1] / input_shape[1],
#     attention=attention,
#     loss_type="bce",
# )

detector.eval()
detector.head.eval()

detector.backend.eval()      
#detector.backend = reparameterize_model(detector.backend)


#input_sample = {"x": torch.randn((1, 3, 512, 512)), "altitudes": torch.randn(1)}
input_sample = torch.randn((1, 3, 512, 512))
# onnx_backend = torch.onnx.dynamo_export(backend, input_sample
# onnx_backend.save("backend_trunc_512.onnx")

onnx_model = detector.to_onnx("/data/results/quantization/mobOne_notReparamterized.onnx", input_sample, export_params=True)