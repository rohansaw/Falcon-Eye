import torch
from model import Detector
from heads.fcn_head import FCNHead
from backends.mobileone import mobileone, reparameterize_model
import time

num_classes = 1
input_shape = (3, 2688, 1536)
device = "cuda:0"
checkpoint = "/data/artifacts/model-uysyrv59:v48/model.ckpt"


backend = mobileone(variant="s0")
feat_map_shape = backend.get_feature_map_shape(input_shape)
head = FCNHead(
    num_classes=num_classes, in_channels=feat_map_shape[0], middle_channels=32
)


detector = (
    Detector.load_from_checkpoint(checkpoint, head=head, backend=backend)
    .type(torch.FloatTensor)
    .to(device)
)


pytorch_total_params = sum(p.numel() for p in detector.parameters())
print("Total number of parameters: ", pytorch_total_params)
print("Total number of parameters in head: ", sum(p.numel() for p in detector.head.parameters()))
print("Total number of parameters in backend: ", sum(p.numel() for p in detector.backend.parameters()))

detector.eval()
detector.head.eval()
detector.backend.eval()      
detector.backend = reparameterize_model(detector.backend)



times = []
for i in range(110):
    sample = torch.rand(input_shape).to(device)
    torch.cuda.synchronize(device)
    start_time = time.time()
    res = detector(sample)
    torch.cuda.synchronize(device)
    end_time = time.time()
    if i > 10:
        times.append(end_time - start_time)
    print("Inference time: ", end_time - start_time)

print("AVG Inference time: ", sum(times)/len(times))