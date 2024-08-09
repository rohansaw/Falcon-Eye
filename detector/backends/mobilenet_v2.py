from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from backends.mobnet_v2_unofficial import mobilenetv2 as mobilenet_v2_unofficial
import torch
import torch.nn as nn
import lightning as L
import os

pretrained_models_dir = "pretrained_models"


class MobileNetV2(L.LightningModule):
    def __init__(self, bn_momentum, alpha=1.0):
        super(MobileNetV2, self).__init__()

        # Initialize MobileNetV2
        if alpha == 1.0:
            net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            # currently available 0.35, more downloadable here: https://github.com/d-li14/mobilenetv2.pytorch?tab=readme-ov-file
            weights_file = os.path.join(
                pretrained_models_dir, f"mobilenetv2_{alpha}.pth"
            )
            if not os.path.exists(weights_file):
                raise FileNotFoundError(
                    f"File {weights_file} not found. Please download a corresponding pretrained model"
                )
            weights = torch.load(weights_file)
            net = mobilenet_v2_unofficial(width_mult=alpha)
            net.load_state_dict(weights)

        # Apply momentum to batchnorm
        for m in net.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum

        self.mobilenet_truncated = net.features[:8]
        # self.mobilenet_truncated[7] = self.mobilenet_truncated[7].conv[:1]

    def forward(self, x):
        return self.mobilenet_truncated(x)

    def get_feature_map_shape(self, input_shape):
        dummy_input = torch.rand(1, *input_shape)
        output = self.forward(dummy_input)
        return output.shape[1:]
