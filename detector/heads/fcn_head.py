import torch
import torch.nn as nn
import lightning as L


class FCNHead(L.LightningModule):
    def __init__(self, num_classes, in_channels, middle_channels):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes + 1  # Adding 1 for the background class
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.dropout = nn.Dropout(p=0.5)

        # Fully Convolutional Classification Head
        self.head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=int(self.in_channels * 0.5),
            kernel_size=1,
        )
        
        self.head2 = nn.Conv2d(
            in_channels=int(self.in_channels * 0.5),
            out_channels=self.middle_channels,
            kernel_size=1,
        )
        
        self.relu = nn.ReLU()
        self.logits = nn.Conv2d(
            in_channels=self.middle_channels,
            out_channels=self.num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.head(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.head2(x)
        x = self.relu(x)
        x = self.logits(x)
        return x
