import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

class MpoxCNN(nn.Module):
    def __init__(self, input_c=3, output=6, dropout=0.3):
        super(MpoxCNN, self).__init__()
        self.backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, output)
        )

    def forward(self, x):
        return self.backbone(x)
