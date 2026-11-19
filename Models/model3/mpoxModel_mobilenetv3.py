import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

class MpoxCNN(nn.Module):
    def __init__(self, input_c=3, output=6, dropout=0.3):
        super(MpoxCNN, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=dropout)
        self.backbone.classifier[3] = nn.Linear(in_features, output)

    def forward(self, x):
        return self.backbone(x)
