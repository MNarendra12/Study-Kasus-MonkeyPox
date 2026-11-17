import torch
from Models.mpoxModel import MpoxCNN

model = MpoxCNN(input_c=3, output=6, dropout=0.3)
x = torch.randn(32, 3, 128, 128)
out = model(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
print("Total params:", sum(p.numel() for p in model.parameters()))