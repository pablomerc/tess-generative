import torch
import torchvision.models as models
import torch.nn as nn


# # Load pre-trained Resnet18 model

# model = models.resnet18(weights="DEFAULT")

# example = torch.randn((100, 3, 256, 256))

# with torch.no_grad():
#         logits = model(example)

# print('Logits shape', logits.shape)

# print('='*10)
# print('Going to modify the model to try to extract latents now')

# model.fc = nn.Identity()

# x = torch.randn(100,3,256,256)

# with torch.no_grad():
#     z = model(x)

# print('Shape of latent', z.shape)

model = models.resnet18(weights=None)

print(model)

model.conv1 = nn.Conv2d(
    in_channels=4,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
model.fc = nn.Identity()

print('\n', '='*10)
print(model)

# Tesing the inputs of the model
x = torch.randn(100,4,256,256)

with torch.no_grad():
    z = model(x)

print(f'Shape of z {z.shape}')
