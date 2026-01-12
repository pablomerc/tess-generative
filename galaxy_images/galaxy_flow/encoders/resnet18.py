import torch
import torchvision.models as models

# Load pre-trained Resnet18 model

model = models.resnet18(weights="DEFAULT")

example = torch.randn((100, 3, 256, 256))

logits = model(example)



print('Logits shape', logits.shape)