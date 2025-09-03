"""
Custom transform to normalize data from [0,1] to [-1,1] range
This is needed for flow matching decoders that expect data in [-1,1] range
"""

import torch
from torchvision import transforms

class NormalizeToFlowRange:
    """
    Transform to convert tensor from [0,1] to [-1,1] range
    """
    def __call__(self, tensor):
        return 2.0 * tensor - 1.0

class DenormalizeFromFlowRange:
    """
    Transform to convert tensor from [-1,1] to [0,1] range
    """
    def __call__(self, tensor):
        return (tensor + 1.0) / 2.0

def get_flow_transforms():
    """
    Get transforms that output data in [-1,1] range for flow matching
    """
    return transforms.Compose([
        transforms.ToTensor(),
        NormalizeToFlowRange()
    ])

def get_visualization_transforms():
    """
    Get transforms that output data in [0,1] range for visualization
    """
    return transforms.Compose([
        transforms.ToTensor()
    ])
