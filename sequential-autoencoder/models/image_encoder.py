import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights


class ImageEncoder(nn.Module):

    def __init__(self, model_type='resnet18', embedding_size=512, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.pretrained = pretrained
        if model_type == 'resnet18':
            self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.embedding_size)
        elif model_type == 'resnet50':
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.embedding_size)
        elif model_type == 'vit':
            self.model = torchvision.models.vision_transformer.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.embedding_size)
        else:
            raise ValueError('Invalid model type: {}'.format(model_type))


        print(model_type, "- TOTAL PARAMETERS:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    
    def forward(self, x):
        x = self.model(x)
        # TODO: Check if this is necessary
        x = x.view(x.size(0), -1)
        return x
