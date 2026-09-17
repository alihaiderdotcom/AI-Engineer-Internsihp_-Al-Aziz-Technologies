"""
Week 3 - Day 5 Weekly Project
Model Architecture: VisionClassifierNet
A modular, deep convolutional neural network for visual pattern recognition.
"""

import torch
import torch.nn as nn


class VisionClassifierNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, dropout_rate=0.4):
        super(VisionClassifierNet, self).__init__()

        # Conv Block 1: Input (C, H, W) -> (32, H/2, W/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2: (32, H/2, W/2) -> (64, H/4, W/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3: (64, H/4, W/4) -> (128, H/8, W/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive spatial pooling to ensure fixed output dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model):
    """Return total and trainable parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model(in_channels=1, num_classes=5, dropout_rate=0.4):
    return VisionClassifierNet(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate)
