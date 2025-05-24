# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_ct = models.densenet201(weights=None)
        base_mri = models.densenet201(weights=None)
        self.ct_features = base_ct.features
        self.mri_features = base_mri.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_ct, x_mri):
        f_ct = self.ct_features(x_ct)
        f_mri = self.mri_features(x_mri)
        fused = torch.cat((f_ct, f_mri), dim=1)
        return self.classifier(fused)
