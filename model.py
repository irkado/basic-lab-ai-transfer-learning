import torch
import torch.nn as nn
from torchvision import models
from typing import List

class TransferModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)

            feat_dim = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )

        elif backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)

            feat_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            raise ValueError("backbone must be 'resnet18' or 'mobilenet_v2'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def train_head_only(self) -> None:
        self.freeze_all()
        head = self.model.fc if self.backbone_name == "resnet18" else self.model.classifier
        for p in head.parameters():
            p.requires_grad = True

    def fine_tune_last_block(self) -> None:
        self.freeze_all()
        if self.backbone_name == "resnet18":
            for p in self.model.layer4.parameters():
                p.requires_grad = True
            for p in self.model.fc.parameters():
                p.requires_grad = True
        else:
            for p in self.model.features[-3:].parameters():
                p.requires_grad = True
            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]