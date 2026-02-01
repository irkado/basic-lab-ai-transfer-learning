import torch
import torch.nn as nn
from torchvision import models
from typing import List

class TransferModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = models.densenet121(weights=weights)

            feat_dim = self.model.fc.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)

            feat_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            raise ValueError("backbone must be 'densenet121' or 'efficientnet_b0'")

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
        for p in self.model.classifier.parameters():
            p.requires_grad = True

    def fine_tune_last_block(self) -> None:
        self.freeze_all()

        if self.backbone_name == "densenet121":
            for p in self.model.features.denseblock4.parameters():
                p.requires_grad = True
            for p in self.model.features.norm5.parameters():
                p.requires_grad = True
            for p in self.model.classifier.parameters():
                p.requires_grad = True
        else:  #efficientnet_b0
            for p in self.model.features[-3:].parameters():
                p.requires_grad = True
            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]