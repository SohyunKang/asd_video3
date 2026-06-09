import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TimeSformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained_model_name="facebook/timesformer-base-finetuned-k400",
        freeze_encoder=False
    ):
        super().__init__()

        from transformers import TimesformerModel

        self.encoder = TimesformerModel.from_pretrained(
            pretrained_model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        입력 x: [B, C, T, H, W]
        TimeSformer 입력: [B, T, C, H, W]
        """

        x = x.permute(0, 2, 1, 3, 4)

        outputs = self.encoder(pixel_values=x)

        cls_token = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_token)

        return logits


def build_model(
    model_name="simple3dcnn",
    num_classes=2,
    freeze_encoder=False
):
    if model_name == "simple3dcnn":
        return Simple3DCNN(num_classes=num_classes)

    elif model_name == "timesformer":
        return TimeSformerClassifier(
            num_classes=num_classes,
            freeze_encoder=freeze_encoder
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")