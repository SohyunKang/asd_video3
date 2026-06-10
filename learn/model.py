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

        self.feature_dim = 128
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return x

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


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

        self.feature_dim = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        """
        x: [B, C, T, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


class MeanMaxVideoClassifier(nn.Module):
    """
    입력:
        x: [B, N, C, T, H, W]
           B = video batch
           N = clips per video

    동작:
        clip feature 추출
        mean pooling + max pooling
        video-level classification
    """
    def __init__(self, clip_encoder, num_classes=2):
        super().__init__()

        self.clip_encoder = clip_encoder
        feature_dim = clip_encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b, n, c, t, h, w = x.shape

        x = x.view(b * n, c, t, h, w)

        clip_feat = self.clip_encoder.extract_features(x)
        clip_feat = clip_feat.view(b, n, -1)

        mean_feat = clip_feat.mean(dim=1)
        max_feat = clip_feat.max(dim=1).values

        video_feat = torch.cat(
            [mean_feat, max_feat],
            dim=1
        )

        logits = self.classifier(video_feat)

        return logits


def build_model(
    model_name="simple3dcnn",
    num_classes=2,
    freeze_encoder=False,
    video_level=False
):
    if model_name == "simple3dcnn":
        clip_model = Simple3DCNN(num_classes=num_classes)

    elif model_name == "timesformer":
        clip_model = TimeSformerClassifier(
            num_classes=num_classes,
            freeze_encoder=freeze_encoder
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if video_level:
        return MeanMaxVideoClassifier(
            clip_encoder=clip_model,
            num_classes=num_classes
        )

    return clip_model