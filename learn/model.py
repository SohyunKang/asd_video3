import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = nn.Sequential(
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
        x = self.encoder(x)
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
    def __init__(
        self,
        clip_encoder,
        num_classes=2,
        classifier_num_layers=0,
        classifier_hidden_dim=512,
        classifier_dropout=0.3,
    ):
        super().__init__()

        self.clip_encoder = clip_encoder
        feature_dim = clip_encoder.feature_dim

        input_dim = feature_dim * 2

        if classifier_num_layers == 0:
            self.classifier = nn.Linear(input_dim, num_classes)

        else:
            layers = []
            prev_dim = input_dim

            for _ in range(classifier_num_layers):
                layers.extend([
                    nn.Linear(prev_dim, classifier_hidden_dim),
                    nn.LayerNorm(classifier_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(classifier_dropout),
                ])
                prev_dim = classifier_hidden_dim

            layers.append(
                nn.Linear(prev_dim, num_classes)
            )

            self.classifier = nn.Sequential(*layers)

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
    video_level=False,
    classifier_num_layers=0,
    classifier_hidden_dim=512,
    classifier_dropout=0.3,
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
            num_classes=num_classes,
            classifier_num_layers=classifier_num_layers,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
        )

    return clip_model