import torch
import pandas as pd

from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from learn.model import build_model
from inference_utils import predict_video_segments


class CFG:
    checkpoint_path = "./experiments/best_model_preprocessed_clips_person_2.0_16.pt"

    device = "cuda"

    clip_duration = 2.0
    stride = 0.5
    num_frames = 16
    image_size = 224

    crop_mode = "center"
    crop_ratio = 0.8
    person_margin = 0.2

    threshold = 0.4


video_path = "/storage/sohyunkang/video_data/IF2001_3_1_1023101971_0.mp4.mp4"
inference_path = "./inference/results"

if not os.path.exists(inference_path):
    os.makedirs(inference_path)


model = build_model(
    model_name="timesformer",
    num_classes=2
)

checkpoint = torch.load(
    CFG.checkpoint_path,
    map_location=CFG.device
)

model.load_state_dict(checkpoint)

model = model.to(CFG.device)

pred_df, segment_df = predict_video_segments(
    model=model,
    video_path=video_path,
    device=CFG.device,

    clip_duration=CFG.clip_duration,
    stride=CFG.stride,
    num_frames=CFG.num_frames,
    image_size=CFG.image_size,

    crop_mode=CFG.crop_mode,
    crop_ratio=CFG.crop_ratio,
    person_margin=CFG.person_margin,

    threshold=CFG.threshold,
)

pred_df.to_csv(f"{inference_path}/test_clip_predictions_{video_path.split('/')[-1].split('.')[0]}.csv", index=False)
segment_df.to_csv(f"{inference_path}/test_segments_{video_path.split('/')[-1].split('.')[0]}.csv", index=False)

print(segment_df)