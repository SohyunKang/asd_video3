import torch
import pandas as pd
from tqdm import tqdm

from utils import read_clip, get_video_duration


@torch.no_grad()
def predict_video_segments(
    model,
    video_path,
    device,

    clip_duration=2.0,
    stride=0.5,
    num_frames=16,
    image_size=224,

    crop_mode="center",
    crop_ratio=0.8,
    person_margin=0.2,

    threshold=0.5,
    merge_gap=0,
):
    model.eval()

    duration = get_video_duration(video_path)

    rows = []

    t = 0.0
    total_steps = int((duration - clip_duration) / stride) + 1

    pbar = tqdm(total=total_steps, desc="Predict")

    while t + clip_duration <= duration:
        frames = read_clip(
            video_path=video_path,
            start_time=t,
            clip_duration=clip_duration,
            num_frames=num_frames,
            image_size=image_size,
            crop_mode=crop_mode,
            crop_ratio=crop_ratio,
            person_margin=person_margin,
        )

        clip = torch.from_numpy(frames).float()
        clip = clip.permute(3, 0, 1, 2)
        clip = clip.unsqueeze(0).to(device)

        logits = model(clip)

        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(prob >= threshold)

        rows.append({
            "start_time": t,
            "end_time": t + clip_duration,
            "prob": prob,
            "pred": pred,
        })

        t += stride
        pbar.update(1)

    pbar.close()

    pred_df = pd.DataFrame(rows)

    segments = []
    current_start = None
    current_end = None

    for _, row in pred_df.iterrows():
        if row["pred"] == 1:
            if current_start is None:
                current_start = row["start_time"]
                current_end = row["end_time"]
            else:
                gap = row["start_time"] - current_end

                if gap <= merge_gap:
                    current_end = row["end_time"]
                else:
                    segments.append({
                        "start_time": current_start,
                        "end_time": current_end,
                    })

                    current_start = row["start_time"]
                    current_end = row["end_time"]

    if current_start is not None:
        segments.append({
            "start_time": current_start,
            "end_time": current_end,
        })

    segment_df = pd.DataFrame(segments)

    return pred_df, segment_df