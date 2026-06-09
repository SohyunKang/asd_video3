import pandas as pd
import torch
from torch.utils.data import Dataset

from learn.utils import temporal_iou, get_video_duration, read_clip
import numpy as np

LABEL_MAP = {
    "non_eye_contact": 0,
    "eye_contact": 1
}

class PreprocessedClipDataset(Dataset):
    def __init__(self, clip_df):
        self.clip_df = clip_df.reset_index(drop=True)

    def __len__(self):
        return len(self.clip_df)

    def __getitem__(self, idx):
        row = self.clip_df.iloc[idx]

        frames = np.load(row["npy_path"]).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]

        label = 1 if row["label"] == "eye_contact" else 0
        label = torch.tensor(label, dtype=torch.long)

        return frames, label

def build_clip_table(
    label_df,
    split,
    clip_duration=2.0,
    stride=0.5,
    iou_label_threshold=0.5
):
    if split is None:
        split_df = label_df.copy()
        split_name = "all"
    else:
        split_df = label_df[label_df["split"] == split].copy()
        split_name = split

    rows = []

    videos = split_df[
        ["patient_id", "video_id", "video_path"]
    ].drop_duplicates()

    print(f"[INFO] {split_name} videos: {len(videos)}")

    for _, video in videos.iterrows():
        duration = get_video_duration(video["video_path"])

        event_df = split_df[
            (split_df["video_id"] == video["video_id"])
            & (split_df["label"] == "eye_contact")
        ]

        t = 0.0

        while t + clip_duration <= duration:
            clip_start = t
            clip_end = t + clip_duration

            max_iou = 0.0

            for _, event in event_df.iterrows():
                iou = temporal_iou(
                    clip_start,
                    clip_end,
                    event["start_time"],
                    event["end_time"]
                )
                max_iou = max(max_iou, iou)

            label = (
                "eye_contact"
                if max_iou >= iou_label_threshold
                else "non_eye_contact"
            )

            rows.append({
                "patient_id": video["patient_id"],
                "video_id": video["video_id"],
                "video_path": video["video_path"],
                "clip_start": clip_start,
                "clip_end": clip_end,
                "label": label,
                "target": LABEL_MAP[label]
            })

            t += stride

    clip_df = pd.DataFrame(rows)

    print(f"[INFO] {split_name} clips: {len(clip_df)}")
    print(clip_df["label"].value_counts())

    return clip_df


class VideoClipDataset(Dataset):
    def __init__(
        self,
        clip_df,
        clip_duration=2.0,
        num_frames=16,
        image_size=224
    ):
        self.clip_df = clip_df.reset_index(drop=True)
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.clip_df)

    def __getitem__(self, idx):
        row = self.clip_df.iloc[idx]

        frames = read_clip(
            video_path=row["video_path"],
            start_time=row["clip_start"],
            clip_duration=self.clip_duration,
            num_frames=self.num_frames,
            image_size=self.image_size
        )

        # T, H, W, C → C, T, H, W
        clip = torch.tensor(frames).permute(3, 0, 1, 2).float()

        label = torch.tensor(row["target"], dtype=torch.long)

        return clip, label