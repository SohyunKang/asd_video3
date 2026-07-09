import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn.utils import temporal_iou, get_video_duration, read_clip_stable_person

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

        patient_id = str(row["patient_id"])
        video_id = str(row["video_id"])
        clip_start = float(row["clip_start"])
        clip_end = float(row["clip_end"])

        return frames, label, patient_id, video_id, clip_start, clip_end

class VideoDiagnosisDataset(Dataset):
    def __init__(
        self,
        clip_df,
        num_clips_per_video=8,
        random_window=False
    ):
        self.video_ids = list(
            clip_df["video_id"].unique()
        )

        self.random_window = random_window
        self.video_map = {}

        for video_id, g in clip_df.groupby("video_id"):
            self.video_map[video_id] = (
                g.sort_values("clip_start")
                .reset_index(drop=True)
            )

        self.num_clips_per_video = num_clips_per_video

    def select_pseudo_centered_sparse_clips(self, g, k):
        n = len(g)

        if n <= k:
            return g

        indices = np.arange(min(k, n))

        return g.iloc[indices]

    def pad_to_fixed_length(self, g, k):
        n = len(g)

        if n >= k:
            return g

        pad_n = k - n

        pad_df = pd.concat(
            [g.iloc[[-1]]] * pad_n,
            ignore_index=True
        )

        return pd.concat(
            [g, pad_df],
            ignore_index=True
        )

    def get_selected_clip_df(self, video_id):
        g = self.video_map[video_id]

        g = self.select_pseudo_centered_sparse_clips(
            g,
            self.num_clips_per_video
        )

        g = self.pad_to_fixed_length(
            g,
            self.num_clips_per_video
        )

        return g.reset_index(drop=True)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        g = self.get_selected_clip_df(video_id)

        patient_id = str(g.iloc[0]["patient_id"])

        clips = []

        for _, row in g.iterrows():
            frames = np.load(row["npy_path"]).astype(np.float32) / 255.0
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(3, 0, 1, 2)

            clips.append(frames)

        clips = torch.stack(clips)

        label = torch.tensor(
            int(g.iloc[0]["target"]),
            dtype=torch.long
        )

        return clips, label, patient_id


class FrameNpyClipDataset(torch.utils.data.Dataset):
    def __init__(self, clip_df, num_frames=8):
        self.clip_df = clip_df.reset_index(drop=True)
        self.num_frames = num_frames
        self.cache = {}

    def _load_video(self, npy_path):
        if npy_path not in self.cache:
            self.cache[npy_path] = np.load(npy_path, mmap_mode="r")
        return self.cache[npy_path]

    def __len__(self):
        return len(self.clip_df)

    def __getitem__(self, idx):
        row = self.clip_df.iloc[idx]

        arr = self._load_video(row["video_npy_path"])

        fps = float(row["fps"])
        start_frame = int(float(row["clip_start"]) * fps)
        end_frame = int(float(row["clip_end"]) * fps)

        end_frame = min(end_frame, len(arr) - 1)

        frame_idxs = np.linspace(
            start_frame,
            end_frame,
            self.num_frames,
            endpoint=False,
            dtype=int
        )

        frame_idxs = np.clip(frame_idxs, 0, len(arr) - 1)

        clip = arr[frame_idxs]  # [T, H, W, C], uint8

        clip = torch.from_numpy(clip.copy()).float() / 255.0
        clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]

        label = torch.tensor(int(row["target"]), dtype=torch.long)

        return (
            clip,
            label,
            str(row["patient_id"]),
            str(row["video_id"]),
            float(row["clip_start"]),
            float(row["clip_end"]),
        )

class VideoFrameNpyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clip_df,
        num_clips_per_video=15,
        num_frames=8,
        random_window=False,
    ):
        self.clip_df = clip_df.reset_index(drop=True)
        self.video_ids = list(self.clip_df["video_id"].unique())
        self.num_clips_per_video = num_clips_per_video
        self.num_frames = num_frames
        self.random_window = random_window
        self.cache = {}

        self.video_map = {}

        for video_id, g in self.clip_df.groupby("video_id"):
            self.video_map[video_id] = (
                g.sort_values("clip_start")
                .reset_index(drop=True)
            )

    def _load_video(self, npy_path):
        if npy_path not in self.cache:
            self.cache[npy_path] = np.load(npy_path, mmap_mode="r")
        return self.cache[npy_path]

    def get_selected_clip_df(self, video_id):
        g = self.video_map[video_id]

        if len(g) >= self.num_clips_per_video:
            if self.random_window:
                max_start = len(g) - self.num_clips_per_video
                start = np.random.randint(0, max_start + 1)
                return g.iloc[start:start + self.num_clips_per_video].reset_index(drop=True)
            else:
                return g.head(self.num_clips_per_video).reset_index(drop=True)

        return g.reset_index(drop=True)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        g = self.get_selected_clip_df(video_id)

        clips = []

        for _, row in g.iterrows():
            arr = self._load_video(row["video_npy_path"])

            fps = float(row["fps"])
            start_frame = int(float(row["clip_start"]) * fps)
            end_frame = int(float(row["clip_end"]) * fps)
            end_frame = min(end_frame, len(arr) - 1)

            frame_idxs = np.linspace(
                start_frame,
                end_frame,
                self.num_frames,
                endpoint=False,
                dtype=int
            )

            frame_idxs = np.clip(frame_idxs, 0, len(arr) - 1)

            clip = arr[frame_idxs]  # [T,H,W,C]
            clip = torch.from_numpy(clip.copy()).float() / 255.0
            clip = clip.permute(3, 0, 1, 2)  # [C,T,H,W]

            clips.append(clip)

        # clip 개수가 부족하면 마지막 clip 반복 padding
        while len(clips) < self.num_clips_per_video:
            clips.append(clips[-1].clone())

        clips = torch.stack(clips, dim=0)  # [N,C,T,H,W]

        label = torch.tensor(int(g["target"].max()), dtype=torch.long)
        patient_id = str(g["patient_id"].iloc[0])

        return clips, label, patient_id
    
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