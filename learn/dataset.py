import pandas as pd
import torch
from torch.utils.data import Dataset

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


class VideoDiagnosisDataset(Dataset):
    def __init__(
        self,
        clip_df,
        num_clips_per_video=8,
        random_window=True
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

        if "pseudo_target" in g.columns:
            pseudo = g["pseudo_target"].values
            positive_idx = np.where(pseudo == 1)[0]
        else:
            positive_idx = []

        if len(positive_idx) > 0:
            # 첫 eye-contact 시점 중심
            center = int(positive_idx[0])

            # stride=0.5초 기준, ±12 clip = 약 ±6초
            offsets = np.linspace(
                -12,
                12,
                k,
                dtype=int
            )

            indices = center + offsets
            indices = np.clip(indices, 0, n - 1)

        else:
            # eye-contact가 없으면 전체 영상에서 균등 샘플링
            indices = np.linspace(
                0,
                n - 1,
                k,
                dtype=int
            )

        indices = np.unique(indices)

        # unique 때문에 k보다 줄어들면 마지막 index 반복
        if len(indices) < k:
            pad = np.full(
                k - len(indices),
                indices[-1]
            )
            indices = np.concatenate([indices, pad])

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

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        g = self.video_map[video_id]

        patient_id = str(g.iloc[0]["patient_id"])

        # pseudo eye-contact가 많은 연속 window 선택
        g = self.select_pseudo_centered_sparse_clips(
            g,
            self.num_clips_per_video
        )

        # clip 개수가 부족하면 마지막 clip 반복
        g = self.pad_to_fixed_length(
            g,
            self.num_clips_per_video
        )

        clips = []

        for _, row in g.iterrows():
            frames = np.load(
                row["npy_path"]
            ).astype(np.float32) / 255.0

            frames = torch.from_numpy(frames).float()

            frames = frames.permute(
                3, 0, 1, 2
            )  # C,T,H,W

            clips.append(frames)

        clips = torch.stack(clips)

        label = torch.tensor(
            int(g.iloc[0]["target"]),
            dtype=torch.long
        )

        return clips, label, patient_id