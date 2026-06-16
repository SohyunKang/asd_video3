import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn.dataset import VideoDiagnosisDataset

CLIP_CSV = "./preprocessing/results/preprocessed_clips_person_1.0_8.csv"
PATIENT_CSV = "/home/sohyunkang/asd_video3/experiments/diagnosis_labels_timesformer_20260615_144530/val_predictions.csv"

OUT_DIR = Path("./diagnosis_input_check")
OUT_DIR.mkdir(exist_ok=True)

clip_df = pd.read_csv(CLIP_CSV)
patient_df = pd.read_csv(PATIENT_CSV)

patient_df["patient_id"] = patient_df["patient_id"].astype(str).str.strip()

pred_map = (
    patient_df
    .drop_duplicates(subset=["patient_id"])
    .set_index("patient_id")[["true", "pred", "prob"]]
    .to_dict("index")
)

target_patients = set(pred_map.keys())

dataset = VideoDiagnosisDataset(
    clip_df,
    num_clips_per_video=8,
    random_window=False,
)

for video_id, g in dataset.video_map.items():
    patient_id = str(g.iloc[0]["patient_id"]).strip()

    if patient_id not in target_patients:
        continue

    selected = dataset.select_pseudo_centered_sparse_clips(
        g,
        dataset.num_clips_per_video
    )

    selected = dataset.pad_to_fixed_length(
        selected,
        dataset.num_clips_per_video
    )

    true = int(pred_map[patient_id]["true"])
    pred = int(pred_map[patient_id]["pred"])
    prob = float(pred_map[patient_id]["prob"])

    montage_frames = []

    for _, row in selected.iterrows():
        arr = np.load(row["npy_path"])
        center_frame = arr[len(arr) // 2]

        montage_frames.append(
            cv2.cvtColor(
                center_frame,
                cv2.COLOR_RGB2BGR
            )
        )

    montage = np.concatenate(
        montage_frames,
        axis=1
    )

    safe_video_id = str(video_id).replace("/", "_")
    save_name = (
        f"{patient_id}_{safe_video_id}_"
        f"true{true}_pred{pred}_prob{prob:.3f}.jpg"
    )

    cv2.imwrite(
        str(OUT_DIR / save_name),
        montage
    )

    print(
        save_name,
        list(zip(selected["clip_start"], selected["clip_end"]))
    )

print("done")