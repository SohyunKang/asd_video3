import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn.utils import build_label_table_from_jsons, read_clip_stable_person
from learn.dataset import build_clip_table

def save_patient_montage(patient_id, rows, montage_dir, max_show=16):
    frames_for_montage = []

    rows = rows[:max_show]

    for row in rows:
        npy_path = row["npy_path"]

        if not os.path.exists(npy_path):
            continue

        arr = np.load(npy_path)
        frame = arr[len(arr) // 2]

        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames_for_montage.append(frame_bgr)

    if len(frames_for_montage) == 0:
        return

    montage = np.concatenate(frames_for_montage, axis=1)

    montage_path = os.path.join(
        montage_dir,
        f"{patient_id}_montage.jpg"
    )

    cv2.imwrite(montage_path, montage)

class CFG:
    json_root = "./data"
    video_root = "/storage/sohyunkang/video_data"

    clip_duration = 1.0
    stride = 0.5
    num_frames = 8
    image_size = 224

    crop_mode = "person"   # "none", "center", "person"
    crop_ratio = 0.8
    person_margin = 0.2

    iou_label_threshold = 0.5

    min_event_duration_sec = 0.3
    max_gap_sec = 0.2

    csv_save_every = 100

    max_clips_per_patient = 32

    out_dir = f"/storage/sohyunkang/renew_yolol_preprocessed_clips_{crop_mode}_{clip_duration}_{num_frames}_{stride}_{max_clips_per_patient}"
    
    save_patient_montage = True


os.makedirs(CFG.out_dir, exist_ok=True)
montage_dir = os.path.join(CFG.out_dir, "patient_montages")
os.makedirs(montage_dir, exist_ok=True)

config_dict = {
    "json_root": CFG.json_root,
    "video_root": CFG.video_root,
    "clip_duration": CFG.clip_duration,
    "stride": CFG.stride,
    "num_frames": CFG.num_frames,
    "image_size": CFG.image_size,
    "crop_mode": CFG.crop_mode,
    "crop_ratio": CFG.crop_ratio,
    "person_margin": CFG.person_margin,
    "iou_label_threshold": CFG.iou_label_threshold,
    "min_event_duration_sec": CFG.min_event_duration_sec,
    "max_gap_sec": CFG.max_gap_sec,
}

config_path = os.path.join(CFG.out_dir, "preprocess_config.json")
with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=4)

print(f"[INFO] Saved config: {config_path}")

csv_path = (
    f"./preprocessing/results/"
    f"renew_yolol_preprocessed_clips_"
    f"{CFG.crop_mode}_{CFG.clip_duration}_{CFG.num_frames}_"
    f"{CFG.stride}_{CFG.max_clips_per_patient}.csv"
)

# 시작 부분
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    existing_df["patient_id"] = existing_df["patient_id"].astype(str)
    existing_keys = set(
        zip(
            existing_df["patient_id"],
            existing_df["clip_start"],
            existing_df["clip_end"]
        )
    )
    processed_rows = existing_df.to_dict("records")
else:
    existing_df = pd.DataFrame()
    existing_keys = set()
    processed_rows = []
    

label_df = build_label_table_from_jsons(
    json_root=CFG.json_root,
    video_root=CFG.video_root,
    min_duration_sec=CFG.min_event_duration_sec,
    max_gap_sec=CFG.max_gap_sec,
)

# annotation에는 있는데 JSON에 없는 환자도 포함
ANNOTATION_EXCEL = "./demographics/0626 호명_시간기록.xlsx"
id_col = "연구대상자ID"

anno_df = pd.read_excel(ANNOTATION_EXCEL, header=1)
anno_df.columns = anno_df.columns.astype(str).str.strip()

annotated_ids = (
    anno_df[id_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

existing_ids = set(label_df["patient_id"].astype(str).str.strip())

extra_rows = []

for pid in annotated_ids:
    if pid in existing_ids:
        continue

    video_files = sorted(Path(CFG.video_root).glob(f"*_{pid}_*.mp4*"))

    if len(video_files) == 0:
        print("[WARN] no video for annotated patient:", pid)
        continue

    video_path = video_files[0]
    video_id = video_path.name

    while video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]

    extra_rows.append({
        "patient_id": pid,
        "video_id": video_id,
        "video_path": str(video_path),
        "start_time": -1.0,
        "end_time": -1.0,
        "label": "non_eye_contact",
        "split": "all",
    })

extra_df = pd.DataFrame(extra_rows)

if len(extra_df) > 0:
    label_df = pd.concat([label_df, extra_df], ignore_index=True)

print("[INFO] added annotated patients without json:", len(extra_df))
print("[INFO] label_df patients after adding:", label_df["patient_id"].nunique())

print("[INFO] Total labels:", len(label_df))
print("[INFO] Unique patients:", label_df["patient_id"].nunique())

clip_df = build_clip_table(
    label_df=label_df,
    split=None,
    clip_duration=CFG.clip_duration,
    stride=CFG.stride,
    iou_label_threshold=CFG.iou_label_threshold,
)

print("[INFO] Total clips:", len(clip_df))
print(clip_df["label"].value_counts())

if CFG.max_clips_per_patient is not None:
    clip_df = (
        clip_df
        .sort_values(["patient_id", "clip_start"])
        .groupby("patient_id", group_keys=False)
        .head(CFG.max_clips_per_patient)
        .reset_index(drop=True)
    )

print("[INFO] Clips after patient limit:", len(clip_df))
print("[INFO] Clips per patient:")
print(clip_df.groupby("patient_id").size().describe())

patient_clip_counter = {}

saved_count = 0
skipped_count = 0


current_patient_id = None
current_patient_rows = []

for idx, row in tqdm(clip_df.iterrows(), total=len(clip_df), desc="Preprocessing"):
    patient_id = str(row["patient_id"])

    if current_patient_id is None:
        current_patient_id = patient_id

    if patient_id != current_patient_id:
        if CFG.save_patient_montage:
            save_patient_montage(
                current_patient_id,
                current_patient_rows,
                montage_dir
            )

        current_patient_id = patient_id
        current_patient_rows = []

    if patient_id not in patient_clip_counter:
        patient_clip_counter[patient_id] = 0

    clip_idx = patient_clip_counter[patient_id]
    patient_clip_counter[patient_id] += 1

    save_name = f"{patient_id}_{clip_idx:06d}.npy"
    save_path = os.path.join(CFG.out_dir, save_name)


    row_dict = row.to_dict()
    row_dict["npy_path"] = save_path

    key = (
        str(row_dict["patient_id"]),
        float(row_dict["clip_start"]),
        float(row_dict["clip_end"])
    )

    if key in existing_keys:
        skipped_count += 1
        continue

    if os.path.exists(save_path):
        skipped_count += 1
        processed_rows.append(row_dict)
        existing_keys.add(key)
        current_patient_rows.append(row_dict)

        if len(processed_rows) % CFG.csv_save_every == 0:
            temp_df = pd.DataFrame(processed_rows)
            temp_df = temp_df[temp_df["npy_path"].apply(os.path.exists)].reset_index(drop=True)
            temp_df = temp_df.drop_duplicates(
                subset=["patient_id", "clip_start", "clip_end"],
                keep="first"
            )
            temp_df.to_csv(csv_path, index=False)
            print(f"[INFO] Intermediate csv saved: {len(temp_df)} rows")

        continue

    frames = read_clip_stable_person(
        video_path=row["video_path"],
        start_time=row["clip_start"],
        clip_duration=CFG.clip_duration,
        num_frames=CFG.num_frames,
        image_size=CFG.image_size,
        crop_mode=CFG.crop_mode,
        crop_ratio=CFG.crop_ratio,
        person_margin=CFG.person_margin,
    )


    frames = (frames * 255).astype(np.uint8)
    np.save(save_path, frames)

    processed_rows.append(row_dict)
    current_patient_rows.append(row_dict)
    existing_keys.add(key)

    if len(processed_rows) % CFG.csv_save_every == 0:
        temp_df = pd.DataFrame(processed_rows)
        temp_df = temp_df[temp_df["npy_path"].apply(os.path.exists)].reset_index(drop=True)
        temp_df.to_csv(csv_path, index=False)
        print(f"[INFO] Intermediate csv saved: {len(temp_df)} rows")

    saved_count += 1

final_df = pd.DataFrame(processed_rows)

final_df = final_df[
    final_df["npy_path"].apply(os.path.exists)
].reset_index(drop=True)

final_df = final_df.drop_duplicates(
    subset=["patient_id", "clip_start", "clip_end"],
    keep="first"
).reset_index(drop=True)

final_df.to_csv(csv_path, index=False)

print(f"[INFO] Saved csv: {csv_path}")
print(f"[INFO] Newly saved npy files: {saved_count}")
print(f"[INFO] Skipped existing npy files: {skipped_count}")
print(f"[INFO] Total clips in csv: {len(final_df)}")