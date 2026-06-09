import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import build_label_table_from_jsons, read_clip
from dataset import build_clip_table


class CFG:
    json_root = "./data"
    video_root = "./video_data"

    clip_duration = 2.0
    stride = 0.5
    num_frames = 16
    image_size = 224

    crop_mode = "none"   # "none", "center", "person"
    crop_ratio = 0.8
    person_margin = 0.2

    iou_label_threshold = 0.5

    min_event_duration_sec = 0.3
    max_gap_sec = 0.2

    save_debug_images = True
    num_debug_images = 20

    csv_save_every = 100

    out_dir = f"/storage/sohyunkang/preprocessed_clips_{crop_mode}_{clip_duration}_{num_frames}"


os.makedirs(CFG.out_dir, exist_ok=True)

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

csv_path = f"preprocessed_clips_{CFG.crop_mode}_{CFG.clip_duration}_{CFG.num_frames}.csv"


processed_rows = []

debug_dir = os.path.join(CFG.out_dir, "debug_images")
os.makedirs(debug_dir, exist_ok=True)

label_df = build_label_table_from_jsons(
    json_root=CFG.json_root,
    video_root=CFG.video_root,
    min_duration_sec=CFG.min_event_duration_sec,
    max_gap_sec=CFG.max_gap_sec,
)

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

npy_paths = []
patient_clip_counter = {}

saved_count = 0
skipped_count = 0
saved_debug_count = 0

for idx, row in tqdm(clip_df.iterrows(), total=len(clip_df), desc="Preprocessing"):
    patient_id = str(row["patient_id"])

    if patient_id not in patient_clip_counter:
        patient_clip_counter[patient_id] = 0

    clip_idx = patient_clip_counter[patient_id]
    patient_clip_counter[patient_id] += 1

    save_name = f"{patient_id}_{clip_idx:06d}.npy"
    save_path = os.path.join(CFG.out_dir, save_name)

    npy_paths.append(save_path)

    row_dict = row.to_dict()
    row_dict["npy_path"] = save_path

    if os.path.exists(save_path):
        skipped_count += 1
        processed_rows.append(row_dict)

        if len(processed_rows) % CFG.csv_save_every == 0:
            temp_df = pd.DataFrame(processed_rows)
            temp_df = temp_df[temp_df["npy_path"].apply(os.path.exists)].reset_index(drop=True)
            temp_df.to_csv(csv_path, index=False)
            print(f"[INFO] Intermediate csv saved: {len(temp_df)} rows")

        continue

    frames = read_clip(
        video_path=row["video_path"],
        start_time=row["clip_start"],
        clip_duration=CFG.clip_duration,
        num_frames=CFG.num_frames,
        image_size=CFG.image_size,
        crop_mode=CFG.crop_mode,
        crop_ratio=CFG.crop_ratio,
        person_margin=CFG.person_margin,
    )

    if CFG.save_debug_images and saved_debug_count < CFG.num_debug_images:
        debug_frame = frames[0]
        debug_frame = (debug_frame * 255).astype(np.uint8)
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)

        debug_path = os.path.join(
            debug_dir,
            save_name.replace(".npy", ".jpg")
        )

        cv2.imwrite(debug_path, debug_frame)
        saved_debug_count += 1

    frames = (frames * 255).astype(np.uint8)
    np.save(save_path, frames)

    processed_rows.append(row_dict)

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

final_df.to_csv(csv_path, index=False)

print(f"[INFO] Saved csv: {csv_path}")
print(f"[INFO] Newly saved npy files: {saved_count}")
print(f"[INFO] Skipped existing npy files: {skipped_count}")
print(f"[INFO] Total clips in csv: {len(final_df)}")