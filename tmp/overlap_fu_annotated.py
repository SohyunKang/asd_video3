import re
from pathlib import Path

import pandas as pd


# ==========================
# 설정
# ==========================
CSV_PATH = "/home/sohyunkang/asd_video3/experiments/annotated_labels_timesformer_20260706_181008/clip_df_with_folds.csv"   # clip csv

FU_MONTAGE_DIR = "/storage/sohyunkang/preprocessed_video_frames_person_track_landscape_224/montages/fu"

OUT_CSV = "baseline_csv_patients_with_fu_montage.csv"


def extract_patient_id_from_video_id(video_id):
    parts = str(video_id).split("_")
    if len(parts) >= 4:
        return parts[3]
    return None


def extract_patient_id_from_montage_name(path):
    name = Path(path).stem.replace("_montage", "")

    # IF2001_1_1_1024032882_0_fu
    if name.endswith("_fu"):
        name = name[:-3]

    m = re.search(r"IF\d+_\d+_\d+_(\d+)_\d+$", name)
    if m:
        return m.group(1)

    return None


# =========================
# CSV baseline 환자
# =========================
df = pd.read_csv(CSV_PATH)

df["patient_id"] = df["patient_id"].astype(str).str.strip()
df["visit"] = df["visit"].astype(str).str.strip().str.lower()

baseline_patient_ids = set(
    df.loc[df["visit"] == "baseline", "patient_id"]
)

print("CSV baseline patients:", len(baseline_patient_ids))


# =========================
# FU montage 환자
# =========================
fu_montage_rows = []

for p in Path(FU_MONTAGE_DIR).glob("*_fu_montage.jpg"):
    patient_id = extract_patient_id_from_montage_name(p)

    if patient_id is None:
        print("[WARN] cannot parse:", p.name)
        continue

    fu_montage_rows.append({
        "patient_id": patient_id,
        "fu_montage_file": p.name,
        "fu_montage_path": str(p),
    })

fu_df = pd.DataFrame(fu_montage_rows)

fu_patient_ids = set(fu_df["patient_id"].astype(str))

print("FU montage patients:", len(fu_patient_ids))


# =========================
# 겹치는 환자
# =========================
overlap_patient_ids = sorted(
    baseline_patient_ids & fu_patient_ids
)

print("Baseline CSV patients with FU montage:", len(overlap_patient_ids))

overlap_df = pd.DataFrame({
    "patient_id": overlap_patient_ids
})

overlap_df = overlap_df.merge(
    fu_df,
    on="patient_id",
    how="left"
)

overlap_df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", OUT_CSV)
print(overlap_df)