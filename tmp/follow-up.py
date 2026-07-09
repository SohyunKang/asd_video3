import re
from pathlib import Path

import pandas as pd

import shutil

# =========================
# 경로 설정
# =========================
DIR_A = Path("/storage/sohyunkang/video_data")
DIR_B = Path("/storage/sohyunkang/ASD_VIDEO_FU")

DIAG_EXCEL = Path(
    "/home/sohyunkang/asd_video3/demographics/rpmp_검사지_result_20241219.xlsx"
)

OUT_DIR = Path("/home/sohyunkang/asd_video3/tmp/check_overlap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_OVERLAP_CSV = OUT_DIR / "overlap_video_list.csv"
OUT_SUMMARY_CSV = OUT_DIR / "overlap_group_summary.csv"

OUT_COPY_LOG_CSV = OUT_DIR / "copied_fu_files.csv"
COPY_OVERLAP_TO_VIDEO_DATA = True

# =========================
# utils
# =========================
def extract_video_id(path):
    """
    예:
    IF2001_3_1_1023092761_0.mp4
    IF2001_3_1_1023092761_0.mp4.mp4
    IF2001_3_1_1023092761_0_gazed.mp4
    """
    stem = Path(path).stem

    # .mp4.mp4 대응
    if stem.endswith(".mp4"):
        stem = Path(stem).stem

    m = re.search(r"(IF\d+_\d+_\d+_\d+_\d+)", stem)

    if m:
        return m.group(1)

    return stem


def extract_patient_id(video_id):
    """
    IF2001_3_1_1023092761_0
                ↓
          1023092761
    """
    parts = str(video_id).split("_")

    if len(parts) >= 4:
        return parts[3]

    return None


def find_col(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand

        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def collect_mp4s(folder):
    paths = []

    for pattern in ["*.mp4", "*.MP4"]:
        paths.extend(folder.glob(pattern))

    rows = []

    for p in sorted(paths):
        if p.name.startswith("._"):
            continue

        video_id = extract_video_id(p)
        patient_id = extract_patient_id(video_id)
        if not patient_id.startswith("10"):
            continue
        
        rows.append(
            {
                "video_id": video_id,
                "patient_id": patient_id,
                "filename": p.name,
                "path": str(p),
            }
        )

    return pd.DataFrame(rows)


# =========================
# 1) 두 폴더 mp4 수집
# =========================
df_a = collect_mp4s(DIR_A)
df_b = collect_mp4s(DIR_B)

print("[INFO] DIR_A files:", len(df_a))
print("[INFO] DIR_B files:", len(df_b))

ids_a = set(df_a["video_id"])
ids_b = set(df_b["video_id"])

overlap_ids = sorted(ids_a & ids_b)

print("[INFO] overlapping video_id:", len(overlap_ids))


# =========================
# 2) 겹치는 리스트 만들기
# =========================
a_sub = df_a[df_a["video_id"].isin(overlap_ids)].copy()
b_sub = df_b[df_b["video_id"].isin(overlap_ids)].copy()

overlap_df = a_sub.merge(
    b_sub,
    on=["video_id", "patient_id"],
    how="inner",
    suffixes=("_dir_a", "_dir_b"),
)

# 중복 파일명 때문에 row가 늘어날 수 있으니 video_id 기준 정렬
overlap_df = overlap_df.sort_values(["patient_id", "video_id"]).reset_index(drop=True)


# =========================
# 3) 질병유무 엑셀 붙이기
# =========================
diag_df = pd.read_excel(DIAG_EXCEL)

id_col = find_col(
    diag_df,
    [
        "연구대상자ID",
        "subject_id",
        "patient_id",
        "ID",
        "id",
    ],
)

group_col = find_col(
    diag_df,
    [
        "구분",
        "group",
        "diagnosis",
        "질병유무",
        "진단",
    ],
)

if id_col is None:
    raise ValueError(f"ID column not found in diagnosis excel: {list(diag_df.columns)}")

if group_col is None:
    raise ValueError(f"group column not found in diagnosis excel: {list(diag_df.columns)}")

diag_df = diag_df[[id_col, group_col]].copy()
diag_df.columns = ["patient_id", "group"]

diag_df["patient_id"] = diag_df["patient_id"].astype(str).str.strip()
overlap_df["patient_id"] = overlap_df["patient_id"].astype(str).str.strip()

overlap_df = overlap_df.merge(
    diag_df,
    on="patient_id",
    how="left",
)

# =========================
# 4) 환자/정상군 수 계산
# =========================
unique_patient_df = overlap_df[
    ["patient_id", "group"]
].drop_duplicates()

print("\n[INFO] overlapped unique patients:", unique_patient_df["patient_id"].nunique())
print("\n[INFO] group counts by patient:")
print(unique_patient_df["group"].value_counts(dropna=False))

print("\n[INFO] group counts by video:")
print(overlap_df["group"].value_counts(dropna=False))

# =========================
# 5) 겹치는 follow-up 영상을 video_data에 _fu 붙여서 복사
# =========================
copy_rows = []

if COPY_OVERLAP_TO_VIDEO_DATA:
    for _, row in overlap_df.iterrows():
        src_path = Path(row["path_dir_b"])  # ASD_VIDEO_FU 쪽 파일
        video_id = row["video_id"]

        if not src_path.exists():
            copy_rows.append({
                "video_id": video_id,
                "patient_id": row["patient_id"],
                "src_path": str(src_path),
                "dst_path": None,
                "status": "source_missing",
            })
            continue

        # 원본 suffix 유지: .mp4 또는 .mp4.mp4
        suffix = "".join(src_path.suffixes)

        # 예: IF2001_3_1_1023092761_0_fu.mp4
        dst_name = f"{video_id}_fu{suffix}"
        dst_path = DIR_A / dst_name

        if dst_path.exists():
            copy_rows.append({
                "video_id": video_id,
                "patient_id": row["patient_id"],
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "status": "skipped_exists",
            })
            continue

        shutil.copy2(src_path, dst_path)

        copy_rows.append({
            "video_id": video_id,
            "patient_id": row["patient_id"],
            "src_path": str(src_path),
            "dst_path": str(dst_path),
            "status": "copied",
        })

copy_df = pd.DataFrame(copy_rows)

copy_df.to_csv(
    OUT_COPY_LOG_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("\n[INFO] copied FU files:")
print(copy_df["status"].value_counts(dropna=False))
print("[DONE] saved copy log:", OUT_COPY_LOG_CSV)

# =========================
# 5) 저장
# =========================
summary_patient = (
    unique_patient_df["group"]
    .value_counts(dropna=False)
    .rename_axis("group")
    .reset_index(name="n_patients")
)

summary_video = (
    overlap_df["group"]
    .value_counts(dropna=False)
    .rename_axis("group")
    .reset_index(name="n_videos")
)

summary_df = summary_patient.merge(
    summary_video,
    on="group",
    how="outer",
)

overlap_df.to_csv(
    OUT_OVERLAP_CSV,
    index=False,
    encoding="utf-8-sig",
)

summary_df.to_csv(
    OUT_SUMMARY_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("\n[DONE] saved overlap list:", OUT_OVERLAP_CSV)
print("[DONE] saved group summary:", OUT_SUMMARY_CSV)