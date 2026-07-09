import argparse
import json
import os
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from datetime import datetime

import threading

SKIP_CURRENT_FOLD = False
STOP_ALL = False

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (
    FrameNpyClipDataset,
    VideoFrameNpyDataset,
)
from losses import FocalLoss
from model import build_model
from metrics import compute_metrics

class CFG:
    frame_metadata_path = "/storage/sohyunkang/preprocessed_video_frames_person_track_landscape_224/video_frame_metadata.csv"

    clip_duration = 1.0
    stride = 0.5
    num_frames = 8

    clip_csv_path = None  # 이제 안 씀

    checkpoint_path = './experiments'
    model_name = "videomae"  # "3dcnn", "timesformer", "videomae"

    label_mode = "diagnosis_labels"
    # "pseudo_labels" or "annotated_labels" or "diagnosis_labels"
    video_level = True

    freeze_encoder = False

    num_classes = 2
    n_splits = 5
    # None이면 모든 fold 실행. 예: [0] 또는 [0, 2, 4]만 실행/저장
    target_folds = None
    num_clips_per_video = 15

    classifier_num_layers = 3
    classifier_hidden_dim = 512
    classifier_dropout = 0.3
    pooling = "meanmax"
    # 'meanmax' or "attention_mil"
    mil_hidden_dim = 64
    return_attention = False

    iou_label_threshold = 0.2

    seed = 42

    batch_size = 128
    num_workers = 0

    epochs = 30

    encoder_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 1e-4

    debug_mode = False
    debug_train_n = 8000
    debug_val_n = 2000

    # target_video_list_path = "./demographics/all_prefix_union.csv"
    target_video_list_path = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

# data QC - 데이터 이상 제외
EXCLUDE_PATIENT_IDS = [
        # "1023050311",
        # "1023092434",
        # "1023110861",
        # "1023111472",
        # "1024031034",
        # "1024040343",
        # "1024041272",
        # "1024041433",
        # "1024041681",
        # "1024042581",
        # "1024042906",
        # "1024042909",
        # "1024042943",
        # "1024052692",
        # "1023072931",
        # "1023110671",
        # "1023122041",
        # "1024032743",
        # "1024090941",
        # "1024110641",
        # "1024110931",
        # "1024111491",
        # "1024111984",
        # "1024112742",
        # "1124081215",
        # "1124041312",
        # "1124062612",
        # "1724061202",
        # "1223063011",
        # "1123072711",
        # "1423091451",
        # "1523111651",
        # "1723041201",
        # "1724053101",
        # "1724061202",
        # "1724071201",
        # "1724080605",
        # "1724082302",
        # "1724082306",
        # "1724082310",
        # "1123100912",
        # "1123120611",
        # "1124032311",
        # "1724072403",
        # "1124102811",
        # "1323101711",
        # "1423102481",
        # "1523101351",
        # "1023061961",
        # "1123051111",
    ]

def apply_cli_overrides():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Fold indices to run, e.g. '0', '0,2,4', or 'all'. Overrides CFG.target_folds."
    )

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--clip_duration", type=float, default=None)
    parser.add_argument("--num_clips_per_video", type=int, default=None)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--classifier_num_layers", type=int, default=None)
    parser.add_argument("--classifier_hidden_dim", type=int, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=None)

    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--iou_label_threshold", type=float, default=None)

    args = parser.parse_args()

    if args.folds is not None:
        CFG.target_folds = args.folds

    if args.model_name is not None:
        CFG.model_name = args.model_name

    if args.clip_duration is not None:
        CFG.clip_duration = args.clip_duration

    if args.num_clips_per_video is not None:
        CFG.num_clips_per_video = args.num_clips_per_video

    if args.stride is not None:
        CFG.stride = args.stride

    if args.num_frames is not None:
        CFG.num_frames = args.num_frames

    if args.batch_size is not None:
        CFG.batch_size = args.batch_size

    if args.epochs is not None:
        CFG.epochs = args.epochs

    if args.classifier_num_layers is not None:
        CFG.classifier_num_layers = args.classifier_num_layers

    if args.classifier_hidden_dim is not None:
        CFG.classifier_hidden_dim = args.classifier_hidden_dim

    if args.classifier_dropout is not None:
        CFG.classifier_dropout = args.classifier_dropout

    if args.encoder_lr is not None:
        CFG.encoder_lr = args.encoder_lr

    if args.head_lr is not None:
        CFG.head_lr = args.head_lr

    if args.weight_decay is not None:
        CFG.weight_decay = args.weight_decay

    if args.iou_label_threshold is not None:
        CFG.iou_label_threshold = args.iou_label_threshold

    if CFG.model_name == "videomae" and CFG.num_frames != 16:
        raise ValueError(
            f"VideoMAE는 num_frames=16이어야 합니다. "
            f"현재 num_frames={CFG.num_frames}"
        )


    print("[CLI CONFIG]")
    print("model_name:", CFG.model_name)
    print("clip_duration:", CFG.clip_duration)
    print("stride:", CFG.stride)
    print("num_frames:", CFG.num_frames)
    print("batch_size:", CFG.batch_size)
    print("epochs:", CFG.epochs)
    print("classifier_num_layers:", CFG.classifier_num_layers)
    print("classifier_hidden_dim:", CFG.classifier_hidden_dim)
    print("classifier_dropout:", CFG.classifier_dropout)
    print("encoder_lr:", CFG.encoder_lr)
    print("head_lr:", CFG.head_lr)
    print("weight_decay:", CFG.weight_decay)
    print("target_folds:", CFG.target_folds)

def build_clip_df_from_frame_metadata():
    meta_df = pd.read_csv(CFG.frame_metadata_path)

    meta_df = meta_df[
        meta_df["status"].isin(["saved", "skipped_exists"])
    ].copy()

    meta_df["patient_id"] = meta_df["patient_id"].astype(str).str.strip()
    meta_df["video_id"] = meta_df["video_id"].astype(str).str.strip()

    # 실제 npy가 있는 것만 사용
    meta_df = meta_df[
        meta_df["npy_path"].apply(os.path.exists)
    ].copy()

    rows = []

    for _, row in meta_df.iterrows():
        fps = pd.to_numeric(row.get("fps", np.nan), errors="coerce")
        saved_n_frames = pd.to_numeric(row.get("saved_n_frames", np.nan), errors="coerce")

        if pd.isna(saved_n_frames) or saved_n_frames <= 0:
            arr = np.load(row["npy_path"], mmap_mode="r")
            saved_n_frames = len(arr)
        else:
            saved_n_frames = int(saved_n_frames)

        if pd.isna(fps) or fps <= 0:
            video_path = row.get("video_path", None)

            if video_path is not None and os.path.exists(str(video_path)):
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

            if pd.isna(fps) or fps <= 0:
                print("[WARN] invalid fps, skip:", row.get("video_id"), row.get("npy_path"))
                continue

        max_time = saved_n_frames / fps

        start = 0.0

        while start + CFG.clip_duration <= max_time:
            end = start + CFG.clip_duration

            rows.append({
                "patient_id": str(row["patient_id"]),
                "visit": row.get("visit", "baseline"),
                "video_id": str(row["video_id"]),
                "base_video_id": row.get("base_video_id", row["video_id"]),
                "video_npy_path": row["npy_path"],
                "npy_path": row["npy_path"],
                "fps": fps,
                "saved_n_frames": saved_n_frames,
                "clip_start": start,
                "clip_end": end,
                "label": "unknown",
                "target": -1,
            })

            start += CFG.stride

    clip_df = pd.DataFrame(rows)

    print("[INFO] built clip_df from frame metadata:", len(clip_df))
    print("[INFO] videos:", clip_df["video_id"].nunique())
    print("[INFO] patients:", clip_df["patient_id"].nunique())

    return clip_df

def limit_clips_per_patient_before_split(
    clip_df,
    max_clips_per_patient=15
):
    before_n = len(clip_df)

    clip_df = (
        clip_df
        .sort_values(["patient_id", "clip_start"])
        .groupby("patient_id", group_keys=False)
        .head(max_clips_per_patient)
        .reset_index(drop=True)
    )

    print("[INFO] limit clips per patient before split:")
    print(f"{before_n} -> {len(clip_df)}")
    print("[INFO] label counts after limit:")
    print(clip_df["target"].value_counts().sort_index())

    return clip_df

def parse_time_to_sec(value):
    """
    지원:
    10.25              -> 10.25
    "0:10.25"          -> 10.25
    "S 0:10.25"        -> 10.25
    "E 0:11.30"        -> 11.30

    무시:
    "0:0:12.32"        -> None
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    value = re.sub(r"^[SsEe]\s*", "", value).strip()

    if value == "":
        return None

    # 잘못된 표기: 0:0:12.32 같은 3-part time은 무시
    if value.count(":") >= 2:
        return None

    if ":" not in value:
        try:
            return float(value)
        except ValueError:
            return None

    try:
        minutes, seconds = value.split(":")
        return float(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def parse_times_to_sec_list(value):
    """
    지원:
    "0:09.15, 0:11.25" -> [9.15, 11.25]
    """
    if pd.isna(value):
        return []

    if isinstance(value, (int, float)):
        return [float(value)]

    value = str(value).strip()

    if value == "":
        return []

    chunks = [
        v.strip()
        for v in value.split(",")
        if v.strip() != ""
    ]

    times = []

    for chunk in chunks:
        sec = parse_time_to_sec(chunk)
        if sec is not None:
            times.append(sec)

    return times

def add_diagnosis_labels_from_rpmp(
    clip_df):

    label_map = {
        "정상군": 0,
        "자폐군": 1,
    }

    excel_path = "./demographics/rpmp_검사지_result_20241219.xlsx"
    id_col = "연구대상자ID"
    group_col = "구분"

    clip_df = clip_df.copy()

    rpmp_df = pd.read_excel(excel_path)
    rpmp_df.columns = rpmp_df.columns.astype(str).str.strip()

    rpmp_df[id_col] = (
        rpmp_df[id_col]
        .astype(str)
        .str.strip()
    )

    rpmp_df[group_col] = (
        rpmp_df[group_col]
        .astype(str)
        .str.strip()
    )

    # 정상군 / 자폐군만 사용
    rpmp_df = rpmp_df[
        rpmp_df[group_col].isin(label_map.keys())
    ].copy()

    rpmp_df = (
        rpmp_df[[id_col, group_col]]
        .drop_duplicates(subset=[id_col])
    )

    clip_df["patient_id"] = (
        clip_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    rpmp_df[id_col] = (
        rpmp_df[id_col]
        .astype(str)
        .str.strip()
    )

    before_n = len(clip_df)

    clip_df = clip_df.merge(
        rpmp_df,
        left_on="patient_id",
        right_on=id_col,
        how="inner"
    )

    after_n = len(clip_df)

    clip_df["label"] = clip_df[group_col]
    clip_df["target"] = clip_df[group_col].map(label_map).astype(int)

    clip_df = clip_df.drop(columns=[id_col])

    print("[INFO] Diagnosis label mode: 정상군 (0) vs 자폐군 (1)")
    print(f"[INFO] clips: {before_n} -> {after_n} after merging with RPMP")
    print("[INFO] label counts:")
    print(clip_df["label"].value_counts())
    print("[INFO] subjects per group:")
    print(
        clip_df.groupby("label")["patient_id"]
        .nunique()
        .sort_index()
    )

    return clip_df

def normalize_visit(x):
    x = str(x).strip().lower()

    mapping = {
        "bl": "baseline",
        "fu": "fu",
    }

    return mapping.get(x, x)


def add_annotated_labels_from_excel(
    clip_df,
    label_threshold=0.5
):
    excel_path = "./demographics/0626 호명_시간기록.xlsx"
    id_col = "연구대상자ID"
    visit_col = "방문"

    start_col_idx = 6
    end_col_idx = 7

    clip_start_col = "clip_start"
    clip_end_col = "clip_end"

    clip_df = clip_df.copy()

    anno_df = pd.read_excel(excel_path, header=1)
    anno_df.columns = anno_df.columns.astype(str).str.strip()

    if id_col not in anno_df.columns:
        raise ValueError(f"annotation Excel에 {id_col} 컬럼이 없습니다.")

    if visit_col not in anno_df.columns:
        raise ValueError(
            f"annotation Excel에 {visit_col} 컬럼이 없습니다. "
            "baseline/fu 구분을 위해 '방문' 컬럼을 추가하세요."
        )

    anno_df[id_col] = (
        anno_df[id_col]
        .astype(str)
        .str.strip()
    )

    anno_df[visit_col] = anno_df[visit_col].apply(normalize_visit)

    start_col = anno_df.columns[start_col_idx]
    end_col = anno_df.columns[end_col_idx]

    print(f"[INFO] start column: {start_col}")
    print(f"[INFO] end column: {end_col}")
    print(f"[INFO] visit column: {visit_col}")
    print("[INFO] annotation visit counts:")
    print(anno_df[visit_col].value_counts(dropna=False))

    def is_negative_annotation(start_value, end_value):
        start = str(start_value).strip()
        end = str(end_value).strip()
        return start == "(-)" and end == "(-)"

    def is_valid_time_annotation(start_value, end_value):
        start_list = parse_times_to_sec_list(start_value)
        end_list = parse_times_to_sec_list(end_value)

        n = min(len(start_list), len(end_list))

        if n == 0:
            return False

        for i in range(n):
            if end_list[i] > start_list[i]:
                return True

        return False

    valid_annotation_mask = anno_df.apply(
        lambda row: (
            is_negative_annotation(row[start_col], row[end_col])
            or is_valid_time_annotation(row[start_col], row[end_col])
        ),
        axis=1
    )

    invalid_annotation_df = anno_df[
        ~valid_annotation_mask
    ].copy()

    anno_df_valid = anno_df[
        valid_annotation_mask
    ].copy()

    annotated_keys = set(
        zip(
            anno_df_valid[id_col].astype(str).str.strip(),
            anno_df_valid[visit_col].astype(str).str.strip().str.lower()
        )
    )

    print("[INFO] annotation rows total:", len(anno_df))
    print("[INFO] valid annotation rows:", len(anno_df_valid))
    print("[INFO] invalid annotation rows excluded:", len(invalid_annotation_df))
    print("[INFO] annotated patient+visit used:", len(annotated_keys))

    anno_df_valid["start_list"] = anno_df_valid[start_col].apply(parse_times_to_sec_list)
    anno_df_valid["end_list"] = anno_df_valid[end_col].apply(parse_times_to_sec_list)

    expanded_rows = []

    for _, row in anno_df_valid.iterrows():
        patient_id = str(row[id_col]).strip()
        visit = normalize_visit(row[visit_col])

        start_list = row["start_list"]
        end_list = row["end_list"]

        n = min(len(start_list), len(end_list))

        # (-), (-)인 경우: interval은 없지만 annotated_keys에는 남아있음
        for i in range(n):
            start_sec = start_list[i]
            end_sec = end_list[i]

            if end_sec > start_sec:
                expanded_rows.append({
                    id_col: patient_id,
                    visit_col: visit,
                    start_col: start_sec,
                    end_col: end_sec,
                })

    anno_interval_df = pd.DataFrame(expanded_rows)

    if "patient_id" not in clip_df.columns:
        raise ValueError("clip_df에 patient_id 컬럼이 필요합니다.")

    if "visit" not in clip_df.columns:
        raise ValueError("clip_df에 visit 컬럼이 필요합니다.")

    if clip_start_col not in clip_df.columns:
        raise ValueError(f"clip_df에 {clip_start_col} 컬럼이 없습니다.")

    if clip_end_col not in clip_df.columns:
        raise ValueError(f"clip_df에 {clip_end_col} 컬럼이 없습니다.")

    clip_df["patient_id"] = (
        clip_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    clip_df["visit"] = clip_df["visit"].apply(normalize_visit)

    clip_df[clip_start_col] = pd.to_numeric(
        clip_df[clip_start_col],
        errors="coerce"
    )

    clip_df[clip_end_col] = pd.to_numeric(
        clip_df[clip_end_col],
        errors="coerce"
    )

    before_n = len(clip_df)

    available_keys = set(
        zip(
            clip_df["patient_id"].astype(str).str.strip(),
            clip_df["visit"].astype(str).str.strip().str.lower()
        )
    )

    missing_keys = sorted(
        annotated_keys - available_keys
    )

    print("[DEBUG] available patient+visit before annotation filter:", len(available_keys))
    print("[DEBUG] annotated patient+visit:", len(annotated_keys))
    print("[DEBUG] annotated but missing in clip_df:", len(missing_keys))
    print("[DEBUG] first missing patient+visit:")
    print(missing_keys[:50])

    missing_df = pd.DataFrame(
        missing_keys,
        columns=["patient_id", "visit"]
    )

    missing_df["excluded_patient"] = missing_df["patient_id"].isin(EXCLUDE_PATIENT_IDS)

    missing_df.to_csv(
        "missing_annotated_subjects_debug.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("[DEBUG] saved: missing_annotated_subjects_debug.csv")
    print(missing_df["excluded_patient"].value_counts(dropna=False))

    clip_df["anno_key"] = list(
        zip(
            clip_df["patient_id"].astype(str).str.strip(),
            clip_df["visit"].astype(str).str.strip().str.lower()
        )
    )

    clip_df = clip_df[
        clip_df["anno_key"].isin(annotated_keys)
    ].copy()

    after_n = len(clip_df)

    print("\n[INFO] Subjects by visit after annotation filtering")

    visit_patient_count = (
        clip_df
        .groupby("visit")["patient_id"]
        .nunique()
        .sort_index()
    )

    visit_video_count = (
        clip_df
        .groupby("visit")["video_id"]
        .nunique()
        .sort_index()
    )

    visit_clip_count = (
        clip_df["visit"]
        .value_counts()
        .sort_index()
    )

    print("Patients")
    print(visit_patient_count)

    print("\nVideos")
    print(visit_video_count)

    print("\nClips")
    print(visit_clip_count)

    print("[INFO] Annotated mode: using only annotated patient+visit")
    print(f"[INFO] clips before filtering: {before_n}")
    print(f"[INFO] clips after filtering: {after_n}")
    print(f"[INFO] removed clips: {before_n - after_n}")

    if len(clip_df) == 0:
        raise ValueError(
            "annotation의 patient_id+visit와 clip_df가 매칭되지 않아 clip이 0개입니다."
        )

    anno_map = {}

    if len(anno_interval_df) > 0:
        for (subject, visit), g in anno_interval_df.groupby([id_col, visit_col]):
            subject = str(subject).strip()
            visit = normalize_visit(visit)

            anno_map[(subject, visit)] = [
                (
                    float(row[start_col]),
                    float(row[end_col])
                )
                for _, row in g.iterrows()
            ]

    targets = []

    for _, row in clip_df.iterrows():
        patient_id = str(row["patient_id"]).strip()
        visit = normalize_visit(row["visit"])

        clip_start = row[clip_start_col]
        clip_end = row[clip_end_col]

        if pd.isna(clip_start) or pd.isna(clip_end) or clip_end <= clip_start:
            targets.append(0)
            continue

        clip_duration = clip_end - clip_start
        intervals = anno_map.get((patient_id, visit), [])

        max_overlap_ratio = 0.0

        for anno_start, anno_end in intervals:
            overlap_start = max(clip_start, anno_start)
            overlap_end = min(clip_end, anno_end)

            overlap = max(0.0, overlap_end - overlap_start)
            overlap_ratio = overlap / clip_duration

            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        targets.append(
            1 if max_overlap_ratio >= label_threshold else 0
        )

    clip_df["target"] = targets

    clip_df["label"] = [
        "eye_contact" if t == 1 else "non_eye_contact"
        for t in targets
    ]

    clip_df = clip_df.drop(columns=["anno_key"])

    print("[INFO] Annotated labels applied")
    print("[INFO] label counts:")
    print(clip_df["target"].value_counts(dropna=False))

    print("[INFO] label counts by visit:")
    print(
        clip_df.groupby("visit")["target"]
        .value_counts()
        .sort_index()
    )

    return clip_df

def make_patient_folds(
    label_df,
    n_splits=5,
    random_state=42
):
    label_df = label_df.copy()

    label_df["patient_id"] = (
        label_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    patient_label_df = (
        label_df
        .groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "diagnosis_label"})
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_dfs = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(
            patient_label_df["patient_id"],
            patient_label_df["diagnosis_label"]
        ),
        start=1
    ):
        train_patients = patient_label_df.iloc[train_idx]["patient_id"]
        val_patients = patient_label_df.iloc[val_idx]["patient_id"]

        fold_df = label_df.copy()
        fold_df["split"] = None

        fold_df.loc[
            fold_df["patient_id"].isin(train_patients),
            "split"
        ] = "train"

        fold_df.loc[
            fold_df["patient_id"].isin(val_patients),
            "split"
        ] = "val"

        print(f"\n[FOLD {fold_idx}]")
        print("train patients:", len(train_patients))
        print("val patients:", len(val_patients))
        print("train target:")
        print(
            fold_df[fold_df["split"] == "train"]
            .groupby("patient_id")["target"]
            .first()
            .value_counts()
            .sort_index()
        )
        print("val target:")
        print(
            fold_df[fold_df["split"] == "val"]
            .groupby("patient_id")["target"]
            .first()
            .value_counts()
            .sort_index()
        )

        fold_dfs.append(fold_df)

    return fold_dfs

def normalize_video_id(x):
    name = str(x).strip()
    while name.lower().endswith(".mp4"):
        name = name[:-4]
    return name

def load_target_video_ids(path):
    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"target video list not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        if "video_id" in df.columns:
            values = df["video_id"].tolist()
        elif "video_file" in df.columns:
            values = df["video_file"].tolist()
        else:
            values = df.iloc[:, 0].tolist()

    else:
        with open(path, "r") as f:
            values = [
                line.strip()
                for line in f
                if line.strip() != ""
            ]

    target_video_ids = {
        normalize_video_id(v)
        for v in values
    }

    print("[INFO] target videos loaded:", len(target_video_ids))

    return target_video_ids

def extract_patient_id(video_id):
    # IF2001_1_1_1023032001_0 -> 1023032001
    parts = str(video_id).split("_")
    if len(parts) >= 5:
        return parts[3]
    return None


def load_first_call_start_map(call_excel_path):
    call_df = pd.read_excel(call_excel_path)

    call_df["video_id_norm"] = call_df["video_id"].apply(normalize_video_id)
    call_df["patient_id"] = call_df["video_id_norm"].apply(extract_patient_id)

    call_df["is_call_candidate"] = (
        call_df["is_call_candidate"]
        .astype(str)
        .str.upper()
        .eq("TRUE")
    )

    call_df["call_start"] = pd.to_numeric(
        call_df["call_start"],
        errors="coerce"
    )

    call_df = call_df[
        call_df["is_call_candidate"]
        & call_df["patient_id"].notna()
        & call_df["call_start"].notna()
    ].copy()

    first_call_df = (
        call_df
        .sort_values(["patient_id", "call_start"])
        .drop_duplicates(subset=["patient_id"], keep="first")
    )

    call_start_map = dict(
        zip(
            first_call_df["patient_id"].astype(str),
            first_call_df["call_start"].astype(float)
        )
    )

    print("[INFO] patients with call_start:", len(call_start_map))

    return call_start_map

def load_first_call_start_map_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    call_start_map = {}

    for item in data:
        video_id = str(item.get("id", "")).strip()

        patient_id = extract_patient_id(video_id)

        if patient_id is None:
            continue

        call_segments = item.get("call_segments", [])

        call_starts = []

        for seg in call_segments:
            if seg.get("callname", False):
                start = seg.get("start")

                if start is not None:
                    call_starts.append(float(start))

        if len(call_starts) == 0:
            continue

        first_call_start = min(call_starts)

        if patient_id not in call_start_map:
            call_start_map[patient_id] = first_call_start
        else:
            call_start_map[patient_id] = min(
                call_start_map[patient_id],
                first_call_start
            )

    print(
        "[INFO] patients with call_start (json):",
        len(call_start_map)
    )

    return call_start_map

def load_first_call_start_map_from_json_by_video(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    call_start_map = {}

    for item in data:
        video_id = normalize_video_id(str(item.get("id", "")).strip())

        call_segments = item.get("call_segments", [])
        call_starts = []

        for seg in call_segments:
            if seg.get("callname", False):
                start = seg.get("start")
                if start is not None:
                    call_starts.append(float(start))

        if len(call_starts) == 0:
            continue

        call_start_map[video_id] = min(call_starts)

    print("[INFO] videos with call_start json:", len(call_start_map))
    return call_start_map

#importance > 0
# → 그 clip을 제거하니 자폐군 확률이 떨어짐
# → 자폐군 판단에 기여한 clip일 가능성

# importance < 0
# → 그 clip을 제거하니 자폐군 확률이 올라감
# → 자폐군 판단을 방해하거나 정상군 쪽 정보였을 가능성
@torch.no_grad()
def save_val_clip_ablation_importance(
    model,
    val_dataset,
    save_path,
    device,
    target_class=1
):
    model.eval()

    rows = []

    for idx in range(len(val_dataset)):
        clips, label, patient_id = val_dataset[idx]

        video_id = val_dataset.video_ids[idx]
        selected_g = val_dataset.get_selected_clip_df(video_id)
        clips_batch = clips.unsqueeze(0).to(device)

        logits = model(clips_batch)
        base_prob = torch.softmax(logits, dim=1)[0, target_class].item()
        pred = int(base_prob >= 0.5)

        for clip_i in range(clips.shape[0]):
            masked = clips_batch.clone()
            masked[:, clip_i] = 0

            masked_logits = model(masked)
            masked_prob = torch.softmax(
                masked_logits,
                dim=1
            )[0, target_class].item()

            importance = base_prob - masked_prob

            row_info = selected_g.iloc[clip_i]

            rows.append({
                "patient_id": patient_id,
                "video_id": video_id,
                "true": int(label),
                "pred": pred,
                "base_prob": base_prob,

                "clip_index": clip_i,
                "clip_start": row_info["clip_start"],
                "clip_end": row_info["clip_end"],
                "npy_path": row_info["npy_path"],

                "prob_without_clip": masked_prob,
                "importance": importance,
            })

    importance_df = pd.DataFrame(rows)

    importance_df.to_csv(
        save_path,
        index=False
    )

    print(f"[INFO] Saved val clip ablation importance: {save_path}")

import cv2

def save_val_montages(
    val_dataset,
    save_dir,
    max_videos=None
):
    montage_dir = os.path.join(save_dir, "val_montages")
    os.makedirs(montage_dir, exist_ok=True)

    n = len(val_dataset)
    if max_videos is not None:
        n = min(n, max_videos)

    for idx in range(n):
        clips, label, patient_id = val_dataset[idx]
        video_id = val_dataset.video_ids[idx]

        frames_for_montage = []

        for clip_i in range(clips.shape[0]):
            clip = clips[clip_i]
            mid_t = clip.shape[1] // 2

            frame = clip[:, mid_t]
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(
                frame_bgr,
                f"{clip_i}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            frames_for_montage.append(frame_bgr)

        montage = np.concatenate(frames_for_montage, axis=1)

        save_path = os.path.join(
            montage_dir,
            f"{patient_id}_{video_id}_true{int(label)}.jpg"
        )

        cv2.imwrite(save_path, montage)

    print(f"[INFO] Saved validation montages: {montage_dir}")


def add_patient_wise_folds(
    label_df,
    n_splits=5,
    random_state=42
):
    label_df = label_df.copy()

    label_df["patient_id"] = (
        label_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    patient_label_df = (
        label_df
        .groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "patient_label"})
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    label_df["fold"] = -1

    for fold, _, val_idx in [
        (fold, train_idx, val_idx)
        for fold, (train_idx, val_idx)
        in enumerate(
            skf.split(
                patient_label_df["patient_id"],
                patient_label_df["patient_label"]
            )
        )
    ]:
        val_patients = patient_label_df.iloc[val_idx]["patient_id"]

        label_df.loc[
            label_df["patient_id"].isin(val_patients),
            "fold"
        ] = fold

    print("\n[5-FOLD QC]")
    for fold in range(n_splits):
        fold_df = label_df[label_df["fold"] == fold]

        patient_df = (
            fold_df
            .groupby("patient_id")["target"]
            .max()
        )

        print(f"\n[FOLD {fold}]")
        print("clips:", len(fold_df))
        print("patients:", fold_df["patient_id"].nunique())
        print("patient target counts:")
        print(patient_df.value_counts().sort_index())

    return label_df

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

def add_patient_group_clip_stratified_folds(
    label_df,
    n_splits=5,
    random_state=42
):
    label_df = label_df.copy().reset_index(drop=True)

    label_df["patient_id"] = (
        label_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    label_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        sgkf.split(
            X=label_df,
            y=label_df["target"],
            groups=label_df["patient_id"]
        )
    ):
        label_df.loc[val_idx, "fold"] = fold

    print("\n[5-FOLD QC: STRATIFIED GROUP K-FOLD]")
    for fold in range(n_splits):
        fold_df = label_df[label_df["fold"] == fold]

        print(f"\n[FOLD {fold}]")
        print("clips:", len(fold_df))
        print("patients:", fold_df["patient_id"].nunique())
        print("target counts:")
        print(fold_df["target"].value_counts().sort_index())
        print("target ratio:")
        print(fold_df["target"].mean())

    return label_df

def train_one_epoch(model, loader, optimizer, criterion, threshold=0.5):
    from torch.amp import autocast, GradScaler

    scaler = GradScaler()

    model.train()

    prediction_rows = []

    total_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        if len(batch) == 6:
            clips, labels, patient_ids, video_ids, clip_starts, clip_ends = batch
        elif len(batch) == 3:
            clips, labels, patient_ids = batch
            video_ids = ["unknown"] * clips.size(0)
            clip_starts = [-1.0] * clips.size(0)
            clip_ends = [-1.0] * clips.size(0)
        elif len(batch) == 2:
            clips, labels = batch
            patient_ids = ["unknown"] * clips.size(0)
            video_ids = ["unknown"] * clips.size(0)
            clip_starts = [-1.0] * clips.size(0)
            clip_ends = [-1.0] * clips.size(0)
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        if SKIP_CURRENT_FOLD:
            print("[MANUAL SKIP] Stop requested during training batch.")
            break

        clips = clips.to(CFG.device)
        labels = labels.to(CFG.device)

        # optimizer.zero_grad()

        # logits = model(clips)
        # loss = criterion(logits, labels)

        # loss.backward()
        # optimizer.step()

        optimizer.zero_grad()

        with autocast('cuda'):
            logits = model(clips)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        total_loss += loss.item() * clips.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

        running_acc = accuracy_score(
            y_true,
            y_pred,
        )

        pos_rate_pred = sum(y_pred) / len(y_pred)
        pos_rate_true = sum(y_true) / len(y_true)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{running_acc:.4f}",
            pos_pred=int(sum(y_pred)),
            pos_true=int(sum(y_true)),
            pred_rate=f"{pos_rate_pred:.3f}",
            true_rate=f"{pos_rate_true:.3f}",
        )

        for pid, vid, cs, ce, true, pred, prob in zip(
            patient_ids,
            video_ids,
            clip_starts,
            clip_ends,
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            probs.detach().cpu().numpy()
        ):
            prediction_rows.append({
                "patient_id": str(pid),
                "video_id": str(vid),
                "clip_start": float(cs),
                "clip_end": float(ce),
                "true": int(true),
                "pred": int(pred),
                "prob": float(prob),
            })

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    prediction_df = pd.DataFrame(prediction_rows)

    return metrics, prediction_df


@torch.no_grad()
def validate(model, loader, criterion, threshold=0.5):
    model.eval()

    prediction_rows = []

    total_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        if len(batch) == 6:
            clips, labels, patient_ids, video_ids, clip_starts, clip_ends = batch

        elif len(batch) == 3:
            clips, labels, patient_ids = batch
            video_ids = ["unknown"] * clips.size(0)
            clip_starts = [-1.0] * clips.size(0)
            clip_ends = [-1.0] * clips.size(0)

        elif len(batch) == 2:
            clips, labels = batch
            patient_ids = ["unknown"] * clips.size(0)
            video_ids = ["unknown"] * clips.size(0)
            clip_starts = [-1.0] * clips.size(0)
            clip_ends = [-1.0] * clips.size(0)

        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        
        clips = clips.to(CFG.device)
        labels = labels.to(CFG.device)

        logits = model(clips)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        total_loss += loss.item() * clips.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

        running_acc = accuracy_score(
            y_true,
            y_pred
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{running_acc:.4f}",
            pos_true=int(sum(y_true)),
            pos_pred=int(sum(y_pred))
        )

        for pid, vid, cs, ce, true, pred, prob in zip(
            patient_ids,
            video_ids,
            clip_starts,
            clip_ends,
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            probs.cpu().numpy()
        ):
            prediction_rows.append({
                "patient_id": str(pid),
                "video_id": str(vid),
                "clip_start": float(cs),
                "clip_end": float(ce),
                "true": int(true),
                "pred": int(pred),
                "prob": float(prob),
            })

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    prediction_df = pd.DataFrame(prediction_rows)

    return metrics, prediction_df



def save_json(obj, save_path):
    with open(save_path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def make_config_dict():
    config_dict = {}

    for k, v in CFG.__dict__.items():
        if not k.startswith("__"):
            try:
                json.dumps(v)
                config_dict[k] = v
            except TypeError:
                config_dict[k] = str(v)

    return config_dict

def build_datasets(train_clip_df, val_clip_df):
    if CFG.video_level:
        train_dataset = VideoFrameNpyDataset(
            train_clip_df,
            num_clips_per_video=CFG.num_clips_per_video,
            num_frames=CFG.num_frames
        )
        val_dataset = VideoFrameNpyDataset(
            val_clip_df,
            num_clips_per_video=CFG.num_clips_per_video,
            num_frames=CFG.num_frames
        )
    else:
        train_dataset = FrameNpyClipDataset(
            train_clip_df,
            num_frames=CFG.num_frames
        )
        val_dataset = FrameNpyClipDataset(
            val_clip_df,
            num_frames=CFG.num_frames
        )

    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )

    return train_loader, val_loader


def build_fresh_model():
    model = build_model(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        freeze_encoder=CFG.freeze_encoder,
        video_level=CFG.video_level,
        pooling=CFG.pooling,
        classifier_num_layers=CFG.classifier_num_layers,
        classifier_hidden_dim=CFG.classifier_hidden_dim,
        classifier_dropout=CFG.classifier_dropout,
        mil_hidden_dim=CFG.mil_hidden_dim,
        return_attention=CFG.return_attention,
    )

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(CFG.device)

    return model


def build_criterion(train_clip_df):
    if CFG.label_mode == "diagnosis_labels":
        video_target_df = (
            train_clip_df
            .groupby("video_id")["target"]
            .first()
            .reset_index()
        )
        counts = video_target_df["target"].value_counts().sort_index()
    else:
        counts = train_clip_df["target"].value_counts().sort_index()

    neg = int(counts.get(0, 0))
    pos = int(counts.get(1, 0))

    if pos == 0:
        raise ValueError("Positive samples are 0 in this training fold.")
    if neg == 0:
        raise ValueError("Negative samples are 0 in this training fold.")

    # pos_weight = np.sqrt(neg / pos)
    pos_weight = neg / pos

    class_weights = torch.tensor(
        [1.0, pos_weight],
        dtype=torch.float32
    ).to(CFG.device)

    print("[INFO] neg:", neg)
    print("[INFO] pos:", pos)
    print("[INFO] class weights:", class_weights)

    if CFG.label_mode == "diagnosis_labels":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    return criterion, {
        "neg": neg,
        "pos": pos,
        "pos_weight": float(pos_weight),
    }


def build_optimizer(model):
    encoder_params = []
    head_params = []

    base_model = model.module if hasattr(model, "module") else model

    for name, param in base_model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": CFG.encoder_lr},
            {"params": head_params, "lr": CFG.head_lr},
        ],
        weight_decay=CFG.weight_decay
    )

    return optimizer


def prepare_clip_dataframe():
    clip_df = build_clip_df_from_frame_metadata()
    add_qc_step("metadata", after_df=clip_df)

    clip_df["patient_id"] = (
        clip_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    before_df = clip_df.copy()

    clip_df = clip_df[
        ~clip_df["patient_id"].isin(EXCLUDE_PATIENT_IDS)
    ].copy()

    add_qc_step("exclude_patients", before_df, clip_df)

    # Calling excel에서 호명 시작 시간 불러와서 clip_df에 병합
    # CALL_PATH = "./calling/results/call_detection_all.xlsx"
    CALL_PATH = "./demographics/260707_add_new_videos.json"
    
    before_df = clip_df.copy()
    if CALL_PATH.lower().endswith(".json"):
        call_start_map = load_first_call_start_map_from_json_by_video(CALL_PATH)
    else:
        raise ValueError("현재는 JSON call_start를 video_id 기준으로 쓰도록 수정하세요.")

    clip_df["video_id_norm"] = clip_df["video_id"].apply(normalize_video_id)
    clip_df["call_start"] = clip_df["video_id_norm"].map(call_start_map)

    clip_df["has_call_start"] = clip_df["call_start"].notna()

    print("[INFO] videos with call_start in clip_df:",
        clip_df.loc[clip_df["has_call_start"], "video_id"].nunique())
    print("[INFO] videos without call_start in clip_df:",
        clip_df.loc[~clip_df["has_call_start"], "video_id"].nunique())

    print("[INFO] patient+visit with call_start:")
    print(
        clip_df[clip_df["has_call_start"]]
        .drop_duplicates(["patient_id", "visit"])
        ["visit"]
        .value_counts()
    )

    clip_df["has_call_start"] = clip_df["call_start"].notna()

    # call_start 없는 경우는 기존처럼 0초 기준 유지
    clip_df = clip_df[
        clip_df["has_call_start"]
    ].copy()


    prev_clip_df = (
        clip_df[clip_df["clip_start"] < clip_df["call_start"]]
        .sort_values(["patient_id", "visit", "clip_start"])
        .groupby("patient_id")
        .tail(1)
    )

    after_clip_df = clip_df[
        clip_df["clip_start"] >= clip_df["call_start"]
    ]

    clip_df = pd.concat(
        [after_clip_df, prev_clip_df],
        ignore_index=True
    )

    clip_df = (
        clip_df
        .drop_duplicates(subset=["video_id", "clip_start", "clip_end"])
        .sort_values(["patient_id", "clip_start"])
        .reset_index(drop=True)
    )

    clip_df["clip_start_from_call"] = (
        clip_df["clip_start"] - clip_df["call_start"]
    )

    clip_df["clip_end_from_call"] = (
        clip_df["clip_end"] - clip_df["call_start"]
    )

    add_qc_step("call_start_filter", before_df, clip_df)


    before_df = clip_df.copy()

    if CFG.label_mode == "pseudo_labels":
        print("[INFO] Using pseudo labels from clip CSV")
        print(clip_df["target"].value_counts(dropna=False))

    elif CFG.label_mode == "annotated_labels":
        clip_df = add_annotated_labels_from_excel(
            clip_df=clip_df,
            label_threshold=CFG.iou_label_threshold
        )

    elif CFG.label_mode == "diagnosis_labels":
        clip_df["pseudo_label"] = clip_df["label"].copy()
        clip_df["pseudo_target"] = clip_df["target"].copy()
        clip_df = add_diagnosis_labels_from_rpmp(clip_df=clip_df)

    else:
        raise ValueError(
            f"Unknown label_mode: {CFG.label_mode}. "
            "Use 'pseudo_labels', 'annotated_labels', or 'diagnosis_labels'."
        )
    
    
    add_qc_step("annotation_filter", before_df, clip_df)

    target_video_ids = load_target_video_ids(
        CFG.target_video_list_path
    )

    clip_df = filter_by_target_videos(
        clip_df,
        target_video_ids
    )

    before_df = clip_df.copy()
    clip_df = limit_clips_per_patient_before_split(
        clip_df,
        max_clips_per_patient=CFG.num_clips_per_video
    )
    
    add_qc_step("clip_limit", before_df, clip_df)

    print_qc_summary()
    print("\n[DATA SPLIT]")

    # clip_df = add_patient_wise_folds(
    #     clip_df,
    #     n_splits=CFG.n_splits,
    #     random_state=CFG.seed
    # )

    clip_df = add_patient_group_clip_stratified_folds(
        clip_df,
        n_splits=CFG.n_splits,
        random_state=CFG.seed
    )

    return clip_df

def filter_by_target_videos(clip_df, target_video_ids):
    if target_video_ids is None:
        return clip_df

    clip_df = clip_df.copy()

    clip_df["video_id_norm"] = clip_df["video_id"].apply(
        normalize_video_id
    )

    before_n = len(clip_df)
    before_videos = clip_df["video_id_norm"].nunique()
    before_patients = clip_df["patient_id"].nunique()

    clip_df = clip_df[
        clip_df["video_id_norm"].isin(target_video_ids)
    ].copy()

    after_n = len(clip_df)
    after_videos = clip_df["video_id_norm"].nunique()
    after_patients = clip_df["patient_id"].nunique()

    print("\n[TARGET VIDEO FILTER]")
    print("clips:", before_n, "->", after_n)
    print("videos:", before_videos, "->", after_videos)
    print("patients:", before_patients, "->", after_patients)

    missing_videos = sorted(
        target_video_ids - set(clip_df["video_id_norm"].unique())
    )

    print("[INFO] target videos not found:", len(missing_videos))
    available_video_ids = set(clip_df["video_id_norm"].unique())
    missing_videos = sorted(target_video_ids - available_video_ids)

    missing_df = pd.DataFrame({
        "video_id": missing_videos
    })

    missing_df["patient_id"] = missing_df["video_id"].apply(extract_patient_id)
    missing_df["excluded_patient"] = missing_df["patient_id"].isin(EXCLUDE_PATIENT_IDS)

    missing_df.to_csv(
        "missing_target_videos_debug.csv",
        index=False
    )

    print("[INFO] saved missing debug: missing_target_videos_debug.csv")
    print(missing_df["excluded_patient"].value_counts(dropna=False))

    if len(missing_videos) > 0:
        print("[INFO] first missing videos:")
        print(missing_videos[:20])

    clip_df = clip_df.drop(columns=["video_id_norm"])

    if len(clip_df) == 0:
        raise ValueError(
            "After filtering by target videos, clip_df is empty."
        )

    return clip_df

def split_fold_data(clip_df, fold):
    train_clip_df = clip_df[
        clip_df["fold"] != fold
    ].reset_index(drop=True)

    val_clip_df = clip_df[
        clip_df["fold"] == fold
    ].reset_index(drop=True)

    if CFG.debug_mode:
        train_clip_df = train_clip_df.sample(
            n=min(CFG.debug_train_n, len(train_clip_df)),
            random_state=CFG.seed + fold
        ).reset_index(drop=True)

        val_clip_df = val_clip_df.sample(
            n=min(CFG.debug_val_n, len(val_clip_df)),
            random_state=CFG.seed + fold
        ).reset_index(drop=True)

        print("[DEBUG] Using small subset")

    print("[INFO] train clips:", len(train_clip_df))
    print("[INFO] val clips:", len(val_clip_df))
    print("[INFO] train patients:", train_clip_df["patient_id"].nunique())
    print("[INFO] val patients:", val_clip_df["patient_id"].nunique())
    print("[INFO] train target counts:")
    print(train_clip_df["target"].value_counts().sort_index())
    print("[INFO] val target counts:")
    print(val_clip_df["target"].value_counts().sort_index())

    return train_clip_df, val_clip_df


def save_state_dict(model, save_path):
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )
    torch.save(state_dict, save_path)


def run_one_fold(clip_df, save_dir, fold):
    print(f"\n{'=' * 60}")
    print(f"[FOLD {fold + 1}/{CFG.n_splits}]")
    print(f"{'=' * 60}")

    fold_save_dir = os.path.join(save_dir, f"fold_{fold}")
    os.makedirs(fold_save_dir, exist_ok=True)

    train_clip_df, val_clip_df = split_fold_data(clip_df, fold)

    train_clip_df.to_csv(
        os.path.join(fold_save_dir, "train_clip_df.csv"),
        index=False
    )
    val_clip_df.to_csv(
        os.path.join(fold_save_dir, "val_clip_df.csv"),
        index=False
    )

    train_dataset, val_dataset = build_datasets(train_clip_df, val_clip_df)
    if CFG.video_level:
        save_val_montages(
            val_dataset=val_dataset,
            save_dir=fold_save_dir,
        )

    train_loader, val_loader = build_loaders(train_dataset, val_dataset)
    model = build_fresh_model()
    criterion, class_balance = build_criterion(train_clip_df)
    optimizer = build_optimizer(model)

    best_score = -1.0
    best_metrics = None
    epoch_rows = []

    for epoch in range(CFG.epochs):
        global SKIP_CURRENT_FOLD

        if SKIP_CURRENT_FOLD:
            print(f"[MANUAL SKIP] Fold {fold}")
            SKIP_CURRENT_FOLD = False
            break

        print(f"\n[Fold {fold} | Epoch {epoch + 1}/{CFG.epochs}]")

        train_metrics, train_pred_df = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_metrics, val_pred_df = validate(
            model,
            val_loader,
            criterion
        )

        print("[Train]", train_metrics)
        print("[Val]", val_metrics)

        epoch_row = {
            "fold": fold,
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        epoch_rows.append(epoch_row)

        if CFG.label_mode == "annotated_labels":
            current_score = val_metrics.get("f1", 0.0)
            score_name = "f1"
        else:
            current_score = val_metrics.get("auroc", 0.0)
            score_name = "auroc"

        if current_score > best_score:
            best_score = current_score

            save_state_dict(
                model,
                os.path.join(fold_save_dir, "model.pt")
            )

            best_metrics = {
                "fold": fold,
                "epoch": epoch + 1,
                "best_val_score": best_score,
                "selection_metric": score_name,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "class_balance": class_balance,
                "model_name": CFG.model_name,
                "clip_csv_path": CFG.clip_csv_path,
                "batch_size": CFG.batch_size,
                "encoder_lr": CFG.encoder_lr,
                "head_lr": CFG.head_lr,
                "epochs": CFG.epochs,
                "debug_mode": CFG.debug_mode,
            }

            save_json(
                best_metrics,
                os.path.join(fold_save_dir, "metrics.json")
            )

            train_pred_df.to_csv(
                os.path.join(fold_save_dir, "train_predictions.csv"),
                index=False
            )

            val_pred_df.to_csv(
                os.path.join(fold_save_dir, "val_predictions.csv"),
                index=False
            )

            print(
                "[INFO] Saved best model:",
                os.path.join(fold_save_dir, "model.pt")
            )
            print(
                "[INFO] Saved best metrics:",
                os.path.join(fold_save_dir, "metrics.json")
            )

            # save_val_clip_ablation_importance(
            #     model=model,
            #     val_dataset=val_dataset,
            #     save_path=os.path.join(
            #         fold_save_dir,
            #         "val_clip_ablation_importance.csv"
            #     ),
            #     device=CFG.device,
            #     target_class=1
            # )

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(
        os.path.join(fold_save_dir, "epoch_metrics.csv"),
        index=False
    )

    if best_metrics is None:
        raise RuntimeError(f"No best metrics were saved for fold {fold}.")

    return {
        "fold": fold,
        "best_epoch": best_metrics["epoch"],
        "best_val_score": best_metrics["best_val_score"],
        "selection_metric": best_metrics["selection_metric"],
        **{
            f"best_val_{k}": v
            for k, v in best_metrics["val_metrics"].items()
        },
    }


def parse_target_folds(value, n_splits):
    """
    value examples:
    - None        -> [0, 1, ..., n_splits-1]
    - 0           -> [0]
    - [0, 2, 4]   -> [0, 2, 4]
    - "0,2,4"     -> [0, 2, 4]
    - "all"       -> [0, 1, ..., n_splits-1]
    """
    if value is None:
        folds = list(range(n_splits))
    elif isinstance(value, int):
        folds = [value]
    elif isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "all", "none"}:
            folds = list(range(n_splits))
        else:
            folds = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    else:
        folds = [int(x) for x in value]

    folds = sorted(set(folds))

    invalid_folds = [fold for fold in folds if fold < 0 or fold >= n_splits]
    if invalid_folds:
        raise ValueError(
            f"Invalid target_folds: {invalid_folds}. "
            f"Valid fold index range is 0 to {n_splits - 1}."
        )

    return folds


def keyboard_listener():
    global SKIP_CURRENT_FOLD, STOP_ALL

    while True:
        cmd = input().strip().lower()

        if cmd == "s":
            SKIP_CURRENT_FOLD = True

        elif cmd == "q":
            STOP_ALL = True
            break

QC_ROWS = []

def get_unit_counts(df):
    patient_visit = (
        df[["patient_id", "visit"]]
        .drop_duplicates()
        .shape[0]
        if "visit" in df.columns else df["patient_id"].nunique()
    )

    return {
        "clips": len(df),
        "patients": df["patient_id"].nunique(),
        "patient_visit": patient_visit,
        "baseline_pv": df[df["visit"] == "baseline"][["patient_id", "visit"]].drop_duplicates().shape[0] if "visit" in df.columns else 0,
        "fu_pv": df[df["visit"] == "fu"][["patient_id", "visit"]].drop_duplicates().shape[0] if "visit" in df.columns else 0,
    }


def add_qc_step(step, before_df=None, after_df=None):
    if before_df is None:
        after = get_unit_counts(after_df)
        row = {
            "step": step,
            "before_clips": None,
            "after_clips": after["clips"],
            "dropped_clips": None,
            "before_patient_visit": None,
            "after_patient_visit": after["patient_visit"],
            "dropped_patient_visit": None,
            "after_patients": after["patients"],
            "after_baseline_pv": after["baseline_pv"],
            "after_fu_pv": after["fu_pv"],
        }
    else:
        before = get_unit_counts(before_df)
        after = get_unit_counts(after_df)

        row = {
            "step": step,
            "before_clips": before["clips"],
            "after_clips": after["clips"],
            "dropped_clips": before["clips"] - after["clips"],
            "before_patient_visit": before["patient_visit"],
            "after_patient_visit": after["patient_visit"],
            "dropped_patient_visit": before["patient_visit"] - after["patient_visit"],
            "after_patients": after["patients"],
            "after_baseline_pv": after["baseline_pv"],
            "after_fu_pv": after["fu_pv"],
        }

    QC_ROWS.append(row)


def print_qc_summary():
    qc_df = pd.DataFrame(QC_ROWS)

    print("\n" + "=" * 100)
    print("[DATA QC SUMMARY]")
    print("=" * 100)
    print(qc_df.to_string(index=False))

def main():
    
    print("[INFO] Device:", CFG.device)
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed)
    
    exp_id = (
        f"{CFG.label_mode}_"
        f"{CFG.model_name}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )

    save_dir = os.path.join(CFG.checkpoint_path, exp_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Save dir: {save_dir}")

    preprocess_dir = os.path.dirname(CFG.frame_metadata_path)

    with open(os.path.join(preprocess_dir, "preprocess_config.json"), "r") as f:
        preprocess_cfg = json.load(f)

    save_json(
        preprocess_cfg,
        os.path.join(save_dir, "preprocess_config.json")
    )

    print("[INFO] Loaded preprocess config:")
    print(preprocess_cfg)

    save_json(
        make_config_dict(),
        os.path.join(save_dir, "config.json")
    )

    clip_df = prepare_clip_dataframe()

    clip_df.to_csv(
        os.path.join(save_dir, "clip_df_with_folds.csv"),
        index=False
    )

    target_folds = parse_target_folds(
        CFG.target_folds,
        CFG.n_splits
    )

    print("[INFO] Target folds:", target_folds)

    fold_results = []

    for fold in target_folds:
        fold_result = run_one_fold(
            clip_df=clip_df,
            save_dir=save_dir,
            fold=fold
        )
        fold_results.append(fold_result)

       
        if STOP_ALL:
            break

        # fold가 끝난 뒤 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_result_df = pd.DataFrame(fold_results)
    fold_result_df.to_csv(
        os.path.join(save_dir, "cv_results.csv"),
        index=False
    )

    if CFG.label_mode == "annotated_labels":
        selection_metric = "f1"
    else:
        selection_metric = "auroc"

    summary = {
        "n_splits": CFG.n_splits,
        "target_folds": target_folds,
        "selection_metric": selection_metric,
        f"mean_best_val_{selection_metric}": float(
            fold_result_df["best_val_score"].mean()
        ),
        f"std_best_val_{selection_metric}": float(
            fold_result_df["best_val_score"].std()
        ),
        "fold_results": fold_results,
    }

    metric_cols = [
        col for col in fold_result_df.columns
        if col.startswith("best_val_")
    ]

    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(fold_result_df[col]):
            summary[f"mean_{col}"] = float(fold_result_df[col].mean())
            summary[f"std_{col}"] = float(fold_result_df[col].std())

    save_json(
        summary,
        os.path.join(save_dir, "cv_summary.json")
    )

    print("\n[INFO] Selected fold training finished.")
    print(fold_result_df)
    print(
        f"[INFO] Mean best validation {selection_metric.upper()}:",
        summary[f"mean_best_val_{selection_metric}"]
    )
    print(
        f"[INFO] Std best validation {selection_metric.upper()}:",
        summary[f"std_best_val_{selection_metric}"]
    )
    print("[INFO] Saved CV results:", os.path.join(save_dir, "cv_results.csv"))
    print("[INFO] Saved CV summary:", os.path.join(save_dir, "cv_summary.json"))


if __name__ == "__main__":
    apply_cli_overrides()

    threading.Thread(
        target=keyboard_listener,
        daemon=True
    ).start()

    main()
