import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from utils import build_label_table_from_jsons
from dataset import build_clip_table, VideoClipDataset, PreprocessedClipDataset, VideoDiagnosisDataset
from model import build_model

class CFG:
    video_root = "/storage/sohyunkang/video_data"
    clip_csv_path = "./preprocessing/results/preprocessed_clips_person_1.0_8.csv"
    checkpoint_path = './experiments'
    model_name = "timesformer"  # "3dcnn", "timesformer"
    # 추가
    label_mode = "diagnosis_labels"
    # "pseudo_labels" or "annotated_labels" or "diagnosis_labels"
    video_level = True

    diagnosis_excel_path = "./demographics/rpmp_검사지_result_20241219.xlsx"
    diagnosis_id_col = "연구대상자ID"
    diagnosis_group_col = "구분"

    diagnosis_label_map = {
        "정상군": 0,
        "자폐군": 1,
    }

    num_classes = 2
    annotated_excel_path = "./demographics/0607 호명_시간기록.xlsx"

    annotated_id_col = "연구대상자ID"
    annotated_start_col = "눈맞춤시작"
    annotated_end_col = "눈맞춤끝"

    clip_start_col = "clip_start"
    clip_end_col = "clip_end"

    freeze_encoder = False

    iou_label_threshold = 0.5

    min_event_duration_sec = 0.3
    max_gap_sec = 0.2

    val_ratio = 0.2
    seed = 42

    batch_size = 2
    num_workers = 0
    epochs = 15
    lr = 1e-4

    debug_mode = False
    debug_train_n = 8000
    debug_val_n = 2000

    device = "cuda" if torch.cuda.is_available() else "cpu"

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()
    
def parse_time_to_sec(value):
    """
    지원 예:
    10.25        -> 10.25
    "10.25"      -> 10.25
    "00:10.25"   -> 10.25
    "01:02.50"   -> 62.5
    "01:02:03.5" -> 3723.5
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if value == "":
        return None

    if ":" not in value:
        try:
            return float(value)
        except ValueError:
            return None

    parts = value.split(":")

    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    return None

def add_diagnosis_labels_from_rpmp(
    clip_df,
    excel_path,
    id_col="연구대상자ID",
    group_col="구분",
    label_map=None
):
    if label_map is None:
        label_map = {
            "정상군": 0,
            "자폐군": 1,
        }

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

    print("[INFO] Diagnosis label mode: 정상군 vs 자폐군")
    print(f"[INFO] clips: {before_n} -> {after_n}")
    print("[INFO] label counts:")
    print(clip_df["label"].value_counts())
    print("[INFO] target counts:")
    print(clip_df["target"].value_counts().sort_index())
    print("[INFO] subjects per group:")
    print(
        clip_df.groupby("label")["patient_id"]
        .nunique()
        .sort_index()
    )

    return clip_df

def add_annotated_labels_from_excel(
    clip_df,
    excel_path,
    id_col="연구대상자ID",
    start_col="눈맞춤시작",
    end_col="눈맞춤끝",
    clip_start_col="clip_start",
    clip_end_col="clip_end",
    label_threshold=0.5
):
    """
    annotation excel의 눈맞춤 시작/끝을 이용해
    기존 preprocessed clip의 clip_start/clip_end 기준으로 target만 새로 생성.
    clip_start/clip_end는 절대 annotation 시간으로 덮어쓰지 않음.
    """

    clip_df = clip_df.copy()

    anno_df = pd.read_excel(excel_path)
    anno_df.columns = anno_df.columns.astype(str).str.strip()

    anno_df[id_col] = (
        anno_df[id_col]
        .astype(str)
        .str.strip()
    )

    def is_negative_annotation(start_value, end_value):
        start = str(start_value).strip()
        end = str(end_value).strip()

        return start == "(-)" and end == "(-)"


    def is_valid_time_annotation(start_value, end_value):
        start_sec = parse_time_to_sec(start_value)
        end_sec = parse_time_to_sec(end_value)

        if start_sec is None or end_sec is None:
            return False

        if end_sec <= start_sec:
            return False

        return True


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

    annotated_subjects = set(
        anno_df_valid[id_col]
        .astype(str)
        .str.strip()
    )

    print("[INFO] annotation rows total:", len(anno_df))
    print("[INFO] valid annotation rows:", len(anno_df_valid))
    print("[INFO] invalid annotation rows excluded:", len(invalid_annotation_df))
    print("[INFO] annotated subjects used:", len(annotated_subjects))

    # 실제 eye-contact interval은 정상 시간 형식인 행만 사용
    anno_df = anno_df_valid.copy()

    anno_df[start_col] = anno_df[start_col].apply(parse_time_to_sec)
    anno_df[end_col] = anno_df[end_col].apply(parse_time_to_sec)

    anno_df = anno_df[
        anno_df[start_col].notna()
        & anno_df[end_col].notna()
        & (anno_df[end_col] > anno_df[start_col])
    ].copy()

    if "patient_id" not in clip_df.columns:
        raise ValueError("clip_df에 patient_id 컬럼이 필요합니다.")

    if clip_start_col not in clip_df.columns:
        raise ValueError(f"clip_df에 {clip_start_col} 컬럼이 없습니다.")

    if clip_end_col not in clip_df.columns:
        raise ValueError(f"clip_df에 {clip_end_col} 컬럼이 없습니다.")

    clip_df["patient_id"] = (
        clip_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    clip_df[clip_start_col] = pd.to_numeric(
        clip_df[clip_start_col],
        errors="coerce"
    )

    clip_df[clip_end_col] = pd.to_numeric(
        clip_df[clip_end_col],
        errors="coerce"
    )

    before_n = len(clip_df)

    # annotation 있는 subject만 사용
    clip_df = clip_df[
        clip_df["patient_id"].isin(annotated_subjects)
    ].copy()

    after_n = len(clip_df)

    print("[INFO] Annotated mode: using only annotated subjects")
    print(f"[INFO] clips before filtering: {before_n}")
    print(f"[INFO] clips after filtering: {after_n}")
    print(f"[INFO] removed clips: {before_n - after_n}")

    if len(clip_df) == 0:
        raise ValueError(
            "Annotated subjects와 clip_df['patient_id']가 매칭되지 않아 clip이 0개입니다."
        )

    anno_map = {}

    for subject, g in anno_df.groupby(id_col):
        subject = str(subject).strip()

        anno_map[subject] = [
            (
                float(row[start_col]),
                float(row[end_col])
            )
            for _, row in g.iterrows()
        ]

    targets = []

    for _, row in clip_df.iterrows():
        patient_id = str(row["patient_id"]).strip()

        # 이것은 preprocessed clip의 시간
        clip_start = row[clip_start_col]
        clip_end = row[clip_end_col]

        if pd.isna(clip_start) or pd.isna(clip_end) or clip_end <= clip_start:
            targets.append(0)
            continue

        clip_duration = clip_end - clip_start
        intervals = anno_map.get(patient_id, [])

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

    # target은 binary
    clip_df["target"] = targets

    # label은 기존 dataset 코드와 맞게 문자열
    clip_df["label"] = [
        "eye_contact" if t == 1 else "non_eye_contact"
        for t in targets
    ]

    print("[INFO] Annotated labels applied")
    print("[INFO] label counts:")
    print(clip_df["target"].value_counts(dropna=False))

    return clip_df

def add_patient_wise_split(
    label_df,
    test_patient_ids=None,
    val_size=0.2,
    random_state=42
):
    if test_patient_ids is None:
        test_patient_ids = []

    label_df = label_df.copy()

    label_df["patient_id"] = (
        label_df["patient_id"]
        .astype(str)
        .str.strip()
    )

    test_patient_ids = [
        str(x).strip()
        for x in test_patient_ids
    ]

    label_df["split"] = None

    # -------------------------
    # test patients 분리
    # -------------------------

    test_mask = label_df["patient_id"].isin(test_patient_ids)
    label_df.loc[test_mask, "split"] = "test"

    remain_df = label_df[~test_mask].copy()

    # -------------------------
    # 환자별 eye-contact 유무 계산
    # target 중 하나라도 1이면 has_eye_contact = 1
    # -------------------------

    patient_label_df = (
        remain_df
        .groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "has_eye_contact"})
    )

    print("\n[PATIENT LABEL QC]")
    print(patient_label_df["has_eye_contact"].value_counts().sort_index())

    # -------------------------
    # 환자 단위 stratified split
    # -------------------------

    train_patients, val_patients = train_test_split(
        patient_label_df["patient_id"],
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=patient_label_df["has_eye_contact"]
    )

    label_df.loc[
        label_df["patient_id"].isin(train_patients),
        "split"
    ] = "train"

    label_df.loc[
        label_df["patient_id"].isin(val_patients),
        "split"
    ] = "val"

    # -------------------------
    # split QC
    # -------------------------

    print("[INFO] patient-wise stratified split")
    print("[INFO] train patients:", len(train_patients))
    print("[INFO] val patients:", len(val_patients))
    print("[INFO] test patients:", len(test_patient_ids))

    for split in ["train", "val", "test"]:
        split_df = label_df[label_df["split"] == split]

        if len(split_df) == 0:
            continue

        split_patient_df = (
            split_df
            .groupby("patient_id")["target"]
            .max()
            .reset_index()
            .rename(columns={"target": "has_eye_contact"})
        )

        print(f"\n[{split.upper()}]")
        print("clips:", len(split_df))
        print("patients:", split_df["patient_id"].nunique())
        print("clip target counts:")
        print(split_df["target"].value_counts().sort_index())
        print("patient eye-contact counts:")
        print(
            split_patient_df["has_eye_contact"]
            .value_counts()
            .sort_index()
        )

    return label_df

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = None

    return metrics

from sklearn.metrics import f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, threshold=0.5):
    model.train()

    prediction_rows = []

    total_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)

    for clips, labels, patient_ids in pbar:
        clips = clips.to(CFG.device)
        labels = labels.to(CFG.device)

        optimizer.zero_grad()

        logits = model(clips)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        total_loss += loss.item() * clips.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

        running_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        pos_rate_pred = sum(y_pred) / len(y_pred)
        pos_rate_true = sum(y_true) / len(y_true)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            f1=f"{running_f1:.4f}",
            pos_pred=int(sum(y_pred)),
            pos_true=int(sum(y_true)),
            pred_rate=f"{pos_rate_pred:.3f}",
            true_rate=f"{pos_rate_true:.3f}",
        )

        for pid, true, pred, prob in zip(
            patient_ids,
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            probs.detach().cpu().numpy()
        ):
            prediction_rows.append({
                "patient_id": pid,
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

    for clips, labels, patient_ids in pbar:
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

        running_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            f1=f"{running_f1:.4f}",
            pos_true=int(sum(y_true)),
            pos_pred=int(sum(y_pred))
        )

        for pid, true, pred, prob in zip(
            patient_ids,
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            probs.cpu().numpy()
        ):
            prediction_rows.append({
                "patient_id": pid,
                "true": int(true),
                "pred": int(pred),
                "prob": float(prob),
            })

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    prediction_df = pd.DataFrame(prediction_rows)

    return metrics, prediction_df


def main():
    print("[INFO] Device:", CFG.device)

    from datetime import datetime

    exp_id = (
        f"{CFG.label_mode}_"
        f"{CFG.model_name}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )

    save_dir = os.path.join(
        CFG.checkpoint_path,
        exp_id
    )

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(
        save_dir,
        "model.pt"
    )

    metrics_save_path = os.path.join(
        save_dir,
        "metrics.json"
    )

    config_save_path = os.path.join(
        save_dir,
        "config.json"
    )

    train_pred_path = os.path.join(
        save_dir,
        "train_predictions.csv"
    )

    val_pred_path = os.path.join(
        save_dir,
        "val_predictions.csv"
    )

    csv_name = os.path.splitext(
        os.path.basename(CFG.clip_csv_path)
    )[0]

    preprocess_dir = os.path.join(
        "/storage/sohyunkang",
        csv_name
    )

    config_path = os.path.join(
        preprocess_dir,
        "preprocess_config.json"
    )

    with open(config_path, "r") as f:
        preprocess_cfg = json.load(f)

    with open(
        os.path.join(save_dir, "preprocess_config.json"),
        "w"
    ) as f:
        json.dump(
            preprocess_cfg,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("[INFO] Loaded preprocess config:")
    print(preprocess_cfg)

    if not os.path.exists(CFG.checkpoint_path):
        os.makedirs(CFG.checkpoint_path)

    config_dict = {}

    for k, v in CFG.__dict__.items():
        if not k.startswith("__"):
            try:
                json.dumps(v)
                config_dict[k] = v
            except:
                config_dict[k] = str(v)

    with open(config_save_path, "w") as f:
        json.dump(
            config_dict,
            f,
            indent=4,
            ensure_ascii=False
        )

    clip_df = pd.read_csv(CFG.clip_csv_path)

    if CFG.label_mode == "annotated_labels":
        clip_df = add_annotated_labels_from_excel(
            clip_df=clip_df,
            excel_path=CFG.annotated_excel_path,
            id_col=CFG.annotated_id_col,
            start_col=CFG.annotated_start_col,
            end_col=CFG.annotated_end_col,
            clip_start_col=CFG.clip_start_col,
            clip_end_col=CFG.clip_end_col,
            label_threshold=CFG.iou_label_threshold
        )

    elif CFG.label_mode == "pseudo_labels":
        print("[INFO] Using pseudo labels from clip CSV")
        print(clip_df["target"].value_counts(dropna=False))

    elif CFG.label_mode == "diagnosis_labels":
        clip_df = add_diagnosis_labels_from_rpmp(
            clip_df=clip_df,
            excel_path=CFG.diagnosis_excel_path,
            id_col=CFG.diagnosis_id_col,
            group_col=CFG.diagnosis_group_col,
            label_map=CFG.diagnosis_label_map
        )
        clip_df["pseudo_label"] = clip_df["label"].copy()
        clip_df["pseudo_target"] = clip_df["target"].copy()

    else:
        raise ValueError(
            f"Unknown label_mode: {CFG.label_mode}. "
            "Use 'pseudo_labels' or 'annotated_labels'."
        )

    TEST_PATIENT_IDS = [
        "1023101971",
        "1023102072",
        "1023102561",
        "1023102611",
        "1023102612",
        "1023102941",
        "1023103061",
        "1023103112",
    ]
    TEST_PATIENT_IDS = [
    ]

    clip_df = add_patient_wise_split(
        clip_df,
        test_patient_ids=TEST_PATIENT_IDS
    )

    train_clip_df = clip_df[clip_df["split"] == "train"].reset_index(drop=True)
    val_clip_df = clip_df[clip_df["split"] == "val"].reset_index(drop=True)

    print("\n[DATA SPLIT]")
    print(
        f"train={len(train_clip_df)}, "
        f"val={len(val_clip_df)}"
    )

    print("\n[TRAIN TARGET]")
    print(train_clip_df["target"].value_counts().sort_index())

    print("\n[VAL TARGET]")
    print(val_clip_df["target"].value_counts().sort_index())

    if CFG.debug_mode:
        train_clip_df = train_clip_df.sample(
            n=min(CFG.debug_train_n, len(train_clip_df)),
            random_state=CFG.seed
        ).reset_index(drop=True)

        val_clip_df = val_clip_df.sample(
            n=min(CFG.debug_val_n, len(val_clip_df)),
            random_state=CFG.seed
        ).reset_index(drop=True)

        print("[DEBUG] Using small subset")
        print("[DEBUG] train clips:", len(train_clip_df))
        print("[DEBUG] val clips:", len(val_clip_df))
        print("[DEBUG] train label counts:")
        print(train_clip_df["label"].value_counts())
        print("[DEBUG] val label counts:")
        print(val_clip_df["label"].value_counts())

    if CFG.video_level:
        print("\n[INFO] Using video-level classification")
        print("[INFO] Building video-level model with mean-max pooling")
        train_dataset = VideoDiagnosisDataset(train_clip_df)
        val_dataset = VideoDiagnosisDataset(val_clip_df)
    else:
        print("\n[INFO] Using clip-level classification")
        print("[INFO] Building clip-level model")
        train_dataset = PreprocessedClipDataset(train_clip_df)
        val_dataset = PreprocessedClipDataset(val_clip_df)

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

    model = build_model(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        freeze_encoder=CFG.freeze_encoder,
        video_level=CFG.video_level
    )

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(CFG.device)

    if CFG.label_mode == "diagnosis_labels":
        video_target_df = (
            train_clip_df
            .groupby("video_id")["target"]
            .first()
            .reset_index()
        )

        counts = video_target_df["target"].value_counts().sort_index()

        neg = counts.get(0, 0)
        pos = counts.get(1, 0)


    else:

        counts = train_clip_df["target"].value_counts().sort_index()

        neg = counts[0]
        pos = counts[1]

    pos_weight = np.sqrt(neg / pos)
    pos_weight = neg / pos

    class_weights = torch.tensor(
        [1.0, pos_weight],
        dtype=torch.float32
    ).to(CFG.device)

    print("[INFO] neg:", neg)
    print("[INFO] pos:", pos)
    print("[INFO] class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # criterion = FocalLoss(
    #     alpha=class_weights,
    #     gamma=2.0
    # )
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
            {"params": encoder_params, "lr": CFG.lr * 0.01},
            {"params": head_params, "lr": CFG.lr},
        ],
        weight_decay=1e-4
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_f1 = 0.0

    for epoch in range(CFG.epochs):
        print(f"\n[Epoch {epoch + 1}/{CFG.epochs}]")

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

       
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
    
            state_dict = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )

            torch.save(state_dict, model_save_path)

            best_metrics = {
                "epoch": epoch + 1,
                "best_val_f1": best_f1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model_name": CFG.model_name,
                "clip_csv_path": CFG.clip_csv_path,
                "batch_size": CFG.batch_size,
                "lr": CFG.lr,
                "epochs": CFG.epochs,
                "debug_mode": CFG.debug_mode,
            }

            with open(metrics_save_path, "w") as f:
                json.dump(best_metrics, f, indent=4)

            train_pred_df.to_csv(
                train_pred_path,
                index=False
            )

            val_pred_df.to_csv(
                val_pred_path,
                index=False
            )

            print(f"[INFO] Saved best model: {model_save_path}")
            print(f"[INFO] Saved best metrics: {metrics_save_path}")

    print("\n[INFO] Training finished.")
    print("[INFO] Best validation F1:", best_f1)


if __name__ == "__main__":
    main()