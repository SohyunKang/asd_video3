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
from dataset import build_clip_table, VideoClipDataset, PreprocessedClipDataset
from model import build_model

class CFG:
    # json_root = "/Volumes/SAMSUNG/영유아/eyecont_results_true"
    # video_root = "/Volumes/SAMSUNG/영유아/보류군 외(정상군,고위험군,자폐군) 복호화파일"
    json_root = "/storage/json_data"
    video_root = "/storage/sohyunkang/video_data"
    clip_csv_path = "./preprocessing/results/preprocessed_clips_person_1.0_8.csv"
    checkpoint_path = './experiments'
    model_name = "timesformer"  # "3dcnn", "timesformer"


    iou_label_threshold = 0.5

    min_event_duration_sec = 0.3
    max_gap_sec = 0.2

    val_ratio = 0.2
    seed = 42

    batch_size = 16
    num_workers = 0
    epochs = 10
    lr = 1e-4

    debug_mode = True
    debug_train_n = 8000
    debug_val_n = 2000

    device = "cuda" if torch.cuda.is_available() else "cpu"


def add_patient_wise_split(
    label_df,
    test_patient_ids=None,
    val_size=0.2,
    random_state=42
    ):

    if test_patient_ids is None:
        test_patient_ids = []

    label_df = label_df.copy()

    # -------------------------
    # test patients 분리
    # -------------------------

    test_mask = label_df["patient_id"].isin(test_patient_ids)

    label_df.loc[test_mask, "split"] = "test"

    remain_df = label_df[~test_mask].copy()

    # -------------------------
    # train / val split
    # -------------------------

    remain_patients = remain_df["patient_id"].unique()

    train_patients, val_patients = train_test_split(
        remain_patients,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    label_df.loc[
        label_df["patient_id"].isin(train_patients),
        "split"
    ] = "train"

    label_df.loc[
        label_df["patient_id"].isin(val_patients),
        "split"
    ] = "val"

    print("[INFO] train patients:", len(train_patients))
    print("[INFO] val patients:", len(val_patients))
    print("[INFO] test patients:", len(test_patient_ids))

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

    total_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)

    for clips, labels in pbar:
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

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def validate(model, loader, criterion, threshold=0.5):
    model.eval()

    total_loss = 0.0

    y_true, y_pred, y_prob = [], [], []

    pbar = tqdm(loader, desc="Val", leave=False)

    for clips, labels in pbar:
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

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics

def main():
    print("[INFO] Device:", CFG.device)


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

    print("[INFO] Loaded preprocess config:")
    print(preprocess_cfg)

    if not os.path.exists(CFG.checkpoint_path):
        os.makedirs(CFG.checkpoint_path)

    clip_df = pd.read_csv(CFG.clip_csv_path)

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

    clip_df = add_patient_wise_split(
        clip_df,
        test_patient_ids=TEST_PATIENT_IDS
    )

    train_clip_df = clip_df[clip_df["split"] == "train"].reset_index(drop=True)
    val_clip_df = clip_df[clip_df["split"] == "val"].reset_index(drop=True)

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
        num_classes=2,
        freeze_encoder=False
    )

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(CFG.device)

    counts = train_clip_df["target"].value_counts().sort_index()

    neg = counts[0]
    pos = counts[1]

    pos_weight = np.sqrt(neg / pos)

    class_weights = torch.tensor(
        [1.0, pos_weight],
        dtype=torch.float32
    ).to(CFG.device)

    print("[INFO] neg:", neg)
    print("[INFO] pos:", pos)
    print("[INFO] class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_f1 = 0.0

    for epoch in range(CFG.epochs):
        print(f"\n[Epoch {epoch + 1}/{CFG.epochs}]")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion
        )

        print("[Train]", train_metrics)
        print("[Val]", val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            exp_name = os.path.splitext(os.path.basename(CFG.clip_csv_path))[0]

            model_save_path = os.path.join(
                CFG.checkpoint_path,
                f"best_model_{exp_name}.pt"
            )

            metrics_save_path = os.path.join(
                CFG.checkpoint_path,
                f"best_metrics_{exp_name}.json"
            )

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

            print(f"[INFO] Saved best model: {model_save_path}")
            print(f"[INFO] Saved best metrics: {metrics_save_path}")

    print("\n[INFO] Training finished.")
    print("[INFO] Best validation F1:", best_f1)


if __name__ == "__main__":
    main()