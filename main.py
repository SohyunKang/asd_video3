import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from utils import build_label_table_from_jsons
from dataset import build_clip_table, VideoClipDataset
from model import Simple3DCNN


class CFG:
    json_root = "/Volumes/SAMSUNG/영유아/eyecont_results_true"
    video_root = "/Volumes/SAMSUNG/영유아/보류군 외(정상군,고위험군,자폐군) 복호화파일"

    clip_duration = 2.0
    stride = 0.5
    num_frames = 16
    image_size = 224

    iou_label_threshold = 0.5

    min_event_duration_sec = 0.3
    max_gap_sec = 0.2

    val_ratio = 0.2
    seed = 42

    batch_size = 4
    num_workers = 2
    epochs = 10
    lr = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"


def add_patient_wise_split(label_df):
    patients = label_df["patient_id"].unique()

    train_patients, val_patients = train_test_split(
        patients,
        test_size=CFG.val_ratio,
        random_state=CFG.seed
    )

    label_df["split"] = label_df["patient_id"].apply(
        lambda x: "val" if x in val_patients else "train"
    )

    print(f"[INFO] Train patients: {len(train_patients)}")
    print(f"[INFO] Val patients: {len(val_patients)}")

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


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for clips, labels in loader:
        clips = clips.to(CFG.device)
        labels = labels.to(CFG.device)

        optimizer.zero_grad()

        logits = model(clips)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * clips.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for clips, labels in loader:
        clips = clips.to(CFG.device)
        labels = labels.to(CFG.device)

        logits = model(clips)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * clips.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    print("[INFO] Device:", CFG.device)

    label_df = build_label_table_from_jsons(
        json_root=CFG.json_root,
        video_root=CFG.video_root,
        min_duration_sec=CFG.min_event_duration_sec,
        max_gap_sec=CFG.max_gap_sec
    )

    label_df = add_patient_wise_split(label_df)

    print("[INFO] Total labels:", len(label_df))
    print(label_df["split"].value_counts())

    label_df.to_csv("labels_from_json_with_split.csv", index=False)

    train_clip_df = build_clip_table(
        label_df=label_df,
        split="train",
        clip_duration=CFG.clip_duration,
        stride=CFG.stride,
        iou_label_threshold=CFG.iou_label_threshold
    )

    val_clip_df = build_clip_table(
        label_df=label_df,
        split="val",
        clip_duration=CFG.clip_duration,
        stride=CFG.stride,
        iou_label_threshold=CFG.iou_label_threshold
    )

    train_clip_df.to_csv("train_clips.csv", index=False)
    val_clip_df.to_csv("val_clips.csv", index=False)


    print(train_clip_df["label"].value_counts())
    print(val_clip_df["label"].value_counts())

    train_dataset = VideoClipDataset(
        train_clip_df,
        clip_duration=CFG.clip_duration,
        num_frames=CFG.num_frames,
        image_size=CFG.image_size
    )

    val_dataset = VideoClipDataset(
        val_clip_df,
        clip_duration=CFG.clip_duration,
        num_frames=CFG.num_frames,
        image_size=CFG.image_size
    )

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


    model = Simple3DCNN(num_classes=2).to(CFG.device)

    criterion = nn.CrossEntropyLoss()
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

            torch.save(
                model.state_dict(),
                "best_eye_contact_3dcnn.pt"
            )

            print("[INFO] Saved best model.")

    print("\n[INFO] Training finished.")
    print("[INFO] Best validation F1:", best_f1)


if __name__ == "__main__":
    main()