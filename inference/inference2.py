import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn.dataset import FrameNpyClipDataset, VideoFrameNpyDataset
from learn.model import build_model


def normalize_video_id(x):
    x = str(x).strip()
    while x.lower().endswith(".mp4"):
        x = x[:-4]
    return x


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target_video_ids(args):
    video_ids = []

    if args.video_ids is not None:
        video_ids.extend([
            normalize_video_id(v)
            for v in args.video_ids.split(",")
            if v.strip() != ""
        ])

    if args.video_id_file is not None:
        with open(args.video_id_file, "r", encoding="utf-8") as f:
            video_ids.extend([
                normalize_video_id(line)
                for line in f
                if line.strip() != ""
            ])

    video_ids = sorted(set(video_ids))

    if len(video_ids) == 0:
        raise ValueError("--video_ids 또는 --video_id_file 중 하나는 필요합니다.")

    return video_ids


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

    print("[INFO] videos with call_start in json:", len(call_start_map))
    return call_start_map

def find_video_path(row, video_root="/storage/sohyunkang/video_data"):
    candidates = []

    if "video_path" in row and pd.notna(row["video_path"]):
        candidates.append(str(row["video_path"]))

    video_id = normalize_video_id(row["video_id"])
    candidates += [
        f"{video_root}/{video_id}.mp4",
        f"{video_root}/{video_id}.mp4.mp4",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None

def build_clip_df_from_metadata_for_video_ids(
    metadata_path,
    video_ids,
    clip_duration,
    stride,
    use_calling=True,
    call_json_path="/home/sohyunkang/asd_video3/demographics/260707_add_new_videos.json",
    drop_no_call=True,
    keep_prev_clip=True,
):
    meta_df = pd.read_csv(metadata_path)

    meta_df["video_id"] = meta_df["video_id"].astype(str).str.strip()
    meta_df["video_id_norm"] = meta_df["video_id"].apply(normalize_video_id)

    target_ids = set(normalize_video_id(v) for v in video_ids)

    meta_df = meta_df[
        meta_df["video_id_norm"].isin(target_ids)
    ].copy()

    if len(meta_df) == 0:
        raise ValueError("입력한 video_id가 metadata에서 하나도 안 잡혔습니다.")

    meta_df = meta_df[
        meta_df["npy_path"].apply(os.path.exists)
    ].copy()

    call_start_map = {}

    if use_calling:
        if call_json_path is None:
            raise ValueError("--use_calling 사용 시 --call_json_path가 필요합니다.")
        call_start_map = load_first_call_start_map_from_json_by_video(call_json_path)

    rows = []

    for _, row in meta_df.iterrows():
        video_id = str(row["video_id"])
        video_id_norm = normalize_video_id(video_id)

        fps = pd.to_numeric(row.get("fps", np.nan), errors="coerce")
        saved_n_frames = pd.to_numeric(
            row.get("saved_n_frames", np.nan),
            errors="coerce"
        )

        if pd.isna(fps) or fps <= 0:
            video_path = find_video_path(row)

            if video_path is None:
                print("[WARN] cannot find video file for fps, skip:", video_id)
                continue

            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps is None or fps <= 0:
                print("[WARN] cannot read fps, skip:", video_id)
                continue

            print(f"[INFO] fps recovered: {video_id} fps={fps}")

        if pd.isna(saved_n_frames) or saved_n_frames <= 0:
            arr = np.load(row["npy_path"], mmap_mode="r")
            saved_n_frames = len(arr)
        else:
            saved_n_frames = int(saved_n_frames)

        max_time = saved_n_frames / float(fps)

        call_start = 0.0
        has_call_start = False

        if use_calling:
            if video_id_norm in call_start_map:
                call_start = float(call_start_map[video_id_norm])
                has_call_start = True
            else:
                if drop_no_call:
                    print("[INFO] no call_start, skip:", video_id)
                    continue
                call_start = 0.0
                has_call_start = False

        clip_rows = []
        start = 0.0

        while start + clip_duration <= max_time:
            end = start + clip_duration

            clip_rows.append({
                "patient_id": str(row["patient_id"]),
                "visit": row.get("visit", "baseline"),
                "video_id": video_id,
                "base_video_id": row.get("base_video_id", video_id),
                "video_npy_path": row["npy_path"],
                "npy_path": row["npy_path"],
                "fps": float(fps),
                "saved_n_frames": saved_n_frames,
                "clip_start": start,
                "clip_end": end,
                "call_start": call_start,
                "has_call_start": has_call_start,
                "clip_start_from_call": start - call_start,
                "clip_end_from_call": end - call_start,
                "label": "unknown",
                "target": -1,
            })

            start += stride

        if use_calling:
            before_call = [
                r for r in clip_rows
                if r["clip_start"] < call_start
            ]

            after_call = [
                r for r in clip_rows
                if r["clip_start"] >= call_start
            ]

            selected_rows = after_call

            if keep_prev_clip and len(before_call) > 0:
                selected_rows = [before_call[-1]] + selected_rows

            clip_rows = selected_rows

        rows.extend(clip_rows)

    clip_df = pd.DataFrame(rows)

    found_ids = set(meta_df["video_id_norm"].unique())
    missing_ids = sorted(target_ids - found_ids)

    print("[INFO] requested videos:", len(target_ids))
    print("[INFO] found metadata videos:", len(found_ids))
    print("[INFO] built clips:", len(clip_df))

    if use_calling and len(clip_df) > 0:
        print("[INFO] videos after calling filter:", clip_df["video_id"].nunique())
        print("[INFO] clips with call_start:")
        print(clip_df["has_call_start"].value_counts(dropna=False))

    if len(missing_ids) > 0:
        print("[WARN] missing video_ids in metadata:")
        print(missing_ids[:50])

    if len(clip_df) == 0:
        raise ValueError("clip_df가 비었습니다. video_id/call_start/fps/npy_path를 확인하세요.")

    clip_df = (
        clip_df
        .drop_duplicates(subset=["video_id", "clip_start", "clip_end"])
        .sort_values(["patient_id", "video_id", "clip_start"])
        .reset_index(drop=True)
    )

    return clip_df


def load_model(model_path, cfg, device):
    model = build_model(
        model_name=cfg["model_name"],
        num_classes=cfg.get("num_classes", 2),
        freeze_encoder=False,
        video_level=cfg.get("video_level", False),
        pooling=cfg.get("pooling", "meanmax"),
        classifier_num_layers=cfg.get("classifier_num_layers", 0),
        classifier_hidden_dim=cfg.get("classifier_hidden_dim", 512),
        classifier_dropout=cfg.get("classifier_dropout", 0.3),
        mil_hidden_dim=cfg.get("mil_hidden_dim", 64),
        return_attention=False,
    )

    state_dict = torch.load(model_path, map_location=device)

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def build_dataset(clip_df, cfg):
    if cfg.get("video_level", False):
        return VideoFrameNpyDataset(
            clip_df,
            num_clips_per_video=cfg.get("num_clips_per_video", 30),
            num_frames=cfg.get("num_frames", 8),
            random_window=False,
        )

    return FrameNpyClipDataset(
        clip_df,
        num_frames=cfg.get("num_frames", 8),
    )


@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5):
    rows = []

    for batch in tqdm(loader, desc="Inference"):
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

        clips = clips.to(device)

        logits = model(clips)

        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        for pid, vid, cs, ce, pred, prob in zip(
            patient_ids,
            video_ids,
            clip_starts,
            clip_ends,
            preds.cpu().numpy(),
            probs.cpu().numpy(),
        ):
            rows.append({
                "patient_id": str(pid),
                "video_id": str(vid),
                "clip_start": float(cs),
                "clip_end": float(ce),
                "pred": int(pred),
                "prob": float(prob),
            })

    return pd.DataFrame(rows)


def summarize_video_level(pred_df, threshold=0.5):
    summary_df = (
        pred_df
        .groupby(["patient_id", "video_id"])
        .agg(
            n_clips=("prob", "size"),
            mean_prob=("prob", "mean"),
            max_prob=("prob", "max"),
            positive_clips=("pred", "sum"),
        )
        .reset_index()
    )

    summary_df["mean_pred"] = (summary_df["mean_prob"] >= threshold).astype(int)
    summary_df["max_pred"] = (summary_df["max_prob"] >= threshold).astype(int)

    return summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--video_ids", type=str, default=None)
    parser.add_argument("--video_id_file", type=str, default=None)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--frame_metadata_path", type=str, default=None)

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_summary_csv", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--call_json_path",
        type=str,
        default="./demographics/260707_add_new_videos.json"
    )
    parser.add_argument(
        "--keep_no_call",
        action="store_true",
        help="call_start 없는 영상도 0초부터 inference합니다. 기본은 제거."
    )
    parser.add_argument(
        "--no_prev_clip",
        action="store_true",
        help="call_start 직전 clip 1개를 포함하지 않습니다."
    )

    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)
    exp_dir = fold_dir.parent

    model_path = Path(args.model_path) if args.model_path else fold_dir / "model.pt"
    config_path = Path(args.config_path) if args.config_path else exp_dir / "config.json"

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else fold_dir / "inference_by_video_ids.csv"
    )

    out_summary_csv = (
        Path(args.out_summary_csv)
        if args.out_summary_csv
        else fold_dir / "inference_by_video_ids_summary.csv"
    )

    cfg = load_config(config_path)

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    metadata_path = (
        args.frame_metadata_path
        if args.frame_metadata_path is not None
        else cfg.get(
            "frame_metadata_path",
            "/storage/sohyunkang/preprocessed_video_frames_person_track_landscape_224/video_frame_metadata.csv"
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] device:", device)
    print("[INFO] fold_dir:", fold_dir)
    print("[INFO] model:", model_path)
    print("[INFO] config:", config_path)
    print("[INFO] metadata:", metadata_path)
    print("[INFO] output:", out_csv)

    target_video_ids = load_target_video_ids(args)

    clip_df = build_clip_df_from_metadata_for_video_ids(
        metadata_path=metadata_path,
        video_ids=target_video_ids,
        clip_duration=float(cfg["clip_duration"]),
        stride=float(cfg["stride"]),
        use_calling=True,
        call_json_path=args.call_json_path,
        drop_no_call=not args.keep_no_call,
        keep_prev_clip=not args.no_prev_clip,
    )

    max_clips = int(cfg.get("num_clips_per_video", 30))

    before_n = len(clip_df)

    clip_df = (
        clip_df
        .sort_values(["patient_id", "video_id", "clip_start"])
        .groupby(["patient_id", "video_id"], group_keys=False)
        .head(max_clips)
        .reset_index(drop=True)
    )

    print(f"[INFO] limit clips per video: {before_n} -> {len(clip_df)}")

    dataset = build_dataset(clip_df, cfg)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = load_model(
        model_path=model_path,
        cfg=cfg,
        device=device,
    )

    pred_df = run_inference(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
    )

    keep_cols = [
        "patient_id",
        "visit",
        "video_id",
        "base_video_id",
        "video_npy_path",
        "fps",
        "saved_n_frames",
        "clip_start",
        "clip_end",
        "call_start",
        "has_call_start",
        "clip_start_from_call",
        "clip_end_from_call",
    ]

    keep_cols = [c for c in keep_cols if c in clip_df.columns]
    clip_meta = clip_df[keep_cols].drop_duplicates()

    pred_df = pred_df.merge(
        clip_meta,
        on=["patient_id", "video_id", "clip_start", "clip_end"],
        how="left"
    )

    # ==========================================================
    # 기존 capture_prediction.py와 동일한 형식 저장
    # ==========================================================

    # clip_df 저장
    clip_df.to_csv(
        fold_dir / "inference_clip_df.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # prediction 저장
    pred_save = pred_df.copy()

    pred_save["true"] = -1

    pred_save = pred_save[
        [
            "patient_id",
            "video_id",
            "clip_start",
            "clip_end",
            "call_start",
            "has_call_start",
            "clip_start_from_call",
            "clip_end_from_call",
            "true",
            "pred",
            "prob",
        ]
    ]

    pred_save.to_csv(
        fold_dir / "inference_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 기존 inference 결과도 저장
    pred_df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = summarize_video_level(
        pred_df,
        threshold=args.threshold
    )

    summary_df.to_csv(
        out_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("[DONE] saved :", fold_dir / "inference_clip_df.csv")
    print("[DONE] saved :", fold_dir / "inference_predictions.csv")
    print("[DONE] saved :", out_csv)
    print("[DONE] saved :", out_summary_csv)

if __name__ == "__main__":
    main()