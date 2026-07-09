# train_xgboost_from_npz.py

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =========================
# Utils
# =========================
def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) > 0 else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) > 0 else np.nan


def safe_min(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if len(x) > 0 else np.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) > 0 else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) > 0 else np.nan


def safe_ratio_bool(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return float(np.mean(x.astype(bool)))


def summarize_numeric(prefix, arr):
    return {
        f"{prefix}_mean": safe_mean(arr),
        f"{prefix}_std": safe_std(arr),
        f"{prefix}_min": safe_min(arr),
        f"{prefix}_max": safe_max(arr),
        f"{prefix}_median": safe_median(arr),
    }
def flatten_emotion_top1(label_seq, score_seq=None):
    labels_out = []
    scores_out = []

    label_seq = np.asarray(label_seq, dtype=object)

    if score_seq is None:
        score_seq = [None] * len(label_seq)
    else:
        score_seq = np.asarray(score_seq, dtype=object)

    for labels, scores in zip(label_seq, score_seq):

        # 이미 frame별 top1 label인 경우
        if isinstance(labels, str):
            labels_out.append(labels)
            if scores is None:
                scores_out.append(np.nan)
            else:
                try:
                    scores_out.append(float(scores))
                except Exception:
                    scores_out.append(np.nan)
            continue

        labels_arr = np.asarray(labels, dtype=object).ravel()

        if scores is None:
            scores_arr = np.full(len(labels_arr), np.nan)
        else:
            scores_arr = np.asarray(scores, dtype=float).ravel()

        if len(labels_arr) == 0:
            labels_out.append("")
            scores_out.append(np.nan)
            continue

        if len(scores_arr) == len(labels_arr) and np.isfinite(scores_arr).any():
            best_idx = int(np.nanargmax(scores_arr))
        else:
            best_idx = 0

        labels_out.append(str(labels_arr[best_idx]))
        scores_out.append(
            float(scores_arr[best_idx])
            if len(scores_arr) > best_idx and np.isfinite(scores_arr[best_idx])
            else np.nan
        )

    return np.asarray(labels_out, dtype=str), np.asarray(scores_out, dtype=float)

def summarize_xy(prefix, arr):
    arr = np.asarray(arr, dtype=float)

    out = {}

    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        out[f"{prefix}_x_mean"] = np.nan
        out[f"{prefix}_y_mean"] = np.nan
        out[f"{prefix}_x_std"] = np.nan
        out[f"{prefix}_y_std"] = np.nan
        return out

    out.update(summarize_numeric(f"{prefix}_x", arr[:, 0]))
    out.update(summarize_numeric(f"{prefix}_y", arr[:, 1]))

    return out


def load_npz_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    feat = {}

    # -------------------------
    # diagnosis label
    # -------------------------
    if "diagnosis_target" in data:
        feat["diagnosis_target"] = int(
            np.asarray(data["diagnosis_target"]).reshape(-1)[0]
        )

    if "diagnosis" in data:
        feat["diagnosis"] = str(
            np.asarray(data["diagnosis"]).reshape(-1)[0]
        )

    # -------------------------
    # scalar metadata
    # -------------------------
    for key in [
        "pred_eye_contact_exists",
        "latency_sec",
        "duration_sec",
        # "ec_start_sec",
        # "ec_end_sec",
    ]:
        if key in data:
            feat[key] = float(np.asarray(data[key]).reshape(-1)[0])
        else:
            feat[key] = np.nan

    # -------------------------
    # no eye-contact 처리
    # -------------------------
    if feat.get("pred_eye_contact_exists", 0) == 0:
        feat["latency_sec"] = 999.0
        feat["duration_sec"] = 0.0

    # -------------------------
    # JSON gaze numeric
    # -------------------------
    for key in [
        "gaze_horizontal_offset",
        "gaze_vertical_offset",
        "gaze_left_ear",
        "gaze_right_ear",
        "gaze_ear",
        "json_time_sec",
        "json_time_sec_from_pred_ec_start",
    ]:
        if key in data:
            feat.update(summarize_numeric(key, data[key]))

    # -------------------------
    # pupil / eye center xy
    # -------------------------
    for key in [
        "gaze_left_pupil_xy",
        "gaze_right_pupil_xy",
        "gaze_left_pupil_norm_xy",
        "gaze_right_pupil_norm_xy",
        "gaze_left_eye_center_xy",
        "gaze_right_eye_center_xy",
    ]:
        if key in data:
            feat.update(summarize_xy(key, data[key]))

    # -------------------------
    # gaze boolean ratios
    # -------------------------
    for key in [
        "gaze_is_center",
        "gaze_is_left",
        "gaze_is_right",
        "gaze_is_up",
        "gaze_is_down",
        "gaze_is_blinking",
    ]:
        if key in data:
            feat[f"{key}_ratio"] = safe_ratio_bool(data[key])

    # -------------------------
    # MAT ioprobability
    # -------------------------
    if "ec_ioprobability" in data:
        feat.update(summarize_numeric("ec_ioprobability", data["ec_ioprobability"]))

    # -------------------------
    # aligned features
    # -------------------------
    for key in [
        "aligned_gaze_horizontal_offset",
        "aligned_gaze_vertical_offset",
        "aligned_gaze_ear",
        "aligned_ioprobability",
        "aligned_mat_time_diff_sec",
        "aligned_time_sec_from_pred_ec_start",
    ]:
        if key in data:
            feat.update(summarize_numeric(key, data[key]))

    for key in [
        "aligned_gaze_is_center",
        "aligned_gaze_is_left",
        "aligned_gaze_is_right",
        "aligned_gaze_is_up",
        "aligned_gaze_is_down",
        "aligned_gaze_is_blinking",
    ]:
        if key in data:
            feat[f"{key}_ratio"] = safe_ratio_bool(data[key])

    # -------------------------
    # sequence lengths
    # -------------------------
    for key in [
        "json_frame",
        "ec_ioprobability",
        "aligned_time_sec_from_pred_ec_start",
    ]:
        if key in data:
            feat[f"n_{key}"] = len(data[key])
        else:
            feat[f"n_{key}"] = 0

    if "emotion_label_seq" in data:
        emo, emo_score = flatten_emotion_top1(
            data["emotion_label_seq"],
            data["emotion_score_seq"] if "emotion_score_seq" in data else None,
        )

        emotions = [
            "happy",
            "neutral",
            "sad",
            "angry",
            "fear",
            "surprise",
            "disgust",
        ]

        # emotion 없는 frame 제거
        emo = np.asarray(emo).astype(str)
        emo = emo[
            (emo != "")
            & (emo != "None")
            & (emo != "nan")
        ]

        for e in emotions:
            feat[f"emotion_{e}_ratio"] = (
                float(np.mean(emo == e))
                if len(emo) > 0
                else np.nan
            )

        feat.update(
            summarize_numeric(
                "emotion_score",
                emo_score,
            )
        )
    if "ec_heatmap" in data:
        feat.update(
            summarize_heatmap(
                "ec_heatmap",
                data["ec_heatmap"],
            )
        )
    
    return feat


def build_feature_table(index_csv):
    index_df = pd.read_csv(index_csv)

    rows = []

    for _, row in index_df.iterrows():
        npz_path = row["feature_npz"]

        if not Path(npz_path).exists():
            print(f"[WARN] missing npz: {npz_path}")
            continue

        feat = load_npz_features(npz_path)

        feat["video_id"] = row["video_id"]
        feat["patient_id"] = str(row["patient_id"])
        feat["visit"] = row.get("visit", "unknown")

        rows.append(feat)

    feat_df = pd.DataFrame(rows)

    return feat_df


def make_binary_label(df, label_col):
    """
    기본:
    자폐군=1, 정상군=0
    """
    label_map = {
        "자폐군": 1,
        "정상군": 0,
    }

    y = df[label_col].map(label_map)

    valid = y.notna()

    return y[valid].astype(int), valid
def to_heatmap_3d(heatmap):
    frames = []

    arr = np.asarray(heatmap, dtype=object)

    for item in arr.ravel():
        try:
            h = np.asarray(item, dtype=float)
        except Exception:
            continue

        h = np.squeeze(h)

        if h.ndim == 2:
            frames.append(h)
        elif h.ndim == 3:
            for k in range(h.shape[0]):
                frames.append(np.asarray(h[k], dtype=float))

    if len(frames) == 0:
        return np.empty((0, 0, 0), dtype=float)

    shapes = [f.shape for f in frames]
    common_shape = max(set(shapes), key=shapes.count)

    frames = [
        f for f in frames
        if f.shape == common_shape
    ]

    return np.stack(frames, axis=0)


def summarize_heatmap(prefix, heatmap):
    hm = to_heatmap_3d(heatmap)

    if hm.ndim != 3 or hm.shape[0] == 0:
        return {
            f"{prefix}_peak_mean": np.nan,
            f"{prefix}_peak_std": np.nan,
            f"{prefix}_entropy_mean": np.nan,
            f"{prefix}_center_mass_mean": np.nan,
            f"{prefix}_center_mass_std": np.nan,
        }

    peaks = []
    entropies = []
    center_masses = []

    for h in hm:
        h = np.asarray(h, dtype=float)
        h = np.nan_to_num(h, nan=0.0)
        h = np.maximum(h, 0)

        total = h.sum()
        if total <= 0:
            continue

        p = h / total

        peaks.append(float(h.max()))

        p_flat = p.ravel()
        p_flat = p_flat[p_flat > 0]
        entropies.append(float(-(p_flat * np.log(p_flat)).sum()))

        H, W = h.shape
        y0, y1 = int(H * 0.25), int(H * 0.75)
        x0, x1 = int(W * 0.25), int(W * 0.75)

        center_masses.append(float(p[y0:y1, x0:x1].sum()))

    return {
        f"{prefix}_peak_mean": safe_mean(peaks),
        f"{prefix}_peak_std": safe_std(peaks),
        f"{prefix}_entropy_mean": safe_mean(entropies),
        f"{prefix}_center_mass_mean": safe_mean(center_masses),
        f"{prefix}_center_mass_std": safe_std(center_masses),
    }

def evaluate_fold(y_true, y_pred, y_prob):
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["auc"] = np.nan

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index_csv",
        type=str,
        default="/home/sohyunkang/asd_video3/features/full_feature_index.csv",
    )

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/sohyunkang/asd_video3/features/xgboost_cv_results",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df = build_feature_table(
        index_csv=args.index_csv,
    )

    valid = feat_df["diagnosis_target"].isin([0, 1])

    feat_df = feat_df.loc[valid].reset_index(drop=True)

    y = feat_df["diagnosis_target"].astype(int)


    meta_cols = [
        "video_id",
        "patient_id",
        "visit",
        "diagnosis",
        "diagnosis_target",
    ]

    FEATURE_COLS = [
        # 1. latency / duration
        "pred_eye_contact_exists",
        "latency_sec",
        "duration_sec",

        # # 2. gaze variation
        # "gaze_horizontal_offset_std",
        # "gaze_vertical_offset_std",
        "gaze_is_center_ratio",

        # # 3. emotion ratio
        "emotion_happy_ratio",
        "emotion_neutral_ratio",
        "emotion_sad_ratio",
        "emotion_angry_ratio",
        "emotion_fear_ratio",
        "emotion_surprise_ratio",
        # "emotion_disgust_ratio",

        # # 4. in/out probability
        "ec_ioprobability_mean",
        "ec_ioprobability_std",

        # # 5. heatmap summary
        # "ec_heatmap_peak_mean",
        # "ec_heatmap_peak_std",
        # "ec_heatmap_entropy_mean",
        # "ec_heatmap_center_mass_mean",
        # "ec_heatmap_center_mass_std",
    ]
    
    feature_cols = [
        c for c in FEATURE_COLS
        if c in feat_df.columns
    ]

    missing_features = sorted(set(FEATURE_COLS) - set(feature_cols))
    print("[INFO] selected features:", len(feature_cols))
    print("[INFO] missing features:", missing_features)

    X = feat_df[feature_cols].copy()
    groups = feat_df["patient_id"].astype(str)

    print("[INFO] samples:", len(feat_df))
    print("[INFO] patients:", groups.nunique())
    print("[INFO] features:", len(feature_cols))
    print("[INFO] label counts:")
    print(y.value_counts())

    cv = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    all_pred_rows = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        scale_pos_weight = (
            (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        )

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=args.seed + fold,
                    scale_pos_weight=scale_pos_weight,
                )),
            ]
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_fold(y_val, y_pred, y_prob)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        metrics["train_patients"] = feat_df.iloc[train_idx]["patient_id"].nunique()
        metrics["val_patients"] = feat_df.iloc[val_idx]["patient_id"].nunique()

        fold_metrics.append(metrics)

        pred_part = feat_df.iloc[val_idx][
            [
                "video_id",
                "patient_id",
                "visit",
                "diagnosis",
                "diagnosis_target",
            ]
        ].copy()
        pred_part["fold"] = fold
        pred_part["y_true"] = y_val.values
        pred_part["y_pred"] = y_pred
        pred_part["y_prob"] = y_prob

        all_pred_rows.append(pred_part)

        print(f"\n[FOLD {fold}]")
        print(metrics)
        print(classification_report(y_val, y_pred, digits=4, zero_division=0))

    metrics_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(all_pred_rows, ignore_index=True)

    metrics_df.to_csv(
        out_dir / "cv_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        out_dir / "cv_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feat_df[meta_cols + feature_cols].to_csv(
        out_dir / "feature_table_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print("\n========== CV SUMMARY ==========")
    print(metrics_df[["acc", "precision", "recall", "f1", "auc"]].mean())
    print("\n[DONE] saved:", out_dir)


if __name__ == "__main__":
    main()