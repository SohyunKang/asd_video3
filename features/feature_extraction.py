import json
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat


# =========================
# Paths
# =========================
LATENCY_DURATION_CSV = Path(
    "/home/sohyunkang/asd_video3/features/latency_duration.csv"
)

EYE_TRUE_DIR = Path("/storage/sohyunkang/eyecont_results_true")
EYE_FALSE_DIR = Path("/storage/sohyunkang/eyecont_results_false")
EXACT_MAT_DIR = Path("/storage/sohyunkang/exact_mat")

OUT_DIR = Path("/home/sohyunkang/asd_video3/features/full_features_npz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_INDEX_CSV = Path(
    "/home/sohyunkang/asd_video3/features/full_feature_index.csv"
)


# =========================
# Utils
# =========================
def get_video_id_from_path(path):
    stem = Path(path).stem
    m = re.search(r"(IF\d+_\d+_\d+_\d+_\d+)", stem)
    if m:
        return m.group(1)
    return stem


def get_patient_id_from_video_id(video_id):
    """
    IF2001_3_1_1023092761_0
                ↓
          1023092761
    """
    parts = str(video_id).split("_")

    if len(parts) >= 5:
        return parts[3]

    return str(video_id)


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def find_eye_json(video_id):
    video_id = str(video_id).strip()
    visit = get_visit_from_video_id(video_id)

    candidates = list(EYE_TRUE_DIR.glob(f"{video_id}.json"))
    candidates += list(EYE_TRUE_DIR.glob(f"{video_id}_*.json"))
    candidates = filter_by_visit(candidates, visit)

    source = "true"

    if len(candidates) == 0:
        candidates = list(EYE_FALSE_DIR.glob(f"{video_id}.json"))
        candidates += list(EYE_FALSE_DIR.glob(f"{video_id}_*.json"))
        candidates = filter_by_visit(candidates, visit)
        source = "false"

    if len(candidates) == 0:
        return None, None

    candidates = sorted(candidates, key=lambda p: len(p.name))

    if len(candidates) > 1:
        print(f"[WARN] multiple json found for {video_id}, use best:")
        for c in candidates:
            print("   ", c)
        print("[WARN] selected:", candidates[0])

    return candidates[0], source


def find_exact_mat(video_id, patient_id=None):
    video_id = str(video_id).strip()
    visit = get_visit_from_video_id(video_id)

    patterns = [
        f"{video_id}_gazed.mat",
        f"{video_id}.mat",
        f"{video_id}*.mat",
    ]

    candidates = []

    for pattern in patterns:
        found = sorted([
            p for p in EXACT_MAT_DIR.glob(pattern)
            if not p.name.startswith("._")
        ])

        found = filter_by_visit(found, visit)

        if len(found) > 0:
            candidates = found
            break

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: len(p.name))

    if len(candidates) > 1:
        print(f"[WARN] multiple mat found for {video_id}, use best:")
        for c in candidates:
            print("   ", c)
        print("[WARN] selected:", candidates[0])

    return candidates[0]

def get_visit_from_video_id(video_id):
    video_id = str(video_id).lower()
    if "_fu" in video_id or video_id.endswith("fu"):
        return "fu"
    return "baseline"


def filter_by_visit(candidates, visit):
    candidates = [
        p for p in candidates
        if not p.name.startswith("._")
    ]

    if visit == "fu":
        return [
            p for p in candidates
            if "_fu" in p.stem.lower()
        ]

    # baseline이면 _fu 들어간 파일 제외
    return [
        p for p in candidates
        if "_fu" not in p.stem.lower()
    ]

def find_exact_video(video_id, patient_id=None):
    video_id = str(video_id).strip()
    visit = get_visit_from_video_id(video_id)

    patterns = [
        f"{video_id}.mp4",
        f"{video_id}_gazed.mp4",
        f"{video_id}_gaze_sound.mp4",
        f"{video_id}*.mp4",
    ]

    candidates = []

    for pattern in patterns:
        found = sorted([
            p for p in EXACT_MAT_DIR.glob(pattern)
            if not p.name.startswith("._")
        ])

        found = filter_by_visit(found, visit)

        if len(found) > 0:
            candidates = found
            break

    if len(candidates) == 0:
        return None

    # 우선순위: baseline/fu 맞는 것 중 gaze_sound보다 gazed 우선
    def score(p):
        name = p.name.lower()
        s = 0
        if "_gazed" in name:
            s -= 10
        if "_gaze_sound" in name:
            s -= 5
        s += len(name)
        return s

    candidates = sorted(candidates, key=score)

    if len(candidates) > 1:
        print(f"[WARN] multiple video found for {video_id}, use best:")
        for c in candidates:
            print("   ", c)
        print("[WARN] selected:", candidates[0])

    return candidates[0]


def get_video_fps(video_path):
    if video_path is None:
        return np.nan

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return np.nan

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 0:
        return np.nan

    return float(fps)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_mat(path):
    data = loadmat(path)
    return {
        k: v for k, v in data.items()
        if not k.startswith("__")
    }


def normalize_heatmap(heatmap, idx_len=None):
    """
    heatmap을 [N, H, W] 형태로 변환.
    idx_len과 같은 축을 frame axis로 우선 사용.
    """

    hm = np.asarray(heatmap)
    hm = np.squeeze(hm)

    if hm.ndim == 2:
        return hm[None, :, :]

    if hm.ndim != 3:
        return hm

    if idx_len is not None and idx_len > 0:
        matched_axes = [
            axis for axis, size in enumerate(hm.shape)
            if size == idx_len
        ]

        if len(matched_axes) >= 1:
            frame_axis = matched_axes[0]
            return np.moveaxis(hm, frame_axis, 0)

    # 로그 패턴: (64, T, 64)
    if hm.shape[0] == 64 and hm.shape[2] == 64:
        return np.moveaxis(hm, 1, 0)

    # 일반적인 MATLAB 패턴: (H, W, T)
    if (
        hm.shape[0] > 8
        and hm.shape[1] > 8
        and hm.shape[2] < max(hm.shape[0], hm.shape[1])
    ):
        return np.moveaxis(hm, 2, 0)

    return hm


# =========================
# Empty feature makers
# =========================
def make_empty_json_features():
    return {
        "json_frame": np.array([], dtype=float),
        "json_timestamp_ms": np.array([], dtype=float),

        "gaze_horizontal_offset": np.array([], dtype=float),
        "gaze_vertical_offset": np.array([], dtype=float),

        "gaze_left_pupil_xy": np.empty((0, 2), dtype=float),
        "gaze_right_pupil_xy": np.empty((0, 2), dtype=float),
        "gaze_left_pupil_norm_xy": np.empty((0, 2), dtype=float),
        "gaze_right_pupil_norm_xy": np.empty((0, 2), dtype=float),
        "gaze_left_eye_center_xy": np.empty((0, 2), dtype=float),
        "gaze_right_eye_center_xy": np.empty((0, 2), dtype=float),

        "gaze_direction": np.array([], dtype=object),
        "gaze_is_center": np.array([], dtype=bool),
        "gaze_is_left": np.array([], dtype=bool),
        "gaze_is_right": np.array([], dtype=bool),
        "gaze_is_up": np.array([], dtype=bool),
        "gaze_is_down": np.array([], dtype=bool),
        "gaze_is_blinking": np.array([], dtype=bool),

        "gaze_left_ear": np.array([], dtype=float),
        "gaze_right_ear": np.array([], dtype=float),
        "gaze_ear": np.array([], dtype=float),

        "emotion_label_seq": np.array([], dtype=object),
        "emotion_score_seq": np.array([], dtype=float),
    }


def make_empty_mat_features():
    return {
        "mat_gaze_frame_idx": np.array([], dtype=float),
        "mat_gaze_time_sec": np.array([], dtype=float),
        "ec_ioprobability": np.array([], dtype=float),
        "ec_heatmap": np.array([], dtype=float),
    }


# =========================
# JSON frame-level feature
# =========================
def parse_eye_json_features(json_path, start_sec=None, end_sec=None):
    """
    start_sec/end_sec가 있으면 해당 구간만 추출.
    없으면 JSON 전체 구간 추출.
    """

    empty = make_empty_json_features()

    if json_path is None:
        return empty

    try:
        data = load_json(json_path)
    except Exception as e:
        print(f"[WARN] json load failed: {json_path}, {e}")
        return empty

    frames = []
    timestamps = []

    horizontal_offsets = []
    vertical_offsets = []

    left_pupil_xy = []
    right_pupil_xy = []
    left_pupil_norm_xy = []
    right_pupil_norm_xy = []
    left_eye_center_xy = []
    right_eye_center_xy = []

    directions = []
    is_center = []
    is_left = []
    is_right = []
    is_up = []
    is_down = []
    is_blinking = []

    left_ear = []
    right_ear = []
    ear = []

    emotion_label_seq = []
    emotion_score_seq = []

    for item in data:
        timestamp_ms = safe_float(item.get("timestamp_ms"))
        timestamp_sec = timestamp_ms / 1000.0

        if not np.isfinite(timestamp_sec):
            continue

        if start_sec is not None and timestamp_sec < start_sec:
            continue

        if end_sec is not None and timestamp_sec > end_sec:
            continue

        gaze = item.get("gaze")
        if not isinstance(gaze, dict):
            gaze = {}

        emotion = item.get("emotion")
        if not isinstance(emotion, list):
            emotion = []

        frames.append(safe_float(item.get("frame")))
        timestamps.append(timestamp_ms)

        horizontal_offsets.append(
            safe_float(gaze.get("horizontal_offset"))
        )
        vertical_offsets.append(
            safe_float(gaze.get("vertical_offset"))
        )

        def xy_from_dict(d):
            if not isinstance(d, dict):
                return [np.nan, np.nan]
            return [
                safe_float(d.get("x")),
                safe_float(d.get("y")),
            ]

        left_pupil_xy.append(
            xy_from_dict(gaze.get("left_pupil"))
        )
        right_pupil_xy.append(
            xy_from_dict(gaze.get("right_pupil"))
        )
        left_pupil_norm_xy.append(
            xy_from_dict(gaze.get("left_pupil_norm"))
        )
        right_pupil_norm_xy.append(
            xy_from_dict(gaze.get("right_pupil_norm"))
        )
        left_eye_center_xy.append(
            xy_from_dict(gaze.get("left_eye_center"))
        )
        right_eye_center_xy.append(
            xy_from_dict(gaze.get("right_eye_center"))
        )

        directions.append(str(gaze.get("direction", "")))

        is_center.append(bool(gaze.get("is_center", False)))
        is_left.append(bool(gaze.get("is_left", False)))
        is_right.append(bool(gaze.get("is_right", False)))
        is_up.append(bool(gaze.get("is_up", False)))
        is_down.append(bool(gaze.get("is_down", False)))
        is_blinking.append(bool(gaze.get("is_blinking", False)))

        left_ear.append(safe_float(gaze.get("left_ear")))
        right_ear.append(safe_float(gaze.get("right_ear")))
        ear.append(safe_float(gaze.get("ear")))

        best_label = ""
        best_score = np.nan

        for e in emotion:
            if not isinstance(e, dict):
                continue

            score = safe_float(e.get("score"))

            if not np.isfinite(score):
                continue

            if (not np.isfinite(best_score)) or (score > best_score):
                best_score = score
                best_label = str(e.get("label", ""))

        emotion_label_seq.append(best_label)
        emotion_score_seq.append(best_score)

    return {
        "json_frame": np.asarray(frames, dtype=float),
        "json_timestamp_ms": np.asarray(timestamps, dtype=float),

        "gaze_horizontal_offset": np.asarray(horizontal_offsets, dtype=float),
        "gaze_vertical_offset": np.asarray(vertical_offsets, dtype=float),

        "gaze_left_pupil_xy": np.asarray(left_pupil_xy, dtype=float),
        "gaze_right_pupil_xy": np.asarray(right_pupil_xy, dtype=float),
        "gaze_left_pupil_norm_xy": np.asarray(left_pupil_norm_xy, dtype=float),
        "gaze_right_pupil_norm_xy": np.asarray(right_pupil_norm_xy, dtype=float),
        "gaze_left_eye_center_xy": np.asarray(left_eye_center_xy, dtype=float),
        "gaze_right_eye_center_xy": np.asarray(right_eye_center_xy, dtype=float),

        "gaze_direction": np.asarray(directions, dtype=object),
        "gaze_is_center": np.asarray(is_center, dtype=bool),
        "gaze_is_left": np.asarray(is_left, dtype=bool),
        "gaze_is_right": np.asarray(is_right, dtype=bool),
        "gaze_is_up": np.asarray(is_up, dtype=bool),
        "gaze_is_down": np.asarray(is_down, dtype=bool),
        "gaze_is_blinking": np.asarray(is_blinking, dtype=bool),

        "gaze_left_ear": np.asarray(left_ear, dtype=float),
        "gaze_right_ear": np.asarray(right_ear, dtype=float),
        "gaze_ear": np.asarray(ear, dtype=float),

        "emotion_label_seq": np.asarray(emotion_label_seq, dtype=object),
        "emotion_score_seq": np.asarray(emotion_score_seq, dtype=object),
    }

def to_float_1d_array(x):
    """
    MATLAB cell/object/중첩 list 형태도 1D float array로 평탄화.
    """

    values = []

    def collect(v):
        arr = np.asarray(v)

        if arr.dtype == object:
            for item in arr.ravel():
                collect(item)
        else:
            arr = arr.astype(float).ravel()
            for item in arr:
                values.append(float(item))

    try:
        collect(x)
    except Exception as e:
        print(f"[WARN] failed to convert to float array: {e}")
        return np.array([], dtype=float)

    return np.asarray(values, dtype=float)

# =========================
# MAT feature
# =========================
def extract_mat_features(mat_path, fps, start_sec=None, end_sec=None):
    """
    start_sec/end_sec가 있으면 해당 구간만 추출.
    없으면 MAT 전체 구간 추출.
    """

    empty = make_empty_mat_features()

    if mat_path is None:
        return empty

    try:
        data = read_mat(mat_path)
    except Exception as e:
        print(f"[WARN] mat load failed: {mat_path}, {e}")
        return empty

    if "idx" not in data or "iolist" not in data or "heatmap" not in data:
        print(f"[WARN] missing key in mat: {mat_path}")
        return empty

    idx = to_float_1d_array(data["idx"])
    iolist = to_float_1d_array(data["iolist"])

    heatmap = normalize_heatmap(
        data["heatmap"],
        idx_len=len(idx)
    )

    n = min(len(idx), len(iolist), heatmap.shape[0])

    if n == 0:
        print(
            f"[DEBUG mat empty before mask] {Path(mat_path).name} | "
            f"idx_len={len(idx)} | "
            f"iolist_len={len(iolist)} | "
            f"heatmap_shape={heatmap.shape}"
        )

    idx = idx[:n]
    iolist = iolist[:n]
    heatmap = heatmap[:n]

    if start_sec is None or end_sec is None:
        if np.isfinite(fps) and fps > 0:
            frame_time = idx / fps
        else:
            frame_time = np.full_like(idx, np.nan, dtype=float)

        mask = np.ones(n, dtype=bool)

    else:
        if np.isfinite(fps) and fps > 0:
            frame_time_0based = idx / fps
            mask_0based = (
                (frame_time_0based >= start_sec) &
                (frame_time_0based <= end_sec)
            )

            frame_time_1based = (idx - 1) / fps
            mask_1based = (
                (frame_time_1based >= start_sec) &
                (frame_time_1based <= end_sec)
            )

            if mask_0based.sum() >= mask_1based.sum():
                frame_time = frame_time_0based
                mask = mask_0based
                idx_base = "0based"
            else:
                frame_time = frame_time_1based
                mask = mask_1based
                idx_base = "1based"

            if mask.sum() == 0 and len(idx) > 0:
                print(
                    f"[DEBUG io=0] {Path(mat_path).name} | "
                    f"base={idx_base} | "
                    f"fps={fps} | "
                    f"interval={start_sec:.3f}-{end_sec:.3f} | "
                    f"idx_minmax={idx.min()}-{idx.max()} | "
                    f"time0_minmax={frame_time_0based.min():.3f}-{frame_time_0based.max():.3f} | "
                    f"time1_minmax={frame_time_1based.min():.3f}-{frame_time_1based.max():.3f}"
                )

        else:
            frame_time = np.full_like(idx, np.nan, dtype=float)
            mask = np.ones(n, dtype=bool)

    return {
        "mat_gaze_frame_idx": idx[mask],
        "mat_gaze_time_sec": frame_time[mask],
        "ec_ioprobability": iolist[mask],
        "ec_heatmap": heatmap[mask],
    }


# =========================
# Aligned JSON/MAT feature
# =========================
def make_empty_aligned_features():
    return {
        "aligned_time_sec_from_pred_ec_start": np.array([], dtype=float),

        "aligned_json_frame": np.array([], dtype=float),
        "aligned_json_timestamp_ms": np.array([], dtype=float),

        "aligned_gaze_horizontal_offset": np.array([], dtype=float),
        "aligned_gaze_vertical_offset": np.array([], dtype=float),
        "aligned_gaze_left_pupil_xy": np.empty((0, 2), dtype=float),
        "aligned_gaze_right_pupil_xy": np.empty((0, 2), dtype=float),
        "aligned_gaze_left_pupil_norm_xy": np.empty((0, 2), dtype=float),
        "aligned_gaze_right_pupil_norm_xy": np.empty((0, 2), dtype=float),
        "aligned_gaze_left_eye_center_xy": np.empty((0, 2), dtype=float),
        "aligned_gaze_right_eye_center_xy": np.empty((0, 2), dtype=float),

        "aligned_gaze_direction": np.array([], dtype=object),
        "aligned_gaze_is_center": np.array([], dtype=bool),
        "aligned_gaze_is_left": np.array([], dtype=bool),
        "aligned_gaze_is_right": np.array([], dtype=bool),
        "aligned_gaze_is_up": np.array([], dtype=bool),
        "aligned_gaze_is_down": np.array([], dtype=bool),
        "aligned_gaze_is_blinking": np.array([], dtype=bool),
        "aligned_gaze_left_ear": np.array([], dtype=float),
        "aligned_gaze_right_ear": np.array([], dtype=float),
        "aligned_gaze_ear": np.array([], dtype=float),

        "aligned_emotion_label_seq": np.array([], dtype=object),
        "aligned_emotion_score_seq": np.array([], dtype=object),

        "aligned_mat_frame_idx": np.array([], dtype=float),
        "aligned_mat_time_sec": np.array([], dtype=float),
        "aligned_ioprobability": np.array([], dtype=float),
        "aligned_heatmap": np.array([], dtype=float),
        "aligned_mat_match_idx": np.array([], dtype=int),
        "aligned_mat_time_diff_sec": np.array([], dtype=float),
    }


def add_json_time_features(json_features, pred_ec_start=None):
    if len(json_features["json_timestamp_ms"]) > 0:
        json_time_sec = json_features["json_timestamp_ms"].astype(float) / 1000.0
    else:
        json_time_sec = np.array([], dtype=float)

    json_features["json_time_sec"] = json_time_sec

    if pred_ec_start is not None and np.isfinite(pred_ec_start):
        json_features["json_time_sec_from_pred_ec_start"] = (
            json_time_sec - float(pred_ec_start)
        )
    else:
        json_features["json_time_sec_from_pred_ec_start"] = np.array([], dtype=float)

    return json_features


def nearest_index(source_times, target_times, max_diff_sec=0.05):
    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if len(source_times) == 0 or len(target_times) == 0:
        return np.full(len(target_times), -1, dtype=int)

    valid_source = np.isfinite(source_times)

    if valid_source.sum() == 0:
        return np.full(len(target_times), -1, dtype=int)

    out = []

    for t in target_times:
        if not np.isfinite(t):
            out.append(-1)
            continue

        diff = np.abs(source_times - t)
        diff[~valid_source] = np.inf

        best = int(np.argmin(diff))

        if not np.isfinite(diff[best]):
            out.append(-1)
            continue

        if max_diff_sec is not None and diff[best] > max_diff_sec:
            out.append(-1)
        else:
            out.append(best)

    return np.asarray(out, dtype=int)


def align_json_mat_from_pred_ec_start(
    json_features,
    mat_features,
    pred_ec_start,
    max_time_diff_sec=0.05,
):
    """
    JSON gaze/emotion frame을 기준 time grid로 사용하고,
    MAT ioprobability/heatmap은 가장 가까운 MAT time으로 매칭.

    기준점은 CSV의 pred_eye_contact_start_sec.
    aligned_time_sec_from_pred_ec_start = json_time_sec - pred_ec_start
    """

    if len(json_features["json_timestamp_ms"]) == 0:
        return make_empty_aligned_features()

    json_time_sec = json_features["json_timestamp_ms"].astype(float) / 1000.0
    mat_time_sec = mat_features["mat_gaze_time_sec"].astype(float)

    n = len(json_time_sec)
    aligned_time = json_time_sec - float(pred_ec_start)

    mat_match_idx = nearest_index(
        source_times=mat_time_sec,
        target_times=json_time_sec,
        max_diff_sec=max_time_diff_sec,
    )

    aligned_mat_frame_idx = np.full(n, np.nan, dtype=float)
    aligned_mat_time_sec = np.full(n, np.nan, dtype=float)
    aligned_ioprobability = np.full(n, np.nan, dtype=float)
    aligned_mat_time_diff_sec = np.full(n, np.nan, dtype=float)

    heatmap = mat_features["ec_heatmap"]

    if heatmap.ndim == 3:
        h, w = heatmap.shape[1], heatmap.shape[2]
        aligned_heatmap = np.full((n, h, w), np.nan, dtype=float)
    else:
        aligned_heatmap = np.array([], dtype=float)

    for i, mi in enumerate(mat_match_idx):
        if mi < 0:
            continue

        if mi < len(mat_features["mat_gaze_frame_idx"]):
            aligned_mat_frame_idx[i] = mat_features["mat_gaze_frame_idx"][mi]

        if mi < len(mat_features["mat_gaze_time_sec"]):
            aligned_mat_time_sec[i] = mat_features["mat_gaze_time_sec"][mi]
            aligned_mat_time_diff_sec[i] = abs(
                mat_features["mat_gaze_time_sec"][mi] - json_time_sec[i]
            )

        if mi < len(mat_features["ec_ioprobability"]):
            aligned_ioprobability[i] = mat_features["ec_ioprobability"][mi]

        if heatmap.ndim == 3 and mi < heatmap.shape[0]:
            aligned_heatmap[i] = heatmap[mi]

    return {
        "aligned_time_sec_from_pred_ec_start": aligned_time,

        "aligned_json_frame": json_features["json_frame"],
        "aligned_json_timestamp_ms": json_features["json_timestamp_ms"],

        "aligned_gaze_horizontal_offset": json_features["gaze_horizontal_offset"],
        "aligned_gaze_vertical_offset": json_features["gaze_vertical_offset"],
        "aligned_gaze_left_pupil_xy": json_features["gaze_left_pupil_xy"],
        "aligned_gaze_right_pupil_xy": json_features["gaze_right_pupil_xy"],
        "aligned_gaze_left_pupil_norm_xy": json_features["gaze_left_pupil_norm_xy"],
        "aligned_gaze_right_pupil_norm_xy": json_features["gaze_right_pupil_norm_xy"],
        "aligned_gaze_left_eye_center_xy": json_features["gaze_left_eye_center_xy"],
        "aligned_gaze_right_eye_center_xy": json_features["gaze_right_eye_center_xy"],

        "aligned_gaze_direction": json_features["gaze_direction"],
        "aligned_gaze_is_center": json_features["gaze_is_center"],
        "aligned_gaze_is_left": json_features["gaze_is_left"],
        "aligned_gaze_is_right": json_features["gaze_is_right"],
        "aligned_gaze_is_up": json_features["gaze_is_up"],
        "aligned_gaze_is_down": json_features["gaze_is_down"],
        "aligned_gaze_is_blinking": json_features["gaze_is_blinking"],
        "aligned_gaze_left_ear": json_features["gaze_left_ear"],
        "aligned_gaze_right_ear": json_features["gaze_right_ear"],
        "aligned_gaze_ear": json_features["gaze_ear"],

        "aligned_emotion_label_seq": json_features["emotion_label_seq"],
        "aligned_emotion_score_seq": json_features["emotion_score_seq"],

        "aligned_mat_frame_idx": aligned_mat_frame_idx,
        "aligned_mat_time_sec": aligned_mat_time_sec,
        "aligned_ioprobability": aligned_ioprobability,
        "aligned_heatmap": aligned_heatmap,
        "aligned_mat_match_idx": mat_match_idx,
        "aligned_mat_time_diff_sec": aligned_mat_time_diff_sec,
    }


# =========================
# Main
# =========================
df = pd.read_csv(LATENCY_DURATION_CSV)

video_col = find_col(df, ["video_id", "id", "filename"])
pred_ec_col = find_col(
    df,
    [
        "pred_eye_contact_exists",
        "pred_has_eye_contact",
        "pred_eye_contact",
        "eye_contact_exists",
    ]
)

latency_col = find_col(df, ["pred_latency_sec"])
duration_col = find_col(df, ["pred_eye_contact_duration_sec"])
start_col = find_col(df, ["pred_eye_contact_start_sec"])
end_col = find_col(df, ["pred_eye_contact_end_sec"])

if video_col is None:
    raise ValueError(f"video id column not found. columns={list(df.columns)}")

if pred_ec_col is None:
    raise ValueError(
        f"pred_eye_contact_exists column not found. columns={list(df.columns)}"
    )

if latency_col is None or duration_col is None:
    raise ValueError(
        f"latency/duration column not found. columns={list(df.columns)}"
    )

index_rows = []

for _, row in df.iterrows():

    video_id = str(row[video_col]).strip()
    patient_id = get_patient_id_from_video_id(video_id)
    visit = get_visit_from_video_id(video_id)

    pred_eye_contact_exists = safe_float(row[pred_ec_col])

    if not np.isfinite(pred_eye_contact_exists):
        pred_eye_contact_exists = 0

    pred_eye_contact_exists = int(pred_eye_contact_exists)

    latency_sec = safe_float(row[latency_col])
    duration_sec = safe_float(row[duration_col])

    if start_col is not None:
        ec_start = safe_float(row[start_col])
    else:
        ec_start = latency_sec

    if end_col is not None:
        ec_end = safe_float(row[end_col])
    else:
        ec_end = ec_start + duration_sec

    valid_ec_time = (
        pred_eye_contact_exists == 1
        and np.isfinite(ec_start)
        and np.isfinite(ec_end)
        and ec_end > ec_start
    )

    if not valid_ec_time:
        latency_sec = np.nan
        duration_sec = np.nan
        ec_start = np.nan
        ec_end = np.nan

    json_path, eye_source = find_eye_json(video_id)
    mat_path = find_exact_mat(video_id, patient_id)
    exact_video_path = find_exact_video(video_id, patient_id)
    fps = get_video_fps(exact_video_path)

    # =========================
    # 핵심:
    # EC 있으면 EC 구간 feature
    # EC 없으면 가능한 전체 JSON/MAT feature
    # =========================
    if valid_ec_time:
        pred_ec_start = ec_start
        pred_ec_end = ec_end

        json_features = parse_eye_json_features(
            json_path=json_path,
            start_sec=pred_ec_start,
            end_sec=pred_ec_end,
        )
        json_features = add_json_time_features(
            json_features,
            pred_ec_start=pred_ec_start,
        )

        mat_features = extract_mat_features(
            mat_path=mat_path,
            fps=fps,
            start_sec=pred_ec_start,
            end_sec=pred_ec_end,
        )

        aligned_features = align_json_mat_from_pred_ec_start(
            json_features=json_features,
            mat_features=mat_features,
            pred_ec_start=pred_ec_start,
            max_time_diff_sec=0.05,
        )
    else:
        json_features = parse_eye_json_features(
            json_path=json_path,
            start_sec=None,
            end_sec=None,
        )
        json_features = add_json_time_features(
            json_features,
            pred_ec_start=None,
        )

        mat_features = extract_mat_features(
            mat_path=mat_path,
            fps=fps,
            start_sec=None,
            end_sec=None,
        )

        aligned_features = make_empty_aligned_features()

    out_npz = OUT_DIR / f"{video_id}.npz"

    np.savez_compressed(
        out_npz,

        # metadata
        video_id=np.asarray(video_id),
        patient_id=np.asarray(patient_id),
        visit=np.asarray(visit),
        diagnosis=np.asarray(str(row["diagnosis"])),
        diagnosis_target=np.asarray(int(row["diagnosis_target"]), dtype=int),

        # feature 1
        pred_eye_contact_exists=np.asarray(pred_eye_contact_exists, dtype=int),

        # feature 2~
        latency_sec=np.asarray(latency_sec, dtype=float),
        duration_sec=np.asarray(duration_sec, dtype=float),
        ec_start_sec=np.asarray(ec_start, dtype=float),
        ec_end_sec=np.asarray(ec_end, dtype=float),

        # JSON gaze vector + emotion
        **json_features,

        # MAT ioprobability + heatmap
        **mat_features,

        # time-aligned JSON/MAT features after pred EC start
        **aligned_features,
    )

    index_rows.append(
        {
            "video_id": video_id,
            "patient_id": patient_id,
            "visit": visit,

            "pred_eye_contact_exists": pred_eye_contact_exists,

            "latency_sec": latency_sec,
            "duration_sec": duration_sec,
            "ec_start_sec": ec_start,
            "ec_end_sec": ec_end,
            "fps": fps,

            "feature_npz": str(out_npz),

            "eye_json_path": str(json_path) if json_path is not None else None,
            "eye_json_source": eye_source,
            "mat_path": str(mat_path) if mat_path is not None else None,
            "exact_video_path": str(exact_video_path) if exact_video_path is not None else None,

            "n_json_gaze_frames": len(json_features["json_frame"]),
            "n_emotion_frames": len(json_features["emotion_label_seq"]),
            "n_mat_ioprobability": len(mat_features["ec_ioprobability"]),
            "n_mat_heatmap": (
                int(mat_features["ec_heatmap"].shape[0])
                if mat_features["ec_heatmap"].ndim >= 1
                else 0
            ),
            "heatmap_shape": str(mat_features["ec_heatmap"].shape),

            "has_mat": int(mat_path is not None),
            "has_json": int(json_path is not None),
            "feature_scope": "eye_contact_interval" if valid_ec_time else "full_available_sequence",
            "n_aligned_frames": len(aligned_features["aligned_time_sec_from_pred_ec_start"]),
            "aligned_heatmap_shape": str(aligned_features["aligned_heatmap"].shape),
            "aligned_ioprob_nan_count": (
                int(np.isnan(aligned_features["aligned_ioprobability"]).sum())
                if len(aligned_features["aligned_ioprobability"]) > 0
                else 0
            ),
        }
    )

    print(
        f"[OK] {video_id} | "
        f"patient={patient_id} "
        f"pred_EC={pred_eye_contact_exists} "
        f"scope={'EC' if valid_ec_time else 'FULL'} "
        f"mat={'Y' if mat_path is not None else 'N'} "
        f"json={'Y' if json_path is not None else 'N'} "
        f"json_gaze={len(json_features['json_frame'])} "
        f"emotion={len(json_features['emotion_label_seq'])} "
        f"io={len(mat_features['ec_ioprobability'])} "
        f"heatmap={mat_features['ec_heatmap'].shape} "
        f"aligned={len(aligned_features['aligned_time_sec_from_pred_ec_start'])}"
    )

index_df = pd.DataFrame(index_rows)

# 컬럼 순서 고정: video_id, patient_id, pred_eye_contact_exists
front_cols = [
    "video_id",
    "patient_id",
    "visit",
    "pred_eye_contact_exists",
]

cols = front_cols + [
    c for c in index_df.columns
    if c not in front_cols
]

index_df = index_df[cols]

index_df.to_csv(
    OUT_INDEX_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("[DONE] npz dir:", OUT_DIR)
print("[DONE] index csv:", OUT_INDEX_CSV)
print("[DONE] rows:", len(index_df))
print("[DONE] pred EC counts:")
print(index_df["pred_eye_contact_exists"].value_counts(dropna=False))