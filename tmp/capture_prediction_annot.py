import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LEARN_DIR = ROOT / "learn"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEARN_DIR))


# =========================
# Paths
# =========================
FOLD_DIR = Path(
    "/home/sohyunkang/asd_video3/experiments/annotated_labels_timesformer_20260702_101409/fold_0"
)

VAL_CLIP_CSV = FOLD_DIR / "val_clip_df.csv"
PRED_CSV = FOLD_DIR / "val_predictions.csv"

ALL_CLIP_CSV = Path(
    "/home/sohyunkang/asd_video3/preprocessing/results/renew_preprocessed_clips_person_1.0_8_0.5_32.csv"
)

GAZE_VIDEO_DIR = Path("/storage/sohyunkang/eyecont_results_true")
EXACT_MAT_DIR = Path("/storage/sohyunkang/exact_mat")

OUT_DIR = FOLD_DIR / "annotation_prediction_check"
RAW_OUT_DIR = FOLD_DIR / "annotation_prediction_check_raw_with_before_call"
GAZE_OUT_DIR = FOLD_DIR / "annotation_prediction_check_gaze_video_after_first_EC_correct"
EXACT_OUT_DIR = FOLD_DIR / "annotation_prediction_check_exact_mat_after_first_EC_correct"
EMOTION_OUT_DIR = FOLD_DIR / "annotation_prediction_check_emotion_after_first_EC_correct"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
GAZE_OUT_DIR.mkdir(parents=True, exist_ok=True)
EXACT_OUT_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utils
# =========================
def add_title_bar(image, title, bar_height=50):
    h, w, c = image.shape
    title_bar = np.zeros((bar_height, w, c), dtype=np.uint8)

    cv2.putText(
        title_bar,
        title,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return np.concatenate([title_bar, image], axis=0)


def apply_result_filter(frame_bgr, true, pred, alpha=0.35):
    true = int(true)
    pred = int(pred)

    if true == 1 and pred == 1:
        color = (0, 0, 255)
        text = "EC correct"
    elif true == 0 and pred == 0:
        color = (255, 0, 0)
        text = "No EC correct"
    elif true == 0 and pred == 1:
        color = (64, 64, 192)
        text = "False Positive"
    elif true == 1 and pred == 0:
        color = (192, 64, 64)
        text = "False Negative"

    overlay = np.full_like(frame_bgr, color, dtype=np.uint8)

    frame_bgr = cv2.addWeighted(
        frame_bgr,
        1 - alpha,
        overlay,
        alpha,
        0
    )

    cv2.putText(
        frame_bgr,
        text,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame_bgr


def find_gaze_video(video_id):
    video_id = str(video_id).strip()

    candidates = list(GAZE_VIDEO_DIR.glob(f"{video_id}*.mp4"))

    if len(candidates) == 0:
        candidates = list(GAZE_VIDEO_DIR.glob(f"*{video_id}*.mp4"))

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"[WARN] multiple gaze videos found for {video_id}:")
        for c in candidates:
            print("   ", c)
        print("[WARN] use first one:", candidates[0])

    return candidates[0]


def find_eyecont_json(video_id):
    video_id = str(video_id).strip()

    candidates = list(GAZE_VIDEO_DIR.glob(f"{video_id}*.json"))

    if len(candidates) == 0:
        candidates = list(GAZE_VIDEO_DIR.glob(f"*{video_id}*.json"))

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"[WARN] multiple eyecont json found for {video_id}:")
        for c in candidates:
            print("   ", c)
        print("[WARN] use first one:", candidates[0])

    return candidates[0]


def find_exact_mat_video(patient_id, video_id=None):
    patient_id = str(patient_id).strip()

    candidates = list(EXACT_MAT_DIR.glob(f"*{patient_id}*.mp4"))

    if len(candidates) == 0 and video_id is not None:
        video_id = str(video_id).strip()
        candidates = list(EXACT_MAT_DIR.glob(f"*{video_id}*.mp4"))

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"[WARN] multiple exact_mat videos found for patient={patient_id}:")
        for c in candidates:
            print("   ", c)
        print("[WARN] use first one:", candidates[0])

    return candidates[0]


def read_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"[WARN] invalid fps: {video_path}")
        cap.release()
        return None

    frame_idx = int(round(float(time_sec) * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(
            f"[WARN] cannot read frame: {video_path}, "
            f"time={time_sec:.2f}s, frame_idx={frame_idx}"
        )
        return None

    return frame


def make_time_text(row):
    return f"{float(row['clip_start']):.1f}-{float(row['clip_end']):.1f}s"


def draw_time_text(frame, row):
    cv2.putText(
        frame,
        make_time_text(row),
        (8, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return frame


def resize_like_reference(frame, reference_frames):
    if len(reference_frames) > 0:
        target_h, target_w = reference_frames[0].shape[:2]
        frame = cv2.resize(frame, (target_w, target_h))
    return frame


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_emotion_events(json_data):
    """
    eyecont_results_true json에서 label, score만 추출.

    기대 구조 예:
    [
        {
            "start": 1.2,
            "end": 2.2,
            "label": "happy",
            "score": 0.87
        }
    ]

    반환:
    [
        {
            "start": float or None,
            "end": float or None,
            "emotion": str,
            "prob": float or None
        }
    ]
    """

    events = []

    def walk(obj):
        if isinstance(obj, dict):

            # label, score만 사용
            if "label" in obj and "score" in obj:
                start = (
                    obj.get("start")
                    or obj.get("start_sec")
                    or obj.get("clip_start")
                    or obj.get("time_start")
                    or obj.get("timestamp_start")
                )

                end = (
                    obj.get("end")
                    or obj.get("end_sec")
                    or obj.get("clip_end")
                    or obj.get("time_end")
                    or obj.get("timestamp_end")
                )

                events.append(
                    {
                        "start": safe_float(start),
                        "end": safe_float(end),
                        "emotion": str(obj.get("label")),
                        "prob": safe_float(obj.get("score")),
                    }
                )

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(json_data)

    dedup = []
    seen = set()

    for e in events:
        key = (
            e["start"],
            e["end"],
            e["emotion"],
            e["prob"],
        )

        if key not in seen:
            dedup.append(e)
            seen.add(key)

    return dedup

def get_emotion_for_clip(emotion_events, clip_start, clip_end):
    """
    clip 구간과 가장 많이 겹치는 emotion event 선택.
    시간이 없는 emotion event만 있으면 첫 번째 사용.
    """

    clip_start = float(clip_start)
    clip_end = float(clip_end)

    timed = [
        e for e in emotion_events
        if e["start"] is not None and e["end"] is not None
    ]

    if len(timed) == 0:
        if len(emotion_events) == 0:
            return None, None
        e = emotion_events[0]
        return e["emotion"], e["prob"]

    best = None
    best_overlap = 0.0

    for e in timed:
        s = float(e["start"])
        t = float(e["end"])

        overlap = max(
            0.0,
            min(clip_end, t) - max(clip_start, s)
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best = e

    if best is None:
        center = (clip_start + clip_end) / 2.0

        candidates = [
            e for e in timed
            if e["start"] <= center <= e["end"]
        ]

        if len(candidates) > 0:
            best = candidates[0]

    if best is None:
        return None, None

    return best["emotion"], best["prob"]


def draw_emotion_text(frame, emotion, prob):
    if emotion is None:
        text = "emotion: NA"
    else:
        if prob is None:
            text = f"emotion: {emotion}"
        else:
            text = f"emotion: {emotion} ({prob:.2f})"

    cv2.putText(
        frame,
        text,
        (8, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame

def draw_emotion_text_small(frame, emotion, prob):
    if emotion is None:
        text = "label: NA"
    else:
        if prob is None:
            text = f"{emotion}"
        else:
            text = f"{emotion} {prob:.2f}"

    cv2.putText(
        frame,
        text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame

def draw_time_text_small(frame, row):
    cv2.putText(
        frame,
        make_time_text(row),
        (6, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame

# =========================
# Load
# =========================
val_df = pd.read_csv(VAL_CLIP_CSV)
pred_df = pd.read_csv(PRED_CSV)
all_clip_df = pd.read_csv(ALL_CLIP_CSV)

for df in [val_df, pred_df, all_clip_df]:
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["clip_start"] = pd.to_numeric(df["clip_start"])
    df["clip_end"] = pd.to_numeric(df["clip_end"])

merge_cols = [
    "patient_id",
    "video_id",
    "clip_start",
    "clip_end",
]

val_df = val_df.merge(
    pred_df[
        merge_cols + ["true", "pred", "prob"]
    ],
    on=merge_cols,
    how="left"
)

missing_pred = val_df["pred"].isna().sum()

print("[INFO] val clips:", len(val_df))
print("[INFO] missing predictions:", missing_pred)

if missing_pred > 0:
    raise ValueError(
        "prediction이 val_clip_df와 일부 merge되지 않았습니다. "
        "val_clip_df.csv와 val_predictions.csv가 같은 fold에서 나온 파일인지 확인하세요."
    )


# =========================
# Main
# =========================
for patient_id, patient_df in val_df.groupby("patient_id"):

    patient_df = patient_df.sort_values(
        ["video_id", "clip_start"]
    ).copy()

    val_video_ids = patient_df["video_id"].unique()

    raw_patient_df = all_clip_df[
        (all_clip_df["patient_id"] == str(patient_id)) &
        (all_clip_df["video_id"].isin(val_video_ids))
    ].copy()

    raw_patient_df = raw_patient_df.sort_values(
        ["video_id", "clip_start"]
    )

    montage_frames = []
    raw_montage_frames = []

    # =========================
    # 1) Filtered prediction montage
    # =========================
    for _, row in patient_df.iterrows():

        arr = np.load(row["npy_path"])
        center = arr[len(arr) // 2]

        frame = cv2.cvtColor(center, cv2.COLOR_RGB2BGR)

        filtered_frame = apply_result_filter(
            frame.copy(),
            true=row["true"],
            pred=row["pred"],
            alpha=0.35
        )

        filtered_frame = draw_time_text(
            filtered_frame,
            row
        )

        montage_frames.append(filtered_frame)

    if len(montage_frames) > 0:
        montage = np.concatenate(
            montage_frames,
            axis=1
        )

        title = (
            f"{patient_id} | "
            f"red=TP(eye)  "
            f"blue=TN(no eye)  "
            f"gray=wrong"
        )

        montage = add_title_bar(
            montage,
            title
        )

        cv2.imwrite(
            str(OUT_DIR / f"{patient_id}.png"),
            montage
        )

    # =========================
    # 2) Raw montage
    # 호명 이전 포함 전체 clip 기준
    # =========================
    if len(raw_patient_df) == 0:
        print(f"[WARN] raw clips not found: {patient_id}")
    else:
        for _, row in raw_patient_df.iterrows():

            arr = np.load(row["npy_path"])
            center = arr[len(arr) // 2]

            raw_frame = cv2.cvtColor(center, cv2.COLOR_RGB2BGR)

            raw_frame = draw_time_text(
                raw_frame,
                row
            )

            raw_montage_frames.append(raw_frame)

        raw_montage = np.concatenate(
            raw_montage_frames,
            axis=1
        )

        raw_title = f"{patient_id} | raw frames including before call"

        raw_montage = add_title_bar(
            raw_montage,
            raw_title
        )

        cv2.imwrite(
            str(RAW_OUT_DIR / f"{patient_id}.png"),
            raw_montage
        )

    # =========================
    # 첫 EC correct 찾기
    # =========================
    tp_df = patient_df[
        (patient_df["true"].astype(int) == 1) &
        (patient_df["pred"].astype(int) == 1)
    ].copy()

    if len(tp_df) == 0:
        print(f"[INFO] no EC correct: {patient_id}")
        continue

    first_ec_start = float(tp_df["clip_start"].min())

    after_ec_rows = patient_df[
        patient_df["clip_start"] >= first_ec_start
    ].copy()

    after_ec_rows = after_ec_rows.sort_values(
        ["video_id", "clip_start"]
    )

    # =========================
    # 3) Gaze video montage
    # EC correct가 처음 나온 시점 이후
    # =========================
    gaze_montage_frames = []

    for _, row in after_ec_rows.iterrows():

        video_id = row["video_id"]

        gaze_video_path = find_gaze_video(video_id)

        if gaze_video_path is None:
            print(
                f"[WARN] gaze video not found: "
                f"patient={patient_id}, video_id={video_id}"
            )
            continue

        center_time = (
            float(row["clip_start"]) +
            float(row["clip_end"])
        ) / 2.0

        gaze_frame = read_frame_at_time(
            gaze_video_path,
            center_time
        )

        if gaze_frame is None:
            continue

        gaze_frame = resize_like_reference(
            gaze_frame,
            montage_frames
        )

        cv2.putText(
            gaze_frame,
            "gaze video",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        gaze_frame = draw_time_text(
            gaze_frame,
            row
        )

        gaze_montage_frames.append(gaze_frame)

    if len(gaze_montage_frames) == 0:
        print(f"[WARN] no gaze montage frames saved: {patient_id}")
    else:
        gaze_montage = np.concatenate(
            gaze_montage_frames,
            axis=1
        )

        gaze_title = (
            f"{patient_id} | gaze video after first EC correct "
            f"from {first_ec_start:.1f}s"
        )

        gaze_montage = add_title_bar(
            gaze_montage,
            gaze_title
        )

        cv2.imwrite(
            str(GAZE_OUT_DIR / f"{patient_id}.png"),
            gaze_montage
        )

    # =========================
    # 4) exact_mat montage
    # 첫 EC correct 이후 val clip과 같은 시간 간격
    # =========================
    exact_video_path = find_exact_mat_video(
        patient_id=patient_id,
        video_id=val_video_ids[0] if len(val_video_ids) > 0 else None
    )

    if exact_video_path is None:
        print(f"[WARN] exact_mat video not found: patient={patient_id}")
    else:
        exact_montage_frames = []

        for _, row in after_ec_rows.iterrows():

            center_time = (
                float(row["clip_start"]) +
                float(row["clip_end"])
            ) / 2.0

            exact_frame = read_frame_at_time(
                exact_video_path,
                center_time
            )

            if exact_frame is None:
                continue

            exact_frame = resize_like_reference(
                exact_frame,
                montage_frames
            )

            cv2.putText(
                exact_frame,
                "exact_mat",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            exact_frame = draw_time_text(
                exact_frame,
                row
            )

            exact_montage_frames.append(exact_frame)

        if len(exact_montage_frames) == 0:
            print(f"[WARN] no exact_mat montage frames saved: {patient_id}")
        else:
            exact_montage = np.concatenate(
                exact_montage_frames,
                axis=1
            )

            exact_title = (
                f"{patient_id} | exact_mat video after first EC correct "
                f"from {first_ec_start:.1f}s"
            )

            exact_montage = add_title_bar(
                exact_montage,
                exact_title
            )

            cv2.imwrite(
                str(EXACT_OUT_DIR / f"{patient_id}.png"),
                exact_montage
            )

        # =========================
        # 5) Emotion montage
        # prediction filter 없이 감정/확률만 표시
        # =========================
        emotion_montage_frames = []
        emotion_cache = {}

        for _, row in after_ec_rows.iterrows():

            video_id = row["video_id"]

            if video_id not in emotion_cache:
                json_path = find_eyecont_json(video_id)

                if json_path is None:
                    print(
                        f"[WARN] eyecont json not found: "
                        f"patient={patient_id}, video_id={video_id}"
                    )
                    emotion_cache[video_id] = []
                else:
                    try:
                        json_data = load_json(json_path)
                        emotion_events = extract_emotion_events(json_data)
                        emotion_cache[video_id] = emotion_events

                        print(
                            f"[INFO] emotion events: "
                            f"patient={patient_id}, video_id={video_id}, "
                            f"n={len(emotion_events)}, json={json_path.name}"
                        )

                    except Exception as e:
                        print(
                            f"[WARN] failed to load emotion json: "
                            f"{json_path}, error={e}"
                        )
                        emotion_cache[video_id] = []

            emotion_events = emotion_cache[video_id]

            arr = np.load(row["npy_path"])
            center = arr[len(arr) // 2]

            # 원본 frame 그대로 사용
            emotion_frame = cv2.cvtColor(center, cv2.COLOR_RGB2BGR)

            emotion, emotion_prob = get_emotion_for_clip(
                emotion_events,
                clip_start=row["clip_start"],
                clip_end=row["clip_end"]
            )

            emotion_frame = draw_emotion_text_small(
                emotion_frame,
                emotion,
                emotion_prob
            )

            emotion_frame = draw_time_text_small(
                emotion_frame,
                row
            )

            emotion_montage_frames.append(emotion_frame)

        if len(emotion_montage_frames) == 0:
            print(f"[WARN] no emotion montage frames saved: {patient_id}")
        else:
            emotion_montage = np.concatenate(
                emotion_montage_frames,
                axis=1
            )

            emotion_title = (
                f"{patient_id} | emotion only from eyecont json "
                f"after first EC correct from {first_ec_start:.1f}s"
            )

            emotion_montage = add_title_bar(
                emotion_montage,
                emotion_title
            )

            cv2.imwrite(
                str(EMOTION_OUT_DIR / f"{patient_id}.png"),
                emotion_montage
            )

print(f"done filtered: {OUT_DIR}")
print(f"done raw: {RAW_OUT_DIR}")
print(f"done gaze: {GAZE_OUT_DIR}")
print(f"done exact_mat: {EXACT_OUT_DIR}")
print(f"done emotion: {EMOTION_OUT_DIR}")