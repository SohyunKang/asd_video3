import os
import json
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


class CFG:
    video_root = "/storage/sohyunkang/video_data"
    annotation_excel = "./demographics/0626 호명_시간기록.xlsx"
    annotation_header = 1
    id_col = "연구대상자ID"

    # 새 폴더 권장: 기존 npy skip 방지
    out_dir = "/storage/sohyunkang/preprocessed_video_frames_person_track_landscape_224"

    image_size = 224
    frame_stride = 1
    max_duration_sec = 15.0

    yolo_weight = "yolov8l.pt"
    conf_threshold = 0.2

    smooth_alpha = 0.8
    max_missing = 30

    crop_mode = "person"
    crop_ratio = 0.8
    person_margin = 0.2

    # 세로 영상이면 반시계 90도 회전해서 가로로 맞춤
    rotate_portrait_to_landscape = True

    save_metadata_every = 20
    save_montage = True
    montage_max_frames = 16
    montage_dirname = "montages"


# ============================================================
# ID / file utils
# ============================================================

def normalize_video_id(video_path):
    name = Path(video_path).name
    video_id = name

    while video_id.lower().endswith(".mp4"):
        video_id = video_id[:-4]

    return video_id


def get_visit_type(video_path):
    video_id = normalize_video_id(video_path)
    return "fu" if video_id.endswith("_fu") else "baseline"


def get_base_video_id(video_path):
    video_id = normalize_video_id(video_path)
    return video_id[:-3] if video_id.endswith("_fu") else video_id


def extract_patient_id_from_filename(video_path):
    """
    IF2001_4_1_1724083008_0.mp4.mp4
    IF2001_4_1_1724083008_0_fu.mp4.mp4
    -> 1724083008
    """
    base_video_id = get_base_video_id(video_path)
    parts = base_video_id.split("_")

    if len(parts) >= 4:
        return parts[3]

    return base_video_id


def find_all_videos(video_root):
    video_root = Path(video_root)

    videos = []
    videos.extend(video_root.rglob("*.mp4"))
    videos.extend(video_root.rglob("*.MP4"))

    return sorted(set(str(v) for v in videos))


def load_annotation_ids(excel_path, id_col, header=1):
    if not os.path.exists(excel_path):
        print("[WARN] annotation excel not found:", excel_path)
        return []

    anno_df = pd.read_excel(excel_path, header=header)
    anno_df.columns = anno_df.columns.astype(str).str.strip()

    if id_col not in anno_df.columns:
        raise ValueError(
            f"Cannot find id_col={id_col}. "
            f"Available columns: {list(anno_df.columns)}"
        )

    ids = (
        anno_df[id_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return ids


def is_annotated_video(video_path, annotation_ids):
    patient_id = extract_patient_id_from_filename(video_path)
    return patient_id in set(str(x).strip() for x in annotation_ids)


def sort_videos_annotation_first(video_paths, annotation_ids):
    annotated = []
    remaining = []

    for vp in video_paths:
        if is_annotated_video(vp, annotation_ids):
            annotated.append(vp)
        else:
            remaining.append(vp)

    return sorted(annotated) + sorted(remaining), sorted(annotated), sorted(remaining)


def get_npy_path(video_path):
    video_id = normalize_video_id(video_path)
    visit = get_visit_type(video_path)

    out_subdir = Path(CFG.out_dir) / visit
    out_subdir.mkdir(parents=True, exist_ok=True)

    return out_subdir / f"{video_id}.npy"


# ============================================================
# Rotation utils
# ============================================================

def get_rotation_angle_by_video_shape(cap):
    """
    원본 video shape 기준:
    height > width 이면 세로 영상으로 보고 반시계 90도 회전.
    cv2 기준 angle=270이 반시계 90도.
    """
    if not CFG.rotate_portrait_to_landscape:
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if height > width:
        return 270

    return 0


def rotate_frame(frame, angle):
    if angle == 0:
        return frame
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unknown rotation angle: {angle}")


# ============================================================
# Montage
# ============================================================

def save_video_montage(arr, video_id, visit):
    """
    전체 저장 frame을 일정 구간으로 나누고,
    각 구간의 가운데 frame을 montage로 저장.
    """
    if not CFG.save_montage:
        return None

    if arr is None or len(arr) == 0:
        return None

    montage_dir = Path(CFG.out_dir) / CFG.montage_dirname / visit
    montage_dir.mkdir(parents=True, exist_ok=True)

    n = len(arr)
    k = min(CFG.montage_max_frames, n)

    boundaries = np.linspace(0, n, k + 1, dtype=int)

    idxs = []
    for i in range(k):
        start = boundaries[i]
        end = boundaries[i + 1]

        if end <= start:
            idx = start
        else:
            idx = (start + end - 1) // 2

        idxs.append(idx)

    frames = []

    for idx in idxs:
        frame_rgb = arr[idx]

        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

    montage = np.concatenate(frames, axis=1)

    montage_path = montage_dir / f"{video_id}_montage.jpg"
    cv2.imwrite(str(montage_path), montage)

    return str(montage_path)


# ============================================================
# Crop utils
# ============================================================

def crop_center(frame, crop_ratio=0.8):
    h, w, _ = frame.shape

    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    cx = w // 2
    cy = h // 2

    x1 = max(cx - crop_w // 2, 0)
    x2 = min(cx + crop_w // 2, w)

    y1 = max(cy - crop_h // 2, 0)
    y2 = min(cy + crop_h // 2, h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame

    return crop


def expand_box(box, frame_shape, margin=0.2):
    h, w, _ = frame_shape

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * margin
    my = bh * margin

    x1 = int(max(x1 - mx, 0))
    y1 = int(max(y1 - my, 0))
    x2 = int(min(x2 + mx, w))
    y2 = int(min(y2 + my, h))

    return x1, y1, x2, y2


def crop_by_box(frame, box, margin=0.2):
    x1, y1, x2, y2 = expand_box(box, frame.shape, margin)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame

    return crop


# ============================================================
# Person detection / tracking
# ============================================================

_PERSON_DETECTOR = None


def get_person_detector():
    global _PERSON_DETECTOR

    if _PERSON_DETECTOR is None:
        _PERSON_DETECTOR = YOLO(CFG.yolo_weight)

    return _PERSON_DETECTOR


def detect_person_candidates(frame_rgb, conf_threshold=0.2):
    detector = get_person_detector()
    results = detector(frame_rgb, verbose=False)[0]

    candidates = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # COCO class 0 = person
        if cls != 0 or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        candidates.append({
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "conf": conf,
        })

    return candidates


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def select_initial_person_box(candidates, frame_shape, center_weight=0.7):
    h, w, _ = frame_shape
    cx = w / 2
    cy = h / 2

    best_box = None
    best_score = -1e9

    for c in candidates:
        box = c["box"]
        conf = c["conf"]

        x1, y1, x2, y2 = box
        bx = (x1 + x2) / 2
        by = (y1 + y2) / 2

        dist = np.sqrt(
            ((bx - cx) / w) ** 2
            + ((by - cy) / h) ** 2
        )

        score = conf * (1.0 - center_weight * dist)

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


def select_by_iou(candidates, prev_box):
    best_box = None
    best_score = -1e9

    for c in candidates:
        box = c["box"]
        conf = c["conf"]

        iou = box_iou(box, prev_box)

        score = iou + 0.05 * conf

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


# ============================================================
# Main video processing
# ============================================================

def process_one_video(video_path):
    video_path = str(video_path)

    video_id = normalize_video_id(video_path)
    base_video_id = get_base_video_id(video_path)
    patient_id = extract_patient_id_from_filename(video_path)
    visit = get_visit_type(video_path)

    npy_path = get_npy_path(video_path)

    expected_montage_path = (
        Path(CFG.out_dir)
        / CFG.montage_dirname
        / visit
        / f"{video_id}_montage.jpg"
    )

    # 이미 처리된 영상은 skip
    if npy_path.exists():
        montage_path = None

        if expected_montage_path.exists():
            montage_path = str(expected_montage_path)

        elif CFG.save_montage:
            arr = np.load(str(npy_path), mmap_mode="r")
            montage_path = save_video_montage(
                arr=arr,
                video_id=video_id,
                visit=visit
            )

        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rotation_angle = get_rotation_angle_by_video_shape(cap)
            cap.release()
        else:
            fps = np.nan
            original_n_frames = np.nan
            original_width = np.nan
            original_height = np.nan
            rotation_angle = np.nan

        arr = np.load(str(npy_path), mmap_mode="r")
        saved_n_frames = len(arr)

        duration_sec = (
            original_n_frames / fps
            if fps is not None and not pd.isna(fps) and fps > 0
            and original_n_frames is not None and not pd.isna(original_n_frames)
            else np.nan
        )

        return {
            "patient_id": patient_id,
            "visit": visit,
            "video_id": video_id,
            "base_video_id": base_video_id,
            "video_path": video_path,
            "npy_path": str(npy_path),
            "montage_path": montage_path,
            "status": "skipped_exists",
            "rotation_angle": rotation_angle,
            "fps": fps,
            "original_width": original_width,
            "original_height": original_height,
            "original_n_frames": original_n_frames,
            "saved_n_frames": saved_n_frames,
            "duration_sec": duration_sec,
            "person_detected_frames": np.nan,
            "used_prev_box_frames": np.nan,
            "fallback_center_frames": np.nan,
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "patient_id": patient_id,
            "visit": visit,
            "video_id": video_id,
            "base_video_id": base_video_id,
            "video_path": video_path,
            "npy_path": str(npy_path),
            "montage_path": None,
            "status": "cannot_open",
            "rotation_angle": np.nan,
            "fps": np.nan,
            "original_width": np.nan,
            "original_height": np.nan,
            "original_n_frames": np.nan,
            "saved_n_frames": 0,
            "duration_sec": np.nan,
            "person_detected_frames": 0,
            "used_prev_box_frames": 0,
            "fallback_center_frames": 0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotation_angle = get_rotation_angle_by_video_shape(cap)

    if fps <= 0 or original_n_frames <= 0:
        cap.release()

        return {
            "patient_id": patient_id,
            "visit": visit,
            "video_id": video_id,
            "base_video_id": base_video_id,
            "video_path": video_path,
            "npy_path": str(npy_path),
            "montage_path": None,
            "status": "invalid_fps_or_frames",
            "rotation_angle": rotation_angle,
            "fps": fps,
            "original_width": original_width,
            "original_height": original_height,
            "original_n_frames": original_n_frames,
            "saved_n_frames": 0,
            "duration_sec": np.nan,
            "person_detected_frames": 0,
            "used_prev_box_frames": 0,
            "fallback_center_frames": 0,
        }

    if CFG.max_duration_sec is None:
        max_frame_to_read = original_n_frames
    else:
        max_frame_to_read = min(
            original_n_frames,
            int(CFG.max_duration_sec * fps)
        )

    frames = []

    prev_box = None
    missing_count = 0

    person_detected_frames = 0
    used_prev_box_frames = 0
    fallback_center_frames = 0

    frame_idx = 0

    pbar = tqdm(
        total=max_frame_to_read,
        desc=f"{visit}/{video_id}",
        leave=False
    )

    while frame_idx < max_frame_to_read:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        if frame_idx % CFG.frame_stride != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 핵심: 세로 영상이면 반시계 90도 회전
        frame_rgb = rotate_frame(frame_rgb, rotation_angle)

        selected_box = None

        if CFG.crop_mode == "person":
            candidates = detect_person_candidates(
                frame_rgb,
                conf_threshold=CFG.conf_threshold
            )

            if len(candidates) > 0:
                if prev_box is None:
                    selected_box = select_initial_person_box(
                        candidates,
                        frame_rgb.shape
                    )
                else:
                    selected_box = select_by_iou(
                        candidates,
                        prev_box
                    )

                if prev_box is not None and selected_box is not None:
                    selected_box = (
                        CFG.smooth_alpha * prev_box
                        + (1.0 - CFG.smooth_alpha) * selected_box
                    )

                prev_box = selected_box
                missing_count = 0
                person_detected_frames += 1

            else:
                if prev_box is not None and missing_count < CFG.max_missing:
                    selected_box = prev_box.copy()
                    missing_count += 1
                    used_prev_box_frames += 1
                else:
                    selected_box = None
                    fallback_center_frames += 1

            if selected_box is not None:
                frame_rgb = crop_by_box(
                    frame_rgb,
                    selected_box,
                    margin=CFG.person_margin
                )
            else:
                frame_rgb = crop_center(
                    frame_rgb,
                    crop_ratio=CFG.crop_ratio
                )

        elif CFG.crop_mode == "center":
            frame_rgb = crop_center(
                frame_rgb,
                crop_ratio=CFG.crop_ratio
            )

        elif CFG.crop_mode == "none":
            pass

        else:
            raise ValueError(f"Unknown crop_mode: {CFG.crop_mode}")

        frame_rgb = cv2.resize(
            frame_rgb,
            (CFG.image_size, CFG.image_size)
        )

        frames.append(frame_rgb.astype(np.uint8))

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if len(frames) == 0:
        return {
            "patient_id": patient_id,
            "visit": visit,
            "video_id": video_id,
            "base_video_id": base_video_id,
            "video_path": video_path,
            "npy_path": str(npy_path),
            "montage_path": None,
            "status": "no_frames_saved",
            "rotation_angle": rotation_angle,
            "fps": fps,
            "original_width": original_width,
            "original_height": original_height,
            "original_n_frames": original_n_frames,
            "saved_n_frames": 0,
            "duration_sec": original_n_frames / fps,
            "person_detected_frames": person_detected_frames,
            "used_prev_box_frames": used_prev_box_frames,
            "fallback_center_frames": fallback_center_frames,
        }

    arr = np.stack(frames, axis=0).astype(np.uint8)
    np.save(str(npy_path), arr)

    montage_path = save_video_montage(
        arr=arr,
        video_id=video_id,
        visit=visit
    )

    return {
        "patient_id": patient_id,
        "visit": visit,
        "video_id": video_id,
        "base_video_id": base_video_id,
        "video_path": video_path,
        "npy_path": str(npy_path),
        "montage_path": montage_path,
        "status": "saved",
        "rotation_angle": rotation_angle,
        "fps": fps,
        "original_width": original_width,
        "original_height": original_height,
        "original_n_frames": original_n_frames,
        "saved_n_frames": arr.shape[0],
        "duration_sec": original_n_frames / fps,
        "person_detected_frames": person_detected_frames,
        "used_prev_box_frames": used_prev_box_frames,
        "fallback_center_frames": fallback_center_frames,
    }


# ============================================================
# Run
# ============================================================

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    config = {
        "video_root": CFG.video_root,
        "annotation_excel": CFG.annotation_excel,
        "id_col": CFG.id_col,
        "out_dir": CFG.out_dir,
        "image_size": CFG.image_size,
        "frame_stride": CFG.frame_stride,
        "max_duration_sec": CFG.max_duration_sec,
        "yolo_weight": CFG.yolo_weight,
        "conf_threshold": CFG.conf_threshold,
        "smooth_alpha": CFG.smooth_alpha,
        "max_missing": CFG.max_missing,
        "crop_mode": CFG.crop_mode,
        "crop_ratio": CFG.crop_ratio,
        "person_margin": CFG.person_margin,
        "rotate_portrait_to_landscape": CFG.rotate_portrait_to_landscape,
        "rotation_rule": "if original height > width, rotate 90 degrees counterclockwise",
        "patient_id_rule": "split base video_id by '_' and use index 3",
        "followup_rule": "video_id ending with _fu is visit=fu",
        "skip_rule": "if target npy already exists, skip",
        "montage_rule": "sample segment-center frames evenly across saved frames",
    }

    config_path = Path(CFG.out_dir) / "preprocess_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    metadata_path = Path(CFG.out_dir) / "video_frame_metadata.csv"

    if metadata_path.exists():
        old_meta = pd.read_csv(metadata_path)
        metadata_rows = old_meta.to_dict("records")
    else:
        metadata_rows = []

    video_paths = find_all_videos(CFG.video_root)

    annotation_ids = load_annotation_ids(
        CFG.annotation_excel,
        CFG.id_col,
        header=CFG.annotation_header
    )

    ordered_videos, annotated_videos, remaining_videos = sort_videos_annotation_first(
        video_paths,
        annotation_ids
    )

    print("[INFO] video_root:", CFG.video_root)
    print("[INFO] total videos in storage:", len(video_paths))
    print("[INFO] annotation ids:", len(annotation_ids))
    print("[INFO] annotated videos first:", len(annotated_videos))
    print("[INFO] remaining videos:", len(remaining_videos))
    print("[INFO] output:", CFG.out_dir)

    for i, video_path in enumerate(tqdm(ordered_videos, desc="Videos")):
        row = process_one_video(video_path)
        metadata_rows.append(row)

        if (i + 1) % CFG.save_metadata_every == 0:
            temp_df = pd.DataFrame(metadata_rows)
            temp_df = temp_df.drop_duplicates(
                subset=["video_id", "visit", "npy_path"],
                keep="last"
            ).reset_index(drop=True)

            temp_df.to_csv(metadata_path, index=False)
            print("[INFO] intermediate metadata saved:", metadata_path)

    meta_df = pd.DataFrame(metadata_rows)

    if len(meta_df) > 0:
        meta_df = meta_df.drop_duplicates(
            subset=["video_id", "visit", "npy_path"],
            keep="last"
        ).reset_index(drop=True)

    meta_df.to_csv(metadata_path, index=False)

    print("[INFO] saved metadata:", metadata_path)
    print("[INFO] total metadata rows:", len(meta_df))

    if len(meta_df) > 0:
        print()
        print("[INFO] status counts:")
        print(meta_df["status"].value_counts(dropna=False))

        print()
        print("[INFO] visit counts:")
        print(meta_df["visit"].value_counts(dropna=False))

        print()
        print("[INFO] rotation counts:")
        print(meta_df["rotation_angle"].value_counts(dropna=False))

        print()
        print("[INFO] newly saved:")
        print((meta_df["status"] == "saved").sum())

        print("[INFO] skipped existing:")
        print((meta_df["status"] == "skipped_exists").sum())


if __name__ == "__main__":
    main()