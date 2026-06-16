import os
import glob
import json
import cv2
import numpy as np
import pandas as pd


def find_json_files(json_root):
    return glob.glob(os.path.join(json_root, "**", "*.json"), recursive=True)


def find_matching_video(video_root, json_path):
    """
    json 파일명과 같은 이름의 .mp4.mp4 영상 탐색

    예:
    IF2001_0.json
    → IF2001_0.mp4.mp4
    """

    base = os.path.splitext(os.path.basename(json_path))[0]
    target_name = base + ".mp4.mp4"

    matches = glob.glob(
        os.path.join(video_root, "**", target_name),
        recursive=True
    )

    if len(matches) == 0:
        return None

    return matches[0]


def has_valid_pupil(item):
    gaze = item.get("gaze", None)

    if not item.get("face_detected", False):
        return False

    if gaze is None:
        return False

    return (
        gaze.get("left_pupil") is not None
        and gaze.get("right_pupil") is not None
    )


def temporal_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    return n_frames / fps

def json_to_eye_contact_events(
    json_path,
    video_id,
    patient_id,
    min_duration_sec=0.3,
    max_gap_sec=0.2
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []

    current_start = None
    current_end = None
    last_valid_time = None

    for item in data:
        timestamp_sec = item["timestamp_ms"] / 1000.0
        valid = has_valid_pupil(item)

        if valid:
            if current_start is None:
                current_start = timestamp_sec

            current_end = timestamp_sec
            last_valid_time = timestamp_sec

        else:
            if current_start is not None:
                gap = timestamp_sec - last_valid_time

                if gap > max_gap_sec:
                    duration = current_end - current_start

                    if duration >= min_duration_sec:
                        events.append({
                            "patient_id": patient_id,
                            "video_id": video_id,
                            "start_time": current_start,
                            "end_time": current_end,
                            "label": "eye_contact"
                        })

                    current_start = None
                    current_end = None
                    last_valid_time = None

    if current_start is not None:
        duration = current_end - current_start

        if duration >= min_duration_sec:
            events.append({
                "patient_id": patient_id,
                "video_id": video_id,
                "start_time": current_start,
                "end_time": current_end,
                "label": "eye_contact"
            })

    return events


def build_label_table_from_jsons(
    json_root,
    video_root,
    min_duration_sec=0.3,
    max_gap_sec=0.2
):
    json_files = find_json_files(json_root)

    print(f"[INFO] Found JSON files: {len(json_files)}")

    rows = []
    matched_count = 0
    missing_video_count = 0

    for json_path in json_files:
        video_path = find_matching_video(video_root, json_path)

        if video_path is None:
            missing_video_count += 1
            print(f"[WARN] Matching video not found for: {json_path}")
            continue

        matched_count += 1

        video_id = os.path.splitext(os.path.basename(json_path))[0]
        
        filename = os.path.basename(video_path)
        parts = filename.split("_")

        if len(parts) >= 4:
            patient_id = parts[3]
        else:
            patient_id = video_id

        events = json_to_eye_contact_events(
            json_path=json_path,
            video_id=video_id,
            patient_id=patient_id,
            min_duration_sec=min_duration_sec,
            max_gap_sec=max_gap_sec
        )

        for e in events:
            e["json_path"] = json_path
            e["video_path"] = video_path

        rows.extend(events)

    label_df = pd.DataFrame(rows)

    print(f"[INFO] Matched videos: {matched_count}")
    print(f"[INFO] Missing videos: {missing_video_count}")
    print(f"[INFO] Generated eye-contact events: {len(label_df)}")

    if len(label_df) == 0:
        raise ValueError("No labels generated from JSON files.")

    return label_df


def temporal_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    return n_frames / fps


def get_person_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


_PERSON_DETECTOR = None


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


def detect_person_box(
    frame,
    conf_threshold=0.3,
    center_weight=0.7,
):
    detector = get_person_detector()

    h, w, _ = frame.shape
    cx = w / 2
    cy = h / 2

    results = detector(frame, verbose=False)[0]

    candidates = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != 0 or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        area = (x2 - x1) * (y2 - y1)

        bx = (x1 + x2) / 2
        by = (y1 + y2) / 2

        dist = np.sqrt(
            ((bx - cx) / w) ** 2
            + ((by - cy) / h) ** 2
        )

        # 큰 사람 + 중앙에 가까운 사람 우선
        score = area * (1.0 - center_weight * dist)

        candidates.append(
            {
                "score": score,
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            }
        )

    if len(candidates) == 0:
        return None

    best = max(candidates, key=lambda x: x["score"])

    return best["box"]


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
    x1, y1, x2, y2 = expand_box(
        box,
        frame.shape,
        margin=margin
    )

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame

    return crop

_VIDEO_BOX_CACHE = {}


def get_video_median_person_box(
    video_path,
    sample_every_sec=1.0,
    conf_threshold=0.3,
):
    if video_path in _VIDEO_BOX_CACHE:
        return _VIDEO_BOX_CACHE[video_path]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    boxes = []

    t = 0.0

    while t < duration:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()

        if not ret:
            t += sample_every_sec
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        box = detect_person_box(
            frame,
            conf_threshold=conf_threshold
        )

        if box is not None:
            boxes.append(box)

        t += sample_every_sec

    cap.release()

    if len(boxes) == 0:
        median_box = None
    else:
        median_box = np.median(
            np.stack(boxes),
            axis=0
        )

    _VIDEO_BOX_CACHE[video_path] = median_box

    return median_box

def read_clip_stable_person(
    video_path,
    start_time,
    clip_duration,
    num_frames,
    image_size,
    crop_mode="person",
    crop_ratio=0.8,
    person_margin=0.2,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    times = np.linspace(
        start_time,
        start_time + clip_duration,
        num_frames,
        endpoint=False
    )

    raw_frames = []
    boxes = []

    # -------------------------
    # 1차 pass: frame 읽고 person box 수집
    # -------------------------
    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()

        if not ret:
            if len(raw_frames) > 0:
                frame = raw_frames[-1].copy()
            else:
                frame = np.zeros(
                    (image_size, image_size, 3),
                    dtype=np.uint8
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame)

        if crop_mode == "person":
            box = detect_person_box(frame)
            if box is not None:
                boxes.append(box)

    cap.release()

    # -------------------------
    # clip 단위 median box
    # -------------------------
    video_box = None

    if crop_mode == "person":
        video_box = get_video_median_person_box(
            video_path,
            sample_every_sec=1.0,
            conf_threshold=0.3
        )

    frames = []

    # -------------------------
    # 2차 pass: 같은 crop box로 안정화
    # -------------------------
    for frame in raw_frames:
        if crop_mode == "person":
            if video_box is not None:
                frame = crop_by_box(
                    frame,
                    video_box,
                    margin=person_margin
                )
            else:
                frame = crop_center(
                    frame,
                    crop_ratio=crop_ratio
                )

        elif crop_mode == "center":
            frame = crop_center(
                frame,
                crop_ratio=crop_ratio
            )

        elif crop_mode == "none":
            pass

        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")

        frame = cv2.resize(
            frame,
            (image_size, image_size)
        )

        frames.append(frame)

    frames = np.stack(frames).astype(np.float32) / 255.0

    return frames

# def crop_person_with_yolo(frame, margin=0.2, conf_threshold=0.3):
#     global _PERSON_DETECTOR

#     if _PERSON_DETECTOR is None:
#         _PERSON_DETECTOR = get_person_detector()

#     h, w, _ = frame.shape

#     results = _PERSON_DETECTOR(frame, verbose=False)[0]

#     person_boxes = []

#     for box in results.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])

#         # COCO class 0 = person
#         if cls == 0 and conf >= conf_threshold:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             area = (x2 - x1) * (y2 - y1)
#             person_boxes.append((area, x1, y1, x2, y2))

#     if len(person_boxes) == 0:
#         return frame

#     _, x1, y1, x2, y2 = max(person_boxes, key=lambda x: x[0])

#     bw = x2 - x1
#     bh = y2 - y1

#     mx = bw * margin
#     my = bh * margin

#     x1 = int(max(x1 - mx, 0))
#     y1 = int(max(y1 - my, 0))
#     x2 = int(min(x2 + mx, w))
#     y2 = int(min(y2 + my, h))

#     crop = frame[y1:y2, x1:x2]

#     if crop.size == 0:
#         return frame

#     return crop


# def read_clip(
#     video_path,
#     start_time,
#     clip_duration,
#     num_frames,
#     image_size,
#     crop_mode="center",
#     crop_ratio=0.8,
#     person_margin=0.2,
# ):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video: {video_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS)

#     times = np.linspace(
#         start_time,
#         start_time + clip_duration,
#         num_frames,
#         endpoint=False
#     )

#     frames = []

#     for t in times:
#         frame_idx = int(t * fps)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

#         ret, frame = cap.read()

#         if not ret:
#             if len(frames) > 0:
#                 frame = frames[-1]
#             else:
#                 frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         if crop_mode == "center":
#             frame = crop_center(frame, crop_ratio=crop_ratio)

#         elif crop_mode == "person":
#             frame = crop_person_with_yolo(
#                 frame,
#                 margin=person_margin,
#                 conf_threshold=0.3
#             )

#         elif crop_mode == "none":
#             pass

#         else:
#             raise ValueError(f"Unknown crop_mode: {crop_mode}")

#         frame = cv2.resize(frame, (image_size, image_size))

#         frames.append(frame)

#     cap.release()

#     frames = np.stack(frames).astype(np.float32) / 255.0

#     return frames

