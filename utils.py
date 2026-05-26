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
        patient_id = os.path.basename(os.path.dirname(video_path))

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


def read_clip(video_path, start_time, clip_duration, num_frames, image_size):
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

    frames = []

    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()

        if not ret:
            if len(frames) > 0:
                frame = frames[-1]
            else:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))

        frames.append(frame)

    cap.release()

    frames = np.stack(frames).astype(np.float32) / 255.0

    return frames
