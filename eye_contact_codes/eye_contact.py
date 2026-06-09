# pip install opencv-python mediapipe numpy
import os
import glob

import cv2
import json
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image
from transformers import pipeline

import time
import shutil


VIDEO_ROOT = "./video_data"
LABEL_ROOT = "./labels"
RESULT_TRUE_ROOT = "/storage/sohyunkang/eyecont_results_true"
RESULT_FALSE_ROOT = "/storage/sohyunkang/eyecont_results_false"

MODEL_PATH = "face_landmarker.task"
EMOTION_MODEL_NAME = "trpakov/vit-face-expression"

emotion_classifier = pipeline(
    "image-classification",
    model=EMOTION_MODEL_NAME,
    device=0  # GPU 사용. CPU면 -1
)

DETECT_SCALE = 3.0   # 얼굴 detection용 확대 배율. 2.0~3.0 추천

os.makedirs(LABEL_ROOT, exist_ok=True)
os.makedirs(RESULT_TRUE_ROOT, exist_ok=True)
os.makedirs(RESULT_FALSE_ROOT, exist_ok=True)

MIRROR_X = False

LEFT_EYE_POINTS = [
    33, 246, 161, 160, 159, 158, 157, 173,
    7, 163, 144, 145, 153, 154, 155
]

RIGHT_EYE_POINTS = [
    263, 466, 388, 387, 386, 385, 384, 398,
    249, 390, 373, 374, 380, 381, 382
]


class Calibration:
    def __init__(self, nb_frames=5):
        self.nb_frames = nb_frames
        self.thresholds_left = []
        self.thresholds_right = []

    @property
    def is_complete(self):
        return (
            len(self.thresholds_left) >= self.nb_frames
            and len(self.thresholds_right) >= self.nb_frames
        )

    def threshold(self, side):
        arr = self.thresholds_left if side == 0 else self.thresholds_right
        return int(np.mean(arr)) if arr else 50

    def evaluate(self, eye_frame, side):
        th = self.find_best_threshold(eye_frame)
        if side == 0:
            self.thresholds_left.append(th)
        else:
            self.thresholds_right.append(th)

    def find_best_threshold(self, eye_frame):
        average_iris_size = 0.48
        best_th = 50
        best_diff = float("inf")

        for th in range(5, 100, 5):
            iris_frame = self.image_processing(eye_frame, th)
            size = self.iris_size(iris_frame)
            diff = abs(size - average_iris_size)

            if diff < best_diff:
                best_diff = diff
                best_th = th

        return best_th

    def image_processing(self, eye_frame, th):
        filtered = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(filtered, kernel, iterations=3)
        _, binary = cv2.threshold(eroded, th, 255, cv2.THRESH_BINARY)
        return binary

    def iris_size(self, frame):
        h, w = frame.shape[:2]
        if h <= 10 or w <= 10:
            return 0.0

        roi = frame[5:h - 5, 5:w - 5]
        nb_pixels = roi.size
        nb_whites = cv2.countNonZero(roi)
        nb_blacks = nb_pixels - nb_whites

        return nb_blacks / nb_pixels if nb_pixels > 0 else 0.0

class Eye:
    def __init__(self, eye_region=None):
        self.eye_region = eye_region

    @staticmethod
    def build(landmarks, side):
        points = LEFT_EYE_POINTS if side == 0 else RIGHT_EYE_POINTS
        region = np.array([landmarks[i] for i in points], dtype=np.int32)
        return Eye(eye_region=region)

class GazeTracker:
    def __init__(self):
        self.eye_left = None
        self.eye_right = None

        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

    def update(self, landmarks):
        self.eye_left = Eye.build(landmarks, 0)
        self.eye_right = Eye.build(landmarks, 1)

def build_iris_gaze_result(
        landmarks,
        width,
        height,
        gaze_tracker
    ):

    left_eye_center = polygon_centroid(
        np.array([landmarks[i] for i in LEFT_EYE_POINTS], dtype=np.int32)
    )
    right_eye_center = polygon_centroid(
        np.array([landmarks[i] for i in RIGHT_EYE_POINTS], dtype=np.int32)
    )

    left_iris_pts = np.array([landmarks[i] for i in LEFT_IRIS], dtype=np.int32)
    right_iris_pts = np.array([landmarks[i] for i in RIGHT_IRIS], dtype=np.int32)

    left_pupil = tuple(np.mean(left_iris_pts, axis=0).astype(int))
    right_pupil = tuple(np.mean(right_iris_pts, axis=0).astype(int))

    left_eye_width = np.ptp(np.array([landmarks[i][0] for i in LEFT_EYE_POINTS]))
    right_eye_width = np.ptp(np.array([landmarks[i][0] for i in RIGHT_EYE_POINTS]))

    left_eye_height = np.ptp(np.array([landmarks[i][1] for i in LEFT_EYE_POINTS]))
    right_eye_height = np.ptp(np.array([landmarks[i][1] for i in RIGHT_EYE_POINTS]))

    left_dx = (left_pupil[0] - left_eye_center[0]) / max(left_eye_width, 1)
    right_dx = (right_pupil[0] - right_eye_center[0]) / max(right_eye_width, 1)

    left_dy = (left_pupil[1] - left_eye_center[1]) / max(left_eye_height, 1)
    right_dy = (right_pupil[1] - right_eye_center[1]) / max(right_eye_height, 1)

    raw_dx = float((left_dx + right_dx) / 2)
    raw_dy = float((left_dy + right_dy) / 2)

    # smoothing
    alpha = 0.82

    gaze_tracker.smooth_dx = (
        alpha * gaze_tracker.smooth_dx
        + (1 - alpha) * raw_dx
    )

    gaze_tracker.smooth_dy = (
        alpha * gaze_tracker.smooth_dy
        + (1 - alpha) * raw_dy
    )

    horizontal_offset = gaze_tracker.smooth_dx
    vertical_offset = gaze_tracker.smooth_dy

    # threshold는 영상 보고 조정 가능
    is_left = horizontal_offset < -0.1
    is_right = horizontal_offset > 0.1
    is_up = vertical_offset < -0.1
    is_down = vertical_offset > 0.1

    is_center = not is_left and not is_right and not is_up and not is_down

    if is_left:
        horizontal = "LEFT"
    elif is_right:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"

    if is_up:
        vertical = "UP"
    elif is_down:
        vertical = "DOWN"
    else:
        vertical = "CENTER"

    if horizontal == "CENTER" and vertical == "CENTER":
        direction = "CENTER"
    else:
        direction = f"{vertical}-{horizontal}"

    try:
        left_ear = eye_aspect_ratio(landmarks, "left")
        right_ear = eye_aspect_ratio(landmarks, "right")
        ear = float((left_ear + right_ear) / 2)
    except Exception:
        left_ear = 1.0
        right_ear = 1.0
        ear = 1.0

    is_blinking = ear < 0.07

    return {
        "source": "iris_landmark",

        "left_pupil": {
            "x": float(left_pupil[0]),
            "y": float(left_pupil[1]),
        },
        "right_pupil": {
            "x": float(right_pupil[0]),
            "y": float(right_pupil[1]),
        },

        "left_pupil_norm": {
            "x": float(left_pupil[0] / max(width, 1)),
            "y": float(left_pupil[1] / max(height, 1)),
        },
        "right_pupil_norm": {
            "x": float(right_pupil[0] / max(width, 1)),
            "y": float(right_pupil[1] / max(height, 1)),
        },

        "left_eye_center": {
            "x": float(left_eye_center[0]),
            "y": float(left_eye_center[1]),
        },
        "right_eye_center": {
            "x": float(right_eye_center[0]),
            "y": float(right_eye_center[1]),
        },

        "horizontal_offset": horizontal_offset,
        "vertical_offset": vertical_offset,

        "is_left": is_left,
        "is_right": is_right,
        "is_up": is_up,
        "is_down": is_down,
        "is_center": is_center,

        "direction": direction,
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "ear": ear,
        "is_blinking": is_blinking,
    }

def make_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.FaceLandmarker.create_from_options(options)


def contour_centroid(contour):
    if contour is None:
        return None

    m = cv2.moments(contour)

    if m["m00"] == 0:
        return None

    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def polygon_centroid(points):
    if points is None or len(points) == 0:
        return None

    pts = np.array(points, dtype=np.float32)

    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    return cx, cy

def eye_aspect_ratio(landmarks, side):
    if side == "left":
        # MediaPipe left eye points
        outer = landmarks[33]
        inner = landmarks[133]
        top1 = landmarks[159]
        top2 = landmarks[160]
        bottom1 = landmarks[145]
        bottom2 = landmarks[144]

    else:
        # MediaPipe right eye points
        outer = landmarks[263]
        inner = landmarks[362]
        top1 = landmarks[386]
        top2 = landmarks[387]
        bottom1 = landmarks[374]
        bottom2 = landmarks[373]

    eye_width = math.dist(outer, inner)

    if eye_width < 1e-6:
        return 1.0

    eye_height1 = math.dist(top1, bottom1)
    eye_height2 = math.dist(top2, bottom2)

    ear = (eye_height1 + eye_height2) / (2.0 * eye_width)

    return float(ear)

def draw_eye_debug(frame, eye, eye_color, pupil_color=None, label=""):
    if eye is None:
        return

    # eye contour polygon만 그림
    if eye.eye_region is not None:
        eye_contour = eye.eye_region.reshape((-1, 1, 2)).astype(np.int32)

        cv2.polylines(
            frame,
            [eye_contour],
            isClosed=True,
            color=eye_color,
            thickness=1
        )

        # eye contour 중심점
        eye_center = polygon_centroid(eye.eye_region)

        if eye_center is not None:
            cv2.circle(frame, eye_center, 2, eye_color, -1)
            cv2.circle(frame, eye_center, 5, eye_color, 1)


def draw_iris_debug(frame, landmarks):
    left_iris_pts = np.array(
        [landmarks[i] for i in LEFT_IRIS],
        dtype=np.int32
    )

    right_iris_pts = np.array(
        [landmarks[i] for i in RIGHT_IRIS],
        dtype=np.int32
    )

    # left iris contour
    cv2.polylines(
        frame,
        [left_iris_pts.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    # right iris contour
    cv2.polylines(
        frame,
        [right_iris_pts.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 165, 255),
        thickness=2
    )

    left_center = tuple(np.mean(left_iris_pts, axis=0).astype(int))
    right_center = tuple(np.mean(right_iris_pts, axis=0).astype(int))

    # iris center는 X로 표시
    cv2.drawMarker(
        frame,
        left_center,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=1
    )

    cv2.drawMarker(
        frame,
        right_center,
        (0, 165, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=1
    )

    return {
        "left": {
            "x": float(left_center[0]),
            "y": float(left_center[1])
        },
        "right": {
            "x": float(right_center[0]),
            "y": float(right_center[1])
        }
    }

def draw_gaze_vector(frame, gaze, length=100):
    if gaze is None:
        return

    left = gaze.get("left_pupil")
    right = gaze.get("right_pupil")

    if left is None or right is None:
        return

    cx = int((left["x"] + right["x"]) / 2)
    cy = int((left["y"] + right["y"]) / 2)

    dx = gaze.get("horizontal_offset", 0.0)
    dy = gaze.get("vertical_offset", 0.0)

    end_x = int(cx + dx * length * 3)
    end_y = int(cy + dy * length * 3)

    cv2.arrowedLine(
        frame,
        (cx, cy),
        (end_x, end_y),
        (255, 0, 255),
        2,
        tipLength=0.3
    )

def get_gaze_direction_label(gaze):
    if gaze is None:
        return "NO GAZE"
    return gaze.get("direction", "UNKNOWN")

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def iris_fallback_from_landmarks(landmarks, width, height):
    def mean_point(indices):
        pts = np.array([landmarks[i] for i in indices], dtype=np.float32)
        x, y = pts.mean(axis=0)
        return {
            "x": float(x),
            "y": float(y),
            "norm": {
                "x": float(x / max(width, 1)),
                "y": float(y / max(height, 1)),
            }
        }

    return {
        "left": mean_point(LEFT_IRIS),
        "right": mean_point(RIGHT_IRIS),
    }

def detect_face_on_center_zoom(
    landmarker,
    frame,
    timestamp_ms,
    crop_ratio=0.5
):
    height, width = frame.shape[:2]

    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)

    cx = width // 2
    cy = height // 2

    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)
    x2 = min(cx + crop_w // 2, width)
    y2 = min(cy + crop_h // 2, height)

    cropped = frame[y1:y2, x1:x2]

    zoomed = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=zoomed_rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]

    landmarks = []

    for lm in face:
        zx = lm.x * width
        zy = lm.y * height

        ox = zx * (crop_w / width) + x1
        oy = zy * (crop_h / height) + y1

        landmarks.append((int(ox), int(oy)))

    return landmarks

def classify_emotion_from_face(frame, bbox, top_k=3):
    """
    frame: BGR 원본 frame
    bbox: (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = bbox

    h, w = frame.shape[:2]

    margin = 0.15

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = max(x1 - mx, 0)
    y1 = max(y1 - my, 0)
    x2 = min(x2 + mx, w - 1)
    y2 = min(y2 + my, h - 1)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    results = emotion_classifier(pil_img, top_k=top_k)

    return [
        {
            "label": r["label"],
            "score": float(r["score"])
        }
        for r in results
    ]

def process_video_single_crop(video_path, crop_ratio):
    has_pupil_detection = False

    base_name = os.path.basename(video_path)
    video_id = base_name.replace(".mp4.mp4", "")

    temp_json = os.path.join(
        LABEL_ROOT,
        f"{video_id}_crop{crop_ratio}.json"
    )

    temp_video = os.path.join(
        RESULT_TRUE_ROOT,
        f"{video_id}_crop{crop_ratio}_temp.mp4"
    )

    temp_final_video = os.path.join(
        RESULT_TRUE_ROOT,
        f"{video_id}_crop{crop_ratio}_gaze_result.mp4"
    )

    print(f"\n[PROCESS] {video_id} | crop_ratio={crop_ratio}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return False, temp_json, temp_final_video

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    gaze_tracker = GazeTracker()
    all_results = []

    with make_landmarker() as landmarker:
        frame_idx = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)

            landmarks = detect_face_on_center_zoom(
                landmarker=landmarker,
                frame=frame,
                timestamp_ms=timestamp_ms,
                crop_ratio=crop_ratio
            )

            record = {
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "face_detected": False,
                "gaze": None,
                "crop_ratio": crop_ratio,
            }

            if landmarks is not None:
                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]

                x1 = max(min(xs), 0)
                y1 = max(min(ys), 0)
                x2 = min(max(xs), width - 1)
                y2 = min(max(ys), height - 1)

                emotion = classify_emotion_from_face(
                    frame=frame,
                    bbox=(x1, y1, x2, y2),
                    top_k=3
                )

                record["emotion"] = emotion

                if emotion is not None and len(emotion) > 0:
                    emo_label = emotion[0]["label"]
                    emo_score = emotion[0]["score"]

                    cv2.putText(
                        frame,
                        f"EMO: {emo_label} ({emo_score:.2f})",
                        (x1, min(y2 + 25, height - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                gaze_tracker.update(landmarks)

                gaze = build_iris_gaze_result(
                    landmarks,
                    width,
                    height,
                    gaze_tracker
                )

                if not gaze["is_blinking"]:

                    draw_eye_debug(
                        frame,
                        gaze_tracker.eye_left,
                        eye_color=(0, 255, 0),
                        label="L"
                    )

                    draw_eye_debug(
                        frame,
                        gaze_tracker.eye_right,
                        eye_color=(255, 255, 0),
                        label="R"
                    )

                    draw_iris_debug(frame, landmarks)

                    draw_gaze_vector(frame, gaze, length=100)

                    gaze_label = get_gaze_direction_label(gaze)

                    cv2.putText(
                        frame,
                        f"GAZE: {gaze_label} | CROP: {crop_ratio}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                else:

                    cv2.putText(
                        frame,
                        "BLINK",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                record["face_detected"] = True
                record["gaze"] = gaze

                if (
                    gaze["left_pupil"] is not None
                    and gaze["right_pupil"] is not None
                ):
                    has_pupil_detection = True

            all_results.append(record)
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    ffmpeg_cmd = f'''
    ffmpeg -loglevel error -y \
    -i "{temp_video}" \
    -i "{video_path}" \
    -c:v copy \
    -c:a aac \
    -map 0:v:0 \
    -map 1:a:0? \
    -shortest \
    "{temp_final_video}"
    '''

    os.system(ffmpeg_cmd)

    if os.path.exists(temp_video):
        os.remove(temp_video)

    return has_pupil_detection, temp_json, temp_final_video


def process_video(video_path):
    base_name = os.path.basename(video_path)
    video_id = base_name.replace(".mp4.mp4", "")

    crop_ratios = [0.8, 0.5, 0.35]

    temp_files = []

    final_true_json = os.path.join(
        RESULT_TRUE_ROOT,
        f"{video_id}.json"
    )

    final_true_video = os.path.join(
        RESULT_TRUE_ROOT,
        f"{video_id}_gaze_result.mp4"
    )

    final_false_json = os.path.join(
        RESULT_FALSE_ROOT,
        f"{video_id}.json"
    )

    final_false_video = os.path.join(
        RESULT_FALSE_ROOT,
        f"{video_id}_gaze_result.mp4"
    )

    last_json = None
    last_video = None

    for crop_ratio in crop_ratios:
        detected, temp_json, temp_video = process_video_single_crop(
            video_path=video_path,
            crop_ratio=crop_ratio
        )

        temp_files.append((temp_json, temp_video))
        last_json = temp_json
        last_video = temp_video

        if detected:
            print(
                f"[SUCCESS] {video_id} "
                f"crop_ratio={crop_ratio}"
            )

            if os.path.exists(temp_json):
                shutil.move(temp_json, final_true_json)

            if os.path.exists(temp_video):
                shutil.move(temp_video, final_true_video)

            # 성공한 crop 외 이전 실패 crop 임시 파일 삭제
            for j, v in temp_files:
                if j != final_true_json and os.path.exists(j):
                    os.remove(j)
                if v != final_true_video and os.path.exists(v):
                    os.remove(v)

            print(f"[TRUE JSON] {final_true_json}")
            print(f"[TRUE VIDEO] {final_true_video}")

            return True

        print(
            f"[RETRY] {video_id} "
            f"failed at crop_ratio={crop_ratio}"
        )

    print(
        f"[FINAL FALSE] {video_id} "
        f"failed at all crop ratios"
    )

    if last_json is not None and os.path.exists(last_json):
        shutil.move(last_json, final_false_json)

    if last_video is not None and os.path.exists(last_video):
        shutil.move(last_video, final_false_video)

    # 마지막 false 저장본 외 이전 crop 임시 파일 삭제
    for j, v in temp_files:
        if j != final_false_json and os.path.exists(j):
            os.remove(j)
        if v != final_false_video and os.path.exists(v):
            os.remove(v)

    print(f"[FALSE JSON] {final_false_json}")
    print(f"[FALSE VIDEO] {final_false_video}")

    return False

def main():

    pattern = os.path.join(
        VIDEO_ROOT,
        "**",
        "IF2001*.mp4.mp4"
    )

    video_paths = glob.glob(
        pattern,
        recursive=True
    )
    
    print(f"[FOUND VIDEOS] {len(video_paths)}")

    process_count = 0
    skip_count = 0

    detected_videos = []
    undetected_videos = []
    skipped_videos = []


    for video_path in video_paths:
        # if "1023103112" not in video_path:
        #     continue
        start = time.time()

        base_name = os.path.basename(video_path)
        video_id = base_name.replace(".mp4.mp4", "")

        true_json_path = os.path.join(
            RESULT_TRUE_ROOT,
            f"{video_id}.json"
        )

        false_json_path = os.path.join(
            RESULT_FALSE_ROOT,
            f"{video_id}.json"
        )

        if (
            os.path.exists(true_json_path)
            or
            os.path.exists(false_json_path)
        ):
            print(f"[SKIP] {video_id}")
            skip_count += 1
            skipped_videos.append(video_id)
            continue

        detected = process_video(video_path)

        print(
            f"[PROCESS RESULT] "
            f"{video_id} "
            f"pupil_detected={detected} "
            f"time={time.time()-start}"
        )

        process_count += 1

        if detected:
            detected_videos.append(video_id)
        else:
            undetected_videos.append(video_id)

    print("\n========== DONE ==========")
    print(f"Processed: {process_count}")
    print(f"Skipped: {skip_count}")

    print("\n========== PUPIL DETECTION SUMMARY ==========")

    print(f"\nDetected videos: {len(detected_videos)}")
    # for name in detected_videos:
    #     print(f"  - {name}")

    print(f"\nUndetected videos: {len(undetected_videos)}")
    # for name in undetected_videos:
    #     print(f"  - {name}")

    print(f"\nSkipped videos: {len(skipped_videos)}")
    # for name in skipped_videos:
    #     print(f"  - {name}")

if __name__ == "__main__":
    main()