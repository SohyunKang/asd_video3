import json
import pandas as pd
from pathlib import Path

# Eye contact detection 학습 결과가 저장된 폴더
EXP_DIR = Path(
    "/home/sohyunkang/asd_video3/experiments/annotated_labels_timesformer_20260708_105748"
)

CALL_JSON_PATH = Path(
    "/home/sohyunkang/asd_video3/demographics/260707_add_new_videos.json"
)

EYE_TRUE_ROOT = Path("/storage/sohyunkang/eyecont_results_true")
EYE_FALSE_ROOT = Path("/storage/sohyunkang/eyecont_results_false")

OUT_CSV = Path(
    "/home/sohyunkang/asd_video3/features/latency_duration.csv"
)


def normalize_video_id(x):
    x = str(x)
    x = Path(x).name
    x = x.replace("_gaze_result.mp4", "")
    x = x.replace(".mp4.mp4", "")
    x = x.replace(".mp4", "")
    x = x.replace(".json", "")
    return x


def find_eye_json_path(video_id):
    video_id = normalize_video_id(video_id)

    candidates = [
        EYE_TRUE_ROOT / f"{video_id}.json",
        EYE_FALSE_ROOT / f"{video_id}.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def get_first_call_segment(item):
    for seg in item.get("call_segments", []):
        if seg.get("callname") is True:
            return float(seg["start"]), float(seg["end"]), seg.get("word")

    return None, None, None


def get_first_face_detect_after_call(video_id, call_start):
    eye_json_path = find_eye_json_path(video_id)

    if eye_json_path is None:
        return None, "EYE_JSON_NOT_FOUND"

    try:
        with open(eye_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            face_detected = r.get("face_detected", False)

            if face_detected is True:
                face_time = float(r["timestamp_ms"]) / 1000.0

                if face_time >= call_start:
                    return face_time, ""

        return None, "NO_FACE_DETECT_AFTER_CALL"

    except Exception as e:
        return None, f"EYE_JSON_ERROR: {e}"


def load_val_predictions(exp_dir):
    dfs = []

    for fold_dir in sorted(exp_dir.glob("fold_*")):
        pred_path = fold_dir / "val_predictions.csv"

        if not pred_path.exists():
            continue

        df = pd.read_csv(pred_path)
        df["fold"] = fold_dir.name
        dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError("fold_*/val_predictions.csv 파일이 없습니다.")

    df = pd.concat(dfs, ignore_index=True)

    df["video_id"] = df["video_id"].apply(normalize_video_id)
    df["clip_start"] = pd.to_numeric(df["clip_start"], errors="coerce")
    df["clip_end"] = pd.to_numeric(df["clip_end"], errors="coerce")
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    return df


def get_first_real_eye_contact_after_call(video_id, call_start, val_df, target_col="true"):
    video_id = normalize_video_id(video_id)

    g = (
        val_df[
            (val_df["video_id"] == video_id)
            & (val_df["clip_end"] > call_start)
        ]
        .sort_values("clip_start")
        .reset_index(drop=True)
    )

    pos_g = g[g[target_col] == 1].copy()

    if len(pos_g) == 0:
        return (
            False,
            None,
            None,
            None,
            0.0,
            None,
            None,
            None,
            "NO_PREDICTED_EYE_CONTACT",
        )
    
    first_idx = pos_g.index[0]
    first_row = g.loc[first_idx]

    start = float(first_row["clip_start"])
    end = float(first_row["clip_end"])

    # 연속 true clip duration 계산
    duration_start = start
    duration_end = end

    prev_end = end

    for j in range(first_idx + 1, len(g)):
        row = g.loc[j]

        if int(row[target_col]) != 1:
            break

        next_start = float(row["clip_start"])
        next_end = float(row["clip_end"])

        # 연속 clip이면 이어붙임
        if next_start <= prev_end + 1e-6:
            duration_end = max(duration_end, next_end)
            prev_end = duration_end
        else:
            break

    real_eye_contact_sec = (start + end) / 2.0
    real_eye_contact_sec = max(real_eye_contact_sec, float(call_start))

    real_eye_contact_duration_sec = duration_end - duration_start


    return (
        True,
        real_eye_contact_sec,
        duration_start,
        duration_end,
        real_eye_contact_duration_sec,
        first_row.get("fold"),
        int(first_row["pred"]),      # 항상 1
        float(first_row["prob"]),
        "",
    )

RPMP_PATH = Path(
    "/home/sohyunkang/asd_video3/demographics/rpmp_검사지_result_20241219.xlsx"
)


def extract_patient_id(video_id):
    parts = str(video_id).split("_")
    if len(parts) >= 5:
        return parts[3]
    return ""


def load_diagnosis_map():
    label_map = {
        "정상군": 0,
        "자폐군": 1,
    }

    df = pd.read_excel(RPMP_PATH)

    df["연구대상자ID"] = (
        df["연구대상자ID"]
        .astype(str)
        .str.strip()
    )

    df["구분"] = (
        df["구분"]
        .astype(str)
        .str.strip()
    )

    df = df[df["구분"].isin(label_map)]

    return {
        pid: {
            "diagnosis": group,
            "diagnosis_target": label_map[group],
        }
        for pid, group in zip(
            df["연구대상자ID"],
            df["구분"],
        )
    }

def main():
    val_df = load_val_predictions(EXP_DIR)
    diagnosis_map = load_diagnosis_map()
    valid_val_video_ids = set(val_df["video_id"].unique())

    with open(CALL_JSON_PATH, "r", encoding="utf-8") as f:
        calling_data = json.load(f)

    rows = []

    skipped_not_in_val = 0
    skipped_no_eye_json = 0
    skipped_no_callname = 0

    for item in calling_data:
        video_id = normalize_video_id(item.get("id"))

        patient_id = extract_patient_id(video_id)

        diag = diagnosis_map.get(
            patient_id,
            {
                "diagnosis": None,
                "diagnosis_target": -1,
            }
        )

        # validation prediction에 있는 영상만 사용
        if video_id not in valid_val_video_ids:
            skipped_not_in_val += 1
            continue

        # face detect JSON이 있는 영상만 사용
        if find_eye_json_path(video_id) is None:
            skipped_no_eye_json += 1
            continue

        call_start, call_end, call_word = get_first_call_segment(item)

        if call_start is None:
            skipped_no_callname += 1
            continue

        first_face, face_error = get_first_face_detect_after_call(
            video_id=video_id,
            call_start=call_start,
        )

        if first_face is None:
            face_latency = None
            face_latency_error = face_error
        else:
            face_latency = first_face - call_start
            face_latency_error = ""

        # GT annotation 기준
        (
            gt_eye_exists,
            gt_eye,
            gt_start,
            gt_end,
            gt_duration,
            gt_fold,
            gt_pred,
            gt_prob,
            gt_error,
        ) = get_first_real_eye_contact_after_call(
            video_id=video_id,
            call_start=call_start,
            val_df=val_df,
            target_col="true",
        )

        # 모델 prediction 기준
        (
            pred_eye_exists,
            pred_eye,
            pred_start,
            pred_end,
            pred_duration,
            pred_fold,
            pred_label,
            pred_prob,
            pred_error,
        ) = get_first_real_eye_contact_after_call(
            video_id=video_id,
            call_start=call_start,
            val_df=val_df,
            target_col="pred",
        )

        gt_latency = None if gt_eye is None else gt_eye - call_start
        pred_latency = None if pred_eye is None else pred_eye - call_start


        rows.append({
            "video_id": video_id,
            "transcription": item.get("transcription"),

            "diagnosis": diag["diagnosis"],
            "diagnosis_target": diag["diagnosis_target"],

            "first_call_start": call_start,
            "first_call_end": call_end,
            "first_call_word": call_word,

            "first_face_detect_sec": first_face,
            "face_detect_latency_sec": face_latency,
            "face_detect_latency_error": face_latency_error,

            # annotation / GT 기준
            "real_eye_contact_exists": gt_eye_exists,
            "real_eye_contact_sec": gt_eye,
            "real_eye_contact_start_sec": gt_start,
            "real_eye_contact_end_sec": gt_end,
            "real_eye_contact_duration_sec": gt_duration,
            "real_latency_sec": gt_latency,
            "real_latency_error": gt_error,

            # model prediction 기준
            "pred_eye_contact_exists": pred_eye_exists,
            "pred_eye_contact_sec": pred_eye,
            "pred_eye_contact_start_sec": pred_start,
            "pred_eye_contact_end_sec": pred_end,
            "pred_eye_contact_duration_sec": pred_duration,
            "pred_latency_sec": pred_latency,

            "pred_fold": pred_fold,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
            "pred_latency_error": pred_error,
        })

    df = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("[DONE]")
    print("saved:", OUT_CSV)
    print("n rows:", len(df))

    print("\n[SKIPPED]")
    print("not in validation predictions:", skipped_not_in_val)
    print("eye json not found:", skipped_no_eye_json)
    print("no callname:", skipped_no_callname)

    print("\n[face detect latency error]")
    print(df["face_detect_latency_error"].value_counts(dropna=False))

    print("\n[pred latency error]")
    print(df["pred_latency_error"].value_counts(dropna=False))


    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        mean_absolute_error,
    )

    print("\n==============================")
    print("Prediction Performance")
    print("==============================")

    # -----------------------------
    # Video-level eye-contact existence
    # -----------------------------
    valid_pred = df.dropna(
        subset=[
            "real_eye_contact_exists",
            "pred_eye_contact_exists",
        ]
    ).copy()

    if len(valid_pred) > 0:
        y_true = valid_pred["real_eye_contact_exists"].astype(bool)
        y_pred = valid_pred["pred_eye_contact_exists"].astype(bool)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Eye-contact detection accuracy : {acc:.4f}")
        print(f"Eye-contact detection precision: {precision:.4f}")
        print(f"Eye-contact detection recall   : {recall:.4f}")
        print(f"Eye-contact detection F1       : {f1:.4f}")
        print(f"Detection samples              : {len(valid_pred)}")
    else:
        print("Eye-contact detection metrics  : N/A")


    # -----------------------------
    # Timing MAE
    # 둘 다 eye contact가 있다고 판단된 경우만 사용
    # -----------------------------
    valid_timing = df[
        (df["real_eye_contact_exists"] == True)
        & (df["pred_eye_contact_exists"] == True)
    ].copy()

    valid_timing = valid_timing.dropna(
        subset=[
            "real_latency_sec",
            "pred_latency_sec",
            "real_eye_contact_duration_sec",
            "pred_eye_contact_duration_sec",
        ]
    )

    if len(valid_timing) > 0:
        latency_mae = mean_absolute_error(
            valid_timing["real_latency_sec"],
            valid_timing["pred_latency_sec"],
        )

        duration_mae = mean_absolute_error(
            valid_timing["real_eye_contact_duration_sec"],
            valid_timing["pred_eye_contact_duration_sec"],
        )

        print(f"Latency MAE                  : {latency_mae:.4f} sec")
        print(f"Duration MAE                 : {duration_mae:.4f} sec")
        print(f"Timing samples               : {len(valid_timing)}")
    else:
        print("Timing MAE                   : N/A")

if __name__ == "__main__":
    main()