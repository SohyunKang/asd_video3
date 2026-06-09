import re
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import matplotlib.font_manager as fm
import matplotlib as mpl

# Linux에서 자주 있는 한글 폰트 후보
font_candidates = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
]

for font_path in font_candidates:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = font_name
        break

mpl.rcParams["axes.unicode_minus"] = False

# =========================
# 설정
# =========================

JSON_DIRS = [
    r"/storage/sohyunkang/eyecont_results_true",
    r"/storage/sohyunkang/eyecont_results_false",
]

EXCEL_PATH = r"./demographics/rpmp_검사지_result_20241219.xlsx"

OUTPUT_FEATURES = "./eye_contact/results/eye_contact_features.xlsx"
OUTPUT_GROUP_SUMMARY = "./eye_contact/results/group_summary.xlsx"
OUTPUT_POSTHOC = "./eye_contact/results/posthoc_dunn.xlsx"
OUTPUT_EMOTION_GROUP = "./eye_contact/results/emotion_group_comparison.xlsx"
OUTPUT_MISSING_GROUP = "./eye_contact/results/missing_group_subjects.xlsx"
TARGET_LIST_XLSX = "./demographics/251216_EXACT_FILENAMES.xlsx"
OUTPUT_NOT_IN_DATA = "./eye_contact/results/excluded_not_in_data.xlsx"

ID_COL = "연구대상자ID"
GROUP_COL = "구분"

FILENAME_PATTERN = re.compile(
    r"^IF2001_\d+_\d+_(?P<patient_id>[^_]+)_\d+\.json$"
)


# =========================
# 함수
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_top_emotion(frame):
    emotions = frame.get("emotion")

    if not emotions:
        return None, None

    top = max(emotions, key=lambda x: x.get("score", -1))
    return top.get("label"), top.get("score")


def is_eye_contact(frame):
    return frame.get("face_detected", False)

def finalize_segment(segment):
    frames = segment["frames"]
    emotion_votes = {}

    for f in frames:
        label = f["emotion_label"]
        score = f["emotion_score"]

        if label is None:
            continue

        emotion_votes[label] = emotion_votes.get(label, 0) + (
            score if score is not None else 1
        )

    representative_emotion = (
        max(emotion_votes, key=emotion_votes.get)
        if emotion_votes else "Unknown"
    )

    start_ms = segment["start_ms"]
    end_ms = segment["end_ms"]

    return {
        "file_name": segment["file_name"],
        "patient_id": segment["patient_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_sec": start_ms / 1000,
        "end_sec": end_ms / 1000,
        "duration_ms": end_ms - start_ms,
        "duration_sec": (end_ms - start_ms) / 1000,
        "n_frames": len(frames),
        "representative_emotion": representative_emotion,
        "has_eye_contact": True,
    }


def extract_eye_contact_segments(data, file_name, patient_id):
    segments = []
    current_segment = None

    for frame in data:
        timestamp_ms = frame.get("timestamp_ms")

        if timestamp_ms is None:
            continue

        contact = is_eye_contact(frame)
        emotion_label, emotion_score = get_top_emotion(frame)

        if contact:
            if current_segment is None:
                current_segment = {
                    "file_name": file_name,
                    "patient_id": patient_id,
                    "start_ms": timestamp_ms,
                    "end_ms": timestamp_ms,
                    "frames": [],
                }

            current_segment["end_ms"] = timestamp_ms
            current_segment["frames"].append({
                "timestamp_ms": timestamp_ms,
                "emotion_label": emotion_label,
                "emotion_score": emotion_score,
            })

        else:
            if current_segment is not None:
                segments.append(finalize_segment(current_segment))
                current_segment = None

    if current_segment is not None:
        segments.append(finalize_segment(current_segment))

    return segments


def get_video_duration_sec(data):
    timestamps = [
        f.get("timestamp_ms")
        for f in data
        if f.get("timestamp_ms") is not None
    ]

    if len(timestamps) == 0:
        return 0

    return (max(timestamps) - min(timestamps)) / 1000


def summarize_emotions(data, only_eye_contact=False):
    emotion_counts = {}
    total = 0

    for frame in data:
        if only_eye_contact and not is_eye_contact(frame):
            continue

        label, score = get_top_emotion(frame)

        if label is None:
            continue

        emotion_counts[label] = emotion_counts.get(label, 0) + 1
        total += 1

    result = {
        "n_emotion_frames": total,
    }

    emotions = ["happy", "sad", "neutral", "angry", "fear", "surprise", "disgust"]

    for emotion in emotions:
        count = emotion_counts.get(emotion, 0)
        result[f"{emotion}_count"] = count
        result[f"{emotion}_ratio"] = count / total if total > 0 else 0

    result["dominant_emotion"] = (
        max(emotion_counts, key=emotion_counts.get)
        if total > 0 else "No_Emotion"
    )

    return result
def run_kruskal_safe(df, metrics, group_col="group"):
    results = []

    for metric in metrics:
        tmp = df[[group_col, metric]].dropna().copy()

        if tmp[group_col].nunique() < 2:
            continue

        # 모든 값이 동일하면 Kruskal 실행 불가
        if tmp[metric].nunique() <= 1:
            results.append({
                "metric": metric,
                "kruskal_H": None,
                "p_value": None,
                "note": "Skipped: all values are identical"
            })
            continue

        groups = [
            g[metric].values
            for _, g in tmp.groupby(group_col)
            if len(g[metric].dropna()) > 0
        ]

        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                results.append({
                    "metric": metric,
                    "kruskal_H": stat,
                    "p_value": p,
                    "note": ""
                })
            except ValueError as e:
                results.append({
                    "metric": metric,
                    "kruskal_H": None,
                    "p_value": None,
                    "note": str(e)
                })

    return pd.DataFrame(results)


def run_dunn_safe(df, metric, group_col="group"):
    tmp = df[[group_col, metric]].dropna().copy()

    if tmp[group_col].nunique() < 2:
        return None

    # 모든 값이 동일하면 Dunn test 생략
    if tmp[metric].nunique() <= 1:
        return None

    valid_groups = tmp.groupby(group_col)[metric].count()
    valid_groups = valid_groups[valid_groups > 0].index
    tmp = tmp[tmp[group_col].isin(valid_groups)]

    if tmp[group_col].nunique() < 2:
        return None

    try:
        return sp.posthoc_dunn(
            tmp,
            val_col=metric,
            group_col=group_col,
            p_adjust="bonferroni"
        )
    except ValueError:
        return None

# =========================
# 1. JSON 파일 읽기 + 대상 여부 정리
# =========================

target_df = pd.read_excel(TARGET_LIST_XLSX, header=None)

valid_file_ids = set()
for fname in target_df.iloc[:, 0].dropna():
    fname = str(fname).strip()
    valid_file_ids.add(fname)

print(f"\n[1] Exact 대상 목록 파일 수: {len(valid_file_ids)}")

all_segments = []
file_summary_rows = []

excluded_filename_rows = []
exact_exist_rows = []
exact_not_exist_rows = []

json_total_count = 0
json_pattern_match_count = 0

for folder in JSON_DIRS:
    folder = Path(folder)

    json_files = list(folder.glob("*.json"))
    print(f"[1] JSON 폴더: {folder}")
    print(f"    JSON 파일 수: {len(json_files)}")

    for json_path in json_files:
        json_total_count += 1

        match = FILENAME_PATTERN.match(json_path.name)

        if match is None:
            excluded_filename_rows.append({
                "folder": str(folder),
                "file_name": json_path.name,
                "reason": "filename_pattern_mismatch"
            })
            continue

        json_pattern_match_count += 1

        file_id = json_path.stem
        patient_id = str(match.group("patient_id"))

        base_row = {
            "folder": str(folder),
            "file_name": json_path.name,
            "file_id": file_id,
            "patient_id": patient_id,
        }

        if file_id in valid_file_ids:
            exact_exist_rows.append(base_row)
        else:
            exact_not_exist_rows.append({
                **base_row,
                "reason": "not_in_exact_list"
            })
            continue

        data = load_json(json_path)

        video_duration_sec = get_video_duration_sec(data)

        segments = extract_eye_contact_segments(
            data=data,
            file_name=json_path.name,
            patient_id=patient_id
        )

        # 첫 번째 eye contact만 사용
        if len(segments) > 0:
            segments = [segments[0]]

        emotion_all = summarize_emotions(data, only_eye_contact=False)

        emotion_eye = summarize_emotions(data, only_eye_contact=True)
        emotion_eye = {
            f"eye_contact_{k}": v
            for k, v in emotion_eye.items()
        }

        if len(segments) == 0:
            all_segments.append({
                "file_name": json_path.name,
                "patient_id": patient_id,
                "start_ms": None,
                "end_ms": None,
                "start_sec": None,
                "end_sec": None,
                "duration_ms": 0,
                "duration_sec": 0,
                "n_frames": 0,
                "representative_emotion": "No_EyeContact",
                "has_eye_contact": False,
            })

            eye_contact_count = 0
            total_eye_contact_sec = 0
            mean_eye_contact_sec = 0
            median_eye_contact_sec = 0
            max_eye_contact_sec = 0
            eye_contact_ratio = 0
            representative_emotion_file = "No_EyeContact"

        else:
            all_segments.extend(segments)

            durations = [s["duration_sec"] for s in segments]

            eye_contact_count = len(segments)
            total_eye_contact_sec = sum(durations)
            mean_eye_contact_sec = pd.Series(durations).mean()
            median_eye_contact_sec = pd.Series(durations).median()
            max_eye_contact_sec = max(durations)
            eye_contact_ratio = (
                total_eye_contact_sec / video_duration_sec
                if video_duration_sec > 0 else 0
            )

            emotion_duration = {}
            for s in segments:
                emo = s["representative_emotion"]
                emotion_duration[emo] = emotion_duration.get(emo, 0) + s["duration_sec"]

            representative_emotion_file = max(
                emotion_duration,
                key=emotion_duration.get
            )

        file_summary_rows.append({
            "file_name": json_path.name,
            "file_id": file_id,
            "patient_id": patient_id,
            "folder": str(folder),
            "in_exact_list": True,
            "video_duration_sec": video_duration_sec,
            "eye_contact_count": eye_contact_count,
            "total_eye_contact_sec": total_eye_contact_sec,
            "mean_eye_contact_sec": mean_eye_contact_sec,
            "median_eye_contact_sec": median_eye_contact_sec,
            "max_eye_contact_sec": max_eye_contact_sec,
            "eye_contact_ratio": eye_contact_ratio,
            "representative_emotion_file": representative_emotion_file,
            **emotion_all,
            **emotion_eye,
        })


features_df = pd.DataFrame(all_segments)
file_summary_df = pd.DataFrame(file_summary_rows)

exact_exist_df = pd.DataFrame(exact_exist_rows)
exact_not_exist_df = pd.DataFrame(exact_not_exist_rows)
excluded_filename_df = pd.DataFrame(excluded_filename_rows)

print("\n[1] JSON 전체 요약")
print(f"전체 JSON 파일 수: {json_total_count}")
print(f"파일명 패턴 일치 JSON 수: {json_pattern_match_count}")
print(f"파일명 패턴 불일치 JSON 수: {len(excluded_filename_df)}")
print(f"Exact 목록에 존재하는 JSON 수: {len(exact_exist_df)}")
print(f"Exact 목록에 존재하지 않는 JSON 수: {len(exact_not_exist_df)}")

if len(exact_exist_df) > 0:
    print(f"Exact 존재 대상자 수: {exact_exist_df['patient_id'].nunique()}")

if len(exact_not_exist_df) > 0:
    print(f"Exact 미존재 대상자 수: {exact_not_exist_df['patient_id'].nunique()}")


# =========================
# 2. 엑셀에서 group 정보 붙이기
# =========================

group_df = pd.read_excel(EXCEL_PATH)

group_df[ID_COL] = group_df[ID_COL].astype(str).str.strip()
group_df[GROUP_COL] = group_df[GROUP_COL].astype(str).str.strip()

features_df["patient_id"] = features_df["patient_id"].astype(str).str.strip()
file_summary_df["patient_id"] = file_summary_df["patient_id"].astype(str).str.strip()

if len(exact_exist_df) > 0:
    exact_exist_df["patient_id"] = exact_exist_df["patient_id"].astype(str).str.strip()

if len(exact_not_exist_df) > 0:
    exact_not_exist_df["patient_id"] = exact_not_exist_df["patient_id"].astype(str).str.strip()

group_df = group_df[[ID_COL, GROUP_COL]].drop_duplicates()

features_df = features_df.merge(
    group_df,
    left_on="patient_id",
    right_on=ID_COL,
    how="left"
)
features_df = features_df.rename(columns={GROUP_COL: "group"})
features_df = features_df.drop(columns=[ID_COL])

file_summary_df = file_summary_df.merge(
    group_df,
    left_on="patient_id",
    right_on=ID_COL,
    how="left"
)
file_summary_df = file_summary_df.rename(columns={GROUP_COL: "group"})
file_summary_df = file_summary_df.drop(columns=[ID_COL])

exact_exist_df = exact_exist_df.merge(
    group_df,
    left_on="patient_id",
    right_on=ID_COL,
    how="left"
).rename(columns={GROUP_COL: "group"})

if ID_COL in exact_exist_df.columns:
    exact_exist_df = exact_exist_df.drop(columns=[ID_COL])

exact_not_exist_df = exact_not_exist_df.merge(
    group_df,
    left_on="patient_id",
    right_on=ID_COL,
    how="left"
).rename(columns={GROUP_COL: "group"})

if ID_COL in exact_not_exist_df.columns:
    exact_not_exist_df = exact_not_exist_df.drop(columns=[ID_COL])


# =========================
# 3. 제외 대상 정리
# =========================

file_summary_df_all = file_summary_df.copy()
features_df_all = features_df.copy()

missing_group_exact_exist_df = (
    exact_exist_df[exact_exist_df["group"].isna()]
    .sort_values(["patient_id", "file_name"])
    .copy()
)

missing_group_exact_not_exist_df = (
    exact_not_exist_df[exact_not_exist_df["group"].isna()]
    .sort_values(["patient_id", "file_name"])
    .copy()
)

hold_group_exact_exist_df = (
    exact_exist_df[exact_exist_df["group"] == "보류"]
    .sort_values(["patient_id", "file_name"])
    .copy()
)

hold_group_exact_not_exist_df = (
    exact_not_exist_df[exact_not_exist_df["group"] == "보류"]
    .sort_values(["patient_id", "file_name"])
    .copy()
)

print("\n[2] RPMP group 정보 요약")
print(f"Exact 존재 중 group 없음 파일 수: {len(missing_group_exact_exist_df)}")
print(f"Exact 존재 중 group 없음 대상자 수: {missing_group_exact_exist_df['patient_id'].nunique() if len(missing_group_exact_exist_df) > 0 else 0}")
print(f"Exact 미존재 중 group 없음 파일 수: {len(missing_group_exact_not_exist_df)}")
print(f"Exact 미존재 중 group 없음 대상자 수: {missing_group_exact_not_exist_df['patient_id'].nunique() if len(missing_group_exact_not_exist_df) > 0 else 0}")

print("\n[3] 보류 그룹 요약")
print(f"Exact 존재 중 보류 파일 수: {len(hold_group_exact_exist_df)}")
print(f"Exact 존재 중 보류 대상자 수: {hold_group_exact_exist_df['patient_id'].nunique() if len(hold_group_exact_exist_df) > 0 else 0}")
print(f"Exact 미존재 중 보류 파일 수: {len(hold_group_exact_not_exist_df)}")
print(f"Exact 미존재 중 보류 대상자 수: {hold_group_exact_not_exist_df['patient_id'].nunique() if len(hold_group_exact_not_exist_df) > 0 else 0}")

# 실제 분석에서는 기존처럼 group 없음 + 보류 제외
file_summary_df = file_summary_df[
    file_summary_df["group"].notna()
].copy()

features_df = features_df[
    features_df["group"].notna()
].copy()

file_summary_df = file_summary_df[
    file_summary_df["group"] != "보류"
].copy()

features_df = features_df[
    features_df["group"] != "보류"
].copy()

print("\n[4] 최종 분석 포함 데이터")
print(f"최종 분석 파일 수: {file_summary_df['file_name'].nunique()}")
print(f"최종 분석 대상자 수: {file_summary_df['patient_id'].nunique()}")
print("최종 분석 그룹별 대상자 수:")
print(file_summary_df.groupby("group")["patient_id"].nunique())


# =========================
# 제외/포함 목록 저장
# =========================

OUTPUT_DATA_AUDIT = "./eye_contact/results/data_inclusion_audit.xlsx"

with pd.ExcelWriter(OUTPUT_DATA_AUDIT) as writer:
    exact_exist_df.to_excel(writer, sheet_name="exact_exist", index=False)
    exact_not_exist_df.to_excel(writer, sheet_name="exact_not_exist", index=False)

    missing_group_exact_exist_df.to_excel(
        writer,
        sheet_name="missing_group_exact_exist",
        index=False
    )
    missing_group_exact_not_exist_df.to_excel(
        writer,
        sheet_name="missing_group_exact_not_exist",
        index=False
    )

    hold_group_exact_exist_df.to_excel(
        writer,
        sheet_name="hold_group_exact_exist",
        index=False
    )
    hold_group_exact_not_exist_df.to_excel(
        writer,
        sheet_name="hold_group_exact_not_exist",
        index=False
    )

    if len(excluded_filename_df) > 0:
        excluded_filename_df.to_excel(
            writer,
            sheet_name="filename_pattern_mismatch",
            index=False
        )

print(f"\n포함/제외 점검 파일 저장: {OUTPUT_DATA_AUDIT}")

# =========================
# 4. segment feature 저장
# =========================

features_df.to_excel(OUTPUT_FEATURES, index=False)


# =========================
# 5. 환자 단위 요약
# =========================

numeric_cols = [
    "eye_contact_count",
    "total_eye_contact_sec",
    "mean_eye_contact_sec",
    "median_eye_contact_sec",
    "max_eye_contact_sec",
    "eye_contact_ratio",

    "happy_ratio",
    "sad_ratio",
    "neutral_ratio",
    "angry_ratio",
    "fear_ratio",
    "surprise_ratio",
    "disgust_ratio",

    "eye_contact_happy_ratio",
    "eye_contact_sad_ratio",
    "eye_contact_neutral_ratio",
    "eye_contact_angry_ratio",
    "eye_contact_fear_ratio",
    "eye_contact_surprise_ratio",
    "eye_contact_disgust_ratio",
]

patient_summary = (
    file_summary_df
    .groupby(["patient_id", "group"])[numeric_cols]
    .mean()
    .reset_index()
)

group_summary = (
    patient_summary
    .groupby("group")
    .agg(
        n_subjects=("patient_id", "nunique"),

        mean_eye_contact_count=("eye_contact_count", "mean"),
        sd_eye_contact_count=("eye_contact_count", "std"),

        mean_total_eye_contact_sec=("total_eye_contact_sec", "mean"),
        sd_total_eye_contact_sec=("total_eye_contact_sec", "std"),

        mean_eye_contact_ratio=("eye_contact_ratio", "mean"),
        sd_eye_contact_ratio=("eye_contact_ratio", "std"),

        mean_happy_ratio=("happy_ratio", "mean"),
        sd_happy_ratio=("happy_ratio", "std"),

        mean_sad_ratio=("sad_ratio", "mean"),
        sd_sad_ratio=("sad_ratio", "std"),

        mean_neutral_ratio=("neutral_ratio", "mean"),
        sd_neutral_ratio=("neutral_ratio", "std"),

        mean_eye_contact_happy_ratio=("eye_contact_happy_ratio", "mean"),
        sd_eye_contact_happy_ratio=("eye_contact_happy_ratio", "std"),

        mean_eye_contact_sad_ratio=("eye_contact_sad_ratio", "mean"),
        sd_eye_contact_sad_ratio=("eye_contact_sad_ratio", "std"),

        mean_eye_contact_neutral_ratio=("eye_contact_neutral_ratio", "mean"),
        sd_eye_contact_neutral_ratio=("eye_contact_neutral_ratio", "std"),
    )
    .reset_index()
)

with pd.ExcelWriter(OUTPUT_GROUP_SUMMARY) as writer:
    file_summary_df.to_excel(writer, sheet_name="file_summary_used", index=False)
    file_summary_df_all.to_excel(writer, sheet_name="file_summary_all", index=False)
    patient_summary.to_excel(writer, sheet_name="patient_summary", index=False)
    group_summary.to_excel(writer, sheet_name="group_summary", index=False)


# =========================
# 6. Eye contact 그룹 비교
# =========================

eye_metrics = [
    "eye_contact_count",
    "total_eye_contact_sec",
    "mean_eye_contact_sec",
    "median_eye_contact_sec",
    "max_eye_contact_sec",
    "eye_contact_ratio",
]

kruskal_df = run_kruskal_safe(patient_summary, eye_metrics)

with pd.ExcelWriter(OUTPUT_POSTHOC) as writer:
    kruskal_df.to_excel(writer, sheet_name="kruskal", index=False)

    for metric in eye_metrics:
        posthoc = run_dunn_safe(patient_summary, metric)

        if posthoc is not None:
            posthoc.to_excel(writer, sheet_name=metric[:31])


# =========================
# 7. 표정 그룹 비교
# =========================

emotion_metrics = [
    "happy_ratio",
    "sad_ratio",
    "neutral_ratio",
    "angry_ratio",
    "fear_ratio",
    "surprise_ratio",
    "disgust_ratio",

    "eye_contact_happy_ratio",
    "eye_contact_sad_ratio",
    "eye_contact_neutral_ratio",
    "eye_contact_angry_ratio",
    "eye_contact_fear_ratio",
    "eye_contact_surprise_ratio",
    "eye_contact_disgust_ratio",
]

emotion_kruskal_df = run_kruskal_safe(patient_summary, emotion_metrics)

with pd.ExcelWriter(OUTPUT_EMOTION_GROUP) as writer:
    emotion_kruskal_df.to_excel(writer, sheet_name="emotion_kruskal", index=False)

    for metric in emotion_metrics:
        posthoc = run_dunn_safe(patient_summary, metric)

        if posthoc is not None:
            posthoc.to_excel(writer, sheet_name=metric[:31])


# =========================
# 8. 최종 출력
# =========================

print("\n완료")
print(f"Segment feature file: {OUTPUT_FEATURES}")
print(f"Group summary file: {OUTPUT_GROUP_SUMMARY}")
print(f"Eye contact posthoc file: {OUTPUT_POSTHOC}")
print(f"Emotion group comparison file: {OUTPUT_EMOTION_GROUP}")

print("\n분석에 포함된 대상자 수:")
print(patient_summary.groupby("group")["patient_id"].nunique())


if len(excluded_filename_rows) > 0:
    excluded_filename_df = pd.DataFrame(excluded_filename_rows)
    print("\n[제외됨] 파일명 형식이 맞지 않는 JSON:")
    print(excluded_filename_df.to_string(index=False))


# =========================
# 9. Plot + t-test 추가 분석
# =========================

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import combinations
from pathlib import Path

OUTPUT_FIG_DIR = Path("./eye_contact/results/eye_contact_figures")
OUTPUT_FIG_DIR.mkdir(exist_ok=True)

GROUP_ORDER = ["정상군", "고위험군", "자폐군"]

# 실제 존재하는 group만 사용
group_order_used = [
    g for g in GROUP_ORDER
    if g in patient_summary["group"].dropna().unique()
]

# -------------------------------------------------
# 9-1. Eye contact 길이 boxplot + t-test
# -------------------------------------------------

# 첫 번째 eye contact만 쓰는 현재 코드에서는
# total_eye_contact_sec == 첫 eye contact duration
BOX_METRIC = "total_eye_contact_sec"

box_data = [
    patient_summary.loc[
        patient_summary["group"] == group, BOX_METRIC
    ].dropna()
    for group in group_order_used
]

plt.figure(figsize=(7, 5))
plt.boxplot(
    box_data,
    labels=group_order_used,
    showmeans=True
)
plt.ylabel("Eye contact duration (sec)")
plt.xlabel("Group")
plt.title("First eye contact duration by group")
plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "boxplot_eye_contact_duration.png", dpi=300)
plt.close()

# pairwise t-test
ttest_results = []

for g1, g2 in combinations(group_order_used, 2):
    x1 = patient_summary.loc[
        patient_summary["group"] == g1, BOX_METRIC
    ].dropna()

    x2 = patient_summary.loc[
        patient_summary["group"] == g2, BOX_METRIC
    ].dropna()

    if len(x1) >= 2 and len(x2) >= 2:
        stat, p = ttest_ind(x1, x2, equal_var=False)

        ttest_results.append({
            "metric": BOX_METRIC,
            "group1": g1,
            "group2": g2,
            "n_group1": len(x1),
            "n_group2": len(x2),
            "mean_group1": x1.mean(),
            "mean_group2": x2.mean(),
            "t_stat": stat,
            "p_value": p,
            "test": "Welch t-test"
        })

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_excel("./eye_contact/results/eye_contact_duration_ttest.xlsx", index=False)


# -------------------------------------------------
# 9-2. Eye contact 유무 percentage bar graph
# -------------------------------------------------

presence_df = patient_summary.copy()
presence_df["has_eye_contact"] = presence_df["eye_contact_count"] > 0

presence_summary = (
    presence_df
    .groupby("group")
    .agg(
        n_subjects=("patient_id", "nunique"),
        n_eye_contact=("has_eye_contact", "sum")
    )
    .reset_index()
)

presence_summary["eye_contact_percentage"] = (
    presence_summary["n_eye_contact"] /
    presence_summary["n_subjects"] * 100
)

presence_summary["group"] = pd.Categorical(
    presence_summary["group"],
    categories=GROUP_ORDER,
    ordered=True
)

presence_summary = presence_summary.sort_values("group")

plt.figure(figsize=(7, 5))
plt.bar(
    presence_summary["group"].astype(str),
    presence_summary["eye_contact_percentage"]
)
plt.ylim(0, 100)
plt.ylabel("Eye contact present (%)")
plt.xlabel("Group")
plt.title("Percentage of subjects with eye contact by group")

for i, row in presence_summary.reset_index(drop=True).iterrows():
    plt.text(
        i,
        row["eye_contact_percentage"] + 2,
        f'{row["eye_contact_percentage"]:.1f}%',
        ha="center"
    )

plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "bar_eye_contact_presence_percentage.png", dpi=300)
plt.close()

presence_summary.to_excel("./eye_contact/results/eye_contact_presence_percentage.xlsx", index=False)


# -------------------------------------------------
# 9-3. 그룹별 emotion 분포 pie chart
# -------------------------------------------------

emotion_count_cols = [
    "happy_count",
    "sad_count",
    "neutral_count",
    "angry_count",
    "fear_count",
    "surprise_count",
    "disgust_count",
]

emotion_labels = [
    "happy",
    "sad",
    "neutral",
    "angry",
    "fear",
    "surprise",
    "disgust",
]

emotion_group_summary = (
    file_summary_df
    .groupby("group")[emotion_count_cols]
    .sum()
    .reset_index()
)

emotion_group_summary["group"] = pd.Categorical(
    emotion_group_summary["group"],
    categories=GROUP_ORDER,
    ordered=True
)

emotion_group_summary = emotion_group_summary.sort_values("group")

for _, row in emotion_group_summary.iterrows():
    group = row["group"]

    values = [row[col] for col in emotion_count_cols]

    # 0인 emotion 제거
    plot_labels = []
    plot_values = []

    for label, value in zip(emotion_labels, values):
        if value > 0:
            plot_labels.append(label)
            plot_values.append(value)

    if sum(plot_values) == 0:
        continue

    plt.figure(figsize=(6, 6))
    plt.pie(
        plot_values,
        labels=plot_labels,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"Emotion distribution - {group}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIG_DIR / f"pie_emotion_distribution_{group}.png",
        dpi=300
    )
    plt.close()

emotion_group_summary.to_excel("./eye_contact/results/emotion_distribution_by_group.xlsx", index=False)


print("\n추가 plot 및 t-test 완료")
print(f"Figure folder: {OUTPUT_FIG_DIR}")
print("1) boxplot_eye_contact_duration.png")
print("2) bar_eye_contact_presence_percentage.png")
print("3) pie_emotion_distribution_정상군/고위험군/자폐군.png")
print("4) eye_contact_duration_ttest.xlsx")
print("5) eye_contact_presence_percentage.xlsx")
print("6) emotion_distribution_by_group.xlsx")
    