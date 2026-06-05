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

EXCEL_PATH = r"rpmp_검사지_result_20241219.xlsx"

OUTPUT_FEATURES = "eye_contact_features.xlsx"
OUTPUT_GROUP_SUMMARY = "group_summary.xlsx"
OUTPUT_POSTHOC = "posthoc_dunn.xlsx"
OUTPUT_EMOTION_GROUP = "emotion_group_comparison.xlsx"
OUTPUT_MISSING_GROUP = "missing_group_subjects.xlsx"
TARGET_LIST_XLSX = "251216_EXACT_FILENAMES.xlsx"
OUTPUT_NOT_IN_DATA = "excluded_not_in_data.xlsx"

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
# 1. JSON 파일 읽기
# =========================

# =========================
# 분석 대상 파일 목록 읽기
# xlsx 첫 번째 열:
# IF2001_1_1_1023092434_0
# =========================

target_df = pd.read_excel(
    TARGET_LIST_XLSX,
    header=None
)

valid_file_ids = set()

for fname in target_df.iloc[:, 0].dropna():

    fname = str(fname).strip()

    valid_file_ids.add(fname)

print(f"\n대상 목록 XLSX 파일 수: {len(valid_file_ids)}")

all_segments = []
file_summary_rows = []
excluded_filename_rows = []
excluded_not_in_data_rows = []

for folder in JSON_DIRS:
    folder = Path(folder)

    for json_path in folder.glob("*.json"):
        match = FILENAME_PATTERN.match(json_path.name)

        if match is None:
            excluded_filename_rows.append({
                "file_name": json_path.name,
                "reason": "filename_pattern_mismatch"
            })
            print(f"파일명 형식 불일치로 제외: {json_path.name}")
            continue

        file_id = json_path.stem

        if file_id not in valid_file_ids:
            excluded_not_in_data_rows.append({
                "file_name": json_path.name,
                "file_id": file_id,
                "patient_id": str(match.group("patient_id")),
                "reason": "not_in_data_folder"
            })
            continue

        patient_id = str(match.group("patient_id"))
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
            "patient_id": patient_id,
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


# =========================
# 2. 엑셀에서 group 정보 붙이기
# =========================

group_df = pd.read_excel(EXCEL_PATH)

group_df[ID_COL] = group_df[ID_COL].astype(str).str.strip()
group_df[GROUP_COL] = group_df[GROUP_COL].astype(str).str.strip()

features_df["patient_id"] = features_df["patient_id"].astype(str).str.strip()
file_summary_df["patient_id"] = file_summary_df["patient_id"].astype(str).str.strip()

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


# =========================
# 3. 엑셀에 없는 대상자 제외
# =========================

missing_group_df = (
    file_summary_df[file_summary_df["group"].isna()]
    [["file_name", "patient_id"]]
    .drop_duplicates()
    .sort_values(["patient_id", "file_name"])
)

if len(missing_group_df) > 0:
    missing_group_df.to_excel(OUTPUT_MISSING_GROUP, index=False)

file_summary_df_all = file_summary_df.copy()
features_df_all = features_df.copy()

# 보류 그룹 제외
file_summary_df = file_summary_df[
    file_summary_df["group"] != "보류"
].copy()

features_df = features_df[
    features_df["group"] != "보류"
].copy()

# group 없는 경우 제외
file_summary_df = file_summary_df.dropna(subset=["group"]).copy()
features_df = features_df.dropna(subset=["group"]).copy()

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

    if len(missing_group_df) > 0:
        missing_group_df.to_excel(writer, sheet_name="excluded_no_group", index=False)


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

if len(missing_group_df) > 0:
    print("\n[제외됨] 엑셀에 group 정보가 없는 대상자/파일:")
    print(missing_group_df.to_string(index=False))
    print(f"\n제외 목록 저장: {OUTPUT_MISSING_GROUP}")
else:
    print("\n엑셀에 없는 대상자는 없습니다.")

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

OUTPUT_FIG_DIR = Path("eye_contact_figures")
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
ttest_df.to_excel("eye_contact_duration_ttest.xlsx", index=False)


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

presence_summary.to_excel("eye_contact_presence_percentage.xlsx", index=False)


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

emotion_group_summary.to_excel("emotion_distribution_by_group.xlsx", index=False)


print("\n추가 plot 및 t-test 완료")
print(f"Figure folder: {OUTPUT_FIG_DIR}")
print("1) boxplot_eye_contact_duration.png")
print("2) bar_eye_contact_presence_percentage.png")
print("3) pie_emotion_distribution_정상군/고위험군/자폐군.png")
print("4) eye_contact_duration_ttest.xlsx")
print("5) eye_contact_presence_percentage.xlsx")
print("6) emotion_distribution_by_group.xlsx")


if len(excluded_not_in_data_rows) > 0:
    excluded_not_in_data_df = pd.DataFrame(excluded_not_in_data_rows)
    excluded_not_in_data_df.to_excel(OUTPUT_NOT_IN_DATA, index=False)

    print("\n[제외됨] ./data 안에 같은 이름의 json이 없는 파일:")
    print(f"제외 파일 수: {len(excluded_not_in_data_df)}")
    print(f"제외 대상자 수: {excluded_not_in_data_df['patient_id'].nunique()}")
    print(f"제외 목록 저장: {OUTPUT_NOT_IN_DATA}")
else:
    print("\n./data 기준으로 제외된 파일은 없습니다.")

print(f"\n./data 기준 분석 대상 파일 수: {len(valid_file_ids)}")
print(f"실제로 분석된 파일 수: {file_summary_df_all['file_name'].nunique()}")
print(f"실제로 분석된 대상자 수: {file_summary_df_all['patient_id'].nunique()}")