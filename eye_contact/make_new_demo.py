from pathlib import Path
import re
import json
import pandas as pd

# =========================
# 경로 설정
# =========================

VIDEO_DIR = Path("/storage/sohyunkang/video_data")

JSON_DIRS = [
    Path("/storage/sohyunkang/eyecont_results_true"),
    Path("/storage/sohyunkang/eyecont_results_false"),
]

TARGET_LIST_XLSX = "./demographics/251216_EXACT_FILENAMES.xlsx"

EXCEL_PATH = "./demographics/rpmp_검사지_result_20241219.xlsx"

OUTPUT_XLSX = "./demographics/260609_new_demo.xlsx"

ID_COL = "연구대상자ID"
GROUP_COL = "구분"
AGE_COL = "월령"

# =========================
# 함수
# =========================

def normalize_video_file_id(file_name):
    """
    예:
    IF2001_1_1_1023092434_0.mp4.mp4
    -> IF2001_1_1_1023092434_0
    """
    name = str(file_name).strip()

    while name.lower().endswith(".mp4"):
        name = name[:-4]

    return name


def normalize_exact_file_id(value):
    """
    EXACT 엑셀 값 정리
    .json, .mp4, .mp4.mp4 모두 제거
    """
    name = str(value).strip()

    while (
        name.lower().endswith(".json")
        or name.lower().endswith(".mp4")
    ):
        if name.lower().endswith(".json"):
            name = name[:-5]
        elif name.lower().endswith(".mp4"):
            name = name[:-4]

    return name


def extract_subject_from_file_id(file_id):
    """
    예:
    IF2001_1_1_1023092434_0
    -> 1023092434
    """
    match = re.match(
        r"^IF2001_\d+_\d+_([^_]+)_\d+$",
        file_id
    )

    if match is None:
        return None

    return str(match.group(1)).strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eye_contact_segment(data):
    """
    첫 번째 eye contact 구간만 반환.
    face_detected=True인 연속 구간 기준.
    """
    current_start = None
    current_end = None

    for frame in data:
        timestamp_ms = frame.get("timestamp_ms")

        if timestamp_ms is None:
            continue

        has_eye_contact = frame.get("face_detected", False)

        if has_eye_contact:
            if current_start is None:
                current_start = timestamp_ms

            current_end = timestamp_ms

        else:
            if current_start is not None:
                break

    if current_start is None:
        return None, None, 0

    start_sec = current_start / 1000
    end_sec = current_end / 1000
    duration_sec = end_sec - start_sec

    return start_sec, end_sec, duration_sec


# =========================
# EXACT 목록 읽기
# =========================

target_df = pd.read_excel(
    TARGET_LIST_XLSX,
    header=None
)

exact_ids = set()

for value in target_df.values.flatten():
    if pd.isna(value):
        continue

    file_id = normalize_exact_file_id(value)

    if file_id.startswith("IF2001"):
        exact_ids.add(file_id)

print(f"EXACT 목록 수: {len(exact_ids)}")


# =========================
# RPMP 정보 읽기
# =========================

rpmp_df = pd.read_excel(EXCEL_PATH)

rpmp_df[ID_COL] = (
    rpmp_df[ID_COL]
    .astype(str)
    .str.strip()
)

subject_info = (
    rpmp_df[[ID_COL, AGE_COL, GROUP_COL]]
    .drop_duplicates(subset=[ID_COL])
    .rename(
        columns={
            ID_COL: "SUBJECT",
            AGE_COL: "AGE",
            GROUP_COL: "GROUP",
        }
    )
)

subject_info["SUBJECT"] = (
    subject_info["SUBJECT"]
    .astype(str)
    .str.strip()
)

print(f"RPMP 대상자 수: {subject_info['SUBJECT'].nunique()}")


# =========================
# JSON 파일 index 만들기
# =========================

json_map = {}

for json_dir in JSON_DIRS:
    for json_path in json_dir.glob("*.json"):
        file_id = json_path.stem

        if file_id.startswith("IF2001"):
            json_map[file_id] = json_path

print(f"JSON 파일 수: {len(json_map)}")


# =========================
# VIDEO_DATA 기준으로 결과 만들기
# =========================

rows = []

for video_path in VIDEO_DIR.rglob("*"):
    if not video_path.is_file():
        continue

    # mp4만 사용
    if ".mp4" not in video_path.name.lower():
        continue

    # IF2001로 시작하는 파일만 사용
    if not video_path.name.startswith("IF2001"):
        continue

    file_id = normalize_video_file_id(video_path.name)
    subject = extract_subject_from_file_id(file_id)

    json_path = json_map.get(file_id)

    eye_contact_json_exists = json_path is not None

    start_sec = None
    end_sec = None
    duration_sec = None

    if eye_contact_json_exists:
        data = load_json(json_path)
        start_sec, end_sec, duration_sec = get_eye_contact_segment(data)

    rows.append({
        "SUBJECT": subject,

        "EYE_CONTACT_START": start_sec,
        "EYE_CONTACT_END": end_sec,
        "DURATION": duration_sec,

        "IN_EXACT_LIST": file_id in exact_ids,
        "EYE_CONTACT_JSON_EXISTS": eye_contact_json_exists,

        "video_file_name": video_path.name,
        "file_id": file_id,
        "video_full_path": str(video_path),

        "json_file_name": json_path.name if json_path is not None else None,
        "json_full_path": str(json_path) if json_path is not None else None,
    })


result_df = pd.DataFrame(
    rows,
    columns=[
        "SUBJECT",
        "EYE_CONTACT_START",
        "EYE_CONTACT_END",
        "DURATION",
        "IN_EXACT_LIST",
        "EYE_CONTACT_JSON_EXISTS",
        "video_file_name",
        "file_id",
        "video_full_path",
        "json_file_name",
        "json_full_path",
    ]
)

# =========================
# AGE / GROUP 붙이기
# =========================

result_df["SUBJECT"] = (
    result_df["SUBJECT"]
    .astype(str)
    .str.strip()
)

result_df = result_df.merge(
    subject_info,
    on="SUBJECT",
    how="left"
)


# =========================
# 컬럼 순서 정리
# =========================

front_cols = [
    "SUBJECT",
    "AGE",
    "GROUP",
    "EYE_CONTACT_START",
    "EYE_CONTACT_END",
    "DURATION",
    "IN_EXACT_LIST",
    "EYE_CONTACT_JSON_EXISTS",
]

other_cols = [
    col for col in result_df.columns
    if col not in front_cols
]

result_df = result_df[
    front_cols + other_cols
]

result_df = result_df.sort_values(
    ["SUBJECT", "file_id"]
)

# =========================
# 저장
# =========================

result_df.to_excel(
    OUTPUT_XLSX,
    index=False
)
rpmp_ids = set(subject_info["SUBJECT"])

json_exists_df = result_df[
    result_df["EYE_CONTACT_JSON_EXISTS"]
].copy()

json_not_exists_df = result_df[
    ~result_df["EYE_CONTACT_JSON_EXISTS"]
].copy()

json_in_exact_df = json_exists_df[
    json_exists_df["IN_EXACT_LIST"]
].copy()

json_not_in_exact_df = json_exists_df[
    ~json_exists_df["IN_EXACT_LIST"]
].copy()

# EXACT 포함
exact_in_rpmp_missing = (
    ~json_in_exact_df["SUBJECT"].isin(rpmp_ids)
).sum()

exact_in_group_missing = (
    json_in_exact_df["SUBJECT"].isin(rpmp_ids)
    & json_in_exact_df["GROUP"].isna()
).sum()

exact_in_hold = (
    json_in_exact_df["GROUP"] == "보류"
).sum()

# EXACT 미포함
exact_out_rpmp_missing = (
    ~json_not_in_exact_df["SUBJECT"].isin(rpmp_ids)
).sum()

exact_out_group_missing = (
    json_not_in_exact_df["SUBJECT"].isin(rpmp_ids)
    & json_not_in_exact_df["GROUP"].isna()
).sum()

exact_out_hold = (
    json_not_in_exact_df["GROUP"] == "보류"
).sum()

final_analysis_df = json_in_exact_df[
    json_in_exact_df["SUBJECT"].isin(rpmp_ids)
    & json_in_exact_df["GROUP"].notna()
    & (json_in_exact_df["GROUP"] != "보류")
].copy()

print("\n========================================")
print("저장 완료")
print("========================================")

print(f"파일: {OUTPUT_XLSX}")

print("\n[VIDEO_DATA]")
print(f"VIDEO_DATA mp4 수: {len(result_df)}")

print("\n[JSON]")
print(f"JSON 존재 수: {len(json_exists_df)}")
print(f"JSON 없음 수: {len(json_not_exists_df)}")

print("\n[JSON 존재 + EXACT 포함]")
print(f"수: {len(json_in_exact_df)}")
print(f"RPMP ID 없음: {exact_in_rpmp_missing}")
print(f"RPMP ID 있음 + GROUP 없음: {exact_in_group_missing}")
print(f"보류: {exact_in_hold}")

print("\n[JSON 존재 + EXACT 미포함]")
print(f"수: {len(json_not_in_exact_df)}")
print(f"RPMP ID 없음: {exact_out_rpmp_missing}")
print(f"RPMP ID 있음 + GROUP 없음: {exact_out_group_missing}")
print(f"보류: {exact_out_hold}")

print("\n[최종 분석 포함]")
print(f"수: {len(final_analysis_df)}")

print("\nGROUP별 개수 (최종 분석 포함)")
print(
    final_analysis_df["GROUP"]
    .value_counts(dropna=False)
)

print("\n========================================")