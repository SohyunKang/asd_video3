import re
import torch
import whisper
import librosa
import pandas as pd
from pathlib import Path


VIDEO_PATH = "/storage/sohyunkang/video_data/IF2001_3_1_1423111761_0.mp4.mp4"
OUTPUT_CSV = "call_detection_result.csv"

SEARCH_START_SEC = 0.0
SEARCH_END_SEC = 15.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_from_video(video_path, sr=16000):
    y, sr = librosa.load(
        video_path,
        sr=sr,
        mono=True
    )
    return y, sr


def run_silero_vad(y, sr):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    get_speech_timestamps = utils[0]

    speech_timestamps = get_speech_timestamps(
        y,
        model,
        sampling_rate=sr,
        return_seconds=True
    )

    return speech_timestamps


def transcribe_with_whisper(video_path):
    model = whisper.load_model(
        "small",
        device=DEVICE
    )

    result = model.transcribe(
        video_path,
        language="ko",
        fp16=(DEVICE == "cuda"),
        verbose=False
    )

    return result["segments"]


def has_call_pattern(text):
    text = str(text).replace(" ", "")
    return bool(
        re.search(
            r"[가-힣]{2,4}(아|야)",
            text
        )
    )


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def find_call_segments(video_path):
    y, sr = load_audio_from_video(video_path)

    vad_segments = run_silero_vad(y, sr)
    asr_segments = transcribe_with_whisper(video_path)

    rows = []

    for asr in asr_segments:
        asr_start = float(asr["start"])
        asr_end = float(asr["end"])
        text = str(asr["text"]).strip()

        if asr_end < SEARCH_START_SEC:
            continue

        if asr_start > SEARCH_END_SEC:
            continue

        matched_vad = False
        max_vad_overlap = 0.0

        for vad in vad_segments:
            vad_start = float(vad["start"])
            vad_end = float(vad["end"])

            ov = overlap(
                asr_start,
                asr_end,
                vad_start,
                vad_end
            )

            if ov > 0:
                matched_vad = True
                max_vad_overlap = max(max_vad_overlap, ov)

        call_pattern = has_call_pattern(text)

        rows.append({
            "asr_start": asr_start,
            "asr_end": asr_end,
            "text": text,
            "matched_vad": matched_vad,
            "vad_overlap_sec": max_vad_overlap,
            "has_call_pattern": call_pattern,
        })

    result_df = pd.DataFrame(rows)

    call_df = result_df[
        result_df["matched_vad"]
        & result_df["has_call_pattern"]
    ].copy()

    return result_df, call_df


result_df, call_df = find_call_segments(VIDEO_PATH)

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n[ALL ASR SEGMENTS]")
print(result_df)

print("\n[CALL CANDIDATES]")
print(call_df)

if len(call_df) > 0:
    first_call = call_df.iloc[0]
    print("\n[FIRST CALL]")
    print("start:", first_call["asr_start"])
    print("end:", first_call["asr_end"])
    print("text:", first_call["text"])
else:
    print("\n[WARNING] 호명 후보를 찾지 못했습니다.")