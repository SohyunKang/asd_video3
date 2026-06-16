import re
import torch
import librosa
import pandas as pd
from pathlib import Path
from transformers import pipeline

ROOT_DIR = Path("/storage/sohyunkang/video_data")
OUTPUT_DIR = Path("./calling/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXCEL = OUTPUT_DIR / "call_detection_all.xlsx"

SEARCH_START_SEC = 0.0
SEARCH_END_SEC = 15.0
SAMPLE_RATE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ner = pipeline(
    "ner",
    model="Leo97/KoELECTRA-small-v3-modu-ner",
    aggregation_strategy="simple",
    device=0 if DEVICE == "cuda" else -1
)


def normalize_text(text):
    text = str(text).replace(" ", "").replace("##", "").strip()
    text = re.sub(r"[^가-힣]", "", text)
    return text


def load_audio_from_video(video_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(str(video_path), sr=sr, mono=True)
    return y, sr

def run_silero_vad(y, sr):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    get_speech_timestamps = utils[0]

    return get_speech_timestamps(
        torch.tensor(y),
        model,
        sampling_rate=sr,
        threshold=0.15,
        min_speech_duration_ms=80,
        min_silence_duration_ms=50,
        speech_pad_ms=100,
        return_seconds=True
    )


def transcribe_with_whisperx(video_path):
    import whisperx

    audio = whisperx.load_audio(str(video_path))

    audio = audio[
        int(SEARCH_START_SEC * SAMPLE_RATE):
        int(SEARCH_END_SEC * SAMPLE_RATE)
    ]

    model = whisperx.load_model(
        "small",
        device=DEVICE,
        language="ko",
    )

    result = model.transcribe(audio)

    align_model, metadata = whisperx.load_align_model(
        language_code="ko",
        device=DEVICE
    )

    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        DEVICE
    )

    return aligned["segments"]


def extract_person_names(text):
    entities = ner(str(text).strip())
    names = []

    for ent in entities:
        label = ent.get("entity_group", "")

        if label in ["PS", "PER", "PERSON"]:
            name = normalize_text(ent["word"])

            if name:
                names.append(name)

    return list(set(names))


def is_call_form(word, names):
    word_norm = normalize_text(word)

    for name in names:
        name_norm = normalize_text(name)

        if word_norm == name_norm:
            return True

        if word_norm == name_norm + "아":
            return True

        if word_norm == name_norm + "야":
            return True
        
        if word_norm == name_norm + "라":
            return True
        
        if word_norm == name_norm + "가":
            return True
        
        if word_norm == name_norm + "나":
            return True

    return False


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def get_vad_overlap(word_start, word_end, vad_segments):
    matched_vad = False
    max_vad_overlap = 0.0

    for vad in vad_segments:
        vad_start = float(vad["start"])
        vad_end = float(vad["end"])

        ov = overlap(
            word_start,
            word_end,
            vad_start,
            vad_end
        )

        if ov > 0:
            matched_vad = True
            max_vad_overlap = max(max_vad_overlap, ov)

    return matched_vad, max_vad_overlap

def is_regex_call_candidate(word):
    word_norm = normalize_text(word)

    if re.fullmatch(r"[가-힣]{2,4}(아|야|라)", word_norm):
        return True

    return False
import subprocess


import re
import subprocess
from pathlib import Path


def get_mean_volume(video_path):
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = result.stderr

    m = re.search(
        r"mean_volume:\s*(-?\d+\.?\d*)\s*dB",
        output
    )

    if m:
        return float(m.group(1))

    return None

def convert_to_wav_16k(video_path):
    tmp_dir = Path("./calling/tmp_wav")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wav_path = tmp_dir / f"{Path(video_path).stem}_16k.wav"

    mean_db = get_mean_volume(video_path)
    print(mean_db)
    target_mean_db = -20.0

    if mean_db is None:
        gain_db = 0.0
    else:
        gain_db = max(0.0, target_mean_db - mean_db)
        gain_db = min(gain_db, 12.0)

    audio_filter = (
        "highpass=f=100,"
        "lowpass=f=4500,"
        "afftdn=nf=-20,"
        f"volume={gain_db}dB"
    )

    print(
        f"[AUDIO] {Path(video_path).name} "
        f"mean={mean_db:.1f}dB "
        f"gain=+{gain_db:.1f}dB"
    )

    # audio_filter = (
    #     "highpass=f=120,"
    #     "lowpass=f=3800,"
    #     "afftdn=nf=-25,"
    #     f"volume={gain_db}dB"
    # )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", audio_filter,
        "-c:a", "pcm_s16le",
        str(wav_path)
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("video:", video_path.name)
    print("ENTER convert_to_wav_16k")

    return wav_path

def find_call_segments(video_path):
    wav_path = convert_to_wav_16k(video_path)

    y, sr = load_audio_from_video(wav_path)

    y = y[
        int(SEARCH_START_SEC * sr):
        int(SEARCH_END_SEC * sr)
    ]

    vad_segments = run_silero_vad(y, sr)
    asr_segments = transcribe_with_whisperx(wav_path)

    rows = []

    if len(vad_segments) == 0:
        rows.append({
            "video_id": video_path.stem,
            "video_file": video_path.name,
            "video_path": str(video_path),
            "call_start": None,
            "call_end": None,
            "call_word": None,
            "person_names_in_segment": None,
            "segment_text": None,
            "matched_vad": False,
            "vad_overlap_sec": 0.0,
            "is_call_candidate": False,
            "review": "",
            "error": "VAD_EMPTY"
        })
        return rows

    if len(asr_segments) == 0:
        for vad in vad_segments:
            vad_start = float(vad["start"])
            vad_end = float(vad["end"])

            rows.append({
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "call_start": vad_start,
                "call_end": vad_end,
                "call_word": None,
                "person_names_in_segment": None,
                "segment_text": None,
                "matched_vad": True,
                "vad_overlap_sec": vad_end - vad_start,
                "is_call_candidate": False,
                "review": "",
                "error": "VAD_FOUND_BUT_ASR_EMPTY"
            })
        return rows

    debug_rows = []

    for seg in asr_segments:
        seg_text = str(seg.get("text", "")).strip()
        names = extract_person_names(seg_text)
        words = seg.get("words", [])

        if len(words) == 0:
            debug_rows.append({
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "call_start": None,
                "call_end": None,
                "call_word": None,
                "person_names_in_segment": ",".join(names),
                "segment_text": seg_text,
                "matched_vad": False,
                "vad_overlap_sec": 0.0,
                "is_call_candidate": False,
                "review": "",
                "error": "ASR_SEGMENT_WITHOUT_WORD_TIMESTAMPS"
            })
            continue

        if len(names) == 0:
            debug_rows.append({
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "call_start": None,
                "call_end": None,
                "call_word": None,
                "person_names_in_segment": "",
                "segment_text": seg_text,
                "matched_vad": False,
                "vad_overlap_sec": 0.0,
                "is_call_candidate": False,
                "review": "",
                "error": "NER_NO_NAME"
            })

        for word in words:
            if "start" not in word or "end" not in word:
                continue

            word_start = float(word["start"])
            word_end = float(word["end"])
            word_text = str(word["word"]).strip()

            if word_end < SEARCH_START_SEC:
                continue

            if word_start > SEARCH_END_SEC:
                continue

            if len(names) > 0:
                is_call = is_call_form(word_text, names)
            else:
                is_call = bool(
                    re.fullmatch(
                        r"[가-힣]{2,4}(아|야)",
                        normalize_text(word_text)
                    )
                )

            best_vad, max_vad_overlap = get_best_vad_segment(
                word_start,
                word_end,
                vad_segments
            )

            matched_vad = best_vad is not None

            call_start = word_start
            call_end = word_end

            if matched_vad:
                call_start = max(word_start, best_vad[0])
                call_end = min(word_end, best_vad[1])

            if not is_call:
                debug_rows.append({
                    "video_id": video_path.stem,
                    "video_file": video_path.name,
                    "video_path": str(video_path),
                    "call_start": word_start,
                    "call_end": word_end,
                    "call_word": word_text,
                    "person_names_in_segment": ",".join(names),
                    "segment_text": seg_text,
                    "matched_vad": matched_vad,
                    "vad_overlap_sec": max_vad_overlap,
                    "is_call_candidate": False,
                    "review": "",
                    "error": "NO_CALL_PATTERN"
                })
                continue

            rows.append({
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "call_start": call_start,
                "call_end": call_end,
                "raw_word_start": word_start,
                "raw_word_end": word_end,
                "call_word": word_text,
                "person_names_in_segment": ",".join(names),
                "segment_text": seg_text,
                "matched_vad": matched_vad,
                "vad_overlap_sec": max_vad_overlap,
                "is_call_candidate": True,
                "status": "DETECTED",
                "error": ""
            })

    if len(rows) == 0:
        return debug_rows

    return rows

def get_best_vad_segment(word_start, word_end, vad_segments):
    best_vad = None
    best_overlap = 0.0

    for vad in vad_segments:
        vad_start = float(vad["start"])
        vad_end = float(vad["end"])

        ov = overlap(word_start, word_end, vad_start, vad_end)

        if ov > best_overlap:
            best_overlap = ov
            best_vad = (vad_start, vad_end)

    return best_vad, best_overlap

def find_target_videos(root_dir):
    root_dir = Path(root_dir)

    videos = sorted(
        root_dir.rglob("IF2001_*.mp4")
    )

    return videos


def make_empty_row(video_path, error="", note="no_call_detected"):
    return {
        "video_id": video_path.stem,
        "video_file": video_path.name,
        "video_path": str(video_path),
        "call_start": None,
        "call_end": None,
        "call_word": None,
        "person_names_in_segment": None,
        "segment_text": None,
        "matched_vad": False,
        "vad_overlap_sec": 0.0,
        "is_call_candidate": False,
        "review": "",
        "note": note,
        "error": error
    }


def save_excel(rows, save_path):
    columns = [
        "video_id",
        "video_file",
        "video_path",
        "call_start",
        "call_end",
        "call_word",
        "person_names_in_segment",
        "segment_text",
        "matched_vad",
        "vad_overlap_sec",
        "is_call_candidate",
        "review",
        "note",
        "error"
    ]

    df = pd.DataFrame(rows)

    if len(df) == 0:
        df = pd.DataFrame(columns=columns)
    else:
        for col in columns:
            if col not in df.columns:
                df[col] = None

        df = df[columns]

        # df = df.sort_values(
        #     ["video_id", "call_start"],
        #     na_position="last"
        # )

    df.to_excel(save_path, index=False)


def main():
    videos = find_target_videos(ROOT_DIR)

    print(f"총 처리 대상 영상 수: {len(videos)}")

    all_rows = []

    for idx, video_path in enumerate(videos, start=1):
        # if not "IF2001_1_1_1023041312_1.mp4.mp4" in str(video_path):
        #     continue

        print(f"\n[{idx}/{len(videos)}]")
        print("video:", video_path.name)
        
        try:
            rows = find_call_segments(video_path)

            all_rows.extend(rows)

        except Exception as e:
            all_rows.append(
                make_empty_row(video_path, error=str(e))
            )

            print("[ERROR]", e)

        if idx % 10 == 0:
            print(f"\n[INFO] 중간 저장: {len(all_rows)} rows")
            save_excel(all_rows, OUTPUT_EXCEL)

    print(all_rows)        
    save_excel(all_rows, OUTPUT_EXCEL)

    print("\n[DONE]")
    print("saved:", OUTPUT_EXCEL)


if __name__ == "__main__":
    main()