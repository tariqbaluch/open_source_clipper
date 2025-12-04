from faster_whisper import WhisperModel
import json, csv, os

def format_time(seconds):
    """Convert seconds to HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def transcribe_with_wait_markers(audio_path, model_size="small.en", device="cpu",
                                 compute_type="int8", silence_threshold=0.6):
    """
    Transcribe + align audio with faster-whisper (word-level timestamps).
    Generate TTS-ready text with [WAIT Xs] markers for pauses.
    """
    print(f"Processing {audio_path} ...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # 1. Transcribe with word timestamps
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Convert to list format compatible with original code
    segments_list = []
    for seg in segments:
        words = []
        for word in seg.words:
            words.append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            })
        segments_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words
        })
    
    print(f"Transcribed {len(segments_list)} segments.")
    segments = segments_list

    # 3. Collect words + generate TTS text
    tts_lines = []
    csv_rows = [["segment", "word", "start", "end", "duration", "gap_to_next"]]

    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        for j, w in enumerate(words):
            start, end = w.get("start"), w.get("end")
            duration = (end - start) if (end is not None and start is not None) else 0
            tts_lines.append(w["word"])
            csv_row = [i + 1, w["word"], start, end, duration, ""]
            # compute pause after this word
            gap = 0
            next_word_start = words[j + 1].get("start") if j + 1 < len(words) else None
            next_seg_start = segments[i + 1].get("start") if i + 1 < len(segments) else None
            if end is not None and next_word_start is not None:
                gap = next_word_start - end
            elif end is not None and next_seg_start is not None:
                gap = next_seg_start - end
            if gap is not None and gap >= silence_threshold:
                tts_lines.append(f"[WAIT {gap:.2f}s]")
                csv_row[-1] = gap
            csv_rows.append(csv_row)

    # 4. Join text and save outputs
    tts_text = " ".join(tts_lines)
    with open("tts_ready_transcript.txt", "w", encoding="utf-8") as f:
        f.write(tts_text)
    print("Saved: tts_ready_transcript.txt")

    # Save pacing CSV
    with open("word_pacing.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print("Saved: word_pacing.csv")

    # Save JSON for deeper analysis
    with open("aligned_transcript.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print("Saved: aligned_transcript.json")

    print("All done! You can now feed `tts_ready_transcript.txt` into your TTS engine.")


if __name__ == "__main__":
    audio_file = "vocals.wav"  # your file
    transcribe_with_wait_markers(audio_file, model_size="small.en", silence_threshold=0.6)