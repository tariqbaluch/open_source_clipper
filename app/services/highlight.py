# 5_Highlight_TimeSnap_Stable.py
# pip install --upgrade openai tqdm
import json
import os
import subprocess
import shlex
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ---------------- CONFIG ----------------
client = OpenAI()

MODEL = "gpt-4o-mini"            # or "gpt-4o" for best accuracy
TOP_K = 10
MAX_WORKERS = 4
OVERLAP = 15.0                   # seconds overlap between chunks
MIN_SEC = 30.0                   # default minimum highlight length
MAX_SEC = 90.0                   # default maximum highlight length
HIGHLIGHT_DIR = "highlights"
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/Arial.ttf" # update for your system

'''
- For each chosen highlight, generate **3 versions**:
   - Short: 15–30s
   - Medium: 30–60s
   - Long: 60–180s

'''

# ---------------- DURATION PRESETS ----------------

def map_duration_preset(preset: str | None) -> (float, float):
    """Translate a UI duration_preset string into (min_sec, max_sec).

    These bounds are used by the highlight generator to enforce clip length
    per job. If an unknown or missing preset is provided, fall back to a
    reasonable default.
    """
    mapping = {
        "lt_30":   (5.0, 29.0),
        "30_59":   (30.0, 59.0),
        "60_89":   (60.0, 89.0),
        "90_180":  (90.0, 180.0),
        "180_300": (180.0, 300.0),
        "300_600": (300.0, 600.0),
        "600_900": (600.0, 900.0),
        "auto":    (MIN_SEC, MAX_SEC),
        None:       (MIN_SEC, MAX_SEC),
    }
    return mapping.get(preset, (MIN_SEC, MAX_SEC))


# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You are a professional short-form video editor.

You will be given transcript lines with timestamps and several simple numeric/audio features per line:
- energy: number where higher means louder/more intense audio.
- scene_id: integer identifying which visual scene the line belongs to (0, 1, 2, ...).
- near_cut: true/false indicating that the line is very close to a natural scene/shot cut.
- tags: zero or more audio tags like "music" or "laughter".
- excitement: a 0–1 score combining energy, cuts and tags (higher means more exciting).

DO NOT invent or rewrite text.
Your job: confidently pick engaging *time ranges* (start_time and end_time in seconds) that mark interesting
30–90 second candidate highlights (hooks, emotional moments, surprising lines) in the provided transcript.

**Important rules (follow exactly):**
- Return JSON only.
- Pay close attention to the transcript *content* first.
- Use the features as guidance:
  - Higher energy and excitement usually mean more intense/emotional moments.
  - near_cut=true is often a good place to start or end a highlight so the cut feels natural.
  - tags like "laughter" suggest funny moments; "music" can suggest hype or build-up.
  - Do NOT rely on any single feature alone; only choose highlights where the spoken content is strong
    AND the features support it.
- You should almost always return several clips when the transcript is non-empty. It is better to
  select a few reasonable candidates than to be over-cautious and return nothing.
- For short or low-intensity videos, still pick the most interesting parts (even if subtle) rather
  than returning zero clips.
- For each clip return: id, start_time, end_time, category, title, caption, hashtags, description,
  scores (hook, surprise_novelty, emotion,
  clarity_self_contained, shareability, cta_potential between 0 and 5), overall virality_score (average), unsafe (true/false), why (short).
- Based on the scores, generate overall "virality_score" (average).
- start_time and end_time should be numeric seconds (e.g., 123.45).
- Use only timestamps visible in the transcript lines (do not invent times outside the range shown).
- The client code will snap your start/end to transcript line boundaries and enforce exact duration.
- Prefer highlight boundaries that align with scene changes (near_cut=true) when possible.
- For each chosen highlight, assign one category:
   - "Hook"
   - "Tip"
   - "Insight"
   - "Story"
   - "Conclusion"
- For each chosen highlight, generate title (catchy 3–6 words, good for social media title overlay),
  caption (short 1-sentence subtitle text, conversational style, emoji-friendly), 5–8 trending hashtags relevant to the highlight
  and longer description (2–3 sentences) suitable for YouTube/Facebook.
- Example output:
{{
  "clips": [
    {{
      "id": "clip_1",
      "start_time": 12.34,
      "end_time": 47.89,
      "category": "Hook",
      "title": "catchy 3–6 words, good for social media title overlay",
      "caption": "short 1-sentence subtitle text, conversational style, emoji-friendly",
      "hashtags": "#financialtips, #wealth",
      "description": "longer description (2–3 sentences) suitable for YouTube/Facebook",
      "scores": {{ "hook":5, "surprise_novelty":4, "emotion":4, "clarity_self_contained":5, "shareability":4, "cta_potential":3, "unsafe": false }},
      "virality_score": 5.1,
      "unsafe": "false",
      "why": "short reason"
    }}
  ]
}}
""".format(TOP_K=TOP_K)

USER_TEMPLATE = """{prompt_hint}Transcript lines (start | end | energy | scene_id | near_cut | tags | excitement | text):
{lines}

Now pick up to {k} interesting highlights and return JSON only as specified. If nothing seems extremely
strong, still choose the best available parts (do NOT return an empty clips list unless the transcript
is completely empty or unusable).
"""

# ---------------- HELPERS ----------------
def build_transcript_block(transcript: List[Dict]) -> str:
    """Format transcript for prompt."""
    lines = []
    for t in transcript:
        text = t["text"].replace("\n", " ").strip()
        energy = t.get("energy")
        scene_id = t.get("scene_id")
        near_cut = t.get("near_cut")
        tags = t.get("tags") or []
        excitement = t.get("excitement")

        energy_str = "" if energy is None else f"{energy:.4f}"
        scene_str = "" if scene_id is None else str(scene_id)
        near_cut_str = "" if near_cut is None else str(bool(near_cut))
        tags_str = ",".join(tags) if tags else ""
        excitement_str = "" if excitement is None else f"{float(excitement):.4f}"

        lines.append(
            f"{t['start']:.2f} | {t['end']:.2f} | energy={energy_str} | scene_id={scene_str} | near_cut={near_cut_str} | tags={tags_str} | excitement={excitement_str} | text={text}"
        )
    return "\n".join(lines)

def choose_chunk_duration(total_duration: float) -> float:
    """Adaptive chunking durations."""
    if total_duration <= 600: return total_duration
    elif total_duration <= 1800: return 600.0
    elif total_duration <= 3600: return 900.0
    elif total_duration <= 7200: return 1200.0
    else: return 1500.0

def split_transcript(transcript: List[Dict], chunk_duration: float, overlap: float) -> List[List[Dict]]:
    chunks = []
    start_time = transcript[0]["start"]
    end_time = transcript[-1]["end"]
    cur_start = start_time
    while cur_start < end_time:
        cur_end = cur_start + chunk_duration
        chunk = [t for t in transcript if t["end"] > cur_start and t["start"] < cur_end]
        if not chunk:
            break
        chunks.append(chunk)
        cur_start = cur_end - overlap
    return chunks

def identify_and_score_chunk(chunk_transcript: List[Dict], chunk_id: int, user_prompt: str | None = None) -> Dict:
    """Ask LLM for start_time/end_time ranges for highlights (single chunk)."""
    prompt_hint = ""
    if user_prompt:
        prompt_hint = f"The user specifically requested: {user_prompt.strip()}\n\n"
    prompt = USER_TEMPLATE.format(prompt_hint=prompt_hint, lines=build_transcript_block(chunk_transcript), k=TOP_K)
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(completion.choices[0].message.content)
        for c in data.get("clips", []):
            c["chunk_id"] = chunk_id
        return data
    except Exception as e:
        print(f"⚠️ Chunk {chunk_id} failed: {e}")
        return {"clips": []}

def snap_time_to_transcript(start_time: float, end_time: float, transcript: List[Dict]) -> (float, float, int, int):
    """
    Snap given float start_time/end_time to nearest transcript line boundaries.
    Returns snapped_start, snapped_end, start_index, end_index.
    """
    n = len(transcript)
    # find first index with end > start_time
    si = next((i for i, t in enumerate(transcript) if t["end"] > start_time), 0)
    # find last index with start < end_time
    ei = next((i for i, t in enumerate(transcript) if t["start"] >= end_time), n-1)
    # adjust ei to be the last whose start < end_time
    if ei == 0 or transcript[ei]["start"] >= end_time:
        # walk backward
        ei = max(0, next((i for i in range(n-1, -1, -1) if transcript[i]["start"] < end_time), n-1))
    snapped_start = transcript[si]["start"]
    snapped_end = transcript[ei]["end"]
    return round(snapped_start, 3), round(snapped_end, 3), si, ei

def enforce_duration(sn_start: float, sn_end: float, si: int, ei: int, transcript: List[Dict],
                     min_sec: float, max_sec: float) -> (float, float, int, int, str):
    """
    Expand/trim snapped start/end by transcript lines to meet min/max durations.
    Returns possibly adjusted start/end indices and reason string.
    """
    reason = ""
    dur = sn_end - sn_start
    n = len(transcript)

    # Expand outward if too short
    if dur < min_sec:
        # try expanding backwards then forwards until >= min_sec
        s_idx, e_idx = si, ei
        while e_idx < n - 1 and (transcript[e_idx]["end"] - transcript[s_idx]["start"]) < min_sec:
            e_idx += 1
        # if still short and can expand backwards, expand backwards
        while s_idx > 0 and (transcript[e_idx]["end"] - transcript[s_idx]["start"]) < min_sec:
            s_idx -= 1
        new_start = transcript[s_idx]["start"]
        new_end = transcript[e_idx]["end"]
        reason = f"expanded {dur:.1f}s→{new_end - new_start:.1f}s"
        return round(new_start, 3), round(new_end, 3), s_idx, e_idx, reason

    # Trim if too long
    if dur > max_sec:
        # limit end to start + max_sec, snap to nearest transcript end <= target
        target_end = sn_start + max_sec
        candidate_ends = [t["end"] for t in transcript if t["end"] <= target_end]
        if candidate_ends:
            new_end = candidate_ends[-1]
            # find new end idx
            new_ei = next(i for i, t in enumerate(transcript) if t["end"] == new_end)
            reason = f"trimmed {dur:.1f}s→{new_end - sn_start:.1f}s"
            return round(sn_start, 3), round(new_end, 3), si, new_ei, reason
    return sn_start, sn_end, si, ei, reason

def rebuild_text_from_indices(si: int, ei: int, transcript: List[Dict]) -> str:
    return " ".join(t["text"].strip() for t in transcript[si:ei+1])

def materialize_from_times(raw_clips: List[Dict], transcript: List[Dict], min_sec: float, max_sec: float) -> List[Dict]:
    """Convert LLM start_time/end_time floats into snapped transcript-aligned clips."""
    materialized = []
    n = len(transcript)
    for clip in raw_clips:
        st = clip.get("start_time")
        et = clip.get("end_time")
        if st is None or et is None:
            print(f"⚠️ Skipping invalid clip (missing start_time/end_time): {clip}")
            continue
        try:
            st = float(st); et = float(et)
        except Exception:
            print(f"⚠️ Skipping invalid clip (non-float times): {clip}")
            continue
        # Ensure within transcript range
        if st < transcript[0]["start"]: st = transcript[0]["start"]
        if et > transcript[-1]["end"]: et = transcript[-1]["end"]
        if et <= st:
            print(f"⚠️ Skipping invalid clip (end <= start): {clip}")
            continue

        # Snap to transcript boundaries
        sn_start, sn_end, si, ei = snap_time_to_transcript(st, et, transcript)
        # Enforce duration by expanding/ trimming to transcript line anchors
        sn_start2, sn_end2, si2, ei2, reason = enforce_duration(sn_start, sn_end, si, ei, transcript, min_sec=min_sec, max_sec=max_sec)

        # Do not allow the clip to drift too far earlier than the LLM's
        # suggested start_time. This avoids cases where expansion jumps all
        # the way to the beginning of the song/video.
        #
        # However, with strict presets like 30–60s we sometimes need to go
        # further back than 5s (especially near the end of the video) to
        # reach the requested minimum duration. Make the backward allowance
        # scale with the requested min_sec while capping it so we still stay
        # reasonably close to the model's choice.
        base_back = 5.0
        scaled_back = min_sec * 0.5  # e.g. 15s when min_sec=30
        MAX_BACK_EXPAND = max(base_back, min(scaled_back, 30.0))
        earliest_allowed = max(transcript[0]["start"], st - MAX_BACK_EXPAND)
        if sn_start2 < earliest_allowed:
            # Clamp start forward and re-snap end/index to keep duration
            clamped_start = earliest_allowed
            sn_start2, sn_end2, si2, ei2 = snap_time_to_transcript(clamped_start, sn_end2, transcript)

        # Clamp final duration into [min_sec, max_sec] to respect the UI preset
        final_dur = sn_end2 - sn_start2
        HARD_MAX = max_sec
        if final_dur > HARD_MAX:
            target_end = sn_start2 + HARD_MAX
            # snap end down to nearest transcript boundary <= target_end
            candidate_ends = [t["end"] for t in transcript if t["end"] <= target_end]
            if candidate_ends:
                new_end = candidate_ends[-1]
                sn_end2 = new_end
                ei2 = next(i for i, t in enumerate(transcript) if t["end"] == new_end)

        # After all adjustments, if we are still well below the requested
        # minimum duration and the transcript has room, try to expand again
        # (prefer forwards, then slightly backwards but never before
        # earliest_allowed). This avoids ultra-short clips like 7s when the
        # user requested 60–90s.
        final_dur = sn_end2 - sn_start2
        if final_dur + 1e-3 < min_sec:
            s_idx, e_idx = si2, ei2
            n_seg = len(transcript)

            # 1) Try extending the end forward while respecting max_sec
            while e_idx < n_seg - 1:
                candidate_end = transcript[e_idx + 1]["end"]
                cand_dur = candidate_end - sn_start2
                if cand_dur > max_sec + 1e-3:
                    break
                e_idx += 1
                sn_end2 = transcript[e_idx]["end"]
                final_dur = sn_end2 - sn_start2
                if final_dur >= min_sec - 1e-3:
                    break

            # 2) If still short and we have space backwards (without
            # violating earliest_allowed), try expanding start backwards.
            if final_dur < min_sec - 1e-3 and s_idx > 0:
                while s_idx > 0 and transcript[s_idx - 1]["start"] >= earliest_allowed:
                    candidate_start = transcript[s_idx - 1]["start"]
                    cand_dur = sn_end2 - candidate_start
                    if cand_dur > max_sec + 1e-3:
                        break
                    s_idx -= 1
                    sn_start2 = transcript[s_idx]["start"]
                    final_dur = sn_end2 - sn_start2
                    if final_dur >= min_sec - 1e-3:
                        break

            si2, ei2 = s_idx, e_idx

        # Rebuild text
        text = rebuild_text_from_indices(si2, ei2, transcript)
        clip_out = dict(clip)  # copy all original fields (scores, why)
        clip_out.update({
            "start": sn_start2,
            "end": sn_end2,
            "duration": round(sn_end2 - sn_start2, 3),
            "start_idx": si2,
            "end_idx": ei2,
            "text": text,
            "adjustment_reason": reason
        })
        materialized.append(clip_out)
    print(f"🧩 Materialized {len(materialized)} valid clips from {len(raw_clips)} raw entries.")
    return materialized

def deduplicate_clips_keep_best(clips: List[Dict], overlap_threshold: float = 0.6) -> List[Dict]:
    """Remove heavy temporal duplicates; keep best overall (or earlier if tie)."""
    if not clips:
        return clips
    clips.sort(key=lambda c: (-c.get("overall", 0), c["start"]))
    unique = []
    for clip in clips:
        keep = True
        for u in unique:
            # temporal overlap fraction relative to min duration
            overlap = max(0, min(u["end"], clip["end"]) - max(u["start"], clip["start"]))
            min_dur = min(u["duration"], clip["duration"]) if min(u["duration"], clip["duration"]) > 0 else 1
            if overlap / min_dur > overlap_threshold:
                keep = False
                break
        if keep:
            unique.append(clip)
    return sorted(unique, key=lambda c: c["start"]) 


def merge_adjacent_clips(clips: List[Dict], max_gap: float = 2.0, max_duration: float = MAX_SEC) -> List[Dict]:
    """Merge clips that are overlapping or very close in time into stronger segments.

    Pass 1: always merge overlapping clips, and clips with gap <= max_gap,
    as long as merged duration <= max_duration.

    Pass 2: clean up very short clips (<25s) by trying to merge them with
    neighbors within a slightly larger gap (5s), still respecting max_duration.
    """
    if not clips:
        return clips

    # Ensure chronological order
    clips = sorted(clips, key=lambda c: c["start"])

    def _merge_pair(a: Dict, b: Dict) -> Dict:
        """Merge two clips a and b into a single clip, preferring higher overall."""
        new_start = min(a["start"], b["start"])
        new_end = max(a["end"], b["end"])
        new_duration = new_end - new_start

        cur_overall = a.get("overall", 0) or 0
        nxt_overall = b.get("overall", 0) or 0
        primary = a if cur_overall >= nxt_overall else b

        merged_clip = dict(primary)
        merged_clip["start"] = round(new_start, 3)
        merged_clip["end"] = round(new_end, 3)
        merged_clip["duration"] = round(new_duration, 3)

        # Concatenate text if available
        text_a = (a.get("text") or "").strip()
        text_b = (b.get("text") or "").strip()
        if text_a or text_b:
            if text_a and text_b:
                merged_clip["text"] = f"{text_a} {text_b}"
            else:
                merged_clip["text"] = text_a or text_b

        # Track provenance if useful for debugging
        source_ids = []
        if a.get("id"):
            source_ids.append(a["id"])
        if b.get("id"):
            source_ids.append(b["id"])
        if source_ids:
            merged_clip["merged_from_ids"] = source_ids

        # Invalidate indices since we are merging ranges; the subtitle generator
        # will fall back to time-based filtering which is safer.
        merged_clip.pop("start_idx", None)
        merged_clip.pop("end_idx", None)

        return merged_clip

    # ----- Pass 1: merge overlaps and very small gaps -----
    merged: List[Dict] = []
    current = dict(clips[0])
    for nxt in clips[1:]:
        gap = nxt["start"] - current["end"]
        new_start = min(current["start"], nxt["start"])
        new_end = max(current["end"], nxt["end"])
        new_duration = new_end - new_start

        # Always merge if they overlap (gap < 0), or if gap <= max_gap
        if (gap <= max_gap) and (new_duration <= max_duration):
            current = _merge_pair(current, nxt)
        else:
            merged.append(current)
            current = dict(nxt)

    merged.append(current)

    # ----- Pass 2: clean up very short clips by merging with neighbors -----
    if len(merged) <= 1:
        return merged

    SHORT_THRESHOLD = 25.0
    EXTRA_GAP = 5.0

    cleaned: List[Dict] = []
    i = 0
    while i < len(merged):
        clip = merged[i]
        duration = clip["duration"] if "duration" in clip else (clip["end"] - clip["start"])
        if duration >= SHORT_THRESHOLD or i == len(merged) - 1:
            cleaned.append(clip)
            i += 1
            continue

        # Try to merge short clip with either previous (last in cleaned) or next
        merged_candidate = None

        # Prefer merging with next if possible (forward in time)
        nxt = merged[i + 1]
        gap_next = nxt["start"] - clip["end"]
        new_start_next = min(clip["start"], nxt["start"])
        new_end_next = max(clip["end"], nxt["end"])
        new_dur_next = new_end_next - new_start_next
        if gap_next <= EXTRA_GAP and new_dur_next <= max_duration:
            merged_candidate = _merge_pair(clip, nxt)
            i += 2  # consumed clip and next
        elif cleaned:
            prev = cleaned[-1]
            gap_prev = clip["start"] - prev["end"]
            new_start_prev = min(prev["start"], clip["start"])
            new_end_prev = max(prev["end"], clip["end"])
            new_dur_prev = new_end_prev - new_start_prev
            if gap_prev <= EXTRA_GAP and new_dur_prev <= max_duration:
                merged_candidate = _merge_pair(prev, clip)
                cleaned[-1] = merged_candidate
                i += 1

        if merged_candidate is None:
            cleaned.append(clip)
            i += 1
        elif merged_candidate not in cleaned:
            cleaned.append(merged_candidate)

    return cleaned

def assign_unique_ids(clips: List[Dict]) -> List[Dict]:
    """Assign sequential clip IDs and filenames in chronological order."""
    clips.sort(key=lambda c: c["start"]) 
    for i, clip in enumerate(clips, start=1):
        clip_id = f"clip_{i:03d}"
        clip["id"] = clip_id
        clip["filename"] = f"highlight_{clip_id}.mp4"
    return clips

def escape_font_path_for_ffmpeg(path: str) -> str:
    """
    Make a Windows font path safe for ffmpeg drawtext.
    - Use forward slashes
    - Escape the drive colon C: -> C\:
    - Return quoted path to be safe with spaces
    """
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":  # Windows drive letter
        p = p[0] + r"\:" + p[2:]
    return f"'{p}'"  # quote the whole thing

def escape_caption(caption: str) -> str:
    safe = (caption
            .replace("\\", r"\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace('"', r'\"'))
    return f'"{safe}"'  # wrap in double quotes for text only


def generate_highlights(input_video: str, transcript_path: str, job_dir: str,
                        min_sec: float = MIN_SEC, max_sec: float = MAX_SEC,
                        user_prompt: str | None = None) -> dict:
    """Run the highlight generation pipeline and return clips + paths.

    This wraps the existing script logic so it can be called from Celery or other
    Python code. It will also write highlights.json into the given job_dir.
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"{transcript_path} not found.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # quick sanity
    if not transcript or "start" not in transcript[0] or "end" not in transcript[0]:
        raise ValueError("transcript.json must be an array of {start: float, end: float, text: str} items.")

    # Ensure transcript is strictly sorted in time. Some upstream generators
    # may emit segments in chunk order (0..N) rather than global time order,
    # which breaks snapping and duration expansion.
    transcript = sorted(transcript, key=lambda t: t["start"])

    total_dur = transcript[-1]["end"] - transcript[0]["start"]
    print(f"📜 Loaded {len(transcript)} transcript lines ({total_dur/60:.1f} min).")

    # Normalize per-job bounds to be sane
    min_sec = max(5.0, float(min_sec))
    max_sec = max(min_sec, float(max_sec))

    # chunking
    CHUNK_DURATION = choose_chunk_duration(total_dur)
    chunks = split_transcript(transcript, CHUNK_DURATION, OVERLAP)
    print(f"✂️ Split into {len(chunks)} chunks (~{CHUNK_DURATION/60:.1f} min, {OVERLAP}s overlap).")

    # run LLM on each chunk in parallel
    raw_clips = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for i, ch in enumerate(chunks):
            futures.append(ex.submit(identify_and_score_chunk, ch, i, user_prompt))
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM chunks"):
            res = future.result()
            if res and "clips" in res:
                raw_clips.extend(res["clips"])

    print(f"✅ LLM returned {len(raw_clips)} raw clip candidates across chunks.")

    # Materialize -> snap times to transcript -> enforce durations -> rebuild text
    clips = materialize_from_times(raw_clips, transcript, min_sec=min_sec, max_sec=max_sec)

    # compute overall scores locally (weights)
    WEIGHTS = {"hook":0.25,"surprise_novelty":0.15,"emotion":0.15,"clarity_self_contained":0.15,"shareability":0.15,"cta_potential":0.15}
    for c in clips:
        sc = c.get("scores", {})
        subtotal = sum(WEIGHTS[k] * max(0, min(5, int(sc.get(k, 0)))) for k in WEIGHTS)
        base_overall = round(subtotal / 5 * 100, 1)

        # position-aware adjustment: gently prefer clips that are not only at the very start
        mid = (c["start"] + c["end"]) / 2.0
        pos_frac = 0.0
        if total_dur > 0:
            pos_frac = max(0.0, min(1.0, mid / total_dur))
        # 80% content-driven, 20% position-driven (later clips can get a small boost)
        adjusted = base_overall * (0.8 + 0.2 * pos_frac)
        c["overall"] = round(adjusted, 1)

    # dedupe near-duplicates (keep best)
    clips = deduplicate_clips_keep_best(clips, overlap_threshold=0.7)

    # optional: enforce time diversity (min gap) to spread clips across timeline
    def enforce_time_diversity_simple(clips_list: List[Dict], min_gap: float = 30.0) -> List[Dict]:
        clips_list.sort(key=lambda x: -x["overall"]) 
        selected = []
        occupied_until = float("-inf")
        for c in clips_list:
            if c["start"] >= occupied_until + min_gap:
                selected.append(c)
                occupied_until = c["end"]
        return sorted(selected, key=lambda c: c["start"]) 

    clips = merge_adjacent_clips(clips, max_gap=2.0, max_duration=MAX_SEC)

    clips_before_diversity = list(clips)
    clips = enforce_time_diversity_simple(clips, min_gap=30.0)

    # assign unique IDs and filenames
    clips = assign_unique_ids(clips)

    # final top-K
    top = clips[:TOP_K]

    # If we still ended up with no clips, try a heuristic fallback so the
    # caller always gets at least one highlight for non-empty transcripts.
    if not top:
        print("⚠️ No clips after LLM + filters. Falling back to heuristic excitement-based segment.")
        start_all = transcript[0]["start"]
        end_all = transcript[-1]["end"]
        total = max(0.0, end_all - start_all)
        if total >= 10.0:
            # scan for a ~30s window with highest average excitement
            target_window = 30.0
            best_score = -1.0
            best_start = start_all
            best_end = min(end_all, start_all + target_window)

            # precompute a simple per-line excitement, defaulting to 0
            ex_values = []
            for seg in transcript:
                val = seg.get("excitement")
                try:
                    ex_values.append(float(val) if val is not None else 0.0)
                except Exception:
                    ex_values.append(0.0)

            # sliding window over transcript lines
            n = len(transcript)
            for i in range(n):
                w_start = transcript[i]["start"]
                w_end = w_start + target_window
                if w_start >= end_all:
                    break
                # extend j until we cover ~target_window or run out of transcript
                total_ex = 0.0
                count = 0
                j = i
                while j < n and transcript[j]["end"] <= w_end:
                    total_ex += ex_values[j]
                    count += 1
                    j += 1
                if count == 0:
                    continue
                avg_ex = total_ex / count
                if avg_ex > best_score:
                    best_score = avg_ex
                    best_start = w_start
                    best_end = min(end_all, max(best_start + 10.0, transcript[min(j, n-1)]["end"]))

            heur_start = best_start
            heur_end = best_end
            if heur_end > heur_start:
                overall = 60.0
                if clips_before_diversity:
                    overall = max(c.get("overall", overall) for c in clips_before_diversity)
                fallback = {
                    "id": "clip_001",
                    "start": round(heur_start, 3),
                    "end": round(heur_end, 3),
                    "duration": round(heur_end - heur_start, 3),
                    "category": "Story",
                    "title": "Highlight segment",
                    "caption": "Most interesting part of this video.",
                    "hashtags": "",
                    "description": "Automatically chosen highlight segment.",
                    "scores": {},
                    "overall": overall,
                    "why": "Fallback heuristic when LLM did not propose any strong clip.",
                }
                top = [fallback]

    print(f"🎯 Final highlights: {len(top)} clips (top {TOP_K}).")

    # ensure job_dir exists and write highlights.json inside it
    os.makedirs(job_dir, exist_ok=True)
    highlights_path = os.path.join(job_dir, "highlights.json")
    with open(highlights_path, "w", encoding="utf-8") as f:
        json.dump({"video_path": input_video, "clips": top}, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {highlights_path}")

    # return structured result (no ffmpeg export here; keep that as a separate concern)
    return {
        "highlights_path": highlights_path,
        "video_path": input_video,
        "clips": top,
    }

# ---------------- MAIN ----------------
if __name__ == "__main__":
    input_video = "input.mp4"         # adjust path
    transcript_path = "transcript.json"

    result = generate_highlights(input_video=input_video, transcript_path=transcript_path, job_dir=HIGHLIGHT_DIR)

    # export with ffmpeg (use -ss before -i for faster keyframe cut; consider re-encoding if needed)
    for c in result["clips"]:
        start, end = c["start"], c["end"]
        caption = c.get("caption") or ""
        output = os.path.join(HIGHLIGHT_DIR, c["category"] + "_" + c["filename"])

        safe_caption = escape_caption(caption)
        safe_font = escape_font_path_for_ffmpeg(FONT_PATH)

        vf = ",".join([
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos",
            "unsharp=5:5:1.0:5:5:0.0",
            "crop=1080:1920",
        ])

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_video,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "copy",
            output
        ]
        print(f"➡️ Exporting {c['id']} {start:.2f}-{end:.2f} → {output}  ({c.get('adjustment_reason','')})")
        subprocess.run(cmd, check=True)

    print("✅ All exports done.")
