# Full Video Highlight Pipeline Guide

## Pipeline Overview

```
1. Video Input (YouTube URL or local file)
   ↓
2. Extract Audio (vocals.wav)
   ↓
3. Transcribe with Word Timestamps (aligned_transcript.json)
   ↓
4. Convert Transcript Format (transcript.json)
   ↓
5. Generate Highlights with ChatGPT (highlights.json + highlight clips)
   ↓
6. Apply Speaker Centering (optional, per clip)
   ↓
7. Apply Karaoke Subtitles (optional, per clip)
```

## Step-by-Step Execution

### Step 1: Get Video
**Option A - YouTube Download:**
```bash
python youtube_downloader.py
# Enter YouTube URL when prompted
# Output: downloads/your_video.mp4
```

**Option B - Use Local Video:**
- Place your video as `input.mp4` in the backend folder

### Step 2: Extract Audio
```bash
python extract_audio.py input.mp4
# Output: vocals.wav
```

### Step 3: Transcribe Audio
```bash
python vad_fast_transcribe3.py
# Output: 
#   - aligned_transcript.json (word-level timestamps)
#   - tts_ready_transcript.txt
#   - word_pacing.csv
```

### Step 4: Convert Transcript Format
```bash
python convert_transcript.py aligned_transcript.json transcript.json
# Converts to format needed by highlight generator
# Output: transcript.json
```

### Step 5: Generate Highlights
```bash
# Set your OpenAI API key first:
# Edit 2e_Highlight_Crop_Video.py line 13, or set environment variable:
# set OPENAI_API_KEY=your-key-here

python 2e_Highlight_Crop_Video.py
# Needs: input.mp4 and transcript.json
# Output:
#   - highlights.json (metadata)
#   - highlights/*.mp4 (video clips)
```

### Step 6: Apply Speaker Centering (Optional)
```bash
# Edit 3a_Speaker_Centering.py to point to a highlight clip
# Or use the batch script:
python batch_speaker_center.py
# Output: highlights/*_centered.mp4
```

### Step 7: Apply Karaoke Subtitles (Optional)
```bash
python apply_karaoke.py
# Output: highlights/*_karaoke.mp4
```

## Quick Test (All-in-One)
```bash
python pipeline_orchestrator.py --video input.mp4
# Runs steps 2-5 automatically
```

## File Structure

```
backend/
├── input.mp4                    # Your source video
├── vocals.wav                   # Extracted audio
├── aligned_transcript.json      # Word-level transcript
├── transcript.json              # Formatted for highlight generator
├── highlights.json             # Highlight metadata
├── highlights/                  # Generated clips
│   ├── Hook_highlight_clip_001.mp4
│   ├── Tip_highlight_clip_002.mp4
│   └── ...
├── downloads/                   # YouTube downloads
└── [script files]
```

## Required Files to Create

1. ✅ `extract_audio.py` - Extract audio from video
2. ✅ `convert_transcript.py` - Convert transcript format
3. ✅ `pipeline_orchestrator.py` - Run full pipeline
4. ✅ `batch_speaker_center.py` - Apply centering to all clips
5. ✅ `apply_karaoke.py` - Add karaoke subtitles

## Testing Checklist

- [ ] Step 1: Video downloaded/available
- [ ] Step 2: Audio extracted successfully
- [ ] Step 3: Transcription completed
- [ ] Step 4: Transcript converted
- [ ] Step 5: Highlights generated (needs OpenAI API key)
- [ ] Step 6: Speaker centering applied (optional)
- [ ] Step 7: Karaoke subtitles added (optional)

