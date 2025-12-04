#!/usr/bin/env python3
"""
Convert aligned_transcript.json (from vad_fast_transcribe3.py) 
to transcript.json format (needed by 2e_Highlight_Crop_Video.py).

Input format: [{"start": float, "end": float, "text": str, "words": [...]}]
Output format: [{"start": float, "end": float, "text": str}]
"""
import json
import sys
import os

def convert_transcript(input_path: str, output_path: str = "transcript.json"):
    """
    Convert transcript from word-level format to segment-level format.
    
    Args:
        input_path: Path to aligned_transcript.json
        output_path: Path to output transcript.json
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Converting transcript: {input_path} -> {output_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        aligned = json.load(f)
    
    # Convert to simple format
    transcript = []
    for seg in aligned:
        transcript.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip()
        })
    
    # Sort by start time (should already be sorted, but just in case)
    transcript.sort(key=lambda x: x["start"])
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(transcript)} segments")
    print(f"   Duration: {transcript[-1]['end'] - transcript[0]['start']:.1f} seconds")
    return transcript

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_transcript.py <aligned_transcript.json> [output.json]")
        print("Example: python convert_transcript.py aligned_transcript.json transcript.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "transcript.json"
    
    convert_transcript(input_path, output_path)

