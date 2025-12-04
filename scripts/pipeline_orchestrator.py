#!/usr/bin/env python3
"""
Full pipeline orchestrator - runs steps 2-5 automatically.
"""
import os
import sys
import subprocess
from extract_audio import extract_audio
from convert_transcript import convert_transcript

def run_pipeline(video_path: str, skip_transcription: bool = False):
    """
    Run the full highlight generation pipeline.
    
    Args:
        video_path: Path to input video file
        skip_transcription: If True, skip transcription (use existing files)
    """
    print("=" * 60)
    print("🎬 Video Highlight Pipeline")
    print("=" * 60)
    
    # Step 1: Extract audio
    print("\n[Step 1/4] Extracting audio...")
    audio_path = extract_audio(video_path, "vocals.wav")
    
    # Step 2: Transcribe
    if not skip_transcription:
        print("\n[Step 2/4] Transcribing audio (this may take a while)...")
        try:
            subprocess.run([sys.executable, "vad_fast_transcribe3.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Transcription failed. Make sure vocals.wav exists.")
            return False
    else:
        print("\n[Step 2/4] Skipping transcription (using existing files)")
    
    if not os.path.exists("aligned_transcript.json"):
        print("❌ aligned_transcript.json not found. Run transcription first.")
        return False
    
    # Step 3: Convert transcript
    print("\n[Step 3/4] Converting transcript format...")
    convert_transcript("aligned_transcript.json", "transcript.json")
    
    # Step 4: Generate highlights
    print("\n[Step 4/4] Generating highlights with ChatGPT...")
    print("⚠️  Make sure OPENAI_API_KEY is set in 2e_Highlight_Crop_Video.py")
    
    # Update input video path in the script
    # We'll need to modify the script or pass it as argument
    # For now, user needs to set input.mp4 manually or we copy it
    if video_path != "input.mp4":
        import shutil
        if os.path.exists("input.mp4"):
            print("⚠️  input.mp4 already exists, skipping copy")
        else:
            shutil.copy(video_path, "input.mp4")
            print(f"📋 Copied {video_path} → input.mp4")
    
    try:
        subprocess.run([sys.executable, "2e_Highlight_Crop_Video.py"], check=True)
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Check highlights/ folder for generated clips")
        print(f"📄 Check highlights.json for metadata")
        return True
    except subprocess.CalledProcessError:
        print("❌ Highlight generation failed. Check OpenAI API key and transcript.json")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full video highlight pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--skip-transcription", action="store_true", 
                       help="Skip transcription step (use existing aligned_transcript.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    success = run_pipeline(args.video, args.skip_transcription)
    sys.exit(0 if success else 1)

