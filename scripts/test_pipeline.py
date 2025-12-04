#!/usr/bin/env python3
"""
Quick test script to verify pipeline setup.
Checks if all required files and dependencies are available.
"""
import os
import sys
import subprocess

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[X] {description}: {filepath} (NOT FOUND)")
        return False

def check_command(cmd, description):
    """Check if a command is available."""
    try:
        candidates = [["--version"], ["-version"], ["-V"]]
        for args in candidates:
            try:
                subprocess.run([cmd, *args],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               check=True)
                print(f"[OK] {description}: {cmd}")
                return True
            except subprocess.CalledProcessError:
                continue
        print(f"[X] {description}: {cmd} (NOT FOUND)")
        return False
    except FileNotFoundError:
        print(f"[X] {description}: {cmd} (NOT FOUND)")
        return False

def check_python_module(module, description):
    """Check if a Python module can be imported."""
    try:
        __import__(module)
        print(f"[OK] {description}: {module}")
        return True
    except ImportError:
        print(f"[X] {description}: {module} (NOT INSTALLED)")
        return False

def main():
    print("=" * 60)
    print("Pipeline Setup Check")
    print("=" * 60)
    
    all_ok = True
    
    # Check commands
    print("\n[Commands]")
    all_ok &= check_command("ffmpeg", "FFmpeg")
    all_ok &= check_command("python", "Python")
    
    # Check Python modules
    print("\n[Python Modules]")
    all_ok &= check_python_module("openai", "OpenAI SDK")
    all_ok &= check_python_module("faster_whisper", "Faster Whisper")
    all_ok &= check_python_module("cv2", "OpenCV")
    all_ok &= check_python_module("mediapipe", "MediaPipe")
    all_ok &= check_python_module("librosa", "Librosa")
    all_ok &= check_python_module("scenedetect", "SceneDetect")
    all_ok &= check_python_module("yt_dlp", "yt-dlp")
    
    # Check script files
    print("\n[Script Files]")
    scripts = [
        ("youtube_downloader.py", "YouTube Downloader"),
        ("extract_audio.py", "Audio Extractor"),
        ("vad_fast_transcribe3.py", "Transcriber"),
        ("convert_transcript.py", "Transcript Converter"),
        ("2e_Highlight_Crop_Video.py", "Highlight Generator"),
        ("3a_Speaker_Centering.py", "Speaker Centering"),
        ("analyze_audio6.py", "Audio Analyzer"),
    ]
    
    for script, desc in scripts:
        all_ok &= check_file(script, desc)
    
    # Check optional files
    print("\n[Optional Files]")
    check_file("input.mp4", "Test Video (optional)")
    check_file("vocals.wav", "Test Audio (optional)")
    check_file("transcript.json", "Test Transcript (optional)")
    
    # Check API key
    print("\n[Configuration]")
    api_key_set = False
    try:
        with open("2e_Highlight_Crop_Video.py", "r", encoding="utf-8") as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content:
                # Check if it's not empty
                import re
                # Match: os.environ["OPENAI_API_KEY"] = "value" or os.environ['OPENAI_API_KEY'] = 'value'
                match = re.search(r'OPENAI_API_KEY["\']\s*=\s*["\']([^"\']+)', content)
                if match and match.group(1) and match.group(1).strip():
                    print("[OK] OpenAI API Key: Set in script")
                    api_key_set = True
                else:
                    # Also check environment variable
                    import os
                    if os.environ.get("OPENAI_API_KEY"):
                        print("[OK] OpenAI API Key: Set in environment")
                        api_key_set = True
                    else:
                        print("[!] OpenAI API Key: Not set")
                        print("   Set it in 2e_Highlight_Crop_Video.py line 13, or:")
                        print("   set OPENAI_API_KEY=your-key-here")
    except Exception as e:
        print(f"[!] Could not check OpenAI API Key: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] All required components are available!")
        print("\nNext steps:")
        print("1. Install FFmpeg (see INSTALL_FFMPEG.md)")
        print("2. Set OPENAI_API_KEY in 2e_Highlight_Crop_Video.py")
        print("3. Run: python pipeline_orchestrator.py <your_video.mp4>")
    else:
        print("[X] Some components are missing.")
        print("\nTo fix:")
        print("- Install FFmpeg: See INSTALL_FFMPEG.md")
        print("- Install Python packages: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()

