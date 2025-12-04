#!/usr/bin/env python3
"""
Apply speaker centering to all highlight clips in the highlights/ folder.
"""
import os
import subprocess
import sys
import cv2
import mediapipe as mp
import numpy as np

# Import settings from 3a_Speaker_Centering.py
alpha = 0.07
headroom_ratio = 0.15
max_hold_frames = 150
pose_fallback_delay = 15

def center_speaker(input_video: str, output_video: str):
    """Apply speaker centering to a single video."""
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"⚠️  Could not open {input_video}, skipping...")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    crop_w = int(H * 9 / 16)
    crop_h = H
    
    temp_video = input_video.replace(".mp4", "_temp_crop.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (crop_w, crop_h))
    
    def clamp(v, a, b): return max(a, min(b, v))
    
    face_x, face_y = W / 2, H / 2
    lost_frames = 0
    face_missing_frames = 0
    
    with mp_face.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as facemesh, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_det:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            results_face = facemesh.process(rgb)
            cx, cy = None, None
            
            if results_face.multi_face_landmarks:
                lm = results_face.multi_face_landmarks[0].landmark
                lx, ly = lm[33].x * w, lm[33].y * h
                rx, ry = lm[263].x * w, lm[263].y * h
                cx, cy = (lx + rx) / 2, (ly + ry) / 2
                face_missing_frames = 0
            else:
                face_missing_frames += 1
            
            # Pose fallback
            if cx is None and face_missing_frames > pose_fallback_delay:
                results_pose = pose_det.process(rgb)
                if results_pose.pose_landmarks:
                    lm = results_pose.pose_landmarks.landmark
                    L = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    R = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    cx = (L.x + R.x) / 2 * w
                    cy = (L.y + R.y) / 2 * h - 0.25 * h
                    lost_frames = 0
                else:
                    lost_frames += 1
            
            # Smooth center
            if cx is not None and cy is not None:
                face_x = (1 - alpha) * face_x + alpha * cx
                face_y = (1 - alpha) * face_y + alpha * cy
            elif lost_frames > max_hold_frames:
                face_x = (1 - alpha) * face_x + alpha * (w / 2)
                face_y = (1 - alpha) * face_y + alpha * (h / 2)
            
            # Apply headroom and crop
            target_y = face_y - headroom_ratio * crop_h
            x1 = clamp(int(face_x - crop_w / 2), 0, w - crop_w)
            y1 = clamp(int(target_y - crop_h / 2), 0, h - crop_h)
            
            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            out.write(cropped)
    
    cap.release()
    out.release()
    
    # Merge audio
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_video
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_video)
    return True

if __name__ == "__main__":
    highlights_dir = "highlights"
    if not os.path.exists(highlights_dir):
        print(f"❌ {highlights_dir} directory not found. Run highlight generator first.")
        sys.exit(1)
    
    video_files = [f for f in os.listdir(highlights_dir) if f.endswith(".mp4") and not f.endswith("_centered.mp4")]
    
    if not video_files:
        print(f"❌ No video files found in {highlights_dir}")
        sys.exit(1)
    
    print(f"🎬 Processing {len(video_files)} highlight clips...")
    
    for video_file in video_files:
        input_path = os.path.join(highlights_dir, video_file)
        output_path = os.path.join(highlights_dir, video_file.replace(".mp4", "_centered.mp4"))
        
        print(f"  Processing: {video_file}...")
        if center_speaker(input_path, output_path):
            print(f"  ✅ Saved: {output_path}")
        else:
            print(f"  ⚠️  Failed: {video_file}")
    
    print("\n✅ Speaker centering complete!")

