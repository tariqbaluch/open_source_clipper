# Installing FFmpeg on Windows

FFmpeg is required for video/audio processing. Here's how to install it:

## Option 1: Using Chocolatey (Recommended)

If you have Chocolatey installed:
```powershell
choco install ffmpeg
```

## Option 2: Manual Installation

1. **Download FFmpeg:**
   - Go to https://www.gyan.dev/ffmpeg/builds/
   - Download "ffmpeg-release-essentials.zip"
   - Or direct link: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

2. **Extract:**
   - Extract the zip file to a location like `C:\ffmpeg`

3. **Add to PATH:**
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click "OK" on all dialogs

4. **Verify:**
   - Open a new PowerShell/Command Prompt
   - Run: `ffmpeg -version`
   - You should see version information

## Option 3: Using winget (Windows 10/11)

```powershell
winget install ffmpeg
```

## After Installation

Restart your terminal/IDE and run:
```bash
python test_pipeline.py
```

FFmpeg should now be detected.

