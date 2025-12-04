# Installing Chocolatey on Windows

Chocolatey is a package manager for Windows that makes installing software (like FFmpeg) much easier.

## Quick Install (Recommended)

1. **Open PowerShell as Administrator:**
   - Press `Win + X`
   - Select "Windows PowerShell (Admin)" or "Terminal (Admin)"
   - Click "Yes" when prompted by User Account Control

2. **Run the installation command:**
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

3. **Verify installation:**
   ```powershell
   choco --version
   ```
   You should see a version number like `2.2.2`

## After Installing Chocolatey

Once Chocolatey is installed, you can easily install FFmpeg:

```powershell
choco install ffmpeg
```

This will:
- Download FFmpeg
- Install it
- Add it to your PATH automatically
- No manual configuration needed!

## Alternative: Install FFmpeg Directly (Without Chocolatey)

If you don't want to install Chocolatey, you can install FFmpeg manually:

1. **Download FFmpeg:**
   - Go to: https://www.gyan.dev/ffmpeg/builds/
   - Download "ffmpeg-release-essentials.zip"
   - Extract to `C:\ffmpeg`

2. **Add to PATH:**
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", select "Path" → "Edit"
   - Click "New" → Add: `C:\ffmpeg\bin`
   - Click "OK" on all dialogs

3. **Restart your terminal/IDE**

## Verify FFmpeg Installation

After installing (either method), verify it works:

```powershell
ffmpeg -version
```

You should see FFmpeg version information.

## Troubleshooting

**If you get "Execution Policy" errors:**
- Make sure you're running PowerShell as Administrator
- Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

**If Chocolatey install fails:**
- Make sure you're using PowerShell (not Command Prompt)
- Check your internet connection
- Try running the command again

**If FFmpeg is not found after installation:**
- Close and reopen your terminal/IDE
- Restart your computer if needed
- Verify PATH: `echo $env:PATH` (should include ffmpeg path)

