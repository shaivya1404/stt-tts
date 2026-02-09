# Install FFmpeg (Required for Audio Processing)

FFmpeg is needed to convert and process audio files.

## Option 1: Using Chocolatey (Recommended)

Open PowerShell as Administrator and run:
```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg -y
```

## Option 2: Using winget (Windows 11)

```powershell
winget install FFmpeg
```

## Option 3: Manual Download

1. Go to: https://github.com/BtbN/FFmpeg-Builds/releases
2. Download: `ffmpeg-master-latest-win64-gpl.zip`
3. Extract to: `C:\ffmpeg`
4. Add to PATH:
   - Press Win + R, type `sysdm.cpl`
   - Advanced → Environment Variables
   - Under System variables, find `Path`
   - Click Edit → New → `C:\ffmpeg\bin`
   - Click OK

## Verify Installation

Open new terminal and run:
```bash
ffmpeg -version
```

You should see version info.

## After Installing FFmpeg

Come back and run:
```bash
cd C:\Users\DELL\Desktop\wxwqxqwxw\tts-stt
python ml-service/data/batch_extract.py --start
```
