@echo off
echo ==========================================
echo TTS-STT Data Collection Setup
echo ==========================================
echo.

echo [1/4] Installing yt-dlp (YouTube downloader)...
pip install yt-dlp

echo.
echo [2/4] Installing audio processing libraries...
pip install soundfile librosa numpy torch torchaudio

echo.
echo [3/4] Installing optional tools...
pip install loguru datasets huggingface_hub

echo.
echo [4/4] Creating data directories...
mkdir data\tts\reference 2>nul
mkdir data\stt\youtube 2>nul
mkdir data\stt\movies 2>nul
mkdir data\stt\public 2>nul

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. For TTS: Record 30 seconds of voice
echo    Save to: data\tts\reference\reference_voice.wav
echo    Read guide: ml-service\data\TTS_VOICE_RECORDING_GUIDE.md
echo.
echo 2. For STT: Extract from YouTube
echo    python ml-service\data\youtube_extractor.py --url "YOUTUBE_URL" --output data\stt\youtube --lang hi
echo.
echo ==========================================
pause
