@echo off
echo ==========================================
echo STT Data Extraction Tool
echo ==========================================
echo.
echo Current data:
type data\stt\extracted\en_test\manifest_transcribed.jsonl 2>nul | find /c /v "" || echo   English: 0 segments
type data\stt\extracted\hi_test\manifest_transcribed.jsonl 2>nul | find /c /v "" || echo   Hindi: 0 segments
echo.
echo ==========================================
echo Options:
echo   1. Extract from YouTube URL (English)
echo   2. Extract from YouTube URL (Hindi)
echo   3. Extract from local video file
echo   4. View current data status
echo   5. Exit
echo ==========================================
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    set /p url="Enter YouTube URL: "
    echo Downloading and extracting English audio...
    python ml-service/data/simple_extract.py --url "%url%" --output data/stt/extracted/english --lang en
    echo.
    echo Transcribing segments...
    python ml-service/data/transcribe_segments.py --manifest data/stt/extracted/english/manifest.jsonl --lang en
    echo Done!
    pause
)

if "%choice%"=="2" (
    set /p url="Enter YouTube URL: "
    echo Downloading and extracting Hindi audio...
    python ml-service/data/simple_extract.py --url "%url%" --output data/stt/extracted/hindi --lang hi
    echo.
    echo Transcribing segments...
    python ml-service/data/transcribe_segments.py --manifest data/stt/extracted/hindi/manifest.jsonl --lang hi
    echo Done!
    pause
)

if "%choice%"=="3" (
    set /p filepath="Enter file path: "
    set /p lang="Enter language (en/hi): "
    echo Extracting audio...
    python ml-service/data/simple_extract.py --file "%filepath%" --output data/stt/extracted/%lang%_local --lang %lang%
    echo.
    echo Transcribing segments...
    python ml-service/data/transcribe_segments.py --manifest data/stt/extracted/%lang%_local/manifest.jsonl --lang %lang%
    echo Done!
    pause
)

if "%choice%"=="4" (
    echo.
    echo === English Data ===
    for /d %%d in (data\stt\extracted\en*) do (
        if exist "%%d\manifest_transcribed.jsonl" (
            echo %%d:
            type "%%d\manifest_transcribed.jsonl" | find /c /v ""
        )
    )
    echo.
    echo === Hindi Data ===
    for /d %%d in (data\stt\extracted\hi*) do (
        if exist "%%d\manifest_transcribed.jsonl" (
            echo %%d:
            type "%%d\manifest_transcribed.jsonl" | find /c /v ""
        )
    )
    pause
)

if "%choice%"=="5" exit
