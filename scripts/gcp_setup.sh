#!/bin/bash
# GCP VM Setup Script for STT Training
# Run this after SSH into your GCP VM

set -e  # Exit on error

echo "=========================================="
echo "  GCP VM Setup for STT Training"
echo "=========================================="

# Check if GPU is available
echo ""
echo "[1/7] Checking GPU..."
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ No GPU detected. Please ensure you created a VM with GPU."
    echo "   Or wait a moment and run: nvidia-smi"
    exit 1
fi

# Update system
echo ""
echo "[2/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo ""
echo "[3/7] Installing system dependencies..."
sudo apt install -y ffmpeg git python3-pip python3-venv screen htop

# Create project directory
echo ""
echo "[4/7] Setting up project directory..."
mkdir -p ~/tts-stt
cd ~/tts-stt

# Create virtual environment
echo ""
echo "[5/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo ""
echo "[6/7] Installing Python packages (this may take 5-10 minutes)..."
pip install --upgrade pip

# PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# ML packages
pip install transformers datasets accelerate peft
pip install librosa soundfile jiwer tensorboard
pip install openai-whisper
pip install numpy scipy

# Verify installation
echo ""
echo "[7/7] Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your data from Windows PC:"
echo "   gcloud compute scp --recurse \"C:\\path\\to\\data\" stt-training-vm:~/tts-stt/ --zone=us-central1-a"
echo ""
echo "2. Or download from Cloud Storage:"
echo "   gsutil -m cp -r gs://YOUR_BUCKET/data ~/tts-stt/"
echo ""
echo "3. Activate environment (do this every time you reconnect):"
echo "   cd ~/tts-stt && source venv/bin/activate"
echo ""
echo "4. Run transcription:"
echo "   python ml-service/data/transcribe_segments.py --manifest data/stt/combined/hi_train.jsonl --lang hi --model medium"
echo ""
echo "5. Run training:"
echo "   python ml-service/training/stt/train_whisper.py --config ml-service/training/stt/configs/whisper_finetune_indic_v1.yaml"
echo ""
echo "TIP: Use 'screen -S training' before long commands so they keep running if SSH disconnects"
echo ""
