# GCP GPU Training Guide - Complete Beginner's Guide

> **Your Budget**: $300 free credit
> **Goal**: Train STT model on GPU
> **Time to Setup**: ~30 minutes

---

## Cost Estimate (Important!)

| GPU Type | Cost/Hour | 10 Hours Training | Recommendation |
|----------|-----------|-------------------|----------------|
| T4 | ~$0.35/hr | ~$3.50 | Budget option |
| V100 | ~$2.50/hr | ~$25 | ✅ Your choice - 2-3x faster |
| A100 | ~$4.00/hr | ~$40 | Overkill for this project |

**With $300 credit, you can train for ~120 hours on V100** - plenty for this project!

---

## Part 1: Initial GCP Setup (One-time)

### Step 1.1: Create GCP Account

1. Go to: https://cloud.google.com/
2. Click **"Get started for free"** or **"Start free"**
3. Sign in with your Google account
4. Enter billing info (won't be charged - just verification)
5. You'll get **$300 free credit** valid for 90 days

### Step 1.2: Create a New Project

1. Go to: https://console.cloud.google.com/
2. Click the project dropdown (top-left, next to "Google Cloud")
3. Click **"New Project"**
4. Enter:
   - Project name: `tts-stt-training`
   - Click **"Create"**
5. Make sure this project is selected

### Step 1.3: Enable Required APIs

1. Go to: https://console.cloud.google.com/apis/library
2. Search and enable these APIs (click each → click "Enable"):
   - **Compute Engine API**
   - **Cloud Storage API** (usually enabled by default)

### Step 1.4: Request GPU Quota (IMPORTANT!)

> ⚠️ **New accounts have 0 GPU quota by default. You MUST request quota increase.**

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. In the **Filter** box, type: `GPUs (all regions)`
3. Find the row that says **"GPUs (all regions)"** with value `0`
4. Check the checkbox next to it
5. Click **"Edit Quotas"** (top of page)
6. Enter:
   - New limit: `1`
   - Request description: "For machine learning training project"
7. Click **"Submit Request"**

> **Wait Time**: Usually approved in 1-15 minutes. Check email for confirmation.

**Alternative - Specific GPU Quota (for V100):**
1. Filter by: `NVIDIA V100 GPUs`
2. Select your preferred region (e.g., `us-central1` or `us-west1`)
3. Request limit: `1`

> **Note**: V100 is available in fewer regions. Best regions: `us-central1`, `us-west1`, `europe-west4`

---

## Part 2: Create GPU Virtual Machine

### Step 2.1: Go to VM Creation Page

1. Go to: https://console.cloud.google.com/compute/instances
2. Click **"Create Instance"**

### Step 2.2: Configure the VM

**Basic Settings:**
```
Name: stt-training-vm
Region: us-central1 (or any with T4 available)
Zone: us-central1-a
```

**Machine Configuration:**
1. Click **"GPU"** tab (under Machine configuration)
2. Click **"Add GPU"**
3. Select:
   - GPU type: **NVIDIA V100**
   - Number of GPUs: **1**
4. Machine type: **n1-standard-8** (8 vCPU, 30 GB memory)

**Boot Disk (IMPORTANT):**
1. Click **"Change"** under Boot disk
2. Select:
   - Operating system: **Deep Learning on Linux**
   - Version: **Deep Learning VM with CUDA 11.8 M1xx** (or latest)
   - Size: **100 GB** (need space for data)
3. Click **"Select"**

**Firewall:**
- ✅ Check "Allow HTTP traffic"
- ✅ Check "Allow HTTPS traffic"

### Step 2.3: Create the VM

1. Review the cost estimate on the right (~$2.50/hour for V100)
2. Click **"Create"**
3. Wait 2-3 minutes for VM to start

---

## Part 3: Connect to Your VM

### Step 3.1: SSH into VM

1. Go to: https://console.cloud.google.com/compute/instances
2. Find your VM `stt-training-vm`
3. Click **"SSH"** button (opens browser terminal)

### Step 3.2: First-time Setup (Run these commands)

When you first connect, it may ask about NVIDIA drivers. Type `y` to install.

```bash
# Check GPU is working
nvidia-smi
```

You should see your T4 GPU listed. If not, wait a minute and try again.

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y ffmpeg git python3-pip python3-venv

# Create project directory
mkdir -p ~/tts-stt
cd ~/tts-stt

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install transformers datasets accelerate peft
pip install librosa soundfile jiwer tensorboard
pip install openai-whisper
```

---

## Part 4: Upload Your Data to GCP

### Option A: Using Google Cloud Storage (Recommended)

**On your Windows PC:**

1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
2. Open Command Prompt and run:

```cmd
# Login to GCP
gcloud auth login

# Set your project
gcloud config set project tts-stt-training

# Create a storage bucket
gsutil mb gs://your-unique-bucket-name-stt

# Upload your data folder
cd C:\Users\DELL\Desktop\wxwqxqwxw\tts-stt
gsutil -m cp -r data gs://your-unique-bucket-name-stt/
gsutil -m cp -r ml-service gs://your-unique-bucket-name-stt/
```

**On your GCP VM:**

```bash
cd ~/tts-stt

# Download from bucket
gsutil -m cp -r gs://your-unique-bucket-name-stt/data .
gsutil -m cp -r gs://your-unique-bucket-name-stt/ml-service .
```

### Option B: Using SCP (Direct Upload)

**On your Windows PC (PowerShell):**

```powershell
# First, get gcloud CLI and authenticate
gcloud compute scp --recurse "C:\Users\DELL\Desktop\wxwqxqwxw\tts-stt\data" stt-training-vm:~/tts-stt/ --zone=us-central1-a
gcloud compute scp --recurse "C:\Users\DELL\Desktop\wxwqxqwxw\tts-stt\ml-service" stt-training-vm:~/tts-stt/ --zone=us-central1-a
```

### Option C: Using Git (If code is on GitHub)

If you push your code to GitHub:

```bash
# On GCP VM
cd ~/tts-stt
git clone https://github.com/YOUR_USERNAME/tts-stt.git .
```

---

## Part 5: Transcribe Your Audio (First Step)

Before training, you need to transcribe your audio files.

**On your GCP VM:**

```bash
cd ~/tts-stt
source venv/bin/activate

# Transcribe Hindi data
python ml-service/data/transcribe_segments.py \
    --manifest data/stt/combined/hi_train.jsonl \
    --lang hi \
    --model medium

# Transcribe English data
python ml-service/data/transcribe_segments.py \
    --manifest data/stt/combined/en_train.jsonl \
    --lang en \
    --model medium
```

This will take ~30-60 minutes on T4 GPU.

---

## Part 6: Train the STT Model

### Step 6.1: Check Training Config

```bash
# View the training config
cat ml-service/training/stt/configs/whisper_finetune_indic_v1.yaml
```

### Step 6.2: Start Training

```bash
cd ~/tts-stt
source venv/bin/activate

# Start training
python ml-service/training/stt/train_whisper.py \
    --config ml-service/training/stt/configs/whisper_finetune_indic_v1.yaml
```

Training will take several hours. You can:
- Watch progress in terminal
- Check GPU usage: `nvidia-smi` (in another SSH window)
- Monitor with TensorBoard (see below)

### Step 6.3: Monitor with TensorBoard (Optional)

```bash
# In another SSH terminal
cd ~/tts-stt
source venv/bin/activate
tensorboard --logdir outputs/logs --port 6006
```

To view TensorBoard in browser:
1. In GCP Console, go to your VM
2. Click the "SSH" dropdown → "View gcloud command"
3. Add port forwarding: `-- -L 6006:localhost:6006`
4. Open http://localhost:6006 in your browser

---

## Part 7: Download Trained Model

After training completes:

**On GCP VM:**
```bash
# Upload model to Cloud Storage
gsutil -m cp -r outputs/checkpoints gs://your-unique-bucket-name-stt/trained_model/
```

**On your Windows PC:**
```cmd
# Download model
gsutil -m cp -r gs://your-unique-bucket-name-stt/trained_model C:\Users\DELL\Desktop\wxwqxqwxw\tts-stt\
```

---

## Part 8: STOP YOUR VM (Save Money!)

> ⚠️ **VERY IMPORTANT**: VMs cost money even when idle. ALWAYS stop when not using!

### Stop VM (Keeps data, stops billing for compute)

```bash
# From your Windows PC
gcloud compute instances stop stt-training-vm --zone=us-central1-a
```

Or in GCP Console:
1. Go to: https://console.cloud.google.com/compute/instances
2. Check your VM
3. Click **"Stop"** (square icon)

### Start VM again later

```bash
gcloud compute instances start stt-training-vm --zone=us-central1-a
```

### Delete VM (When completely done)

```bash
gcloud compute instances delete stt-training-vm --zone=us-central1-a
```

---

## Quick Reference Commands

### On Windows PC (Command Prompt/PowerShell)

```cmd
# Login to GCP
gcloud auth login

# Set project
gcloud config set project tts-stt-training

# SSH to VM
gcloud compute ssh stt-training-vm --zone=us-central1-a

# Upload files
gcloud compute scp --recurse LOCAL_PATH stt-training-vm:~/tts-stt/ --zone=us-central1-a

# Download files
gcloud compute scp --recurse stt-training-vm:~/tts-stt/outputs LOCAL_PATH --zone=us-central1-a

# Stop VM (SAVE MONEY!)
gcloud compute instances stop stt-training-vm --zone=us-central1-a

# Start VM
gcloud compute instances start stt-training-vm --zone=us-central1-a
```

### On GCP VM

```bash
# Activate environment
cd ~/tts-stt && source venv/bin/activate

# Check GPU
nvidia-smi

# Run transcription
python ml-service/data/transcribe_segments.py --manifest data/stt/combined/hi_train.jsonl --lang hi --model medium

# Run training
python ml-service/training/stt/train_whisper.py --config ml-service/training/stt/configs/whisper_finetune_indic_v1.yaml
```

---

## Troubleshooting

### "Quota exceeded" error
- Your GPU quota request hasn't been approved yet
- Wait for email confirmation (usually 1-15 minutes)
- Or try a different region

### "CUDA out of memory" error
- Reduce batch size in training config
- Edit `whisper_finetune_indic_v1.yaml`: change `batch_size: 8` to `batch_size: 4`

### VM won't start
- Check if you have GPU quota in that region
- Try a different zone (us-central1-b, us-central1-c)

### Can't find GPU
- Run `nvidia-smi` - if it fails, drivers aren't installed
- For Deep Learning VMs, type `y` when prompted to install drivers on first login

### SSH connection drops
- Use `screen` or `tmux` to keep processes running:
```bash
# Start screen session
screen -S training

# Run your commands...
# Press Ctrl+A, then D to detach

# Reconnect later
screen -r training
```

---

## Cost Saving Tips

1. **Always stop VM when not using** - #1 money saver! (V100 = $2.50/hr idle!)
2. **Use preemptible/spot VMs** - 60-80% cheaper (~$0.75/hr for V100!)
3. **Delete unused disks** - Storage also costs money
4. **Set budget alerts** - Get email when spending reaches threshold
5. **Your V100 budget**: ~120 hours with $300 (plenty for this project)

### Set Budget Alert

1. Go to: https://console.cloud.google.com/billing
2. Click "Budgets & alerts"
3. Create budget: $50 (you'll get email at 50%, 90%, 100%)

---

## Summary Checklist

- [ ] Create GCP account with $300 credit
- [ ] Create project `tts-stt-training`
- [ ] Enable Compute Engine API
- [ ] Request GPU quota (wait for approval)
- [ ] Create VM with T4 GPU
- [ ] Install dependencies
- [ ] Upload data
- [ ] Run transcription
- [ ] Run training
- [ ] Download trained model
- [ ] **STOP VM!**

---

## Need Help?

If stuck, tell me:
1. What step you're on
2. The exact error message
3. I'll help you fix it!
