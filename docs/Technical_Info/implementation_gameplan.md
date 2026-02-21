# Implementation Game Plan: Ad Detection from Videos

## Overview

This document provides a comprehensive, actionable game plan for an ML engineer to implement the advertisement detection framework from scratch. The implementation is divided into phases, with clear dependencies and deliverables at each step.

---

## Prerequisites and Setup

### 1. Hardware Requirements

**Minimum Requirements**:
- GPU: NVIDIA GPU with ≥12GB VRAM (e.g., RTX 3090, V100, A100)
- CPU: Multi-core processor (Intel Xeon or AMD EPYC recommended)
- RAM: ≥32GB
- Storage: ≥500GB SSD (for datasets and models)

**Recommended Setup**:
- GPU: NVIDIA A100 (40GB) or multiple RTX 3090s
- CPU: 16+ cores
- RAM: 64GB+
- Storage: 1TB+ NVMe SSD

### 2. Software Requirements

**Operating System**:
- Ubuntu 18.04+ or equivalent Linux distribution
- Docker (optional but recommended for reproducibility)

**Python Environment**:
```bash
# Python 3.8+
conda create -n ad-detection python=3.8
conda activate ad-detection
```

**Core Dependencies**:
```bash
# PyTorch (adjust CUDA version as needed)
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Audio processing
pip install librosa==0.9.2
pip install soundfile
pip install pydub

# Video processing
pip install opencv-python
pip install av
pip install ffmpeg-python

# Pre-trained models
pip install timm  # PyTorch Image Models
pip install torchvggish  # For audio features

# Scene detection
pip install scenedetect[opencv]

# Utilities
pip install numpy pandas matplotlib seaborn
pip install tqdm
pip install tensorboard
pip install scikit-learn
```

**Additional Dependencies**:
```bash
# Install ffmpeg (for video/audio extraction)
sudo apt-get update
sudo apt-get install ffmpeg

# Install sox (for audio processing)
sudo apt-get install sox libsox-fmt-all
```

### 3. Pre-trained Model Downloads

Create a directory structure:
```bash
mkdir -p ~/ad-detection-project
cd ~/ad-detection-project
mkdir -p {data,models,pretrained,outputs,logs,notebooks}
```

Download pre-trained models:

**VoxCeleb2 ResNet-34** (for CP Classifier):
```bash
# From Oxford Robotics Lab
cd pretrained
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_v2_ap.model
# Or use their GitHub repo: https://github.com/clovaai/voxceleb_trainer
```

**AvSlowFast** (for Ad Classifier):
```bash
# PyTorch Hub or from Facebook Research
# https://github.com/facebookresearch/SlowFast
cd pretrained
# Download model weights (follow repo instructions)
```

**Vision Transformer**:
```bash
# Using timm library (already included)
# Model: jx_vit_base_resnet50_384
# Will auto-download on first use
```

**PANNs** (for Post-processing):
```bash
# From https://github.com/qiuqiangkong/audioset_tagging_cnn
cd pretrained
wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
```

---

## Phase 1: Data Collection and Preparation

### Timeline: 2-3 weeks

### Step 1.1: Video Data Collection

**Objective**: Collect video playbacks from streaming services

**Implementation**:

1. **Screen Recording Setup**:
```python
# Use OBS Studio or ffmpeg for screen capture
# Example ffmpeg command:
import subprocess

def record_screen(output_path, duration_seconds):
    """
    Record screen for specified duration
    """
    cmd = [
        'ffmpeg',
        '-f', 'x11grab',  # For Linux; use 'avfoundation' for Mac, 'gdigrab' for Windows
        '-r', '60',  # 60 fps
        '-s', '1920x1080',  # Resolution
        '-i', ':0.0',  # Display number
        '-t', str(duration_seconds),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '18',
        output_path
    ]
    subprocess.run(cmd)

# Usage
record_screen('data/raw_videos/movie_001.mp4', 7200)  # Record 2 hours
```

2. **Data Organization**:
```
data/
├── raw_videos/
│   ├── movies/
│   │   ├── movie_001.mp4
│   │   ├── movie_002.mp4
│   │   └── ...
│   ├── sports/
│   │   ├── golf_001.mp4
│   │   └── ...
│   └── tv_shows/
│       └── ...
├── short_clips/  # For training
│   ├── ads/
│   │   ├── ad_001.mp4 (10-30s each)
│   │   └── ...
│   └── content/
│       ├── content_001.mp4
│       └── ...
└── annotations/
    ├── movie_001.json
    └── ...
```

### Step 1.2: Data Annotation

**Objective**: Create ground truth labels

**Annotation Format** (JSON):
```json
{
  "video_id": "movie_001",
  "duration_seconds": 7200,
  "segments": [
    {
      "segment_id": 0,
      "start_time": 0.0,
      "end_time": 234.5,
      "label": "content",
      "type": "opening_scene"
    },
    {
      "segment_id": 1,
      "start_time": 234.5,
      "end_time": 249.5,
      "label": "ad",
      "type": "commercial"
    },
    {
      "segment_id": 2,
      "start_time": 249.5,
      "end_time": 264.2,
      "label": "ad",
      "type": "commercial"
    }
  ],
  "change_points": [234.5, 249.5, 264.2, ...]
}
```

**Annotation Tool**:
```python
# Use VGG Image Annotator (VIA) or Label Studio
# Or create simple custom tool:

import cv2
import json

class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.segments = []

    def annotate(self):
        """
        Interactive annotation tool
        Instructions:
        - Press 'c' to mark change point
        - Press 'a' for ad segment
        - Press 's' for content segment
        - Press 'q' to quit
        """
        current_time = 0
        segment_start = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Display frame with timestamp
            cv2.putText(frame, f"Time: {current_time:.2f}s",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Annotation', frame)

            key = cv2.waitKey(int(1000/self.fps)) & 0xFF

            if key == ord('c'):  # Change point
                segment = {
                    'start_time': segment_start,
                    'end_time': current_time,
                    'label': input("Label (ad/content): ")
                }
                self.segments.append(segment)
                segment_start = current_time

            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return self.segments

    def save_annotations(self, output_path):
        annotation = {
            'video_path': self.video_path,
            'segments': self.segments
        }
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)

# Usage
annotator = VideoAnnotator('data/raw_videos/movie_001.mp4')
segments = annotator.annotate()
annotator.save_annotations('data/annotations/movie_001.json')
```

### Step 1.3: Create Short Clips Dataset

**Objective**: Extract 10-30 second clips for training

```python
import ffmpeg
import json
import os
from pathlib import Path

def extract_clips_from_annotations(video_path, annotation_path, output_dir):
    """
    Extract short clips based on annotations
    """
    with open(annotation_path) as f:
        annotation = json.load(f)

    video_name = Path(video_path).stem

    for i, segment in enumerate(annotation['segments']):
        start_time = segment['start_time']
        end_time = segment['end_time']
        label = segment['label']

        # Split long segments into 10-30s clips
        duration = end_time - start_time
        if duration > 30:
            num_clips = int(duration / 20)  # 20s clips
            for j in range(num_clips):
                clip_start = start_time + j * 20
                clip_end = min(clip_start + 20, end_time)

                output_path = f"{output_dir}/{label}/{video_name}_seg{i}_clip{j}.mp4"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                ffmpeg.input(video_path, ss=clip_start, t=clip_end-clip_start).output(
                    output_path,
                    vcodec='libx264',
                    acodec='aac'
                ).run(quiet=True)
        else:
            # Keep as single clip
            output_path = f"{output_dir}/{label}/{video_name}_seg{i}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            ffmpeg.input(video_path, ss=start_time, t=duration).output(
                output_path,
                vcodec='libx264',
                acodec='aac'
            ).run(quiet=True)

# Process all annotated videos
annotation_dir = Path('data/annotations')
for annotation_file in annotation_dir.glob('*.json'):
    video_name = annotation_file.stem
    video_path = f"data/raw_videos/movies/{video_name}.mp4"
    extract_clips_from_annotations(
        video_path,
        annotation_file,
        'data/short_clips'
    )
```

### Step 1.4: Dataset Splitting

```python
import random
from sklearn.model_selection import train_test_split

def create_dataset_split(clips_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test
    """
    # Get all clips
    ad_clips = list(Path(clips_dir).glob('ads/*.mp4'))
    content_clips = list(Path(clips_dir).glob('content/*.mp4'))

    # Shuffle
    random.shuffle(ad_clips)
    random.shuffle(content_clips)

    # Split ads
    train_ads, temp_ads = train_test_split(ad_clips, train_size=train_ratio, random_state=42)
    val_ads, test_ads = train_test_split(temp_ads, train_size=val_ratio/(1-train_ratio), random_state=42)

    # Split content
    train_content, temp_content = train_test_split(content_clips, train_size=train_ratio, random_state=42)
    val_content, test_content = train_test_split(temp_content, train_size=val_ratio/(1-train_ratio), random_state=42)

    # Save splits
    splits = {
        'train': {'ad': [str(p) for p in train_ads], 'content': [str(p) for p in train_content]},
        'val': {'ad': [str(p) for p in val_ads], 'content': [str(p) for p in val_content]},
        'test': {'ad': [str(p) for p in test_ads], 'content': [str(p) for p in test_content]}
    }

    with open(f"{output_dir}/dataset_splits.json", 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Train: {len(train_ads)} ads, {len(train_content)} content")
    print(f"Val: {len(val_ads)} ads, {len(val_content)} content")
    print(f"Test: {len(test_ads)} ads, {len(test_content)} content")

    return splits

splits = create_dataset_split('data/short_clips', 'data')
```

**Deliverable**:
- 12,000+ short clips (10-30s each) split into train/val/test
- 48+ long videos (20-120 min) with annotations for end-to-end testing

---

## Phase 2: Implement CP Classifier

### Timeline: 2 weeks

### Step 2.1: Audio Preprocessing Module

```python
# File: src/audio_processing.py

import librosa
import numpy as np
import torch
from scipy.signal import butter, sosfilt

class AudioPreprocessor:
    def __init__(self, sr=16000, n_mels=64):
        self.sr = sr
        self.n_mels = n_mels

    def load_audio(self, video_path, start_time=None, duration=None):
        """
        Extract audio from video
        """
        y, sr = librosa.load(video_path, sr=self.sr, offset=start_time, duration=duration)
        return y

    def bandpass_filter(self, audio, lowcut=300, highcut=6000):
        """
        Apply bandpass filter [300Hz, 6000Hz]
        """
        sos = butter(10, [lowcut, highcut], btype='band', fs=self.sr, output='sos')
        filtered = sosfilt(sos, audio)
        return filtered

    def detect_silent_segments(self, audio, min_duration_ms=10, volume_threshold=4):
        """
        Detect silent segments in audio

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Normalize to int16
        audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)

        # Get absolute values
        abs_audio = np.abs(audio_int16)

        # Find silent frames
        min_samples = int(min_duration_ms * self.sr / 1000)
        silent_mask = abs_audio < volume_threshold

        # Find contiguous silent regions
        silent_segments = []
        start = None

        for i, is_silent in enumerate(silent_mask):
            if is_silent and start is None:
                start = i
            elif not is_silent and start is not None:
                if i - start >= min_samples:
                    silent_segments.append((start / self.sr, i / self.sr))
                start = None

        return silent_segments

    def extract_log_mel_spectrogram(self, audio):
        """
        Compute Log Mel Spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=512,
            hop_length=160
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def extract_audio_clips_around_silence(self, audio, silence_time, window_duration=2.0):
        """
        Extract audio clips before and after silent segment

        Args:
            audio: Full audio array
            silence_time: Timestamp of silent segment (seconds)
            window_duration: Duration of clips to extract (seconds)

        Returns:
            wav1, wav2: Audio clips before and after
        """
        window_samples = int(window_duration * self.sr)
        silence_sample = int(silence_time * self.sr)

        # Before silence
        start1 = max(0, silence_sample - window_samples)
        end1 = silence_sample
        wav1 = audio[start1:end1]

        # After silence
        start2 = silence_sample
        end2 = min(len(audio), silence_sample + window_samples)
        wav2 = audio[start2:end2]

        # Pad if necessary
        if len(wav1) < window_samples:
            wav1 = np.pad(wav1, (window_samples - len(wav1), 0), mode='constant')
        if len(wav2) < window_samples:
            wav2 = np.pad(wav2, (0, window_samples - len(wav2)), mode='constant')

        return wav1, wav2

# Test
preprocessor = AudioPreprocessor()
audio = preprocessor.load_audio('data/short_clips/ads/ad_001.mp4')
filtered = preprocessor.bandpass_filter(audio)
silent_segments = preprocessor.detect_silent_segments(filtered)
print(f"Found {len(silent_segments)} silent segments")
```

### Step 2.2: CP Classifier Model

```python
# File: src/models/cp_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CPClassifier(nn.Module):
    """
    Change Point Classifier using Siamese ResNet-34
    """
    def __init__(self, pretrained_path=None):
        super().__init__()

        # Load pre-trained ResNet-34 from VoxCeleb
        # This is a simplified version - use actual VoxCeleb model
        self.feature_extractor = self._build_resnet34()

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _build_resnet34(self):
        """
        Build ResNet-34 for audio (simplified)
        Use actual VoxCeleb model in practice
        """
        import torchvision.models as models

        # Start with standard ResNet-34
        resnet = models.resnet34(pretrained=False)

        # Modify first conv for single-channel input (spectrogram)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final FC layer for 512-dim output
        resnet.fc = nn.Linear(512, 512)

        return resnet

    def load_pretrained(self, path):
        """Load pre-trained weights from VoxCeleb"""
        state_dict = torch.load(path)
        self.feature_extractor.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")

    def forward_one(self, x):
        """
        Extract feature from one spectrogram

        Args:
            x: [batch, 1, n_mels, time_steps]
        Returns:
            features: [batch, 512]
        """
        features = self.feature_extractor(x)
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        return features

    def forward(self, x1, x2):
        """
        Forward pass for Siamese network

        Args:
            x1, x2: [batch, 1, n_mels, time_steps]
        Returns:
            dist: Euclidean distance [batch]
        """
        f1 = self.forward_one(x1)
        f2 = self.forward_one(x2)

        # Euclidean distance
        dist = torch.norm(f1 - f2, p=2, dim=1)
        return dist

    def predict(self, x1, x2, threshold=1.0):
        """
        Predict if x1, x2 are from different videos (CP)

        Returns:
            is_cp: Boolean tensor [batch]
        """
        dist = self.forward(x1, x2)
        is_cp = dist > threshold
        return is_cp

# Test
model = CPClassifier()
x1 = torch.randn(4, 1, 64, 200)  # Batch of 4 spectrograms
x2 = torch.randn(4, 1, 64, 200)
dist = model(x1, x2)
print(f"Distances: {dist}")
```

### Step 2.3: CP Classifier Dataset

```python
# File: src/datasets/cp_dataset.py

import torch
from torch.utils.data import Dataset
import json
import random
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from audio_processing import AudioPreprocessor

class CPDataset(Dataset):
    """
    Dataset for Change Point Classifier

    Uses weak supervision as described in the paper:
    - Positive pairs (non-CP): Audio from same video clip
    - Negative pairs (CP): Audio from different video clips

    NOTE: This approach does NOT use annotated CP boundaries directly.
    The annotation JSON is only used to extract short clips (10-30s) with
    ad/content labels. CP pairs are generated on-the-fly from these clips.

    FUTURE IMPROVEMENT: Strong supervision could be added by extracting
    2s before/after each annotated change_point in the JSON. This would
    provide higher-quality TRUE CP pairs compared to the current weak
    supervision approach (different videos = CP assumption).
    """
    def __init__(self, clip_paths, split='train', sr=16000):
        """
        Args:
            clip_paths: List of SHORT CLIP video file paths (10-30s)
                       These are extracted from annotation JSONs
            split: 'train', 'val', or 'test'
        """
        self.clip_paths = clip_paths
        self.split = split
        self.preprocessor = AudioPreprocessor(sr=sr)

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        """
        Returns positive pair (non-CP) and negative pair (CP)

        For training (following paper's weak supervision approach):
            - Positive: Two clips from same video (around a silence or random point)
            - Negative: Two clips from different videos

        This assumes "different videos = CP" which is a coarse approximation.
        """
        # Load anchor video
        anchor_path = self.clip_paths[idx]
        anchor_audio = self.preprocessor.load_audio(anchor_path)
        anchor_audio = self.preprocessor.bandpass_filter(anchor_audio)

        # Find silent segments
        silent_segs = self.preprocessor.detect_silent_segments(anchor_audio)

        # Create positive pair
        if len(silent_segs) > 0 and random.random() > 0.5:
            # Sample from silent segment
            silence_time = random.choice(silent_segs)[0]
        else:
            # Random timestamp
            silence_time = random.uniform(2.0, len(anchor_audio)/self.preprocessor.sr - 2.0)

        wav1, wav2 = self.preprocessor.extract_audio_clips_around_silence(
            anchor_audio, silence_time, window_duration=2.0
        )

        # Convert to spectrograms
        spec1 = self.preprocessor.extract_log_mel_spectrogram(wav1)
        spec2 = self.preprocessor.extract_log_mel_spectrogram(wav2)

        # Create negative pair (from different video)
        negative_idx = random.choice([i for i in range(len(self.clip_paths)) if i != idx])
        negative_path = self.clip_paths[negative_idx]
        negative_audio = self.preprocessor.load_audio(negative_path)
        negative_audio = self.preprocessor.bandpass_filter(negative_audio)

        # Random clip from negative video
        neg_time = random.uniform(0, max(0, len(negative_audio)/self.preprocessor.sr - 2.0))
        neg_start = int(neg_time * self.preprocessor.sr)
        neg_end = neg_start + int(2.0 * self.preprocessor.sr)
        wav_neg = negative_audio[neg_start:neg_end]

        if len(wav_neg) < int(2.0 * self.preprocessor.sr):
            wav_neg = np.pad(wav_neg, (0, int(2.0 * self.preprocessor.sr) - len(wav_neg)))

        spec_neg = self.preprocessor.extract_log_mel_spectrogram(wav_neg)

        # Convert to tensors
        spec1 = torch.FloatTensor(spec1).unsqueeze(0)  # [1, n_mels, time]
        spec2 = torch.FloatTensor(spec2).unsqueeze(0)
        spec_neg = torch.FloatTensor(spec_neg).unsqueeze(0)

        return {
            'anchor': spec1,
            'positive': spec2,
            'negative': spec_neg,
            'anchor_path': anchor_path
        }

# Test
with open('data/dataset_splits.json') as f:
    splits = json.load(f)

train_clips = splits['train']['ad'] + splits['train']['content']
dataset = CPDataset(train_clips, split='train')
sample = dataset[0]
print(f"Anchor shape: {sample['anchor'].shape}")
print(f"Positive shape: {sample['positive'].shape}")
print(f"Negative shape: {sample['negative'].shape}")
```

### Step 2.4: CP Classifier Training

```python
# File: src/train_cp_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
import sys
sys.path.append('src')
from models.cp_classifier import CPClassifier
from datasets.cp_dataset import CPDataset

class TripletLoss(nn.Module):
    """
    Triplet Loss: L = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin, 0)
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch, feat_dim]
            positive: [batch, feat_dim]
            negative: [batch, feat_dim]
        """
        pos_dist = torch.norm(anchor - positive, p=2, dim=1) ** 2
        neg_dist = torch.norm(anchor - negative, p=2, dim=1) ** 2

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

def train_cp_classifier(config):
    # Load dataset
    with open('data/dataset_splits.json') as f:
        splits = json.load(f)

    train_clips = splits['train']['ad'] + splits['train']['content']
    val_clips = splits['val']['ad'] + splits['val']['content']

    train_dataset = CPDataset(train_clips, split='train')
    val_dataset = CPDataset(val_clips, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CPClassifier(pretrained_path=config.get('pretrained_path')).to(device)

    # Loss and optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Tensorboard
    writer = SummaryWriter(log_dir='logs/cp_classifier')

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            # Forward
            anchor_feat = model.forward_one(anchor)
            positive_feat = model.forward_one(positive)
            negative_feat = model.forward_one(negative)

            loss = criterion(anchor_feat, positive_feat, negative_feat)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                anchor = batch['anchor'].to(device)
                positive = batch['positive'].to(device)
                negative = batch['negative'].to(device)

                anchor_feat = model.forward_one(anchor)
                positive_feat = model.forward_one(positive)
                negative_feat = model.forward_one(negative)

                loss = criterion(anchor_feat, positive_feat, negative_feat)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Logging
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/cp_classifier_best.pth')
            print(f"Saved best model at epoch {epoch+1}")

        scheduler.step()

    writer.close()

# Configuration
config = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 196,
    'pretrained_path': 'pretrained/baseline_v2_ap.model'  # VoxCeleb model
}

if __name__ == '__main__':
    train_cp_classifier(config)
```

**Deliverable**:
- Trained CP classifier model
- Validation loss curve
- Model checkpoint

---

## Phase 3: Implement Ad Classifier

### Timeline: 3 weeks

### Step 3.1: Video Preprocessing Module

```python
# File: src/video_processing.py

import cv2
import torch
import numpy as np
import ffmpeg

class VideoPreprocessor:
    def __init__(self, clip_duration=4.3, fps=30):
        self.clip_duration = clip_duration
        self.fps = fps

    def extract_video_clip(self, video_path, start_time, duration=None):
        """
        Extract video clip from file

        Returns:
            frames: np.array [num_frames, height, width, 3]
        """
        if duration is None:
            duration = self.clip_duration

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        frames = []
        num_frames = int(duration * self.fps)

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) < num_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([last_frame] * (num_frames - len(frames)))

        return np.array(frames)

    def temporal_spatial_sampling(self, frames, num_temporal=5, num_spatial=3):
        """
        Sample frames for data augmentation (following AvSlowFast paper)

        Args:
            frames: [T, H, W, 3]
            num_temporal: Number of temporal samples
            num_spatial: Number of spatial crops (left, center, right)

        Returns:
            sampled_clips: List of [T', H', W', 3] arrays
        """
        T, H, W, C = frames.shape
        target_size = 224

        sampled_clips = []

        # Temporal sampling (uniform)
        temporal_indices = np.linspace(0, T-1, num_temporal, dtype=int)

        for t_idx in temporal_indices:
            # Get frame
            frame = frames[t_idx]

            # Spatial crops
            if num_spatial == 3:
                # Left, center, right
                crop_starts = [0, (W - target_size) // 2, W - target_size]
            else:
                # Only center
                crop_starts = [(W - target_size) // 2]

            for crop_start in crop_starts:
                # Crop and resize
                crop = frame[:, crop_start:crop_start+target_size, :]
                crop = cv2.resize(crop, (target_size, target_size))
                sampled_clips.append(crop)

        return sampled_clips

    def prepare_slowfast_input(self, frames, alpha=4):
        """
        Prepare input for SlowFast network

        Args:
            frames: [T, H, W, 3]
            alpha: Ratio between fast and slow pathway

        Returns:
            slow_frames: [T_slow, H, W, 3]
            fast_frames: [T_fast, H, W, 3]
        """
        T = frames.shape[0]

        # Slow pathway: sample every alpha frames
        slow_indices = np.arange(0, T, alpha)
        slow_frames = frames[slow_indices]

        # Fast pathway: all frames
        fast_frames = frames

        return slow_frames, fast_frames

# Test
processor = VideoPreprocessor()
frames = processor.extract_video_clip('data/short_clips/ads/ad_001.mp4', start_time=0)
print(f"Extracted {len(frames)} frames")
```

### Step 3.2: Ad Classifier Model (Simplified AvSlowFast)

```python
# File: src/models/ad_classifier.py

import torch
import torch.nn as nn
import torchvision.models as models

class SlowFastNetwork(nn.Module):
    """
    Simplified SlowFast network for video + audio
    In practice, use the official Facebook Research implementation
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Slow pathway (ResNet-50)
        self.slow_pathway = models.resnet50(pretrained=True)
        self.slow_pathway.fc = nn.Identity()  # Remove final FC

        # Fast pathway (ResNet-50 with different stride)
        self.fast_pathway = models.resnet50(pretrained=True)
        self.fast_pathway.fc = nn.Identity()

        # Audio pathway (ResNet-50 for spectrogram)
        self.audio_pathway = models.resnet50(pretrained=True)
        self.audio_pathway.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.audio_pathway.fc = nn.Identity()

        # Feature dimensions
        self.slow_dim = 2048
        self.fast_dim = 256  # Reduced channels
        self.audio_dim = 1024

        # Fusion
        self.fusion_dim = self.slow_dim + self.fast_dim + self.audio_dim

    def forward(self, slow_input, fast_input, audio_input):
        """
        Args:
            slow_input: [batch, 3, T_slow, H, W]
            fast_input: [batch, 3, T_fast, H, W]
            audio_input: [batch, 1, n_mels, T_audio]

        Returns:
            fused_features: [batch, fusion_dim, temporal_resolution]
        """
        batch_size = slow_input.size(0)

        # Process slow pathway
        # [B, 3, T_slow, H, W] -> [B*T_slow, 3, H, W]
        slow_frames = slow_input.permute(0, 2, 1, 3, 4).contiguous()
        slow_frames = slow_frames.view(-1, 3, slow_input.size(3), slow_input.size(4))
        slow_feat = self.slow_pathway(slow_frames)  # [B*T_slow, 2048]
        slow_feat = slow_feat.view(batch_size, -1, self.slow_dim)  # [B, T_slow, 2048]

        # Process fast pathway (similar)
        fast_frames = fast_input.permute(0, 2, 1, 3, 4).contiguous()
        fast_frames = fast_frames.view(-1, 3, fast_input.size(3), fast_input.size(4))
        fast_feat = self.fast_pathway(fast_frames)
        fast_feat = fast_feat.view(batch_size, -1, 2048)
        # Reduce channels
        fast_feat = fast_feat[:, :, :self.fast_dim]  # [B, T_fast, 256]

        # Process audio pathway
        audio_feat = self.audio_pathway(audio_input)  # [B, 2048]
        audio_feat = audio_feat[:, :self.audio_dim].unsqueeze(1)  # [B, 1, 1024]
        audio_feat = audio_feat.repeat(1, fast_feat.size(1), 1)  # [B, T_fast, 1024]

        # Fuse (simplified - just concatenate)
        # In practice, use lateral connections
        # Upsample slow to match fast temporal resolution
        slow_feat_upsampled = torch.repeat_interleave(slow_feat, fast_feat.size(1) // slow_feat.size(1), dim=1)

        fused = torch.cat([slow_feat_upsampled, fast_feat, audio_feat], dim=2)  # [B, T_fast, fusion_dim]

        return fused

class TemporalAveragePooling(nn.Module):
    """TAP model"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch, temporal, features]
        pooled = torch.mean(x, dim=1)  # [batch, features]
        out = self.fc(pooled)  # [batch, num_classes]
        return out

class LSTMClassifier(nn.Module):
    """Bi-directional LSTM model"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 512, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(1024, 1000)  # 512*2 from bidirectional
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, temporal, features]
        lstm_out, _ = self.lstm(x)  # [batch, temporal, 1024]

        # Take first timestep
        features = lstm_out[:, 0, :]  # [batch, 1024]

        # FC layers
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

class AdClassifier(nn.Module):
    """
    Complete Ad Classifier = SlowFast + Temporal Attention
    """
    def __init__(self, temporal_model='lstm'):
        super().__init__()

        self.feature_extractor = SlowFastNetwork()
        fusion_dim = self.feature_extractor.fusion_dim

        if temporal_model == 'tap':
            self.temporal_model = TemporalAveragePooling(fusion_dim)
        elif temporal_model == 'lstm':
            self.temporal_model = LSTMClassifier(fusion_dim)
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")

    def forward(self, slow_input, fast_input, audio_input):
        fused_features = self.feature_extractor(slow_input, fast_input, audio_input)
        logits = self.temporal_model(fused_features)
        return logits

# Test
model = AdClassifier(temporal_model='lstm')
slow = torch.randn(2, 3, 4, 224, 224)
fast = torch.randn(2, 3, 32, 224, 224)
audio = torch.randn(2, 1, 64, 256)
output = model(slow, fast, audio)
print(f"Output shape: {output.shape}")  # [2, 2]
```

### Step 3.3: Ad Classifier Dataset

```python
# File: src/datasets/ad_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import sys
sys.path.append('src')
from video_processing import VideoPreprocessor
from audio_processing import AudioPreprocessor

class AdDataset(Dataset):
    """
    Dataset for Ad Classifier
    """
    def __init__(self, video_paths, labels, clip_duration=4.3, split='train'):
        """
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=content, 1=ad)
            clip_duration: Duration of each clip in seconds
            split: 'train', 'val', or 'test'
        """
        self.video_paths = video_paths
        self.labels = labels
        self.clip_duration = clip_duration
        self.split = split

        self.video_processor = VideoPreprocessor(clip_duration=clip_duration)
        self.audio_processor = AudioPreprocessor()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract video frames
        frames = self.video_processor.extract_video_clip(video_path, start_time=0)

        # Prepare slow and fast inputs
        slow_frames, fast_frames = self.video_processor.prepare_slowfast_input(frames, alpha=4)

        # Convert to tensor [C, T, H, W]
        slow_tensor = torch.FloatTensor(slow_frames).permute(3, 0, 1, 2) / 255.0
        fast_tensor = torch.FloatTensor(fast_frames).permute(3, 0, 1, 2) / 255.0

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        slow_tensor = (slow_tensor - mean) / std
        fast_tensor = (fast_tensor - mean) / std

        # Extract audio
        audio = self.audio_processor.load_audio(video_path, duration=self.clip_duration)
        audio = self.audio_processor.bandpass_filter(audio)
        audio_spec = self.audio_processor.extract_log_mel_spectrogram(audio)
        audio_tensor = torch.FloatTensor(audio_spec).unsqueeze(0)  # [1, n_mels, time]

        return {
            'slow': slow_tensor,
            'fast': fast_tensor,
            'audio': audio_tensor,
            'label': torch.LongTensor([label])[0],
            'path': video_path
        }

# Test
video_paths = ['data/short_clips/ads/ad_001.mp4', 'data/short_clips/content/content_001.mp4']
labels = [1, 0]
dataset = AdDataset(video_paths, labels)
sample = dataset[0]
print(f"Slow shape: {sample['slow'].shape}")
print(f"Fast shape: {sample['fast'].shape}")
print(f"Audio shape: {sample['audio'].shape}")
print(f"Label: {sample['label']}")
```

### Step 3.4: Ad Classifier Training

```python
# File: src/train_ad_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import sys
sys.path.append('src')
from models.ad_classifier import AdClassifier
from datasets.ad_dataset import AdDataset

def train_ad_classifier(config):
    # Load dataset
    with open('data/dataset_splits.json') as f:
        splits = json.load(f)

    # Prepare data
    train_videos = splits['train']['ad'] + splits['train']['content']
    train_labels = [1] * len(splits['train']['ad']) + [0] * len(splits['train']['content'])

    val_videos = splits['val']['ad'] + splits['val']['content']
    val_labels = [1] * len(splits['val']['ad']) + [0] * len(splits['val']['content'])

    # Create datasets
    train_dataset = AdDataset(train_videos, train_labels, split='train')
    val_dataset = AdDataset(val_videos, val_labels, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdClassifier(temporal_model=config['temporal_model']).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Tensorboard
    writer = SummaryWriter(log_dir=f"logs/ad_classifier_{config['temporal_model']}")

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            slow = batch['slow'].to(device)
            fast = batch['fast'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(slow, fast, audio)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                slow = batch['slow'].to(device)
                fast = batch['fast'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)

                logits = model(slow, fast, audio)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Logging
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"models/ad_classifier_{config['temporal_model']}_best.pth")
            print(f"Saved best model at epoch {epoch+1} with val_acc={val_acc:.2f}%")

        scheduler.step()

    writer.close()

# Configuration
config = {
    'batch_size': 8,  # Small due to video memory
    'lr': 1e-4,
    'epochs': 196,
    'temporal_model': 'lstm'  # or 'tap'
}

if __name__ == '__main__':
    train_ad_classifier(config)
```

**Deliverable**:
- Trained Ad classifier model (LSTM and TAP versions)
- Accuracy curves
- Model checkpoints

---

## Phase 4: Implement End-to-End Pipeline

### Timeline: 2 weeks

### Step 4.1: Segmentation Pipeline

```python
# File: src/pipeline/segmentation.py

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')
from audio_processing import AudioPreprocessor
from models.cp_classifier import CPClassifier
from pyscenedetect import detect, ContentDetector

class SegmentationPipeline:
    """
    End-to-end segmentation pipeline
    """
    def __init__(self, cp_model_path, threshold=1.0):
        self.audio_processor = AudioPreprocessor()

        # Load CP classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cp_model = CPClassifier().to(self.device)
        checkpoint = torch.load(cp_model_path, map_location=self.device)
        self.cp_model.load_state_dict(checkpoint['model_state_dict'])
        self.cp_model.eval()

        self.threshold = threshold

    def detect_change_points(self, video_path):
        """
        Detect change points in video

        Returns:
            List of change point timestamps
        """
        # Step 1: Extract audio
        audio = self.audio_processor.load_audio(video_path)
        audio = self.audio_processor.bandpass_filter(audio)

        # Step 2: Detect silent segments
        silent_segments = self.audio_processor.detect_silent_segments(
            audio,
            min_duration_ms=10,
            volume_threshold=4
        )

        print(f"Found {len(silent_segments)} silent segments")

        # Step 3: Classify each silent segment as CP or not
        change_points = []

        for silence_start, silence_end in silent_segments:
            silence_mid = (silence_start + silence_end) / 2

            # Extract audio clips around silence
            wav1, wav2 = self.audio_processor.extract_audio_clips_around_silence(
                audio, silence_mid, window_duration=2.0
            )

            # Convert to spectrograms
            spec1 = self.audio_processor.extract_log_mel_spectrogram(wav1)
            spec2 = self.audio_processor.extract_log_mel_spectrogram(wav2)

            # Convert to tensors
            spec1_tensor = torch.FloatTensor(spec1).unsqueeze(0).unsqueeze(0).to(self.device)
            spec2_tensor = torch.FloatTensor(spec2).unsqueeze(0).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                distance = self.cp_model(spec1_tensor, spec2_tensor)
                is_cp = distance.item() > self.threshold

            if is_cp:
                change_points.append(silence_mid)
            else:
                # Step 4: Video-based validation for non-CP segments
                # Use scene detection as backup
                scene_changes = self.detect_scene_changes(
                    video_path,
                    start_time=max(0, silence_mid - 0.5),
                    end_time=silence_mid + 0.5
                )

                if len(scene_changes) > 0:
                    print(f"Video scene change detected at {silence_mid:.2f}s (audio CP missed)")
                    change_points.append(silence_mid)

        print(f"Detected {len(change_points)} change points")
        return sorted(change_points)

    def detect_scene_changes(self, video_path, start_time=None, end_time=None):
        """
        Detect scene changes using PySceneDetect
        """
        try:
            scene_list = detect(video_path, ContentDetector())

            # Filter by time range if specified
            if start_time is not None and end_time is not None:
                scene_list = [
                    scene for scene in scene_list
                    if start_time <= scene[0].get_seconds() <= end_time
                ]

            return [scene[0].get_seconds() for scene in scene_list]
        except:
            return []

    def create_segments(self, change_points, video_duration):
        """
        Create segments from change points

        Returns:
            List of (start_time, end_time) tuples
        """
        segments = []

        # First segment
        if len(change_points) == 0:
            return [(0, video_duration)]

        segments.append((0, change_points[0]))

        # Middle segments
        for i in range(len(change_points) - 1):
            segments.append((change_points[i], change_points[i+1]))

        # Last segment
        segments.append((change_points[-1], video_duration))

        return segments

    def cleanup_short_segments(self, segments, min_duration=8.0):
        """
        Merge or discard segments shorter than min_duration
        """
        cleaned = []
        i = 0

        while i < len(segments):
            start, end = segments[i]
            duration = end - start

            if duration >= min_duration:
                cleaned.append((start, end))
                i += 1
            else:
                # Check if next segment is also short
                if i + 1 < len(segments):
                    next_start, next_end = segments[i+1]
                    next_duration = next_end - next_start

                    if next_duration < min_duration:
                        # Merge
                        merged_end = next_end
                        cleaned.append((start, merged_end))
                        i += 2
                    else:
                        # Discard current short segment
                        i += 1
                else:
                    # Last segment is short, discard
                    i += 1

        return cleaned

    def segment_video(self, video_path):
        """
        Complete segmentation pipeline

        Returns:
            List of segments: [(start, end), ...]
        """
        # Get video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()

        # Detect change points
        change_points = self.detect_change_points(video_path)

        # Create segments
        segments = self.create_segments(change_points, duration)

        # Cleanup short segments
        segments = self.cleanup_short_segments(segments, min_duration=8.0)

        print(f"Final: {len(segments)} segments")
        return segments

# Test
pipeline = SegmentationPipeline('models/cp_classifier_best.pth')
segments = pipeline.segment_video('data/raw_videos/movies/movie_001.mp4')
for i, (start, end) in enumerate(segments):
    print(f"Segment {i}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
```

### Step 4.2: Classification Pipeline

```python
# File: src/pipeline/classification.py

import torch
import numpy as np
import sys
sys.path.append('src')
from models.ad_classifier import AdClassifier
from video_processing import VideoPreprocessor
from audio_processing import AudioPreprocessor

class ClassificationPipeline:
    """
    Classify segments as Ad or Content
    """
    def __init__(self, ad_model_path, temporal_model='lstm'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Ad classifier
        self.ad_model = AdClassifier(temporal_model=temporal_model).to(self.device)
        checkpoint = torch.load(ad_model_path, map_location=self.device)
        self.ad_model.load_state_dict(checkpoint['model_state_dict'])
        self.ad_model.eval()

        self.video_processor = VideoPreprocessor(clip_duration=4.3)
        self.audio_processor = AudioPreprocessor()

        self.long_segment_threshold = 85.0  # seconds
        self.ad_high_threshold = 0.3333
        self.ad_low_threshold = 0.1

    def classify_segment(self, video_path, start_time, end_time):
        """
        Classify a single segment

        Returns:
            'ad' or 'content'
        """
        duration = end_time - start_time

        # Skip classification for very long segments
        if duration > self.long_segment_threshold:
            return 'content'

        # Crop segment into 4.3s clips with 2s hop
        clip_predictions = []
        current_time = start_time
        hop_size = 2.0

        while current_time + 4.3 <= end_time:
            # Extract clip
            frames = self.video_processor.extract_video_clip(
                video_path,
                start_time=current_time,
                duration=4.3
            )
            slow_frames, fast_frames = self.video_processor.prepare_slowfast_input(frames)

            audio = self.audio_processor.load_audio(
                video_path,
                start_time=current_time,
                duration=4.3
            )
            audio = self.audio_processor.bandpass_filter(audio)
            audio_spec = self.audio_processor.extract_log_mel_spectrogram(audio)

            # Convert to tensors
            slow_tensor = torch.FloatTensor(slow_frames).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
            fast_tensor = torch.FloatTensor(fast_frames).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
            audio_tensor = torch.FloatTensor(audio_spec).unsqueeze(0).unsqueeze(0)

            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
            slow_tensor = (slow_tensor - mean) / std
            fast_tensor = (fast_tensor - mean) / std

            # Move to device
            slow_tensor = slow_tensor.to(self.device)
            fast_tensor = fast_tensor.to(self.device)
            audio_tensor = audio_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.ad_model(slow_tensor, fast_tensor, audio_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            clip_predictions.append(pred)
            current_time += hop_size

        # Aggregate predictions
        if len(clip_predictions) == 0:
            return 'content'  # Very short segment

        ad_ratio = sum(clip_predictions) / len(clip_predictions)

        # Double threshold decision with hysteresis
        if ad_ratio >= self.ad_high_threshold:
            return 'ad'  # Strong ad
        elif ad_ratio >= self.ad_low_threshold:
            return 'weak_ad'  # Needs hysteresis tracking
        else:
            return 'content'

    def classify_segments(self, video_path, segments):
        """
        Classify all segments

        Returns:
            List of labels: ['ad', 'content', ...]
        """
        labels = []

        for start, end in segments:
            label = self.classify_segment(video_path, start, end)
            labels.append(label)
            print(f"Segment {start:.2f}s-{end:.2f}s: {label}")

        # Hysteresis tracking for weak_ad
        final_labels = []
        for i, label in enumerate(labels):
            if label == 'weak_ad':
                # Check neighbors
                has_strong_neighbor = False
                if i > 0 and labels[i-1] == 'ad':
                    has_strong_neighbor = True
                if i < len(labels) - 1 and labels[i+1] == 'ad':
                    has_strong_neighbor = True

                if has_strong_neighbor:
                    final_labels.append('ad')
                else:
                    final_labels.append('content')
            else:
                final_labels.append(label)

        return final_labels

# Test
pipeline = ClassificationPipeline('models/ad_classifier_lstm_best.pth')
segments = [(0, 30), (30, 45), (45, 200)]
labels = pipeline.classify_segments('data/raw_videos/movies/movie_001.mp4', segments)
print(labels)
```

### Step 4.3: Complete End-to-End Pipeline

```python
# File: src/pipeline/end_to_end.py

import json
import sys
sys.path.append('src')
from pipeline.segmentation import SegmentationPipeline
from pipeline.classification import ClassificationPipeline

class AdDetectionPipeline:
    """
    Complete end-to-end ad detection pipeline
    """
    def __init__(self, cp_model_path, ad_model_path, temporal_model='lstm'):
        self.segmentation = SegmentationPipeline(cp_model_path)
        self.classification = ClassificationPipeline(ad_model_path, temporal_model)

    def process_video(self, video_path, output_path=None):
        """
        Process video end-to-end

        Returns:
            Dictionary with results
        """
        print(f"Processing {video_path}...")

        # Stage 1: Segmentation
        print("\n=== Stage 1: Segmentation ===")
        segments = self.segmentation.segment_video(video_path)

        # Stage 2: Classification
        print("\n=== Stage 2: Classification ===")
        labels = self.classification.classify_segments(video_path, segments)

        # Stage 3: Post-processing (simplified - implement fully in Phase 5)
        print("\n=== Stage 3: Post-processing ===")
        # TODO: Implement under-segmentation correction

        # Compile results
        results = {
            'video_path': video_path,
            'num_segments': len(segments),
            'segments': [
                {
                    'id': i,
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'label': label
                }
                for i, ((start, end), label) in enumerate(zip(segments, labels))
            ]
        }

        # Count ads
        ad_segments = [s for s in results['segments'] if s['label'] == 'ad']
        results['num_ads'] = len(ad_segments)
        results['total_ad_duration'] = sum(s['duration'] for s in ad_segments)

        print(f"\nResults:")
        print(f"- Total segments: {results['num_segments']}")
        print(f"- Ad segments: {results['num_ads']}")
        print(f"- Total ad duration: {results['total_ad_duration']:.2f}s")

        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {output_path}")

        return results

# Test
pipeline = AdDetectionPipeline(
    cp_model_path='models/cp_classifier_best.pth',
    ad_model_path='models/ad_classifier_lstm_best.pth'
)

results = pipeline.process_video(
    video_path='data/raw_videos/movies/movie_001.mp4',
    output_path='outputs/movie_001_results.json'
)
```

**Deliverable**:
- Working end-to-end pipeline
- Results JSON files
- Performance metrics

---

## Phase 5: Post-Processing and Optimization

### Timeline: 1 week

### Step 5.1: Implement PANNs-Based Post-Processing

```python
# File: src/pipeline/post_processing.py

import torch
import torchaudio
from panns_inference import AudioTagging
import numpy as np
import sys
sys.path.append('src')
from audio_processing import AudioPreprocessor

class PostProcessor:
    """
    Post-processing to fix segmentation errors
    """
    def __init__(self, panns_model_path):
        # Load PANNs
        self.panns = AudioTagging(checkpoint_path=panns_model_path, device='cuda')
        self.audio_processor = AudioPreprocessor()

        # Define continuous vs non-continuous categories
        self.continuous_categories = [
            'Music', 'Musical instrument', 'Guitar', 'Piano', 'Drum',
            'Fire', 'Water', 'Rain', 'Wind', 'Ocean'
        ]
        self.non_continuous_categories = [
            'Speech', 'Male speech', 'Female speech', 'Child speech',
            'Bird', 'Dog', 'Cat'
        ]

    def classify_audio_type(self, audio_path):
        """
        Classify audio as continuous or non-continuous
        """
        # Run PANNs
        clipwise_output, embedding = self.panns.inference(audio_path)

        # Get top predictions
        # clipwise_output has shape [527] for 527 AudioSet classes
        # Check for continuous vs non-continuous

        # Simplified - check if vocal-heavy
        # In practice, use proper category mapping
        speech_score = clipwise_output[0]  # Example
        music_score = clipwise_output[1]  # Example

        if speech_score > music_score:
            return 'non-continuous'
        else:
            return 'continuous'

    def fix_ad_undersegmentation(self, video_path, segments, labels):
        """
        Fix under-segmentation of short ads
        """
        # Re-segment with aggressive thresholds
        # (Implement similar to segmentation pipeline but with stricter params)
        pass

    def fix_content_undersegmentation(self, video_path, segments, labels):
        """
        Fix under-segmentation of long content segments
        """
        pass

# Implement this fully based on paper Section 4
```

### Step 5.2: Optimize Inference Speed

```python
# File: src/pipeline/optimized_inference.py

import torch
from torch.utils.data import DataLoader, Dataset

class BatchInference:
    """
    Batch processing for faster inference
    """
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size

    def process_clips_batch(self, clips):
        """
        Process multiple clips in batch
        """
        # Implement batched inference
        pass

# Use TorchScript for optimization
def export_to_torchscript(model, example_inputs):
    """
    Export model to TorchScript for faster inference
    """
    model.eval()
    traced = torch.jit.trace(model, example_inputs)
    traced.save('models/ad_classifier_scripted.pt')
    return traced
```

**Deliverable**:
- Post-processing module
- Optimized inference pipeline
- Speed benchmarks

---

## Phase 6: Evaluation and Testing

### Timeline: 1 week

### Step 6.1: Implement Evaluation Metrics

```python
# File: src/evaluation/metrics.py

def compute_segment_overlap(pred_seg, gt_seg):
    """
    Compute overlap ratio between predicted and ground truth segments
    """
    pred_start, pred_end = pred_seg
    gt_start, gt_end = gt_seg

    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)

    if overlap_end <= overlap_start:
        return 0.0

    overlap = overlap_end - overlap_start
    pred_duration = pred_end - pred_start

    return overlap / pred_duration

def evaluate_segmentation(predictions, ground_truth, threshold=0.5):
    """
    Evaluate segmentation quality

    Args:
        predictions: List of predicted segments
        ground_truth: List of GT segments
        threshold: Overlap threshold for matching

    Returns:
        metrics dict
    """
    # Map predictions to GT
    matched_pred = set()
    matched_gt = set()

    correct = 0
    overseg = 0
    underseg = 0
    false_pos = 0
    miss = 0

    # For each GT segment
    for gt_idx, gt_seg in enumerate(ground_truth):
        matches = []
        for pred_idx, pred_seg in enumerate(predictions):
            overlap = compute_segment_overlap(pred_seg, gt_seg)
            if overlap > threshold:
                matches.append(pred_idx)

        if len(matches) == 1:
            correct += 1
            matched_gt.add(gt_idx)
            matched_pred.add(matches[0])
        elif len(matches) > 1:
            overseg += 1
            matched_gt.add(gt_idx)
            for m in matches:
                matched_pred.add(m)
        elif len(matches) == 0:
            miss += 1

    # Check for under-segmentation and false positives
    for pred_idx in range(len(predictions)):
        if pred_idx not in matched_pred:
            false_pos += 1

    # Normalize
    total_gt = len(ground_truth)
    total_pred = len(predictions)

    metrics = {
        'correct_rate': correct / total_pred if total_pred > 0 else 0,
        'overseg_rate': overseg / total_gt,
        'underseg_rate': underseg / total_gt,
        'miss_rate': miss / total_gt,
        'false_pos_rate': false_pos / total_pred if total_pred > 0 else 0,
        'num_predicted': total_pred,
        'num_ground_truth': total_gt
    }

    return metrics

# Test
pred_segments = [(0, 30), (30, 45), (45, 200)]
gt_segments = [(0, 28), (28, 46), (46, 200)]
metrics = evaluate_segmentation(pred_segments, gt_segments)
print(metrics)
```

### Step 6.2: Run Full Evaluation

```python
# File: src/evaluation/evaluate.py

import json
from pathlib import Path
import sys
sys.path.append('src')
from pipeline.end_to_end import AdDetectionPipeline
from evaluation.metrics import evaluate_segmentation

def evaluate_on_dataset(test_videos, annotations_dir, pipeline):
    """
    Evaluate pipeline on test dataset
    """
    all_metrics = []

    for video_path in test_videos:
        video_name = Path(video_path).stem
        annotation_path = Path(annotations_dir) / f"{video_name}.json"

        if not annotation_path.exists():
            print(f"Warning: No annotation for {video_name}")
            continue

        # Load ground truth
        with open(annotation_path) as f:
            gt_data = json.load(f)

        gt_segments = [
            (seg['start_time'], seg['end_time'])
            for seg in gt_data['segments']
            if seg['label'] == 'ad'
        ]

        # Run pipeline
        results = pipeline.process_video(video_path)

        pred_segments = [
            (seg['start_time'], seg['end_time'])
            for seg in results['segments']
            if seg['label'] == 'ad'
        ]

        # Compute metrics
        metrics = evaluate_segmentation(pred_segments, gt_segments)
        metrics['video'] = video_name
        all_metrics.append(metrics)

        print(f"\n{video_name}:")
        print(f"  Correct: {metrics['correct_rate']:.2%}")
        print(f"  Overseg: {metrics['overseg_rate']:.2%}")
        print(f"  Miss: {metrics['miss_rate']:.2%}")

    # Aggregate metrics
    avg_metrics = {
        'correct_rate': np.mean([m['correct_rate'] for m in all_metrics]),
        'overseg_rate': np.mean([m['overseg_rate'] for m in all_metrics]),
        'underseg_rate': np.mean([m['underseg_rate'] for m in all_metrics]),
        'miss_rate': np.mean([m['miss_rate'] for m in all_metrics]),
        'false_pos_rate': np.mean([m['false_pos_rate'] for m in all_metrics])
    }

    print("\n=== Overall Metrics ===")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.2%}")

    return all_metrics, avg_metrics

# Run evaluation
pipeline = AdDetectionPipeline(
    'models/cp_classifier_best.pth',
    'models/ad_classifier_lstm_best.pth'
)

test_videos = list(Path('data/raw_videos/movies').glob('*.mp4'))
results, avg_metrics = evaluate_on_dataset(
    test_videos,
    'data/annotations',
    pipeline
)
```

**Deliverable**:
- Evaluation metrics on test set
- Performance report
- Error analysis

---

## Phase 7: Deployment

### Timeline: 1 week

### Step 7.1: Create Inference API

```python
# File: src/api/app.py

from flask import Flask, request, jsonify
import sys
sys.path.append('src')
from pipeline.end_to_end import AdDetectionPipeline

app = Flask(__name__)

# Load models
pipeline = AdDetectionPipeline(
    cp_model_path='models/cp_classifier_best.pth',
    ad_model_path='models/ad_classifier_lstm_best.pth'
)

@app.route('/detect_ads', methods=['POST'])
def detect_ads():
    """
    API endpoint for ad detection

    Request:
        {
            "video_path": "/path/to/video.mp4"
        }

    Response:
        {
            "num_ads": 5,
            "total_ad_duration": 75.5,
            "segments": [...]
        }
    """
    data = request.json
    video_path = data.get('video_path')

    if not video_path:
        return jsonify({"error": "video_path required"}), 400

    # Process video
    results = pipeline.process_video(video_path)

    return jsonify(results)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Step 7.2: Docker Deployment

```dockerfile
# Dockerfile

FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy code
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 5000

# Run API
CMD ["python3", "src/api/app.py"]
```

```bash
# Build and run
docker build -t ad-detection .
docker run --gpus all -p 5000:5000 ad-detection
```

**Deliverable**:
- REST API for inference
- Docker container
- Deployment documentation

---

## Summary Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Data Collection | 2-3 weeks | 12K+ labeled clips, 48+ long videos |
| Phase 2: CP Classifier | 2 weeks | Trained CP model, ~90% accuracy |
| Phase 3: Ad Classifier | 3 weeks | Trained Ad model, ~95% accuracy |
| Phase 4: End-to-End Pipeline | 2 weeks | Complete pipeline |
| Phase 5: Post-Processing | 1 week | Optimized inference |
| Phase 6: Evaluation | 1 week | Performance metrics |
| Phase 7: Deployment | 1 week | Production API |
| **Total** | **12-14 weeks** | **Production-ready system** |

---

## Key Success Metrics

**CP Classifier**:
- AUC ≥ 0.90
- Detection rate ≥ 85% at 10% FPR

**Ad Classifier**:
- Precision ≥ 93%
- Recall ≥ 95%

**End-to-End**:
- Correct rate ≥ 95%
- Over-segmentation ≤ 1%
- Under-segmentation ≤ 1%
- False positive ≤ 2.5%

**Speed**:
- Segmentation: < 30s for 20-min video
- Complete pipeline: 6-10× faster than video-based methods

---

## Troubleshooting Guide

**Issue: Low CP classifier accuracy**
- Solution: Increase training data, adjust window size, try contrastive loss

**Issue: Low Ad classifier accuracy**
- Solution: Check temporal model, increase clip duration, verify data augmentation

**Issue: Slow inference**
- Solution: Use TorchScript, batch processing, skip long segments

**Issue: Over-segmentation**
- Solution: Adjust silent segment threshold, use PANNs filtering

**Issue: Under-segmentation**
- Solution: Lower silent segment threshold, enable video scene detection

---

## Additional Resources

**Pre-trained Models**:
- VoxCeleb2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- SlowFast: https://github.com/facebookresearch/SlowFast
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn

**Tools**:
- PySceneDetect: https://pyscenedetect.readthedocs.io/
- Librosa: https://librosa.org/
- FFmpeg: https://ffmpeg.org/

**Papers**:
- Original paper (this implementation)
- SlowFast Networks: https://arxiv.org/abs/1812.03982
- Vision Transformer: https://arxiv.org/abs/2010.11929

This implementation game plan provides a comprehensive, step-by-step guide for building the ad detection system from scratch. Follow each phase sequentially, validate results at each step, and iterate as needed. Good luck!
