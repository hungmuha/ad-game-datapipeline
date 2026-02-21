# Model Architecture and Training Process

## Overview

This document details the model architectures and training processes for the two-stage advertisement detection framework: the Change Point (CP) Classifier and the Advertisement (Ad) Classifier.

---

## 1. Change Point (CP) Classifier

### Purpose
Detects boundaries between content and advertisements, or between individual ads, by identifying silent segments that represent potential change points.

### Architecture

#### Input Specification
- **Audio clips**: Two temporal windows (wav1, wav2) before and after a silent segment
- **Window duration (win)**: 2 seconds (optimal based on experiments)
- **Audio preprocessing**:
  - Normalize to int16 type
  - Band-pass filter: [300Hz, 6000Hz]
  - Sampling rate: 16kHz+

#### Feature Extraction Network
**Modified ResNet-34** (from Oxford Robotics Lab SpeakerID framework):

1. **Input Processing**:
   - Convert audio clips to Log Mel Spectrograms

2. **Network Layers**:
   - 7×7 convolution layer (initial)
   - 3×3 max pooling layer
   - Four residual net blocks
   - Temporal Self-Attentive Pooling
   - Fully connected layer → **512-dimensional feature vector**

3. **Distance Computation**:
   - Extract 512-dim features for both wav1 and wav2
   - Compute Euclidean distance: D = ||f(wav1) - f(wav2)||²
   - Classification: D > T_CP → Change Point; D ≤ T_CP → Not a Change Point

#### Network Architecture (Siamese)
```
Audio Clip 1 (2s)          Audio Clip 2 (2s)
     ↓                           ↓
Log Mel Spectrogram       Log Mel Spectrogram
     ↓                           ↓
   ResNet-34                 ResNet-34 (shared weights)
     ↓                           ↓
Feature [512]              Feature [512]
     ↓                           ↓
     └──────→ Euclidean Distance ←──────┘
                    ↓
            Compare to Threshold T_CP
                    ↓
              CP / Not CP
```

### Training Process

#### Dataset Preparation
- **Training data**: 70% of short video sub-playbacks (1-5 minutes)
- **Validation data**: 15% of short video sub-playbacks
- **Test data**: 15% of short video sub-playbacks
- **Class distribution**:
  - Train: 5609 Ad clips, 5367 Content clips
  - Test: 1038 Ad clips, 1154 Content clips
  - Validation: 1040 Ad clips, 1154 Content clips

#### Positive and Negative Pair Generation

**Positive Pairs (Non-CP)**:
- Sample from the same video playback
- Find silent segments within a clip
- If no segment found, randomly select a timestamp
- Audio clips before and after this point are temporally adjacent

**Negative Pairs (CP)**:
- Sample audio clips from different playbacks
- For batch size N_B: generates N_B positive pairs and (N_B - 1)² negative pairs
- Apply hard negative mining during training

#### Loss Functions

Two loss functions were tested:

**1. Triplet Loss (Best Performance)**:
```
L(A, P, N) = max(L(P) - L(N) + α, 0)
where:
  L(P) = ||f(A) - f(P)||²  (anchor to positive/non-CP)
  L(N) = ||f(A) - f(N)||²  (anchor to negative/CP)
  α = 1 (margin)
  f = feature extractor (ResNet-34)
```

**2. Contrastive Loss**:
```
L(X1, X2) = L(1) + L(2)
where:
  L(1) = (1 - Y) × ½ × ||f(X1) - f(X2)||²
  L(2) = Y × ½ × max(0, α - ||f(X1) - f(X2)||²)
  Y = 0 for non-CP pairs (same playback)
  Y = 1 for CP pairs (different playbacks)
  α = 1 (margin)
```

#### Transfer Learning
- **Initialization**: Pre-trained model from VoxCeleb2 dataset
  - 1 million utterances from 6,112 speakers
  - Trained for speaker identification

#### Training Hyperparameters
- **Epochs**: 196 iterations
- **Model selection**: Best validation performance
- **Hard negative mining**: Applied to focus on difficult CP pairs
- **Batch size**: N_B (not specified in paper)
- **Optimizer**: Not specified (likely Adam or SGD)

#### Performance Metrics
- **AUC**: ~0.95 (Area Under Curve)
- **Detection rate**: ~90% at 10% false positive rate
- **Optimal window size**: 2 seconds
- **Triplet loss outperforms contrastive loss**

---

## 2. Advertisement (Ad) Classifier

### Purpose
Classifies video segments (produced by CP classifier) as either "Advertisement" or "Content".

### Architecture

The Ad classifier consists of two main components:
1. Feature Extraction Model (Audiovisual SlowFast)
2. Temporal Attention Model (TAP, LSTM, or ViT)

---

### 2.1 Feature Extraction Model: Audiovisual SlowFast Network

#### Input Specification
- **Video clip duration**: 4.3 seconds
- **Video frame rate**: 60 fps
- **Audio sampling rate**: 16kHz+ (48kHz tested)
- **Video resolution**: Resized for ResNet processing

#### Three-Pathway Architecture

The network processes video through three parallel pathways:

**Slow Pathway** (High Channels, Low Temporal Resolution):
- Temporal frames: N_TS = 4
- Channels: N_FS = 2048
- Lower sampling rate for spatial detail

**Fast Pathway** (Low Channels, High Temporal Resolution):
- Temporal frames: N_TF = 32 (doubled from original 2→4 seconds)
- Channels: N_FF = 256
- Higher sampling rate for motion

**Audio Pathway** (Highest Temporal Resolution):
- Temporal frames: N_TA = 256 (doubled from original 128)
- Channels: N_FA = 1024
- Uses 2D Log Mel Spectrogram
- Temporally downsampled to match fast pathway: N_TA = N_TF

#### Feature Fusion Process

```
Input: 4.3-second AV clip

Video (Slow Rate)  →  ResNet-50  →  [N_FS, N_TS, N_Y, N_X]
        ↓
Video (Fast Rate)  →  ResNet-50  →  [N_FF, N_TF, N_Y, N_X]
        ↓                               ↓
Audio (Log Mel)    →  ResNet-50  →  [N_FA, N_TA, 1, N_M]
                                        ↓
                        Temporal Downsampling
                                        ↓
                        Cross-Pathway Fusion
                                        ↓
                         XY Avg Pooling
                                        ↓
         Slow[2048,4,1,1]  Fast[256,32,1,1]  Audio[1024,32,1,1]
```

#### Two Fusion Strategies

**Strategy 1: Fuse into Slow Pathway** (Better for LSTM):
- 3D convolution downsamples Fast & Audio to N_TS = 4
- Concatenate: [2048 + 256 + 1024, 4] = [3328, 4]
- Output temporal resolution: N_T = 4

**Strategy 2: Fuse into Fast Pathway**:
- 3D dilated convolution upsamples Slow to N_TF = 32
- Concatenate: [2048 + 256 + 1024, 32] = [3328, 32]
- Output temporal resolution: N_T = 32

---

### 2.2 Temporal Attention Models

After feature fusion, three temporal attention models were tested:

#### Model 1: Temporal Average Pooling (TAP)

**Architecture**:
```
Input: [N_F, N_T]  where N_F = 3328, N_T = 4 or 32
       ↓
Average Pooling (temporal dimension)
       ↓
Reshape: [N_F, 1] → [N_F]
       ↓
Fully Connected Layer
       ↓
Output: [2]  (Ad vs Content logits)
```

**Characteristics**:
- Simplest model
- No learned temporal weights
- Fast inference
- Best Ad recall (97.5%) and Content precision (97.3%)

---

#### Model 2: Bi-directional LSTM

**Architecture**:
```
Input: [N_F, N_T]
       ↓
Bi-directional LSTM (learns temporal dependencies)
       ↓
Extract feature at T_0 (initial timestamp)
       ↓
Fully Connected Layer 1 → [1000]
       ↓
Fully Connected Layer 2 → [256]
       ↓
Fully Connected Layer 3 → [10]
       ↓
Fully Connected Layer 4 → [2]
```

**Characteristics**:
- Learns temporal attention weights
- Best overall balanced performance
- **Slow pathway fusion**: 97.1% correct rate (best end-to-end)
- Best Ad precision (94.9%) and Content recall (94.9%)

---

#### Model 3: Vision Transformer (ViT)

**Architecture**:
```
Input: [N_F, N_T]
       ↓
1×1 Convolution → [768, N_T]
       ↓
Permute → [N_T, 768]
       ↓
Add positional embeddings
       ↓
Prepend learnable class token
       ↓
Transformer Encoder
       ↓
Extract class token representation → [768]
       ↓
Fully Connected Layer → [2]
```

**Characteristics**:
- Most sophisticated model
- Self-attention mechanism
- Expected to outperform LSTM but didn't (likely due to small dataset)
- Performance: 96.7-96.8% correct rate

---

### 2.3 Training Process for Ad Classifier

#### Dataset Preparation
- Same dataset as CP classifier (shared data)
- Down-sampling for balanced classes (Ad vs Content ratio: 0.9-1.1)

#### Spatial-Temporal Sampling (Following AvSlowFast paper)
For each training clip:
- **Temporal sampling**: 5 uniform samples
- **Spatial sampling**: 3 crops (left, center, right)
- **Total augmentation**: 5 × 3 = 15 sub-clips per video

#### Loss Function

**Cross-Entropy Loss**:
```
L_ce = -(1/N_B) × Σ(i=1 to N_B) Σ(c=1 to C) [y_i(c) × log(x_i(c))]

where:
  N_B = batch size
  C = 2 (number of classes: Ad, Content)
  x_i(c) = predicted probability for class c
  y_i(c) = ground truth probability for class c
```

#### Transfer Learning Initialization

**AvSlowFast Feature Extractor**:
- Pre-trained on large-scale video recognition dataset

**Temporal Attention Models**:
- **TAP & LSTM**: Uniform distribution (Kaiming He initialization)
- **ViT**: Pre-trained jx_vit_base_resnet50_384 model (Ross Wightman's library)

#### Training Hyperparameters
- **Epochs**: 196
- **Model selection**: Best validation performance
- **Optimizer**: Not specified (likely Adam)
- **Learning rate**: Not specified

#### Performance Results

**Short Video Clip Classification** (15% test set):
- 1038 Content clips, 1154 Ad clips from 121 unique playbacks

| Model | Fusion | Ad Precision | Ad Recall | Content Precision | Content Recall |
|-------|--------|--------------|-----------|-------------------|----------------|
| TAP | Slow | 91.9% | **97.5%** | **97.3%** | 91.4% |
| LSTM | Slow | **94.9%** | 96.4% | 96.4% | **94.9%** |
| LSTM | Fast | 93.4% | 96.9% | 96.8% | 93.2% |
| ViT | Slow | 94.7% | 96.8% | 96.8% | 94.6% |
| ViT | Fast | 94.8% | 96.8% | 96.8% | 94.6% |

---

### 2.4 Video Segment Level Classification

For segments longer than 4.3 seconds:

#### Multi-Clip Aggregation
1. **Clip extraction**: Crop segment into 4.3s clips with 2-second hop
2. **Classification**: Run Ad classifier on each clip
3. **Aggregation**: Compute ratio of clips labeled as "Ad"

#### Double Threshold Decision (Canny-style)
```
Ratio ≥ T_ad(high) = 0.3333  → Strong Ad
T_ad(low) = 0.1 ≤ Ratio < T_ad(high)  → Weak Ad
Ratio < T_ad(low)  → Non-Ad
```

#### Hysteresis Tracking
- Suppress weak Ad segments NOT connected to strong Ad segments
- Reduces false positives from noisy predictions

#### Long Segment Optimization
- **Threshold**: 85 seconds
- **Rule**: Segments > 85s automatically labeled as "Content"
- **Rationale**: Single ads rarely exceed 60 seconds
- **Benefit**: Massive computational savings (skip heavy AV processing)

---

## 3. Post-Processing Models

Two additional models are used in post-processing:

### 3.1 PANNs Audio Classifier

**Purpose**: Distinguish continuous vs non-continuous audio to prevent over-segmentation

**Architecture**: Pre-trained Audio Neural Network
- Trained on AudioSet dataset
- Classifies audio into 527 voice types

**Categories**:
- **Continuous**: Instruments, fire, water (less likely to have false silent segments)
- **Non-continuous**: Human vocal, birds, reptiles (may have pauses)

**Processing**:
1. Run PANNs on each segment
2. Apply voice/music separation (Librosa) to weaken vocals
3. Classify as continuous/non-continuous
4. Merge adjacent non-continuous segments

### 3.2 Scene Detection (Video-based)

**Purpose**: Catch under-segmentation when audio-based CP classifier misses boundaries

**Algorithm**: PySceneDetect open-source library
- Only run on frames near detected silent segments
- Low computational cost (targeted processing)

---

## 4. Data Requirements

### Training Data
**Part 1: Short Videos (1-5 minutes)**
- Used for training both CP and Ad classifiers
- Sub-divided into 10-30 second clips
- Total: ~12,000 sub-playbacks
  - Train: 5609 Ad + 5367 Content
  - Validation: 1040 Ad + 1154 Content
  - Test: 1038 Ad + 1154 Content

**Part 2: Long Videos (20-120 minutes)**
- Used for end-to-end evaluation only
- 32 movies, 16 live sports (golf)
- 621 Ad segments (4-60 seconds each)
- Manually labeled ground truth

### Data Diversity
- Multiple domains: Movies, TV shows, live sports
- Multiple platforms: Amazon Freevee, Prime Video
- Various content types: Dramas, comedies, sports events

### Preprocessing Requirements

**Audio**:
- Sampling rate: ≥16kHz (48kHz tested)
- Format: WAV (normalized to int16)
- Band-pass filter: [300Hz, 6000Hz]

**Video**:
- Frame rate: ≥60 fps tested
- Format: MP4 or similar
- Resolution: Sufficient for ResNet-50 processing (224×224 typical)

### Annotation Requirements

**For CP Classifier**:
- Binary labels: "Same video" vs "Different video"
- Silent segment timestamps

**For Ad Classifier**:
- Binary labels: "Ad" vs "Content"
- Segment-level annotations

**For End-to-End Evaluation**:
- Precise start/end timestamps for each Ad segment
- Manual verification required

---

## 5. Training Convergence Analysis

### LSTM Model (Slow Pathway) - Best Performer

**Convergence Behavior**:
- Error rate decreases rapidly in first 90 epochs
- Loss value stabilizes after 90 epochs
- **Best validation performance**: Epoch 190
- Continued training up to 196 epochs

**Final Metrics**:
- Validation error rate: ~2-3%
- Loss value: ~0.02-0.03

---

## 6. Key Design Decisions

### Why Audio-Based Segmentation?
1. **Higher temporal resolution**: 16kHz vs 60 fps
2. **Lightweight**: 19.2M data points (20 min audio) vs 7,465M (20 min RGB video)
3. **Computational efficiency**: 6-12× faster than video-based methods
4. **Natural transition markers**: Silence between ads and content

### Why 4.3-Second Clips for Ad Classification?
1. Doubled from original 2.15s in AvSlowFast
2. Human-level accuracy requires ≥3 seconds of context
3. Sufficient for capturing ad characteristics

### Why Siamese Architecture for CP Classifier?
1. Easy to add new training videos without changing network structure
2. Natural formulation for "same/different" classification
3. Leverages transfer learning from speaker verification

### Why Three Temporal Models?
1. **TAP**: Baseline, computationally efficient
2. **LSTM**: Learns temporal dependencies, best overall performance
3. **ViT**: State-of-art architecture, requires larger dataset to excel

---

## 7. Computational Specifications

### Hardware
- **GPU**: NVIDIA GeForce Titan X Pascal (12GB memory)
- **CPU**: Intel Xeon @ 2.30GHz
- **OS**: Ubuntu 18.04

### Runtime Performance (20-minute video)

**Segmentation** (Audio-based):
- I/O: 9.8s (load full audio)
- CPU: 4.7s (silent segment detection)
- **Total**: ~14.5s

**CP Classifier** (per pair of 2s clips):
- I/O: 0.9s
- GPU: 0.013s

**Ad Classifier** (per 4.3s AV clip):
- I/O: 12s
- GPU: 0.04s
- **Most expensive component** (but skipped for long segments >85s)

### Speed Comparison (Segmentation Module)

| Method | Golf (20 min) | Movie (60 min) | Demo (8 min) |
|--------|---------------|----------------|--------------|
| DNN Multi-modal | 379s | 1136s | 42s |
| CV Color-based | 283s | 839s | 30s |
| **This Work (Audio)** | **25s** | **87s** | **5s** |

**Speedup**: 6-12× faster than existing methods

---

## Summary

This framework employs a two-stage approach with sophisticated architectures:

1. **CP Classifier**: Audio-based Siamese ResNet-34 with triplet loss
2. **Ad Classifier**: Audiovisual SlowFast + LSTM temporal attention

Both models leverage transfer learning and achieve 96-97% accuracy on diverse video content. The audio-first approach provides significant computational advantages while maintaining high accuracy.
