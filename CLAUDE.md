# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a non-intrusive, non-reference deep learning framework for detecting and segmenting individual advertisements from NFL game broadcasts. The system targets 97% accuracy using a two-stage approach:

1. **Stage 1**: Audio-based segmentation to detect change points between content/ads (6-12x faster than video-based methods)
2. **Stage 2**: Audiovisual classification to distinguish ads from content
3. **Stage 3**: Post-processing to fix segmentation errors

**Repository Structure**: This is a unified repository containing the complete end-to-end pipeline from data acquisition to model training. See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for complete workflow:
- **pipeline/01-fubo-scraper**: Extracts stream URLs from Fubo TV
- **pipeline/02-download-games**: Downloads games to local MP4 files
- **pipeline/03-process-videos**: Detects change point candidates (silence segments)
- **src/**: Training models and inference pipeline

**Current Status** (2026-01-30): Repository consolidated. Data pipeline infrastructure complete (35% overall progress). Next: Run pipeline/03-process-videos on downloaded games, then manual annotation.

## Documentation Management Principles

**CRITICAL: Avoid Documentation Sprawl**

- **NEVER EVER create new markdown (.md) documentation files** - This is a HARD RULE
- **ALWAYS update existing documentation** - No exceptions unless user explicitly says "create a new file called X.md"
- When you feel tempted to create a summary/improvements/fix document, STOP and ask yourself: "Which existing doc should this update?"
- If uncertain which file to update, ask the user first - DO NOT create a new file as a "temporary solution"
- Creating files like "PIPELINE_IMPROVEMENTS.md", "AUDIO_REUSE_FIX.md", "CHANGES.md", "UPDATES.md" is WRONG - update existing docs instead

**Existing Documentation Structure**:
- **docs/GETTING_STARTED.md** - Quick overview, current status, next steps
- **docs/DATA_PIPELINE.md** - Complete pipeline workflow
- **docs/ANNOTATION_WORKFLOW.md** - EDL to annotation JSON process
- **docs/TRAINING_APPROACH.md** - CP classifier training methodology
- **docs/PROJECT_STATUS.md** - Progress tracking and weekly updates
- **docs/NFL_PROJECT_PLAN.md** - Detailed implementation plan
- **docs/Technical_Info/** - Research papers and detailed explanations (read-only reference)
- **CLAUDE.md** (this file, root) - Technical reference and AI guidance
- **README.md** (root) - Project overview and installation
- **pipeline/README.md** - Data pipeline component documentation

**When to Update Each File**:
- Progress updates → docs/PROJECT_STATUS.md
- Workflow changes → docs/DATA_PIPELINE.md or docs/ANNOTATION_WORKFLOW.md
- Training methodology → docs/TRAINING_APPROACH.md
- Quick reference for next steps → docs/GETTING_STARTED.md
- Architecture/technical details → CLAUDE.md
- User-facing overview → README.md
- Pipeline components → pipeline/README.md

## Key Architecture Principles

### Two-Stage Pipeline Architecture

**Change Point (CP) Classifier**:
- Siamese ResNet-34 architecture trained with triplet loss
- Processes 2-second audio windows before/after silent segments
- Outputs Euclidean distance; threshold determines if boundary is a change point
- Pre-trained on VoxCeleb2 speaker verification dataset for transfer learning
- Input: Log Mel Spectrograms (16kHz+ audio, band-pass filtered [300Hz, 6000Hz])
- Output: 512-dimensional feature vectors

**Ad Classifier**:
- Audiovisual SlowFast Network with three parallel pathways:
  - Slow pathway: 4 frames, 2048 channels (spatial detail)
  - Fast pathway: 32 frames, 256 channels (motion detail)
  - Audio pathway: 256→32 frames (downsampled), 1024 channels
- Feature fusion produces [3328, N_T] tensor where N_T = 4 or 32
- Temporal attention models: TAP (baseline), LSTM (best balanced), or ViT
- Processes 4.3-second clips with 2-second hop
- Long segment optimization: Auto-label segments >85s as "Content" (saves computation)

### Critical Design Decisions

**Why Audio-First Segmentation**:
- Higher temporal resolution: 16kHz (16k samples/sec) vs 60fps video
- 388x lighter data: 20 min audio = 19.2M points vs 7,465M for RGB video
- Silent transitions naturally mark ad boundaries
- 6-12x faster than video-based methods

**Why 4.3-Second Clips**:
- Doubled from original 2.15s in AvSlowFast
- Sufficient context for human-level ad recognition
- Balances accuracy with computational cost

**Hierarchical Processing Strategy**:
1. Coarse segmentation (audio, fast) → identifies candidate boundaries
2. Fine classification (audiovisual, slower) → distinguishes ad from content
3. Targeted refinement (post-processing) → fixes errors only where needed

### Multi-Modal Fusion Strategy

**Cross-Pathway Fusion**:
- Audio features fused into Fast pathway (temporal alignment)
- Fast features fused into Slow pathway (spatial-temporal integration)
- Two fusion strategies tested:
  - Slow fusion (N_T=4): Better for LSTM, 97.1% correct rate
  - Fast fusion (N_T=32): Alternative for different temporal models

**Double-Threshold Classification** (Canny-inspired):
- Strong Ad: ≥33.33% of clips classified as ad
- Weak Ad: 10-33.33% of clips classified as ad (requires strong neighbor via hysteresis)
- Content: <10% of clips classified as ad

## Data Requirements

### Training Dataset Structure
```
data/
├── raw_videos/           # Long playbacks (20-120 min)
│   ├── movies/
│   ├── sports/
│   └── tv_shows/
├── short_clips/          # Training clips (10-30s)
│   ├── ads/             # 5609 train, 1040 val, 1038 test
│   └── content/         # 5367 train, 1154 val, 1154 test
└── annotations/          # Ground truth timestamps
    └── *.json
```

### Dataset Split
- Train: 70% (5609 Ad, 5367 Content)
- Validation: 15% (1040 Ad, 1154 Content)
- Test: 15% (1038 Ad, 1154 Content)
- Classes balanced via down-sampling (Ad/Content ratio: 0.9-1.1)

## Training Processes

### CP Classifier Training
- Loss: Triplet loss (margin=1.0) outperforms contrastive loss
- Transfer learning: Initialize from VoxCeleb2 ResNet-34
- Hard negative mining applied
- Epochs: 196 (select best validation performance)
- Target metrics: AUC ≥0.90, detection rate ≥85% at 10% FPR

**Training Data Approach (Strong Supervision - Following Paper)**:
- **Positive pairs (non-CP)**: Extract 2s before/after non-CP silent segments (pauses within same content)
- **Negative pairs (CP)**: Extract 2s before/after TRUE CP boundaries (content↔ad transitions)
- **Annotation Required**: Each silent segment must be labeled (CP vs non-CP) in DaVinci Resolve
- **Data Structure**: Matches inference task - comparing audio before/after silent segments

**Critical Insight**: The model compares audio before vs after silent segments at inference.
Training data MUST have the same structure with annotated labels for each silent segment.

### Ad Classifier Training
- Loss: Cross-entropy
- Transfer learning: AvSlowFast feature extractor, ViT (jx_vit_base_resnet50_384)
- Spatial-temporal augmentation: 5 temporal samples × 3 spatial crops = 15 per clip
- Epochs: 196 (select best validation)
- Best model: LSTM + Slow fusion (94.9% Ad precision, 96.4% Ad recall)

## Post-Processing Components

**PANNs Audio Classifier**:
- Distinguishes continuous (instruments, ambient) vs non-continuous (vocals, speech) audio
- Prevents over-segmentation in vocal-heavy scenes
- Uses voice/music separation (Librosa) for mixed audio
- 527 AudioSet categories grouped into continuous/non-continuous

**Video Scene Detection**:
- PySceneDetect for backup validation
- Only runs on frames near rejected silent segments
- Catches visual-only transitions without audio silence

**Under-Segmentation Fixes**:
- Ad segments: Re-run with aggressive thresholds (2ms min silence vs 10ms)
- Content segments >85s: Check head/tail for embedded ads
- Short segment threshold: 4s (post-processing) vs 8s (initial)

## Performance Expectations

### Classification Accuracy (Short Clips)
- LSTM + Slow: 94.9% Ad precision, 96.4% Ad recall
- TAP + Slow: 97.5% Ad recall (highest), 91.9% Ad precision
- ViT models: 96.7-96.8% correct rate (limited by dataset size)

### End-to-End Metrics (Long Videos)
- Overall accuracy: 97%
- Correct rate: ≥95%
- Over-segmentation: ≤1%
- Under-segmentation: ≤1%
- False positive: ≤2.5%

### Runtime Performance (20-min video)
- Audio loading + silent detection: ~14.5s
- CP classification: 0.9s I/O + 0.013s GPU per pair
- Ad classification: 12s I/O + 0.04s GPU per 4.3s clip
- Total: 6-12x faster than DNN multi-modal or CV color-based methods

## Common Development Patterns

### Pipeline Execution Order
1. Audio preprocessing (normalize int16, bandpass [300Hz, 6000Hz])
2. Silent segment detection (min 10ms, volume threshold 4)
3. CP classification on each silent segment (2s windows)
4. Video scene detection on rejected CPs (backup validation)
5. Segment cleanup (merge/discard <8s segments)
6. Ad classification (skip >85s segments, 4.3s clips with 2s hop)
7. Post-processing (fix under-segmentation with PANNs + aggressive re-segmentation)

### Feature Extraction Flow
```
Audio (wav) → Bandpass filter → Log Mel Spectrogram → ResNet-34 → 512-dim features
Video → Slow/Fast sampling → ResNet-50 → [2048, 4] / [256, 32]
Audio → Log Mel → ResNet-50 → [1024, 256] → downsample → [1024, 32]
Fusion → [3328, N_T] → LSTM/TAP/ViT → [2] logits
```

### Threshold Parameters
- CP distance threshold: T_CP (optimized during validation)
- Silent segment volume: 4 (normalized int16)
- Silent segment min duration: 10ms (initial), 2ms (post-processing)
- Short segment: 8s (initial cleanup), 4s (post-processing)
- Long segment: 85s (auto-label as content)
- Ad ratio high: 0.3333 (strong ad)
- Ad ratio low: 0.1 (weak ad, needs hysteresis)

## Hardware Requirements

**Minimum**:
- GPU: NVIDIA GPU with ≥12GB VRAM (RTX 3090, V100, A100)
- CPU: Multi-core processor
- RAM: ≥32GB
- Storage: ≥500GB SSD

**Tested Configuration**:
- GPU: NVIDIA GeForce Titan X Pascal (12GB)
- CPU: Intel Xeon @ 2.30GHz
- OS: Ubuntu 18.04

## Core Dependencies

**Deep Learning**:
- PyTorch 1.12.0+ with CUDA support
- torchvision, torchaudio
- timm (PyTorch Image Models for ViT)
- torchvggish (audio features)

**Audio/Video Processing**:
- librosa 0.9.2+ (audio analysis, voice/music separation)
- opencv-python (video I/O)
- ffmpeg-python (video/audio extraction)
- pydub, soundfile (audio utilities)
- pyscenedetect (scene detection backup)

**Pre-trained Models**:
- VoxCeleb2 ResNet-34: https://github.com/clovaai/voxceleb_trainer
- SlowFast Networks: https://github.com/facebookresearch/SlowFast
- PANNs (Cnn14): https://github.com/qiuqiangkong/audioset_tagging_cnn
- ViT (jx_vit_base_resnet50_384): Auto-download via timm

## Important Constraints and Trade-offs

**Silent Transition Assumption**:
- Method assumes brief silence marks ad boundaries
- May miss seamless transitions (rare in practice)
- Video scene detection provides backup but adds computational cost

**Short Ad Challenge**:
- Ads <4s may be discarded in initial cleanup (8s threshold)
- Requires aggressive post-processing to recover
- Trade-off: initial cleanup improves classification accuracy by providing more context

**Training Data Bottleneck**:
- ViT underperforms LSTM due to small dataset (~12k clips)
- Larger datasets needed for transformer architectures to excel
- Transfer learning mitigates but doesn't eliminate this limitation

**Computational Bottleneck**:
- Ad classifier (audiovisual processing) is most expensive: 12s I/O + 0.04s GPU per clip
- Mitigated by long segment skipping (>85s auto-labeled)
- Further optimization: TorchScript export, batch processing, skip temporal models for very short segments

## Troubleshooting Guide

**Low CP classifier accuracy**: Increase training data, adjust window size (test 1s, 3s), try contrastive loss, verify band-pass filter applied

**Low Ad classifier accuracy**: Check temporal model selection, increase clip duration to 5-6s, verify spatial-temporal augmentation, ensure class balance

**Over-segmentation**: Increase silent segment threshold (10ms→20ms), apply PANNs filtering more aggressively, merge continuous audio segments

**Under-segmentation**: Lower silent segment threshold (10ms→5ms), enable video scene detection for all segments (expensive), reduce short segment cleanup threshold

**Slow inference**: Use TorchScript export, implement batch processing for clips, increase long segment threshold (85s→100s), use TAP instead of LSTM

**False positives on trailers/credits**: Add temporal position features (beginning/end of video), train dedicated trailer classifier, use longer context windows

## Reference Documentation

**Original Paper**: Technical_Info/a-deep-neural-framework-to-detect-individual-advertisement-ad-from-videos.pdf

**Method Explanation**: Technical_Info/method_explanation.md - Complete end-to-end pipeline walkthrough

**Model Architecture**: Technical_Info/model_and_training.md - Detailed network specifications and training procedures

**Implementation Plan**: Technical_Info/implementation_gameplan.md - Step-by-step development roadmap with code examples

**Related Papers**:
- SlowFast Networks: https://arxiv.org/abs/1812.03982
- Vision Transformer: https://arxiv.org/abs/2010.11929
- VoxCeleb2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
