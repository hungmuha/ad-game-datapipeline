# NFL Advertisement Detection System

A production-ready system for detecting and segmenting advertisements from NFL broadcasts using deep learning. Based on the paper "A Deep Neural Framework to Detect Individual Advertisement from Videos."

## Project Overview

This system uses a two-stage approach to detect commercial breaks in NFL game broadcasts from Fubo:

1. **Change Point Detection**: Audio-based segmentation to find boundaries between content and ads
2. **Ad Classification**: Audiovisual classifier to distinguish ad segments from gameplay content
3. **Post-Processing**: Refinement to fix segmentation errors and improve accuracy

**Target Performance**:
- Prototype (1-2 months): 75-85% accuracy
- Production: 95%+ accuracy
- Processing speed: 6-12x faster than video-based methods

## Quick Start

### Prerequisites

- Python 3.8+
- AWS account (S3 + EC2 with GPU)
- FFmpeg installed on system
- Access to NFL game recordings (Fubo or similar)
- 10-15 NFL games for prototype (30-45 hours)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ad-game-explainer

# Create conda environment
conda create -n ad-detection python=3.8
conda activate ad-detection

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg sox libsox-fmt-all

# macOS
brew install ffmpeg sox
```

### Project Structure

```
ad-game-explainer/
├── CLAUDE.md                      # AI development guide
├── README.md                      # Project overview (this file)
├── pipeline/                      # Data acquisition & preprocessing
│   ├── 01-fubo-scraper/          # Extract stream URLs from Fubo TV
│   ├── 02-download-games/        # Download games to MP4
│   ├── 03-process-videos/        # Detect silent segments
│   └── README.md                 # Pipeline documentation
├── scripts/                       # Orchestration scripts
│   ├── 01_scrape_fubo.sh         # Run fubo scraper
│   ├── 02_download_games.sh      # Run game downloader
│   ├── 03_process_videos.sh      # Run video processor
│   └── run_pipeline.sh           # Run complete pipeline
├── data/                          # Dataset (not in git)
│   ├── raw_videos/               # Downloaded MP4 files
│   ├── processed/                # EDL, CSV, WAV outputs
│   ├── annotations/              # Manual labels (JSON)
│   └── short_clips/              # Training clips (10-30s)
│       ├── ads/
│       └── content/
├── src/                           # Training & inference code
│   ├── models/                   # Neural network architectures
│   ├── datasets/                 # PyTorch datasets
│   ├── pipeline/                 # End-to-end inference
│   └── evaluation/               # Metrics and evaluation
├── docs/                          # Documentation
│   ├── GETTING_STARTED.md        # Quick start guide
│   ├── DATA_PIPELINE.md          # Complete workflow
│   ├── ANNOTATION_WORKFLOW.md    # Annotation process
│   ├── TRAINING_APPROACH.md      # Training methodology
│   ├── PROJECT_STATUS.md         # Progress tracking
│   └── Technical_Info/           # Research papers
├── pretrained/                    # Pre-trained model weights
├── models/                        # Trained model checkpoints
├── outputs/                       # Inference results
└── notebooks/                     # Jupyter notebooks
```

## Development Roadmap

**See [NFL_PROJECT_PLAN.md](docs/NFL_PROJECT_PLAN.md) for detailed week-by-week plan**

### Phase 1: Setup & Data Collection (Week 1)
- [x] Setup development environment
- [x] Create download scripts (m3u8 → mp4)
- [x] Create S3 upload scripts
- [ ] Setup AWS (S3 bucket + EC2 instance)
- [ ] Download 10-15 NFL games from Fubo
- [ ] Upload to S3

### Phase 2: Data Annotation (Week 2)
- [ ] Annotate 10-15 NFL games (mark ad breaks)
- [ ] Extract short clips (10-30s) from recordings
- [ ] Create train/val/test splits (70/15/15)
- [ ] Target: 1000-1500 labeled clips for prototype

### Phase 3: Change Point Classifier (Week 4-5)
- [ ] Implement audio preprocessing (bandpass filter, silence detection)
- [ ] Setup Siamese ResNet-34 architecture
- [ ] Download VoxCeleb2 pre-trained weights
- [ ] Train with triplet loss
- [ ] Evaluate: Target 85%+ accuracy

### Phase 4: Ad Classifier (Week 6-7)
- [ ] Implement audio-only ResNet-50 (simplified for prototype)
- [ ] Add bi-directional LSTM temporal model
- [ ] Train with cross-entropy loss
- [ ] Evaluate: Target 85%+ accuracy

### Phase 5: End-to-End Pipeline (Week 8)
- [ ] Build segmentation pipeline
- [ ] Build classification pipeline
- [ ] Integrate both stages
- [ ] Test on long videos (20-60 min)
- [ ] Compute end-to-end metrics

See [NFL_PROJECT_PLAN.md](docs/NFL_PROJECT_PLAN.md) for detailed week-by-week tasks specific to NFL broadcasts.

## Usage

### Quick Start

```bash
# Check prerequisites
./scripts/check_prerequisites.sh

# Run complete pipeline
./scripts/run_pipeline.sh
```

### Data Collection & Processing

```bash
# Step 1: Scrape Fubo TV stream URLs
./scripts/01_scrape_fubo.sh --headless

# Step 2: Download NFL games
./scripts/02_download_games.sh

# Step 3: Process videos (detect silent segments)
./scripts/03_process_videos.sh

# Step 4: Generate annotations (after DaVinci Resolve review)
./scripts/04_create_annotations.sh
```

See `scripts/README.md` for detailed usage and options.

### Annotation (Manual Review)

After processing videos (Step 3), annotate in DaVinci Resolve:

1. Import MP4 file from `data/raw_videos/`
2. Import EDL file from `data/processed/edl/`
3. Review markers: Red = ad boundaries, other colors = non-boundaries
4. Export adjusted EDL to `data/processed/manifests/{video_name}/`
5. Run `./scripts/04_create_annotations.sh`

See `docs/ANNOTATION_WORKFLOW.md` for detailed instructions.

### Training

```bash
# Train Change Point Classifier
python src/train_cp_classifier.py \
  --data data/short_clips/ \
  --pretrained pretrained/voxceleb_resnet34.pth \
  --epochs 50 \
  --batch-size 32

# Train Ad Classifier
python src/train_ad_classifier.py \
  --data data/short_clips/ \
  --epochs 50 \
  --batch-size 16
```

### Inference

```bash
# Process single video
python src/pipeline/detect_ads.py \
  --input data/raw_recordings/test_stream.mp4 \
  --output outputs/test_stream_results.json

# Batch processing
python src/pipeline/batch_detect.py \
  --input data/raw_recordings/ \
  --output outputs/
```

## Key Differences from Paper

This implementation is optimized for NFL broadcasts:

**Domain Match**:
- **Target**: NFL broadcasts from Fubo (paper tested on sports/golf - excellent match!)
- **Ad format**: Traditional TV commercial breaks (same as paper)
- **Advantage**: NFL has predictable ad breaks (timeouts, quarters, two-minute warning)
- **Advantage**: Clear audio transitions (crowd noise → silence → ads)

**Prototype Simplifications**:
- **Audio-only** ad classifier (vs audiovisual SlowFast in paper)
- **Lighter models**: ResNet-18 option (vs ResNet-50)
- **Skip post-processing** initially (PANNs, aggressive re-segmentation)
- **Shorter training**: 50 epochs (vs 196 in paper)
- **Smaller dataset**: 500-1000 clips (vs 12,000 in paper)

**Production Path** (Post-Prototype):
- Scale dataset to 12,000+ clips from 30-40 games
- Add full audiovisual SlowFast network
- Implement PANNs post-processing
- Train for 196 epochs
- Expected: 95%+ accuracy (matching or exceeding paper due to NFL's clearer patterns)

## Performance Benchmarks

### Prototype Targets (1-2 months)
- **Accuracy**: 75-85% F1 score
- **Speed**: <5 min for 20-min video
- **False Positive Rate**: <20%

### Production Targets (post-prototype)
- **Accuracy**: 95%+ F1 score
- **Speed**: <1 min for 20-min video
- **False Positive Rate**: <5%

## Resources

### Documentation
- [CLAUDE.md](CLAUDE.md) - AI assistant guidance for development
- [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) - Complete 4-repository data pipeline workflow
- [ANNOTATION_WORKFLOW.md](docs/ANNOTATION_WORKFLOW.md) - **EDL to annotation JSON workflow** (DaVinci Resolve + scripts)
- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Current progress tracking and next steps
- [TRAINING_APPROACH.md](docs/TRAINING_APPROACH.md) - CP classifier training methodology (strong supervision)
- [GETTING_STARTED.md](docs/GETTING_STARTED.md) - **Start here!** Quick overview and next steps
- [NFL_PROJECT_PLAN.md](docs/NFL_PROJECT_PLAN.md) - Detailed implementation plan for NFL broadcasts
- [Technical_Info/](docs/Technical_Info/) - Research papers and detailed explanations
- [.claude/](.claude/) - Project context for Claude Code sessions

### Pre-trained Models
- [VoxCeleb2 ResNet-34](https://github.com/clovaai/voxceleb_trainer) - Speaker verification
- [SlowFast Networks](https://github.com/facebookresearch/SlowFast) - Video recognition
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) - Audio classification

### Related Papers
- Original paper: `Technical_Info/a-deep-neural-framework-to-detect-individual-advertisement-ad-from-videos.pdf`
- SlowFast: https://arxiv.org/abs/1812.03982
- Vision Transformer: https://arxiv.org/abs/2010.11929

## Contributing

This is a production project. Focus areas:
1. NFL-specific optimizations (timeout detection, quarter transitions)
2. Real-time processing capabilities for live games
3. Multi-network support (CBS, NBC, FOX, ESPN)
4. Ad type classification (automotive, beer, tech, etc.)
5. Multi-sport support (NBA, MLB, NHL)

## License

[Specify license]

## Acknowledgments

Based on research from:
- "A Deep Neural Framework to Detect Individual Advertisement from Videos"
- VoxCeleb2 speaker verification framework
- Facebook Research SlowFast networks
