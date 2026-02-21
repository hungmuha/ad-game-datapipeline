# NFL Ad Detection Project Plan

**Domain**: NFL sports broadcasts from Fubo
**Data Source**: m3u8 streams → ffmpeg → mp4 files
**Infrastructure**: S3 + EC2 (g4dn.xlarge)
**Timeline**: 1-2 months for production-ready system

---

## Data Advantage: You're in an Excellent Position!

**What you have**:
- 150+ NFL games from 2025 season
- ~450-525 hours of broadcast footage
- Original TV commercials embedded
- Professional broadcast quality

**What you need**:
- Prototype: 10-15 games (30-45 hours)
- Production: 30-40 games (90-120 hours)
- Testing: 5-10 games (held out)

**Surplus**: 3-4x more data than required! This enables:
- Multi-network training (CBS, NBC, FOX, ESPN)
- Team/stadium diversity testing
- Potential research publication/benchmark dataset

---

## Phase 0: Data Pipeline Infrastructure ✅ COMPLETE

**Multi-Repository Setup** (Completed over Weeks 1-4)

This project uses a **multi-repository architecture** with 3 upstream data pipeline repos feeding into this training repo. See [DATA_PIPELINE.md](DATA_PIPELINE.md) for complete workflow.

### Repository 1: fubo-scraper ✅
**Location**: `/Users/hungmuhamath/projects/GitHub/fubo-scraper/`
**Purpose**: Extract stream URLs from Fubo TV NFL recordings
**Technology**: Python + Selenium-Wire
**Status**: Operational, 150+ games accessible

**Key Features**:
- Automated login and navigation
- Network traffic interception for m3u8 URLs
- CSV export with metadata (title, air_date, network, stream_url)
- Timestamped outputs for tracking

**Output**: `output/fubo_recordings_YYYYMMDD_HHMMSS.csv`

### Repository 2: GetGamesToLocal ✅
**Location**: `/Users/hungmuhamath/projects/GitHub/GetGamesToLocal/`
**Purpose**: Download m3u8 streams to local MP4 files
**Technology**: Node.js + FFmpeg
**Status**: Operational, multiple games downloaded

**Key Features**:
- Reads CSV from fubo-scraper
- Deduplicates using master.csv tracking
- FFmpeg copy codec (fast, no transcoding)
- DRM detection (skips .mpd streams)
- Absolute file paths for portability

**Output**:
- `videos/*.mp4` (downloaded games)
- `master.csv` (tracking database with status, file paths, timestamps)

### Repository 3: CPPreData ✅
**Location**: `/Users/hungmuhamath/projects/GitHub/CPPreData/`
**Purpose**: Detect silence segments (change point candidates)
**Technology**: Python + Pydub + SciPy + FFmpeg
**Status**: Setup complete, tested on 1 game

**Key Features**:
- Audio extraction (mono, 16kHz WAV)
- Band-pass filtering (300-6000 Hz)
- Silence detection with configurable thresholds
- Time extension (±2.2s context window)
- CSV manifests with HMS timestamps for manual review

**Output**:
- `audio/*.wav` (extracted audio)
- `manifests/*.csv` (CP candidate lists with blank `CP` column for annotation)

**Example**: Cowboys vs Eagles game → 240+ detected silence segments

---

## Phase 0 (Original): Data Pipeline Setup (Week 1)

### Step 1: FFmpeg Download Script

Create `scripts/download_fubo.py`:

```python
#!/usr/bin/env python3
"""
Download Fubo m3u8 streams to mp4 using ffmpeg
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

def download_stream(m3u8_url, output_path, metadata=None):
    """
    Download m3u8 stream to mp4

    Args:
        m3u8_url: URL to m3u8 playlist
        output_path: Path to save mp4 file
        metadata: Dict with game info (team1, team2, date, network)
    """

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command for m3u8 → mp4
    cmd = [
        'ffmpeg',
        '-i', m3u8_url,
        '-c', 'copy',  # Copy streams without re-encoding (fast)
        '-bsf:a', 'aac_adtstoasc',  # Fix audio sync issues
        '-y',  # Overwrite output file if exists
        output_path
    ]

    print(f"Downloading: {metadata.get('description', m3u8_url)}")
    print(f"Output: {output_path}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Download complete: {output_path}")

        # Save metadata
        if metadata:
            metadata_path = output_path.replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e.stderr}")
        return False

def download_nfl_game(m3u8_url, team1, team2, date, network, output_dir='data/raw_recordings'):
    """
    Download NFL game with automatic naming

    Example:
        download_nfl_game(
            m3u8_url='https://...m3u8',
            team1='chiefs',
            team2='bills',
            date='2025-01-21',
            network='cbs'
        )
    """

    # Generate filename: nfl_chiefs_bills_2025-01-21_cbs.mp4
    filename = f"nfl_{team1}_{team2}_{date}_{network}.mp4"
    output_path = Path(output_dir) / filename

    metadata = {
        'sport': 'nfl',
        'team1': team1,
        'team2': team2,
        'date': date,
        'network': network,
        'source': 'fubo',
        'download_time': datetime.now().isoformat(),
        'description': f"NFL: {team1.upper()} vs {team2.upper()} ({date})"
    }

    return download_stream(m3u8_url, str(output_path), metadata)

# Example usage
if __name__ == '__main__':
    # Example: Download single game
    download_nfl_game(
        m3u8_url='YOUR_M3U8_URL_HERE',
        team1='chiefs',
        team2='bills',
        date='2025-01-21',
        network='cbs'
    )
```

### Step 2: Batch Download Script

Create `scripts/batch_download.py`:

```python
#!/usr/bin/env python3
"""
Batch download multiple games from a CSV
"""

import pandas as pd
from download_fubo import download_nfl_game
import time

def batch_download(csv_path):
    """
    Download multiple games from CSV

    CSV format:
    m3u8_url,team1,team2,date,network
    https://...m3u8,chiefs,bills,2025-01-21,cbs
    https://...m3u8,ravens,steelers,2025-01-22,nbc
    """

    df = pd.read_csv(csv_path)

    print(f"Found {len(df)} games to download")

    results = []
    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Downloading game...")

        success = download_nfl_game(
            m3u8_url=row['m3u8_url'],
            team1=row['team1'],
            team2=row['team2'],
            date=row['date'],
            network=row['network']
        )

        results.append({
            'game': f"{row['team1']} vs {row['team2']}",
            'success': success
        })

        # Rate limiting (be nice to Fubo)
        time.sleep(5)

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*50}")
    print(f"Download complete: {successful}/{len(results)} successful")
    print(f"{'='*50}")

    return results

if __name__ == '__main__':
    # Create example CSV
    # games.csv:
    # m3u8_url,team1,team2,date,network
    # https://...m3u8,chiefs,bills,2025-01-21,cbs

    batch_download('games.csv')
```

### Step 3: Upload to S3

Create `scripts/upload_to_s3.py`:

```python
#!/usr/bin/env python3
"""
Upload downloaded games to S3
"""

import boto3
from pathlib import Path
from tqdm import tqdm
import json

def upload_to_s3(local_dir, bucket_name, s3_prefix='raw_recordings'):
    """
    Upload all mp4 files from local directory to S3

    Args:
        local_dir: Path to local data/raw_recordings/
        bucket_name: S3 bucket name (e.g., 'nfl-ad-detection')
        s3_prefix: Folder in S3 bucket
    """

    s3 = boto3.client('s3')
    local_path = Path(local_dir)

    # Find all mp4 and json files
    files = list(local_path.glob('*.mp4')) + list(local_path.glob('*.json'))

    print(f"Found {len(files)} files to upload")

    for file_path in tqdm(files, desc="Uploading"):
        s3_key = f"{s3_prefix}/{file_path.name}"

        # Upload with progress
        s3.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs={'StorageClass': 'STANDARD_IA'}  # Cheaper storage
        )

    print(f"✓ Upload complete: {len(files)} files to s3://{bucket_name}/{s3_prefix}/")

def sync_with_s3(local_dir, bucket_name, s3_prefix='raw_recordings'):
    """
    Sync local directory with S3 (only upload new/changed files)
    """
    import subprocess

    # Use AWS CLI for efficient sync
    cmd = [
        'aws', 's3', 'sync',
        local_dir,
        f's3://{bucket_name}/{s3_prefix}/',
        '--storage-class', 'STANDARD_IA',
        '--exclude', '*.DS_Store'
    ]

    subprocess.run(cmd, check=True)
    print(f"✓ Sync complete")

if __name__ == '__main__':
    # Upload to S3
    upload_to_s3(
        local_dir='data/raw_recordings',
        bucket_name='your-nfl-ad-detection-bucket'
    )
```

---

## Infrastructure Setup

### S3 Bucket Structure

```
s3://nfl-ad-detection/
├── raw_recordings/                # Raw game downloads (450-525 hours)
│   ├── nfl_chiefs_bills_2025-01-21_cbs.mp4
│   ├── nfl_chiefs_bills_2025-01-21_cbs_metadata.json
│   └── ...
├── annotations/                   # Manual annotations (JSON)
│   ├── nfl_chiefs_bills_2025-01-21_cbs.json
│   └── ...
├── short_clips/                   # Extracted 10-30s clips
│   ├── ads/
│   └── content/
├── models/                        # Trained model checkpoints
│   ├── cp_classifier_best.pth
│   └── ad_classifier_lstm_best.pth
└── outputs/                       # Inference results
    └── ...
```

### EC2 Instance Setup

**Instance Type**: `g4dn.xlarge`
- GPU: NVIDIA T4 (16GB)
- vCPU: 4
- RAM: 16GB
- Cost: $0.526/hour on-demand (~$1.50/hr with realistic usage)

**Setup Script** (`scripts/setup_ec2.sh`):

```bash
#!/bin/bash
# Run this on fresh EC2 instance (Ubuntu 20.04 Deep Learning AMI)

# Update system
sudo apt-get update
sudo apt-get install -y ffmpeg sox libsox-fmt-all

# Setup conda environment
source activate pytorch

# Clone repository
git clone https://github.com/YOUR_USERNAME/ad-game-explainer.git
cd ad-game-explainer

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials (if not using IAM role)
aws configure

# Create data directories
mkdir -p data/{raw_recordings,annotations,short_clips/{ads,content}}
mkdir -p pretrained models outputs logs

# Download pre-trained models
cd pretrained
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_v2_ap.model
cd ..

# Mount S3 bucket (optional, for direct access)
# sudo apt-get install -y s3fs
# s3fs nfl-ad-detection ~/s3_mount -o iam_role=auto

echo "✓ EC2 setup complete!"
```

---

## Data Selection Strategy

**For Prototype** (Week 2-3):

Select 10-15 games with diversity:

```python
# Recommended selection for balanced dataset
PROTOTYPE_GAMES = {
    # Different networks
    'cbs': 3,      # CBS has different ad patterns
    'nbc': 3,      # NBC Sunday Night Football
    'fox': 3,      # FOX regional games
    'espn': 2,     # Monday Night Football

    # Different game types
    'close_game': 5,      # Competitive (more timeouts = more ads)
    'blowout': 3,         # One-sided (fewer timeouts)
    'playoff': 2,         # Higher stakes (different ads)

    # Different times
    'early_season': 3,
    'mid_season': 4,
    'late_season': 3,
}

# Target: 10-15 games = 30-45 hours = 1000-1500 clips
```

**Game Selection Criteria**:
1. **Network diversity**: CBS, NBC, FOX, ESPN (different ad insertion patterns)
2. **Game flow**: Mix of close games (many timeouts) and blowouts
3. **Season timing**: Early/mid/late season (different advertiser campaigns)
4. **Teams**: Mix of popular (Chiefs, Bills) and smaller market teams

---

## NFL-Specific Considerations

### Ad Break Patterns in NFL

NFL has **predictable commercial breaks**:
1. **Between quarters**: ~3-5 minutes
2. **Timeouts**: ~30-90 seconds
3. **Two-minute warning**: ~3 minutes
4. **Challenges**: ~30-60 seconds
5. **Halftime**: ~12-15 minutes (exclude from training - too long)
6. **Injury timeouts**: Variable

**Total ads per game**: ~60-80 commercial breaks, ~15-25 minutes of ads

### Audio Characteristics

NFL broadcasts have distinct audio patterns:
- **Content**: Crowd noise, commentary, referee whistles, player sounds
- **Transition to ad**: Crowd fade, "We'll be right back" announcement, silence (0.1-0.5s)
- **Ad**: Music, voiceover, product sounds
- **Transition back**: Network theme music, "Welcome back to..."

**This is IDEAL for the CP Classifier** - clear audio transitions!

### Challenges Specific to NFL

1. **Replays during breaks**: Sometimes show replays instead of cutting to ads immediately
2. **Network branding**: CBS eye logo, NBC peacock (not ads, but similar transitions)
3. **Halftime shows**: Very long segments (12-15 min) - may need special handling
4. **Stadium sounds**: Vary by venue (dome vs outdoor)

**Solutions**:
- Train on multiple networks to handle different patterns
- Use video scene detection for replays vs ads
- Filter out halftime segments (>10 min) as special case

---

## Updated Timeline: NFL-Optimized (Revised 2026-01-25)

### Weeks 1-4: Data Pipeline Infrastructure ✅ COMPLETE
- [x] Project structure created
- [x] Built fubo-scraper (stream URL extraction)
- [x] Built GetGamesToLocal (MP4 download with tracking)
- [x] Built CPPreData (silence detection pipeline)
- [x] Tested complete pipeline on 1 game
- [x] Downloaded multiple NFL games to local storage
- [x] Created comprehensive documentation (DATA_PIPELINE.md)

### Week 5 (Current): Data Preparation 🔄 IN PROGRESS
- [x] Reviewed all repos and documented workflow
- [x] Updated project status (15% → 30%)
- [ ] **NEXT**: Run CPPreData on all downloaded games
- [ ] **NEXT**: Select 3 games for test annotation
- [ ] **NEXT**: Manual annotation of 3 games (Excel/Sheets workflow)
- [ ] **NEXT**: Document annotation time/challenges

### Week 6: Scale Annotation
- [ ] Select 10-15 games for prototype dataset
- [ ] Annotate remaining 7-12 games
- [ ] Create train/val/test splits (70/15/15)
- [ ] Validate annotation quality (random review)
- [ ] **Target**: 1000-1500 annotated CP pairs

### Week 7-8: CP Classifier
- [ ] Install PyTorch environment (conda + requirements.txt)
- [ ] Implement audio preprocessing module
- [ ] Download VoxCeleb2 pre-trained weights
- [ ] Implement CP classifier (Siamese ResNet-34)
- [ ] Train with triplet loss (30-50 epochs)
- [ ] Evaluate: Target 90%+ AUC (NFL should be easier than movies)

### Week 9-10: Ad Classifier
- [ ] Implement audio-only classifier (initial version)
- [ ] Implement LSTM temporal model
- [ ] Train for 50 epochs
- [ ] Evaluate: Target 90%+ precision/recall

### Week 11: End-to-End Pipeline
- [ ] Integrate CP + Ad classifier
- [ ] Test on 3-5 full games (held out)
- [ ] Compute end-to-end metrics
- [ ] Error analysis and debugging
- [ ] Target: 85-90% correct rate (audio-only prototype)

### Week 12-14: Production Refinement (Optional)
- [ ] Upgrade to audiovisual SlowFast Network
- [ ] Implement post-processing (PANNs, video scene detection)
- [ ] Scale dataset to 30 games (if needed)
- [ ] Re-train for production quality
- [ ] Target: 95%+ correct rate

---

## Cost Estimation

### S3 Storage
- Raw games: 50 games × 10GB = 500GB @ $0.0125/GB/month = $6.25/month
- Clips: 12,000 clips × 50MB = 600GB @ $0.0125/GB/month = $7.50/month
- **Total storage**: ~$15/month

### EC2 Compute
- Training: 60 hours @ $0.526/hr = $32
- Inference/testing: 20 hours @ $0.526/hr = $11
- **Total compute**: ~$43

### Data Transfer
- Download from Fubo: Free (local → S3)
- S3 → EC2 (same region): Free
- Results download: <$5

**Total Project Cost**: ~$60-80 + $15/month storage

**Compared to SageMaker**: Would be $200-300 (4-5x more expensive)

---

## Next Steps (This Week) - Revised 2026-01-25

### Day 1: Assess Data Status
```bash
# Check how many games downloaded
cd /Users/hungmuhamath/projects/GitHub/GetGamesToLocal/
wc -l master.csv
grep "downloaded" master.csv | wc -l

# Review downloaded games
cat master.csv
```

### Day 2-3: Run CPPreData Pipeline
```bash
# Process all downloaded games
cd /Users/hungmuhamath/projects/GitHub/CPPreData/
uv sync  # Install dependencies if needed
uv run process_videos.py --input /Users/hungmuhamath/projects/GitHub/GetGamesToLocal/videos

# Verify outputs
ls -lh manifests/  # Check CSV manifests
ls -lh audio/      # Check WAV files
```

### Day 4-5: Start Manual Annotation
1. **Select 3 test games** (diverse networks: CBS, NBC, FOX)
2. **Annotate first game**:
   - Open manifest CSV in Excel/Google Sheets
   - Watch video at each timestamp (use start_time_hms/end_time_hms)
   - Fill `CP` column: `1` if change point (game ↔ ad), `0` if false positive
   - Track time spent
3. **Document workflow** (challenges, time estimates)
4. **Annotate 2 more test games**

### Day 6-7: Review & Plan
- Validate annotation quality (random review)
- Estimate total annotation time for 10-15 games
- Select full prototype dataset (10-15 games)
- Update project timeline based on annotation speed

**By end of this week**:
- All downloaded games processed through CPPreData
- 3 games manually annotated (test dataset)
- Clear workflow for scaling to 10-15 games

---

## Revised Architecture Notes (2026-01-25)

### Data Pipeline Decision: Multi-Repository Approach ✅

**Decision**: Split data acquisition, processing, and training into separate repos
**Rationale**:
- **Separation of Concerns**: Each repo has focused responsibility
- **Reusability**: Data pipeline repos can be used for other ML projects
- **Technology Diversity**: Node.js for downloading, Python for ML
- **Independent Testing**: Can test each stage separately

**Trade-offs**:
- More complex to document (solved with DATA_PIPELINE.md)
- Manual coordination between repos (acceptable for this project size)
- Better than monolithic repo (easier to maintain long-term)

### Annotation Strategy: Excel/Google Sheets ✅

**Decision**: Manual annotation using spreadsheet software
**Rationale**:
- Simple, no custom UI development needed
- Can start immediately
- Suitable for prototype dataset size (10-15 games)
- Easy to share and review

**Limitations**:
- Time-consuming for large datasets
- Requires video player + spreadsheet side-by-side
- For production scaling (30+ games), consider custom annotation UI

### Cloud Infrastructure: Deferred ⏸️

**Decision**: Local development first, cloud deployment later
**Rationale**:
- Data pipeline runs locally (no cloud needed for acquisition)
- Model training can start locally with smaller dataset
- EC2 + S3 only needed for:
  - Large-scale training (30+ games)
  - Production deployment
  - Sharing with collaborators

**Timeline**:
- Local development: Weeks 5-10
- Cloud migration: Week 11+ (if needed)
