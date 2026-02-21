# NFL Ad Detection - Complete Data Pipeline

**Last Updated**: 2026-01-25
**Status**: Data Infrastructure Complete (3/4 repos operational)

---

## Overview

This document describes the **complete end-to-end data pipeline** spanning 4 GitHub repositories that work together to detect and segment advertisements in NFL game broadcasts.

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  fubo-scraper    │ CSV  │ GetGamesToLocal  │ MP4  │   CPPreData      │ CSV  │ ad-game-explainer│
│                  │──────>│                  │──────>│                  │──────>│                  │
│ Extract URLs     │      │ Download games   │      │ Detect CP points │      │ Train models     │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘
     Step 1                    Step 2                    Step 3                   Step 4

     Status: ✅             Status: ✅              Status: ✅ (setup)        Status: 🔄 (in progress)
                                                    🔲 (not run yet)
```

---

## Step 1: Stream URL Extraction (fubo-scraper)

**Repository**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/01-fubo-scraper/`
**Status**: ✅ **COMPLETE & OPERATIONAL**
**Technology**: Python + Selenium-Wire

### Purpose
Automates extraction of video stream URLs (.m3u8 format) from Fubo TV's NFL recordings by logging into user accounts and capturing network traffic.

### What It Does
1. Logs into Fubo TV with user credentials
2. Navigates to NFL recordings page
3. Intercepts network traffic to capture stream URLs
4. Extracts metadata: title, air date, network
5. Exports to timestamped CSV files

### Input
- Fubo TV account credentials (`.env` file)
- Fubo recordings URL

### Output Format
**File**: `output/fubo_recordings_YYYYMMDD_HHMMSS.csv`

| Column | Example |
|--------|---------|
| `title` | "New Orleans Saints at Buffalo Bills" |
| `air_date` | "Sep 28, 2025" |
| `network` | "WIVB" |
| `stream_url` | `https://playlist-nonlive.fubo.tv/v4/finite.m3u8?...` |
| `status` | "SUCCESS" |
| `extraction_time` | "2026-01-23T14:58:24Z" |

### Usage
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/01-fubo-scraper/
python main.py --headless
```

### Documentation
- README.md (400 lines): Comprehensive with installation, usage, troubleshooting
- Excellent inline code documentation

### Current Status
- ✅ Fully operational
- ✅ 150+ NFL games available (2025 season)
- ✅ CSV files generated and tested

---

## Step 2: Game Download (GetGamesToLocal)

**Repository**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/`
**Status**: ✅ **COMPLETE & OPERATIONAL**
**Technology**: Node.js + FFmpeg

### Purpose
Downloads NFL game streams from fubo-scraper CSV output to local MP4 files while managing duplicates and tracking download history.

### What It Does
1. Reads CSV from fubo-scraper (or any compatible CSV)
2. Checks `master.csv` for duplicates (by title + air_date)
3. Downloads .m3u8 streams using FFmpeg (copy codec, no transcoding)
4. Skips DRM-protected streams (.mpd format)
5. Appends download status to `master.csv` with absolute file paths

### Input
**File**: Any CSV with required columns:
- `title` (required)
- `air_date` (required)
- `stream_url` (required)
- `network` (optional)
- `extraction_time` (optional)

### Output Format
**File**: `master.csv` (persistent tracking database)

| Column | Example |
|--------|---------|
| `title` | "New Orleans Saints at Buffalo Bills" |
| `air_date` | "Sep 28, 2025" |
| `network` | "WIVB" |
| `stream_url` | `https://playlist-nonlive.fubo.tv/...` |
| `status` | `downloaded` / `failed` / `skipped_drm` |
| `file_path` | `/full/path/to/videos/New_Orleans_Saints_at_Buffalo_Bills_Sep_28__2025.mp4` |
| `filename` | `New_Orleans_Saints_at_Buffalo_Bills_Sep_28__2025.mp4` |
| `extraction_time` | "2026-01-23 14:58:24" |
| `download_time` | "2026-01-25T10:15:32.123Z" |

**Directory**: `./videos/*.mp4` (downloaded game files)

### Usage
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/
npm install
node download-streams.js /path/to/fubo_recordings.csv
```

### Documentation
- README.md: Clear features, usage, troubleshooting
- master.csv.example: Real-world output example
- Excellent JSDoc comments

### Current Status
- ✅ Fully operational
- ✅ Multiple games successfully downloaded
- ✅ master.csv tracking working correctly
- 🔲 Exact count of downloaded games: **TBD** (check master.csv)

---

## Step 3: Change Point Detection (CPPreData)

**Repository**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/`
**Status**: ✅ **SETUP COMPLETE** | 🔲 **NOT RUN ON DATASET YET**
**Technology**: Python + Pydub + SciPy + FFmpeg

### Purpose
Processes downloaded MP4 game files to detect silent segments (potential change points between game content and ads), generating EDL files for DaVinci Resolve annotation workflow.

### What It Does
1. Extracts audio from MP4 files (mono, 16kHz WAV)
2. Applies band-pass filter (300-6000 Hz) to isolate speech/sound frequencies
3. Detects silent segments based on volume threshold
4. Extends each silence by ±2.2 seconds for context
5. Generates EDL files with markers at each silent segment for DaVinci Resolve
6. Generates CSV manifests with timestamps for reference

### Input
**Directory**: Any folder containing MP4 files (e.g., from GetGamesToLocal)

### Output Format
**Directory**: `edl/*.edl` (one EDL per video for DaVinci Resolve import)
**Directory**: `manifests/*.csv` (one CSV per video for reference)
**Directory**: `audio/*.wav` (extracted audio files)

**EDL Structure** (for DaVinci Resolve):
- Contains markers at each detected silent segment
- Imports directly into DaVinci Resolve timeline
- Allows manual review and identification of true change points

**CSV Structure** (for reference):

| Column | Example | Purpose |
|--------|---------|---------|
| `pair_id` | `cowboysAtEaglesWeek1_001` | Sequential identifier |
| `video_file_path` | `/path/to/game.mp4` | Absolute path to video |
| `audio_file_path` | `/path/to/audio/game.wav` | Absolute path to audio |
| `silent_start_time` | `73.32175` | Extended start (seconds, precise) |
| `silent_end_time` | `78.88187500000001` | Extended end (seconds, precise) |
| `start_time_hms` | `00:01:13` | Human-readable start (HH:MM:SS) |
| `end_time_hms` | `00:01:18` | Human-readable end (HH:MM:SS) |

### Usage
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
uv sync
uv run process_videos.py --input /path/to/GetGamesToLocal/videos
```

**Configuration Parameters**:
```bash
uv run process_videos.py \
  --input /path/to/videos \
  --audio-output audio \
  --manifest-output manifests \
  --filter-low 300 \
  --filter-high 6000 \
  --min-silence-ms 10 \
  --volume-threshold 4 \
  --time-extension 2.2
```

### Smart Skip Logic

The processor intelligently handles re-runs:

**Full Skip** (all 3 files exist):
- CSV manifest exists
- EDL file exists
- WAV audio exists
→ Skips video entirely

**Audio Reuse** (only audio exists):
- WAV audio exists
- CSV/EDL missing
→ Reuses existing audio, only regenerates CSV/EDL

**Fresh Extract** (no audio):
- WAV audio missing
→ Extracts audio from video, then processes

This saves significant time when re-processing with different parameters (e.g., adjusting silence thresholds).

### Documentation
- README.md (754 lines): Exceptional documentation with architecture diagrams
- QUICKSTART.md: 2-minute setup guide
- Comprehensive parameter tuning guide
- Real-world example included (Cowboys vs Eagles)

### Current Status
- ✅ Code fully operational (tested on 1 game)
- ✅ Dependencies installed
- ✅ EDL export functionality added
- ✅ Smart skip logic with audio reuse
- 🔲 **TODO**: Run on all downloaded games from GetGamesToLocal

### Example Output
The repo includes one processed game demonstrating the pipeline:
- **Game**: Cowboys vs Eagles (Week 1)
- **Detected segments**: 240+ silence points
- **Manifest**: `manifests/cowboysAtEaglesWeek1.csv`
- **EDL**: `edl/cowboysAtEaglesWeek1.edl`

---

## Step 3.5: Manual Annotation with DaVinci Resolve

**Repository**: Local workflow with DaVinci Resolve
**Status**: 🔄 **IN PROGRESS** (Annotation workflow defined)
**Technology**: DaVinci Resolve + Manual Review

### Purpose
Manually review EDL markers to identify true segment boundaries (content↔ad transitions) and create ground truth annotations.

### Workflow

**CRITICAL**: For CP classifier training, you must label EVERY silent segment (not just TRUE CPs)!

1. **Import into DaVinci Resolve**:
   - Import MP4 file from GetGamesToLocal
   - Import EDL file from CPPreData (markers appear on timeline)
   - EDL contains ~200-300 markers per game at detected silent segments

2. **Manual Review - Label EVERY Silent Segment**:
   - Play video around each marker (~200-300 markers per game)
   - **For each marker**, classify as:
     - **TRUE CP (is_CP = 1)**: Boundary between content/ad or ad/content
       - Content → Ad Break (start of commercial break)
       - Ad Break → Content (return from commercial break)
     - **Non-CP (is_CP = 0)**: Silence within same content
       - Pauses in commentary
       - Crowd noise dips
       - Natural game flow silences
       - Brief silence during replay

3. **Create Annotation JSON**:
   - **TWO annotation levels required**:
     1. **Silent Segments**: ALL ~200-300 silent segments with is_CP labels (for CP classifier training)
     2. **Segments**: Higher-level ad breaks vs content (for Ad classifier training)
   - Export as JSON in format specified below

### Output Format
**Directory**: `data/annotations/*.json` (one JSON per game)

**Annotation JSON Structure** (UPDATED - Two levels of annotation):
```json
{
  "video_id": "cowboys_eagles_week1",
  "duration_seconds": 10800,

  "silent_segments": [
    {
      "segment_id": "silence_001",
      "timestamp": 73.32,
      "is_CP": 0,
      "notes": "Pause in commentary"
    },
    {
      "segment_id": "silence_002",
      "timestamp": 145.67,
      "is_CP": 0,
      "notes": "Crowd noise dip"
    },
    {
      "segment_id": "silence_003",
      "timestamp": 1245.3,
      "is_CP": 1,
      "notes": "Content → Ad Break (TRUE CP)"
    },
    {
      "segment_id": "silence_004",
      "timestamp": 1335.8,
      "is_CP": 1,
      "notes": "Ad Break → Content (TRUE CP)"
    }
  ],

  "segments": [
    {
      "segment_id": 0,
      "start_time": 0.0,
      "end_time": 1245.3,
      "label": "content",
      "type": "game_play"
    },
    {
      "segment_id": 1,
      "start_time": 1245.3,
      "end_time": 1335.8,
      "label": "ad",
      "type": "commercial_break"
    },
    {
      "segment_id": 2,
      "start_time": 1335.8,
      "end_time": 2890.5,
      "label": "content",
      "type": "game_play"
    }
  ],

  "change_points": [1245.3, 1335.8, 2890.5]
}
```

**Required Fields**:
- `video_id`: Unique identifier for the game
- `duration_seconds`: Total video duration
- **`silent_segments`**: Array of ALL detected silent segments with:
  - `segment_id`: Unique ID for this silent segment
  - `timestamp`: Timestamp of silent segment (seconds, float)
  - **`is_CP`**: **CRITICAL** - 1 if TRUE CP, 0 if non-CP (for CP classifier training)
  - `notes`: Optional description
- `segments`: Array of content/ad segments with:
  - `segment_id`: Sequential ID (starting from 0)
  - `start_time`: Start timestamp in seconds (float)
  - `end_time`: End timestamp in seconds (float)
  - `label`: Either `"ad"` or `"content"`
  - `type`: Optional descriptor (e.g., "commercial_break", "game_play")
- `change_points`: Array of TRUE CP timestamps (where is_CP=1)

**Annotation Strategy**: Two-level annotation required
- **Level 1 (Silent Segments)**: Label ALL ~200-300 silent segments per game (CP vs non-CP)
  - Used for CP classifier training
  - Each silent segment gets is_CP label (0 or 1)
- **Level 2 (Segments)**: Annotate ad breaks vs content (Approach B)
  - Used for Ad classifier training
  - Annotate entire commercial breaks as single "ad" segments
  - Post-processing will handle splitting merged ads later

### Expected Output
**Per Game**:
- ~200-300 silent segment annotations (is_CP labels)
- ~10-20 TRUE CPs (is_CP=1)
- ~180-280 non-CPs (is_CP=0)
- 10-25 content/ad segments

**For 10-15 Games**:
- ~2000-4500 silent segment annotations
- ~100-300 TRUE CP pairs (for negative examples in triplet loss)
- ~1800-4200 non-CP pairs (for positive examples in triplet loss)
- 150-250 content/ad segments (for Ad classifier)

### Current Status
- ✅ Annotation format defined
- ✅ DaVinci Resolve workflow documented
- ✅ `createAnnotationJson.py` script created in CPPreData repo
- ✅ Script generates segment-level annotations from color-coded EDL
- ⚠️ **GAP**: Silent segment annotations (is_CP labels) not yet generated
- 🔲 **TODO**: Extend script to include silent_segments array
- 🔲 **TODO**: Annotate first 3 games (test workflow)
- 🔲 **TODO**: Annotate remaining 7-12 games for prototype

**Annotation Output Location**:
```
CPPreData/manifests/{video_name}/{video_id}_annotation.json
```

**See**: [ANNOTATION_WORKFLOW.md](ANNOTATION_WORKFLOW.md) for complete workflow documentation

---

## Step 4: Model Training (ad-game-explainer)

**Repository**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/` (THIS REPO)
**Status**: 🔄 **IN PROGRESS** (Annotation phase, ~35% complete)
**Technology**: Python + PyTorch + CUDA

### Purpose
Uses segment annotations from DaVinci Resolve review to train:
1. **Change Point (CP) Classifier**: Detects transitions between content/ads
2. **Ad Classifier**: Distinguishes ad segments from game content
3. **End-to-End Pipeline**: Complete inference system

### What It Does (Planned)

**IMPORTANT CLARIFICATION**: CP classifier training requires SILENT SEGMENT annotations,
not just segment boundaries. See corrected workflow below.

1. **Extract CP Training Pairs** (2s before/after silent segments):
   - **For TRUE CPs**: Extract 2s before/after each annotated CP boundary
   - **For non-CPs**: Extract 2s before/after each non-CP silent segment
   - Generate training pairs for triplet loss
   - **Critical**: Must annotate EVERY silent segment (not just boundaries)

2. **Extract Short Clips** (10-30s for Ad Classifier):
   - Read annotation JSONs from Step 3.5
   - Extract clips from annotated segments
   - Generate balanced dataset (ad clips vs content clips)

3. **CP Classifier Training** (Strong Supervision - Following Paper):
   - Positive pairs: 2s before/after non-CP silent segments
   - Negative pairs: 2s before/after TRUE CP boundaries
   - Triplet loss with margin=1.0
   - Model learns to distinguish CP vs non-CP silent segments

4. **Ad Classifier Training**:
   - Siamese ResNet-34 with triplet loss
   - Pre-trained on VoxCeleb2
   - Target: 90%+ AUC
3. **Ad Classifier Training**:
   - Audiovisual SlowFast Network with LSTM
   - Target: 95%+ precision/recall
4. **End-to-End Pipeline**:
   - Integrate CP + Ad classifiers
   - Post-processing with PANNs
   - Target: 97%+ correct rate

### Input
- Segment annotation JSONs from DaVinci Resolve workflow (Step 3.5)
- Raw MP4 files from GetGamesToLocal
- Pre-trained weights (VoxCeleb2, SlowFast, ViT)

### Output
- Trained model checkpoints (`.pth` files)
- Inference pipeline for new videos
- Performance metrics and evaluation results

### Documentation
- CLAUDE.md: Comprehensive technical guidance
- README.md: Project overview
- NFL_PROJECT_PLAN.md: Implementation roadmap
- PROJECT_STATUS.md: Progress tracking
- DATA_PIPELINE.md: This file

### Current Status
- ✅ Project structure created
- ✅ Documentation files created
- ✅ Requirements.txt prepared
- ✅ Data pipeline infrastructure complete (Steps 1-3)
- ✅ Annotation workflow defined (Step 3.5)
- 🔲 **TODO**: Run CPPreData on downloaded games
- 🔲 **TODO**: Manual annotation with DaVinci Resolve (3-5 games to start)
- 🔲 **TODO**: Extract short clips from annotations
- 🔲 **TODO**: Implement CP classifier
- 🔲 **TODO**: Implement Ad classifier
- 🔲 **TODO**: Train models
- 🔲 **TODO**: Build end-to-end pipeline

---

## Complete Workflow Summary

### Phase 0: Infrastructure Setup ✅ (100% Complete)
- [x] fubo-scraper repo operational
- [x] GetGamesToLocal repo operational
- [x] CPPreData repo operational
- [x] ad-game-explainer repo created

### Phase 1: Data Acquisition 🔄 (60% Complete)
- [x] Extract stream URLs from Fubo (fubo-scraper)
- [x] Download games to local storage (GetGamesToLocal)
- [x] Define annotation workflow (DaVinci Resolve)
- [ ] **NEXT**: Run CPPreData on all downloaded games
- [ ] **NEXT**: Manual annotation with DaVinci Resolve (3-5 games)
- [ ] **NEXT**: Extract short clips from annotations

### Phase 2: Model Training 🔲 (0% Complete)
- [ ] Train CP classifier (Siamese ResNet-34)
- [ ] Train Ad classifier (SlowFast + LSTM)
- [ ] Evaluate on test set

### Phase 3: Pipeline Integration 🔲 (0% Complete)
- [ ] Build end-to-end inference pipeline
- [ ] Add post-processing (PANNs)
- [ ] Performance optimization

---

## Data Flow Diagram

```
Fubo TV Recordings (150+ NFL games)
        ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: fubo-scraper                                                │
│ ├── Input:  Fubo account credentials                                │
│ ├── Output: CSV with stream URLs                                    │
│ └── Tech:   Python + Selenium-Wire                                  │
└─────────────────────────────────────────────────────────────────────┘
        ↓ (fubo_recordings_*.csv)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: GetGamesToLocal                                             │
│ ├── Input:  CSV from fubo-scraper                                   │
│ ├── Output: MP4 files + master.csv tracking                         │
│ └── Tech:   Node.js + FFmpeg                                        │
└─────────────────────────────────────────────────────────────────────┘
        ↓ (videos/*.mp4 + master.csv)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: CPPreData                                                   │
│ ├── Input:  MP4 files from GetGamesToLocal                          │
│ ├── Output: EDL files + CSV manifests + WAV audio                   │
│ └── Tech:   Python + Pydub + SciPy                                  │
└─────────────────────────────────────────────────────────────────────┘
        ↓ (edl/*.edl + manifests/*.csv + audio/*.wav)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3.5: Manual Annotation (DaVinci Resolve)                       │
│ ├── Input:  EDL files + MP4 files from CPPreData                    │
│ ├── Process: Import EDL into DaVinci Resolve, review markers        │
│ ├── Output: Segment annotation JSON (ad breaks vs content)          │
│ └── Tech:   DaVinci Resolve + Manual Review                         │
└─────────────────────────────────────────────────────────────────────┘
        ↓ (data/annotations/*.json)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: ad-game-explainer (THIS REPO)                               │
│ ├── Input:  Annotation JSONs + MP4 files                            │
│ ├── Output: Trained models + inference pipeline                     │
│ └── Tech:   Python + PyTorch + CUDA                                 │
└─────────────────────────────────────────────────────────────────────┘
        ↓
  Trained Ad Detection System (97% accuracy target)
```

---

## Next Actions (Priority Order)

### Immediate (This Week)
1. **Check downloaded game count**:
   ```bash
   cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/
   wc -l master.csv  # Count downloaded games
   ```

2. **Run CPPreData on all downloaded games**:
   ```bash
   cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
   uv run process_videos.py --input /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/videos
   ```
   - Expected output: One EDL + CSV per game in `edl/` and `manifests/`
   - Estimated time: ~5-10 min per game

3. **Manual annotation with DaVinci Resolve** (start with 3 games):
   - Import MP4 file into DaVinci Resolve
   - Import EDL file (markers appear on timeline)
   - Review each marker, identify TRUE segment boundaries (content↔ad transitions)
   - Create annotation JSON with segment start/end times and labels
   - Save to `data/annotations/{game_id}.json`

### Short Term (Next 2 Weeks)
4. **Annotate 7-12 more games** (10-15 total for prototype dataset)
5. **Extract short clips (10-30s)** from annotated segments
   - Run clip extraction script on annotation JSONs
   - Generate balanced dataset: `data/short_clips/ads/` and `data/short_clips/content/`
6. **Implement CP classifier** in ad-game-explainer
7. **Download VoxCeleb2 pre-trained weights**
8. **Train CP classifier** on short clips

### Medium Term (Weeks 3-5)
9. **Implement Ad classifier**
10. **Train Ad classifier**
11. **Build end-to-end pipeline**

---

## Repository Links

| Repository | Path | Status |
|------------|------|--------|
| fubo-scraper | `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/01-fubo-scraper/` | ✅ Complete |
| GetGamesToLocal | `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/` | ✅ Complete |
| CPPreData | `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/` | ✅ Setup done, 🔲 Not run |
| ad-game-explainer | `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/` | 🔄 In progress |

---

## Key Files to Track

### fubo-scraper
- `output/fubo_recordings_*.csv` - Stream URL exports

### GetGamesToLocal
- `master.csv` - Download tracking database
- `videos/*.mp4` - Downloaded games

### CPPreData
- `edl/*.edl` - EDL files for DaVinci Resolve import
- `manifests/*.csv` - CP candidate lists (reference)
- `audio/*.wav` - Extracted audio files

### ad-game-explainer
- `data/annotations/*.json` - Segment annotations from DaVinci Resolve
- `data/short_clips/ads/*.mp4` - Short ad clips (10-30s) for training
- `data/short_clips/content/*.mp4` - Short content clips (10-30s) for training
- `models/*.pth` - Trained model checkpoints (future)
- `outputs/*.json` - Inference results (future)

---

## Questions & Support

If you encounter issues at any step:

1. **fubo-scraper issues**: Check README.md troubleshooting section
2. **GetGamesToLocal issues**: Verify FFmpeg installation, check master.csv for errors
3. **CPPreData issues**: Check README.md parameter tuning guide
4. **ad-game-explainer issues**: See PROJECT_STATUS.md and NFL_PROJECT_PLAN.md

---

**Last Updated**: 2026-01-25
**Next Review**: After running CPPreData on downloaded games
