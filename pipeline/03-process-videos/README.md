# CP Detection Data Pipeline

A Python-based data preparation pipeline for training Change Point (CP) detection models. This tool processes NFL game videos (with ads) to extract audio, detect silence segments that may indicate scene changes, and generate CSV manifests and annotation JSON files for model training.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Master Map & Annotation JSON](#master-map--annotation-json)
- [Configuration Parameters](#configuration-parameters)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation (2 minutes)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install ffmpeg
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu/Debian

# 3. Navigate to project and install dependencies
cd /Users/hungmuhamath/projects/GitHub/CPPreData
uv sync
```

### Process Videos

```bash
# Process videos - creates manifests, EDL files, and master_map.csv
uv run process_videos.py --input /path/to/your/videos
```

### Generate Annotation JSON

```bash
# Generate annotation JSON using master map (recommended)
uv run python createAnnotationJson.py \
  --file_path manifests/video_name/video_name_marker.edl \
  --use-master-map
```

---

## Overview

### What Does This Pipeline Do?

The pipeline performs the following operations on each video file:

1. **Audio Extraction**: Extracts the audio track from .mp4 files and converts to WAV format (16 kHz, mono)
2. **Signal Processing**: Applies a band-pass filter (300-6000 Hz) to isolate relevant frequency ranges
3. **Silence Detection**: Identifies silent segments that may indicate transitions between content and ads
4. **Time Extension**: Extends detected segments by ±2.2 seconds to capture context around potential change points
5. **Manifest Generation**: Creates CSV files with metadata for each silence segment
6. **EDL Generation**: Creates EDL marker files for video editing software (DaVinci Resolve)
7. **Master Map Tracking**: Automatically maintains a master_map.csv index of all processed videos
8. **Annotation JSON**: Generates structured JSON files for model training from annotated EDL files

### Key Features

- ✅ **Automated Workflow**: Master map automatically tracks video metadata
- ✅ **Skip Logic**: Automatically skips already processed videos
- ✅ **Configurable Parameters**: All detection parameters can be adjusted via CLI
- ✅ **Boundary Validation**: Extended time windows are clamped to valid video duration
- ✅ **Comprehensive Logging**: Tracks progress, warnings, and errors with timestamps
- ✅ **Modular Design**: Clean, maintainable code following PEP8 standards
- ✅ **Flexible Output**: Supports both automatic and manual annotation workflows

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CP Detection Pipeline                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   1. Audio Extraction Module         │
        │   - ffmpeg/pydub wrapper             │
        │   - Converts to 16kHz mono WAV       │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   2. Signal Processing Module        │
        │   - Band-pass filter [300-6000 Hz]   │
        │   - Butterworth 5th order            │
        │   - Normalize to int16 range         │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   3. Silence Detection Module        │
        │   - Threshold-based detection        │
        │   - Min duration filter (10ms)       │
        │   - Returns (start, end) tuples     │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   4. Time Extension Module           │
        │   - Extends ±2.2s around silence     │
        │   - Clamps to [0, duration]          │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   5. Manifest Generation Module      │
        │   - Sequential pair_id generation    │
        │   - CSV output with metadata         │
        └──────────────────────────────────────┘
```

### Core Functions

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `extract_audio()` | Extract audio from video | Video path, sample rate | WAV file, duration |
| `detect_silence_segments()` | Find silent regions | Audio path, filter params | List of (start, end) tuples |
| `extend_segment_times()` | Add context window | Start, end, extension | Extended (start, end) |
| `create_manifest()` | Generate CSV metadata | Video, audio, segments | CSV file path |
| `process_video_folder()` | Main processing loop | Input folder, config | Processing statistics |

---

## Installation

### Prerequisites

1. **Python 3.8+** (recommended: Python 3.9 or higher)
2. **uv** (modern, fast Python package installer and virtual environment manager)
3. **ffmpeg** (required by pydub for audio processing)

#### Why uv?

[uv](https://github.com/astral-sh/uv) is a blazingly fast Python package installer and resolver written in Rust. Benefits:
- **10-100x faster** than pip for installing packages
- **Automatic virtual environment management** - no need to manually create venvs
- **Better dependency resolution** - handles conflicts more reliably
- **Lock files** - ensures reproducible installations across environments
- **Drop-in replacement** for pip/pip-tools/virtualenv

### Step 1: Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (via pip):**
```bash
pip install uv
```

### Step 2: Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Step 3: Install Python Dependencies with uv

Navigate to the project directory and install dependencies:

```bash
cd /Users/hungmuhamath/projects/GitHub/CPPreData
uv sync
```

This will:
- Create a virtual environment automatically
- Install all dependencies from `pyproject.toml`
- Lock dependencies for reproducibility

**Alternative: Install without creating venv:**
```bash
uv pip install -e .
```

### Step 4: Verify Installation

Check that ffmpeg is accessible:
```bash
ffmpeg -version
```

Test Python imports:
```bash
uv run python -c "import pydub, numpy, scipy, pandas; print('All dependencies installed!')"
```

---

## Usage

### Directory Setup

**Option 1: Place videos in a dedicated folder (Recommended)**

```
your_project/
├── nfl_videos/           # Place your .mp4 files here
│   ├── game1.mp4
│   ├── game2.mp4
│   └── game3.mp4
└── CPPreData/
    ├── process_videos.py
    ├── pyproject.toml    # Project configuration and dependencies
    ├── uv.lock           # Dependency lock file (auto-generated)
    ├── .venv/            # Virtual environment (auto-created by uv)
    ├── audio/            # Created automatically
    └── manifests/        # Created automatically
```

**Option 2: Place videos anywhere and specify absolute path**

```bash
# Videos can be anywhere on your system
/path/to/videos/
├── game1.mp4
├── game2.mp4
└── game3.mp4
```

### Basic Usage

**1. Process videos with default parameters:**

```bash
uv run process_videos.py --input /path/to/nfl_videos
```

This will:
- Create `audio/` and `manifests/` folders in the current directory
- Extract audio from all .mp4 files
- Detect silence segments with default parameters
- Generate CSV manifests

**2. Specify custom output locations:**

```bash
uv run process_videos.py \
  --input /path/to/nfl_videos \
  --audio-output ./extracted_audio \
  --manifest-output ./csv_manifests
```

**3. Adjust detection parameters:**

```bash
uv run process_videos.py \
  --input /path/to/nfl_videos \
  --volume-threshold 5 \
  --min-silence-ms 20 \
  --time-extension 3.0 \
  --filter-low 200 \
  --filter-high 7000
```

**4. Using the installed command (after uv sync):**

```bash
# If you ran 'uv sync', you can use the process-videos command directly
uv run process-videos --input /path/to/nfl_videos
```

### Real-World Example Workflow

```bash
# Step 1: Navigate to project directory
cd /Users/hungmuhamath/projects/GitHub/CPPreData

# Step 2: Install dependencies with uv
uv sync

# Step 3: Create a folder for your videos (if not already created)
mkdir -p ~/nfl_game_videos

# Step 4: Copy or move your .mp4 files to that folder
# (Manually copy your files to ~/nfl_game_videos/)

# Step 5: Run the pipeline
uv run process_videos.py --input ~/nfl_game_videos

# Step 6: Check the outputs
ls -la audio/      # Should contain .wav files
ls -la manifests/  # Should contain .csv files

# Step 7: Review manifests for manual CP annotation
head -n 5 manifests/game1.csv
```

---

## Output Structure

### Folder Organization

After running the pipeline:

```
CPPreData/
├── audio/
│   ├── game1.wav          # 16 kHz mono audio
│   ├── game2.wav
│   └── game3.wav
├── manifests/
│   ├── game1/
│   │   ├── game1.csv              # Metadata for game1
│   │   ├── game1_marker.edl       # EDL with silence markers
│   │   └── game1_annotation.json  # Annotation JSON (optional)
│   ├── game2/
│   │   ├── game2.csv
│   │   ├── game2_marker.edl
│   │   └── game2_annotation.json
│   └── game3/
│       ├── game3.csv
│       ├── game3_marker.edl
│       └── game3_annotation.json
├── master_map.csv         # Master index of all processed videos
├── process_videos.py
├── createAnnotationJson.py
├── pyproject.toml
└── uv.lock
```

### Manifest CSV Format

Each CSV file contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `pair_id` | Sequential identifier | `game1_001`, `game1_002` |
| `video_file_path` | Absolute path to video | `/path/to/game1.mp4` |
| `audio_file_path` | Absolute path to audio | `/path/to/audio/game1.wav` |
| `silent_start_time` | Extended start time (seconds) | `10.523` |
| `silent_end_time` | Extended end time (seconds) | `15.234` |
| `start_time_hms` | Start time in HH:MM:SS format | `00:00:10` |
| `end_time_hms` | End time in HH:MM:SS format | `00:00:15` |
| `CP` | Change Point label (blank) | _(empty for annotation)_ |

**Example CSV content:**

```csv
pair_id,video_file_path,audio_file_path,silent_start_time,silent_end_time,start_time_hms,end_time_hms,CP
game1_001,/Users/user/videos/game1.mp4,/Users/user/CPPreData/audio/game1.wav,8.323,13.034,00:00:08,00:00:13,
game1_002,/Users/user/videos/game1.mp4,/Users/user/CPPreData/audio/game1.wav,45.612,50.823,00:00:45,00:00:50,
game1_003,/Users/user/videos/game1.mp4,/Users/user/CPPreData/audio/game1.wav,102.145,106.556,00:01:42,00:01:46,
```

### Master Map CSV Format

The `master_map.csv` file is automatically generated/updated by `process_videos.py` and serves as a central index of all processed videos:

| Column | Description | Example |
|--------|-------------|---------|
| `video_name` | Video filename without extension | `cowboysAtEaglesWeek1` |
| `video_path` | Absolute path to video file | `/path/to/video.mp4` |
| `edl_path` | Relative path to marker EDL file | `manifests/video/video_marker.edl` |
| `duration_seconds` | Video duration in seconds | `15142.918` |

**Example master_map.csv content:**

```csv
video_name,video_path,edl_path,duration_seconds
game1,/Users/user/videos/game1.mp4,manifests/game1/game1_marker.edl,7200.5
game2,/Users/user/videos/game2.mp4,manifests/game2/game2_marker.edl,8450.3
```

### Annotation JSON Format

Generated by `createAnnotationJson.py`, this file contains structured annotation data for training CP detection models:

```json
{
  "video_id": "cowboysAtEaglesWeek1",
  "duration_seconds": 15142.918,
  "segments": [
    {
      "segment_id": 0,
      "start_time": 0.0,
      "end_time": 76.29,
      "label": "content",
      "type": "game_play"
    },
    {
      "segment_id": 1,
      "start_time": 76.29,
      "end_time": 85.54,
      "label": "ad",
      "type": "commercial_break"
    }
  ],
  "change_points": [0.0, 76.29, 85.54, ...]
}
```

### Understanding the Time Windows

**Original silence segment:** 10.523s - 10.834s (311ms of silence)

**Extended segment (±2.2s):**
- `silent_start_time`: 10.523 - 2.2 = 8.323s (clamped to ≥0)
- `silent_end_time`: 10.834 + 2.2 = 13.034s (clamped to ≤video_duration)

**Total window length:** ~4.4 - 5 seconds (provides context for CP detection)

---

## Configuration Parameters

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | string | *required* | Folder containing .mp4 files |
| `--audio-output` | string | `audio` | Output folder for WAV files |
| `--manifest-output` | string | `manifests` | Output folder for CSV files |
| `--filter-low` | int | `300` | Band-pass filter low cutoff (Hz) |
| `--filter-high` | int | `6000` | Band-pass filter high cutoff (Hz) |
| `--min-silence-ms` | int | `10` | Minimum silence duration (ms) |
| `--volume-threshold` | int | `4` | Volume threshold after int16 normalization |
| `--time-extension` | float | `2.2` | Seconds to extend before/after silence |
| `--sample-rate` | int | `16000` | Audio sample rate (Hz) |

### Parameter Tuning Guide

**Detecting more silence segments:**
- Increase `--volume-threshold` (e.g., 6 or 8)
- Decrease `--min-silence-ms` (e.g., 5)

**Detecting fewer, longer silences:**
- Decrease `--volume-threshold` (e.g., 2 or 3)
- Increase `--min-silence-ms` (e.g., 20 or 50)

**Adjust frequency range:**
- Human speech: 300-3400 Hz (use `--filter-low 300 --filter-high 3400`)
- Broader range: 200-8000 Hz (use `--filter-low 200 --filter-high 8000`)

**Change context window:**
- Longer context: `--time-extension 3.0` (6+ seconds total)
- Shorter context: `--time-extension 1.5` (3+ seconds total)

---

## Development Process

### Project Timeline & Tracking

#### Phase 1: Planning ✅ COMPLETED
**Objective:** Define architecture and implementation strategy

- [x] Define project requirements and specifications
- [x] Design modular architecture (5 core components)
- [x] Specify input/output formats and folder structure
- [x] Identify edge cases and validation rules
- [x] Create implementation plan with dependencies

**Deliverables:**
- Implementation plan document
- Architecture diagram
- Configuration specifications

---

#### Phase 2: Dependency Setup ✅ COMPLETED
**Objective:** Establish development environment

- [x] Create `pyproject.toml` with dependencies
- [x] Migrate to uv package manager
- [x] Document ffmpeg installation requirements
- [x] Verify compatibility across platforms

**Deliverables:**
- `pyproject.toml` (pydub, numpy, scipy, pandas)
- uv-based installation documentation

---

#### Phase 3: Core Module Development ✅ COMPLETED

**3.1 Audio Extraction Module**
- [x] Implement `extract_audio()` function
- [x] Add pydub AudioSegment integration
- [x] Convert to mono, 16 kHz sample rate
- [x] Return duration for boundary validation
- [x] Add error handling (missing audio, corrupted files)

**3.2 Signal Processing Module**
- [x] Implement `detect_silence_segments()` function
- [x] Load WAV files using scipy.io.wavfile
- [x] Design and apply Butterworth band-pass filter
- [x] Normalize audio to int16 range
- [x] Detect contiguous silent regions
- [x] Filter by minimum duration threshold

**3.3 Time Extension Module**
- [x] Implement `extend_segment_times()` function
- [x] Add ±2.2s extension logic
- [x] Implement boundary clamping [0, duration]
- [x] Handle edge cases (negative times, overflow)

**3.4 Manifest Generation Module**
- [x] Implement `create_manifest()` function
- [x] Generate sequential pair IDs (e.g., video_001)
- [x] Create pandas DataFrame with exact column spec
- [x] Use absolute paths for video/audio files
- [x] Save to CSV with proper formatting

**Deliverables:**
- Four modular, documented functions
- Unit-testable components
- Clear separation of concerns

---

#### Phase 4: Integration & Orchestration ✅ COMPLETED

**4.1 Main Processing Loop**
- [x] Implement `process_video_folder()` function
- [x] Create output directories automatically
- [x] Scan for .mp4 files recursively
- [x] Implement skip logic (check manifest existence)
- [x] Coordinate all modules in sequence
- [x] Return processing statistics

**4.2 CLI Interface**
- [x] Implement `main()` with argparse
- [x] Add all configurable parameters as arguments
- [x] Provide sensible defaults
- [x] Add help text and usage examples

**4.3 Logging System**
- [x] Configure logging with timestamps
- [x] Add INFO, WARNING, ERROR levels
- [x] Log progress throughout pipeline
- [x] Provide actionable error messages

**Deliverables:**
- Complete `process_videos.py` script (439 lines)
- Full CLI interface with 9 parameters
- Comprehensive logging system

---

#### Phase 5: Quality Assurance ✅ COMPLETED

- [x] PEP8 compliance check (0 linter errors)
- [x] Verify all edge cases handled
- [x] Confirm output format matches specification
- [x] Test boundary clamping logic
- [x] Validate skip logic prevents reprocessing

**Deliverables:**
- Clean, production-ready code
- Zero linter errors
- Documented edge case handling

---

#### Phase 6: Documentation ✅ COMPLETED

- [x] Create comprehensive README
- [x] Document architecture and data flow
- [x] Provide installation instructions (all platforms)
- [x] Write usage examples and workflows
- [x] Explain output structure and CSV format
- [x] Add configuration parameter guide
- [x] Include troubleshooting section
- [x] Document development process timeline

**Deliverables:**
- `README.md` (this document)
- Complete user and developer documentation

---

### Development Metrics

| Metric | Value |
|--------|-------|
| **Total Development Phases** | 6 |
| **Core Modules** | 5 |
| **Total Functions** | 7 |
| **Lines of Code** | 439 |
| **Configuration Parameters** | 9 |
| **Dependencies** | 4 (+ ffmpeg) |
| **Linter Errors** | 0 |
| **Edge Cases Handled** | 6 |

---

## Troubleshooting

### Common Issues

**1. "ffmpeg not found" error**

**Cause:** ffmpeg is not installed or not in PATH

**Solution:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

---

**2. "No .mp4 files found" warning**

**Cause:** Input folder is empty or contains no .mp4 files

**Solution:**
- Verify you're pointing to the correct folder
- Check that files have `.mp4` extension (case-sensitive on Linux/macOS)
- Use absolute paths to avoid ambiguity

```bash
# Check folder contents
ls -la /path/to/videos/

# Use absolute path
python process_videos.py --input "$(pwd)/videos"
```

---

**3. "Failed to extract audio" error**

**Cause:** Video file is corrupted, unsupported codec, or no audio track

**Solution:**
- Verify video plays correctly in a media player
- Check if video has an audio track: `ffmpeg -i video.mp4`
- Try re-encoding the video:
```bash
ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4
```

---

**4. No silence segments detected**

**Cause:** Detection parameters are too strict, or video genuinely has no silence

**Solution:**
- Increase volume threshold: `--volume-threshold 8`
- Decrease minimum duration: `--min-silence-ms 5`
- Check audio visually: open `audio/your_file.wav` in Audacity

---

**5. Permission denied when creating folders**

**Cause:** No write permissions in current directory

**Solution:**
```bash
# Run from a directory where you have write permissions
cd ~/Documents
python /path/to/process_videos.py --input /path/to/videos

# Or specify writable output locations
python process_videos.py \
  --input /path/to/videos \
  --audio-output ~/output/audio \
  --manifest-output ~/output/manifests
```

---

**6. Script runs but output folders are empty**

**Cause:** All videos already processed (manifests exist)

**Solution:**
- Check for existing manifests: `ls manifests/`
- Delete manifests to reprocess: `rm manifests/*.csv`
- Check logs for "Skipping" messages

---

**7. ImportError or ModuleNotFoundError**

**Cause:** Python dependencies not installed

**Solution:**
```bash
# Reinstall all dependencies with uv
uv sync

# Or use pip to install from pyproject.toml
uv pip install -e .

# Verify imports
uv run python -c "import pydub, numpy, scipy, pandas"
```

---

### Getting Help

**Check logs:** The script provides detailed logging output. Review the console output for specific error messages.

**Verbose debugging:**
```python
# Add to top of process_videos.py temporarily
logging.basicConfig(level=logging.DEBUG)
```

**Test with a single file:**
```bash
# Create a test folder with one video
mkdir test_videos
cp your_video.mp4 test_videos/
uv run process_videos.py --input test_videos
```

---

### Useful uv Commands

**Managing Dependencies:**
```bash
# Sync dependencies (install/update based on pyproject.toml)
uv sync

# Add a new dependency
uv add scipy matplotlib

# Remove a dependency
uv remove matplotlib

# Update all dependencies
uv sync --upgrade

# Show installed packages
uv pip list
```

**Running Scripts:**
```bash
# Run Python script with uv (uses project venv)
uv run process_videos.py --input /path/to/videos

# Run Python command
uv run python -c "print('Hello')"

# Activate the virtual environment manually
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Then run without 'uv run' prefix
python process_videos.py --input /path/to/videos
```

**Environment Management:**
```bash
# Create a fresh virtual environment
uv venv

# Remove virtual environment (then recreate with uv sync)
rm -rf .venv
uv sync
```

---

## Master Map & Annotation JSON

### What is the Master Map?

The `master_map.csv` file is automatically created/updated by `process_videos.py` and serves as a central index of all processed videos. It tracks video metadata including duration, which is essential for generating accurate annotation JSON files.

### Generating Annotation JSON Files

After processing videos, you can generate annotation JSON files from EDL files using `createAnnotationJson.py`.

#### Automatic Mode (Recommended)

Use the `--use-master-map` flag to automatically retrieve video duration:

```bash
# Generate annotation JSON using master map
uv run python createAnnotationJson.py \
  --file_path manifests/video_name/video_name_marker.edl \
  --use-master-map

# Output: manifests/video_name/video_name_annotation.json
```

#### Manual Mode

Specify parameters manually if needed:

```bash
uv run python createAnnotationJson.py \
  --file_path manifests/video/video_marker.edl \
  --video_id my_video \
  --duration 15200.5 \
  --fps 24 \
  --output custom_output.json
```

#### Batch Processing

Generate JSON for all processed videos:

```bash
for edl in manifests/*/*.edl; do
  uv run python createAnnotationJson.py --file_path "$edl" --use-master-map
done
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--file_path` | Yes | Path to EDL file |
| `--use-master-map` | No | Load metadata from master_map.csv (recommended) |
| `--video_id` | No | Video identifier (auto-detected if not provided) |
| `--duration` | No | Video duration in seconds (from master map if --use-master-map) |
| `--fps` | No | Frames per second (default: 24) |
| `--output` | No | Output JSON path (default: video manifest folder) |
| `--master-map-path` | No | Path to master_map.csv (default: master_map.csv) |

---

## Next Steps

After generating manifests:

1. **Manual CP Annotation**: Review each manifest CSV and fill in the `CP` column
   - `1` = Change Point detected (transition from game to ad or vice versa)
   - `0` = Not a Change Point (false positive silence)

2. **Generate Annotation JSON**: Use `createAnnotationJson.py` with manually annotated EDL files to create structured JSON annotations

3. **Data Validation**: Use the annotated manifests to extract video clips for model training

4. **Model Training**: Feed the labeled data into your CP detection model

5. **Iterative Refinement**: Adjust detection parameters based on annotation results

---

## License

This project is part of the CP Detection research pipeline.

---

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated:** December 2, 2025  
**Version:** 1.1.0  
**Status:** Production Ready ✅  
**Package Manager:** uv (blazingly fast! 🚀)

