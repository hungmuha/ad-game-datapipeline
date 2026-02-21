# Pipeline Orchestration Scripts

This directory contains bash scripts to orchestrate the NFL Ad Detection data pipeline.

## Quick Start

### Check Prerequisites
```bash
./scripts/check_prerequisites.sh
```

### Run Complete Pipeline
```bash
./scripts/run_pipeline.sh
```

### Run Individual Steps
```bash
./scripts/01_scrape_fubo.sh          # Extract stream URLs
./scripts/02_download_games.sh       # Download games
./scripts/03_process_videos.sh       # Process videos
./scripts/04_create_annotations.sh   # Generate annotations
```

---

## Script Descriptions

### `generate_pipeline_report.sh`
**Purpose**: Generate comprehensive pipeline status report

**Usage**:
```bash
./scripts/generate_pipeline_report.sh [--output report.md]
```

**Options**:
- `--output` - Custom output path (default: `outputs/pipeline_report_YYYYMMDD_HHMMSS.md`)

**Output**:
- Markdown report with:
  - Pipeline status overview (all 4 steps)
  - File counts and coverage percentages
  - Storage usage breakdown
  - Download statistics
  - Next steps recommendations
  - Complete file listings

**Example**:
```bash
# Generate report with default name
./scripts/generate_pipeline_report.sh

# Generate report with custom name
./scripts/generate_pipeline_report.sh --output my_report.md
```

**Use Cases**:
- Check pipeline progress
- Verify processing coverage
- Monitor storage usage
- Identify what needs to be done next
- Generate status reports for documentation

**Automatically runs** at the end of `run_pipeline.sh`

---

### `check_prerequisites.sh`
**Purpose**: Verify all required dependencies are installed

**Usage**:
```bash
./scripts/check_prerequisites.sh
```

**Checks**:
- Python (required)
- Node.js & npm (required)
- FFmpeg (required)
- uv (required)
- git (required)
- Chrome/Chromium (optional, for scraper)
- DaVinci Resolve (optional, for annotation)

---

### `01_scrape_fubo.sh`
**Purpose**: Extract NFL game stream URLs from Fubo TV

**Usage**:
```bash
./scripts/01_scrape_fubo.sh [--headless]
```

**Prerequisites**:
- Python 3.8+ (uses venv for virtual environments)
- Chrome/Chromium browser
- `.env` file in `pipeline/01-fubo-scraper/` with Fubo credentials
- `.env` file must also have FUBO_RECORDINGS_URL with the recordings page URL

**Output**:
- CSV files in `pipeline/01-fubo-scraper/output/`

**Example**:
```bash
# Run with visible browser
./scripts/01_scrape_fubo.sh

# Run in headless mode
./scripts/01_scrape_fubo.sh --headless
```

---

### `02_download_games.sh`
**Purpose**: Download NFL games from stream URLs to local MP4 files

**Usage**:
```bash
./scripts/02_download_games.sh [path/to/csv]
```

**Prerequisites**:
- Node.js & npm
- FFmpeg
- CSV file from step 1 (or provide path)

**Output**:
- MP4 files in `data/raw_videos/`
- Download log in `pipeline/02-download-games/master.csv`

**Example**:
```bash
# Use most recent CSV from scraper
./scripts/02_download_games.sh

# Use specific CSV
./scripts/02_download_games.sh pipeline/01-fubo-scraper/output/fubo_recordings_20260130.csv
```

---

### `03_process_videos.sh`
**Purpose**: Detect silent segments in videos (change point candidates)

**Usage**:
```bash
./scripts/03_process_videos.sh [--input /path/to/videos]
```

**Prerequisites**:
- uv (Python package manager)
- FFmpeg
- MP4 files from step 2

**Output**:
- EDL files in `data/processed/edl/`
- CSV manifests in `data/processed/manifests/`
- WAV audio in `data/processed/audio/`

**Example**:
```bash
# Process videos from default location (data/raw_videos)
./scripts/03_process_videos.sh

# Process videos from custom location
./scripts/03_process_videos.sh --input /path/to/videos
```

**Technical Notes**:

*Direct Write Behavior*:
- Script writes directly to `data/processed/` (not to local working directory)
- No file moving or cleanup needed
- Prevents file duplication (saves ~8GB)

*Smart Audio Reuse*:
The script intelligently reuses existing audio files:
- **Full skip**: All 3 files exist (CSV + EDL + WAV) → skips video entirely
- **Audio reuse**: Only audio exists → reuses audio, regenerates manifests (~30 sec/video)
- **Fresh extraction**: Audio missing → extracts from video (~5-10 min/video)

*Reprocessing with Parameter Changes*:
If you changed parameters (fps, silence threshold, etc.) and want to regenerate manifests:
```bash
# Delete only manifests (keeps audio for reuse)
rm -rf data/processed/manifests/

# Reprocess (will reuse audio - much faster!)
./scripts/03_process_videos.sh

# Watch for: "Reusing existing audio file: video_name.wav"
```

This saves ~2-3 minutes per video by reusing extracted audio while regenerating manifests with new parameters.

---

### `04_create_annotations.sh`
**Purpose**: Generate annotation JSON files from DaVinci Resolve EDL

**Usage**:
```bash
./scripts/04_create_annotations.sh [--edl /path/to/file.edl]
```

**Prerequisites**:
- Python
- Adjusted EDL files from DaVinci Resolve manual review
- `master_map.csv` in `pipeline/03-process-videos/`

**Output**:
- Annotation JSON files in `data/annotations/`

**Example**:
```bash
# Process all EDL files in data/processed/manifests
./scripts/04_create_annotations.sh

# Process specific EDL file
./scripts/04_create_annotations.sh --edl data/processed/manifests/game1/TimelineTest.edl
```

**Important**: This step requires manual review in DaVinci Resolve before running!

---

### `run_pipeline.sh`
**Purpose**: Run the complete end-to-end pipeline

**Usage**:
```bash
./scripts/run_pipeline.sh [OPTIONS]
```

**Options**:
- `--skip-scraper` - Skip step 1 (use existing CSV)
- `--skip-download` - Skip step 2 (use existing videos)
- `--skip-process` - Skip step 3 (use existing processed outputs)
- `--help` - Show help message

**Features**:
- Automatically generates comprehensive pipeline report at end
- Shows storage usage and progress metrics
- Opens report in default viewer (macOS)

**Example**:
```bash
# Run complete pipeline
./scripts/run_pipeline.sh

# Skip scraping, use existing CSV
./scripts/run_pipeline.sh --skip-scraper

# Process already downloaded videos
./scripts/run_pipeline.sh --skip-scraper --skip-download
```

**Pipeline Flow**:
1. Scrape Fubo TV → CSV
2. Download games → MP4
3. Process videos → EDL/CSV/WAV
4. **[Manual]** Review in DaVinci Resolve
5. Create annotations → JSON

---

## Common Workflows

### First Time Setup
```bash
# 1. Check all dependencies
./scripts/check_prerequisites.sh

# 2. Setup Fubo credentials
# Edit pipeline/01-fubo-scraper/.env with your credentials

# 3. Run complete pipeline
./scripts/run_pipeline.sh
```

### Regular Workflow (Weekly)
```bash
# Scrape new games
./scripts/01_scrape_fubo.sh --headless

# Download new games
./scripts/02_download_games.sh

# Process new videos
./scripts/03_process_videos.sh

# Review in DaVinci Resolve (manual)
# ...

# Generate annotations
./scripts/04_create_annotations.sh
```

### Resume After Manual Review
```bash
# After completing DaVinci Resolve review
./scripts/04_create_annotations.sh
```

### Re-process Existing Videos
```bash
# Re-process with different parameters
./scripts/03_process_videos.sh --input data/raw_videos
```

---

## Troubleshooting

### "Command not found" errors
Run `./scripts/check_prerequisites.sh` to verify all dependencies are installed.

### Fubo scraper fails
- Check `.env` file has correct credentials
- Try running without `--headless` to see browser
- Check `pipeline/01-fubo-scraper/README.md` for troubleshooting

### Download fails
- Verify FFmpeg is installed: `ffmpeg -version`
- Check CSV file has valid stream URLs
- Some streams may be DRM-protected (will be skipped)

### Processing fails
- Verify uv is installed: `uv --version`
- Check input directory has MP4 files
- See `pipeline/03-process-videos/README.md` for parameter tuning

### No annotations generated
- Ensure EDL files were exported from DaVinci Resolve
- Check files are in correct location: `data/processed/manifests/{video_name}/`
- Verify master_map.csv exists in `pipeline/03-process-videos/`

---

## Environment Variables

Scripts use the following environment variables (optional):

```bash
# Override default directories
export RAW_VIDEOS_DIR="/custom/path/to/videos"
export PROCESSED_DIR="/custom/path/to/processed"
export ANNOTATIONS_DIR="/custom/path/to/annotations"
```

---

## Error Handling

All scripts:
- Exit immediately on errors (`set -e`)
- Provide colored output for success/warning/error
- Check prerequisites before running
- Give clear error messages with solutions

---

## Contributing

When adding new scripts:
1. Follow naming convention: `NN_description.sh`
2. Include usage comment at top
3. Use color variables for output
4. Check prerequisites
5. Provide clear error messages
6. Make executable: `chmod +x script.sh`

---

## Documentation

- **Complete Pipeline**: `../docs/DATA_PIPELINE.md`
- **Annotation Workflow**: `../docs/ANNOTATION_WORKFLOW.md`
- **Getting Started**: `../docs/GETTING_STARTED.md`
- **Pipeline Components**: `../pipeline/README.md`
