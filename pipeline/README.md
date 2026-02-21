# Data Pipeline

This directory contains all data acquisition and preprocessing components for the NFL Ad Detection system.

## Pipeline Overview

```
01-fubo-scraper → CSV → 02-download-games → MP4 → 03-process-videos → EDL/CSV
```

## Components

### 01-fubo-scraper
**Purpose**: Extract stream URLs from Fubo TV
**Technology**: Python + Selenium-Wire
**Input**: Fubo TV credentials
**Output**: CSV files with stream URLs

**Usage**:
```bash
cd pipeline/01-fubo-scraper
python main.py --headless
```

See `01-fubo-scraper/README.md` for detailed documentation.

---

### 02-download-games
**Purpose**: Download NFL games to local MP4 files
**Technology**: Node.js + FFmpeg
**Input**: CSV from fubo-scraper
**Output**: MP4 files + master.csv tracking

**Usage**:
```bash
cd pipeline/02-download-games
npm install
node download-streams.js /path/to/fubo_recordings.csv
```

See `02-download-games/README.md` for detailed documentation.

---

### 03-process-videos
**Purpose**: Detect silent segments (change point candidates)
**Technology**: Python + Pydub + FFmpeg
**Input**: MP4 files from download-games
**Output**: EDL files, CSV manifests, WAV audio

**Usage**:
```bash
cd pipeline/03-process-videos
uv run process_videos.py --input /path/to/videos
```

See `03-process-videos/README.md` for detailed documentation.

---

## Quick Start

Run the complete pipeline using orchestration scripts:

```bash
# Run entire pipeline (steps 1-3)
./scripts/run_pipeline.sh

# Or run individual steps
./scripts/01_scrape_fubo.sh
./scripts/02_download_games.sh
./scripts/03_process_videos.sh
```

## Data Flow

**Output Locations**:
- Fubo scraper CSVs: `pipeline/01-fubo-scraper/output/`
- Downloaded videos: `data/raw_videos/`
- Processed outputs: `data/processed/`
  - EDL files for DaVinci Resolve
  - CSV manifests with timestamps
  - WAV audio files
- Annotations: `data/annotations/`

## Next Steps

After running the pipeline:
1. Import EDL files into DaVinci Resolve
2. Manually review and color-code change points
3. Run `pipeline/03-process-videos/createAnnotationJson.py`
4. Copy annotations to `data/annotations/`
5. Begin training (see `docs/GETTING_STARTED.md`)

## Documentation

- **Complete Pipeline**: `docs/DATA_PIPELINE.md`
- **Annotation Workflow**: `docs/ANNOTATION_WORKFLOW.md`
- **Getting Started**: `docs/GETTING_STARTED.md`
