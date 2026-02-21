# Implementation Documentation - Master Map & Annotation JSON

## Overview

This document provides technical details about the master map and annotation JSON features integrated into the CP Detection Data Pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CP Detection Pipeline                       │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   process_videos.py            │
        │   - Extract audio              │
        │   - Detect silence             │
        │   - Generate manifests         │
        │   - Generate EDL markers       │
        │   - Update master_map.csv      │
        └────────────────────────────────┘
                         │
                         ├──────────────────────────┐
                         ▼                          ▼
        ┌────────────────────────────┐  ┌──────────────────────┐
        │  master_map.csv            │  │  Manifest Folders    │
        │  - video_name              │  │  - CSV files         │
        │  - video_path              │  │  - EDL markers       │
        │  - edl_path                │  │                      │
        │  - duration_seconds        │  │                      │
        └────────────────────────────┘  └──────────────────────┘
                         │                          │
                         └──────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  createAnnotationJson.py      │
                    │  - Load master map            │
                    │  - Parse EDL file             │
                    │  - Generate JSON              │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Annotation JSON              │
                    │  - video_id                   │
                    │  - duration_seconds           │
                    │  - segments                   │
                    │  - change_points              │
                    └───────────────────────────────┘
```

## Implementation Details

### 1. Enhanced `process_videos.py`

**Added `update_master_map()` function:**
- Creates/updates `master_map.csv` in project root
- Tracks: video name, video path, EDL path, and duration
- Automatically called after successful video processing
- Uses pandas for CSV management with update/append logic

**Integration:**
- Called in `process_video_folder()` after manifest and EDL creation
- Captures video duration from audio extraction
- Stores relative EDL paths for portability

### 2. Enhanced `createAnnotationJson.py`

**Added Master Map Integration:**
- New function `load_master_map()` - loads master_map.csv
- New function `find_video_in_master_map()` - looks up video metadata by EDL path
- Automatic video duration retrieval when using `--use-master-map` flag

**Updated JSON Output:**
- Default output location: video's manifest folder
- Format: `manifests/{video_name}/{video_name}_annotation.json`
- Custom output path still supported via `--output` argument

**New CLI Arguments:**
- `--use-master-map`: Enable master map lookup
- `--master-map-path`: Custom path to master_map.csv (default: master_map.csv)

### 3. Updated Documentation

**README.md enhancements:**
- Added master_map.csv format documentation
- Added annotation JSON format examples
- Added usage guide for createAnnotationJson.py
- Updated folder structure diagrams
- Added workflow examples

## File Structure

```
CPPreData/
├── master_map.csv                    # NEW: Master index of all videos
├── process_videos.py                 # MODIFIED: Generates master_map.csv
├── createAnnotationJson.py           # MODIFIED: Consumes master_map.csv
├── README.md                         # MODIFIED: Updated documentation
├── audio/
│   └── video.wav
├── manifests/
│   └── video_name/
│       ├── video_name.csv
│       ├── video_name_marker.edl
│       └── video_name_annotation.json  # NEW: Auto-generated JSON
└── ...
```

## Data Flow

1. **Video Processing** (`process_videos.py`)
   - Extracts audio → calculates duration
   - Detects silence segments
   - Creates CSV manifest and EDL marker file
   - Calls `update_master_map()` → updates master_map.csv

2. **Master Map Storage** (`master_map.csv`)
   - Central index of all processed videos
   - Stores: video_name, video_path, edl_path, duration_seconds
   - Updated/appended automatically during processing

3. **Annotation Generation** (`createAnnotationJson.py`)
   - Reads EDL file (manually annotated or auto-generated)
   - Loads master_map.csv (if `--use-master-map` flag used)
   - Looks up video duration by EDL path
   - Generates structured JSON with segments and change points
   - Saves to video's manifest folder

## File Formats

### master_map.csv

```csv
video_name,video_path,edl_path,duration_seconds
cowboysAtEaglesWeek1,/path/to/video.mp4,manifests/video/video_marker.edl,15142.918
```

**Columns:**
- `video_name`: Video filename without extension
- `video_path`: Absolute path to video file
- `edl_path`: Relative path to marker EDL file (from project root)
- `duration_seconds`: Video duration in seconds (from audio extraction)

### Annotation JSON

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

**Fields:**
- `video_id`: Video identifier
- `duration_seconds`: Total video duration
- `segments`: Array of annotated segments with labels
- `change_points`: Array of timestamps where content changes

## Key Functions

### process_videos.py

**`update_master_map(video_name, video_path, edl_path, duration_seconds)`**
- Creates or updates master_map.csv
- Uses pandas for smart CSV management
- Updates existing entries or appends new ones
- Called automatically after successful video processing

### createAnnotationJson.py

**`load_master_map(master_map_path)`**
- Loads master_map.csv into pandas DataFrame
- Returns None if file doesn't exist
- Handles errors gracefully

**`find_video_in_master_map(edl_path, master_map_df)`**
- Looks up video metadata by EDL path
- Tries exact path match first
- Falls back to video name matching
- Returns dictionary with video metadata or None

**`createAnnotationJson(..., use_master_map, master_map_path)`**
- Enhanced to support master map lookup
- Automatically retrieves duration when `use_master_map=True`
- Saves JSON to video's manifest folder by default
- Backward compatible with manual parameter specification

## Testing & Validation

✅ **Master Map Generation**: Verified with existing video data
✅ **Master Map Lookup**: Successfully retrieves video metadata
✅ **JSON Output Location**: Correctly saves to manifest folder
✅ **Backward Compatibility**: Manual mode still functional
✅ **Code Quality**: Zero linter errors, PEP8 compliant

## Benefits

1. **Automated Workflow**: Duration captured automatically during processing
2. **Single Source of Truth**: master_map.csv indexes all videos
3. **Accurate Annotations**: Uses exact durations from audio extraction
4. **Better Organization**: JSON files stored with their manifests
5. **Flexible**: Supports both automatic and manual workflows

## Implementation Date

January 30, 2026

---

## Recent Updates

### Silent Segments Feature (February 1, 2026)

**Added `silent_segments` array to annotation JSON output.**

#### Changes to `createAnnotationJson.py`:

1. **New Function: `create_silent_segments()`** (Lines 198-233)
   - Filters out markers with timestamp 0.0 (excludes starting point)
   - Creates `silent_segments` entries for remaining markers
   - Sets `is_CP: 0` for ResolveColorYellow markers (non-change points)
   - Sets `is_CP: 1` for all other color markers (change points)
   - Generates segment IDs: "silence_001", "silence_002", etc.
   - Leaves `notes` field empty

2. **Updated `createAnnotationJson()` function** (Lines 270-277)
   - Calls `create_silent_segments(markers)` after parsing EDL
   - Includes `silent_segments` in annotation structure

3. **Updated annotation structure** (Lines 302-309)
   - Added `'silent_segments'` to output JSON
   - Order: video_id, duration_seconds, **silent_segments**, segments, change_points

#### Output Format:

```json
{
  "video_id": "cowboysAtEaglesWeek1",
  "duration_seconds": 15142.918,
  "silent_segments": [
    {
      "segment_id": "silence_001",
      "timestamp": 76.29,
      "is_CP": 1,
      "notes": ""
    },
    {
      "segment_id": "silence_002",
      "timestamp": 85.54,
      "is_CP": 0,
      "notes": ""
    }
  ],
  "segments": [...],
  "change_points": [...]
}
```

#### Color Mapping:

| Color | is_CP Value | Description |
|-------|-------------|-------------|
| ResolveColorYellow | 0 | Non-change point |
| ResolveColorBlue | 1 | Change point |
| ResolveColorRed | 1 | Change point |
| All other colors | 1 | Change point |

---

### Dynamic FPS Detection (February 1, 2026)

**Updated `process_videos.py` to dynamically detect video frame rate instead of assuming fixed 59.94 FPS.**

#### Changes to `process_videos.py`:

1. **Added OpenCV Import** (Line 19)
   ```python
   import cv2
   ```

2. **New Function: `get_video_fps()`** (Lines 31-55)
   ```python
   def get_video_fps(video_path: str) -> float:
       """Get the frame rate (FPS) of a video file."""
       try:
           video = cv2.VideoCapture(video_path)
           fps = video.get(cv2.CAP_PROP_FPS)
           video.release()
           
           if fps > 0:
               logging.info(f"Detected video FPS: {fps}")
               return fps
           else:
               logging.warning(f"Could not detect FPS, defaulting to 59.94")
               return 59.94
       except Exception as e:
           logging.error(f"Failed to detect FPS: {e}, defaulting to 59.94")
           return 59.94
   ```

3. **Updated Video Processing** (Lines 520-521, 577-583)
   - Detects FPS for each video before processing
   - Passes detected FPS to `create_edl_file()`
   - Falls back to 59.94 FPS if detection fails

#### Benefits:

- **Accurate Timecodes**: EDL markers use correct FPS for each video
- **Multi-Source Support**: Handles videos with varying frame rates (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)
- **Backward Compatible**: Falls back to 59.94 FPS if detection fails
- **Transparent**: Logs detected FPS for verification

#### Example Log Output:

```
2026-02-01 10:15:23 - INFO - Processing video_name.mp4
2026-02-01 10:15:23 - INFO - Detected video FPS: 29.97
2026-02-01 10:15:24 - INFO - EDL file created: manifests/video_name/video_name_marker.edl (128 markers)
```

---

### Path Resolution Fix (February 1, 2026)

**Fixed relative path calculation error in master_map.csv update.**

#### Change to `process_videos.py` (Line 556):

**Before:**
```python
edl_relative_path = str(edl_path.relative_to(Path.cwd()))
```

**After:**
```python
edl_relative_path = str(edl_path.resolve().relative_to(Path.cwd().resolve()))
```

**Fix**: Converts both paths to absolute before calculating relative path, preventing "one path is relative and the other is absolute" error.

