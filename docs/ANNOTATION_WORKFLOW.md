# Annotation Workflow Documentation

**Last Updated**: 2026-01-29
**Status**: Workflow Defined, Script Created in CPPreData

---

## Overview

This document describes the complete annotation workflow from raw video to training-ready annotation JSON files.

---

## Workflow Pipeline

```
Raw Video (MP4)
    ↓
CPPreData: process_videos.py
    ↓
EDL + CSV (all silent segments detected)
    ↓
DaVinci Resolve: Manual Review & Color Coding
    ↓
Adjusted EDL (color-coded markers)
    ↓
CPPreData: createAnnotationJson.py
    ↓
Annotation JSON (segment-level)
    ↓
ad-game-explainer: Training
```

---

## Component 1: CPPreData Process Videos

**Location**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/process_videos.py`

**Purpose**: Detect all silent segments in raw video

**Input**:
- MP4 files from GetGamesToLocal
- Example: `/path/to/game.mp4`

**Output**:
- EDL file: `manifests/{video_name}/{video_name}_marker.edl`
- CSV file: `manifests/{video_name}/{video_name}.csv`
- WAV file: `audio/{video_name}.wav`

**EDL Contains**: ~200-300 markers at each detected silent segment

---

## Component 2: DaVinci Resolve Manual Review

**Tool**: DaVinci Resolve (Video Editing Software)

**Workflow**:
1. Import MP4 file
2. Import EDL file (markers appear on timeline)
3. Review each marker (~200-300 per game)
4. **Color-code markers**:
   - **Red (ResolveColorRed)**: TRUE CP - boundary between content/ad or ad/content
   - **Other colors**: Non-CP - keep for reference or delete
5. Export adjusted EDL file

**Key Decision**:
Based on `createAnnotationJson.py` line 70-71:
```python
label = "ad" if color == "ResolveColorRed" else "content"
```

**Color Coding Strategy**:
- Red marker at timestamp = start of ad segment
- Non-red marker at timestamp = start of content segment

---

## Component 3: Create Annotation JSON

**Location**: `/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/createAnnotationJson.py`

**Purpose**: Convert color-coded EDL to annotation JSON

### Usage

```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/

# Option 1: Using master_map.csv (recommended)
python createAnnotationJson.py \
  --file_path manifests/cowboys_eagles_week1/cowboys_eagles_week1_marker.edl \
  --use-master-map

# Option 2: Manual metadata
python createAnnotationJson.py \
  --file_path manifests/cowboys_eagles_week1/cowboys_eagles_week1_marker.edl \
  --video_id cowboys_eagles_week1 \
  --duration 10800 \
  --fps 30
```

### Script Functionality

**What It Does**:
1. Parses EDL file and extracts all markers with colors
2. Determines label based on marker color:
   - Red markers → ad segment starts here
   - Other colors → content segment starts here
3. Creates segments between consecutive markers
4. Generates change_points array (marker timestamps)
5. Saves annotation JSON

**Output Location** (from script lines 244-259):
```
manifests/{video_name}/{video_id}_annotation.json
```

**Example Output Path**:
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/manifests/cowboys_eagles_week1/cowboys_eagles_week1_annotation.json
```

### Output Format

```json
{
  "video_id": "cowboys_eagles_week1",
  "duration_seconds": 10800.5,
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
    }
  ],
  "change_points": [1245.3, 1335.8, 2890.5]
}
```

**Fields Generated**:
- `video_id`: From filename or master_map.csv
- `duration_seconds`: From master_map.csv or --duration argument
- `segments`: Array of segments between markers
  - `segment_id`: Sequential index
  - `start_time`: Marker timestamp (seconds)
  - `end_time`: Next marker timestamp (seconds)
  - `label`: "ad" or "content" (based on marker color)
  - `type`: "commercial_break" or "game_play"
- `change_points`: Array of all marker timestamps

---

## CRITICAL GAP: Silent Segment Annotations

### Issue

The current `createAnnotationJson.py` generates **segment-level** annotations (for Ad classifier), but does NOT generate **silent segment** annotations needed for CP classifier training.

### What's Missing

Based on our training approach analysis, we need:

```json
{
  "video_id": "...",
  "duration_seconds": ...,

  "silent_segments": [  // ← MISSING FROM CURRENT SCRIPT
    {
      "segment_id": "silence_001",
      "timestamp": 73.32,
      "is_CP": 0,
      "notes": "Pause in commentary"
    },
    {
      "segment_id": "silence_002",
      "timestamp": 1245.3,
      "is_CP": 1,
      "notes": "Content → Ad (TRUE CP)"
    }
  ],

  "segments": [...],  // ← Currently generated
  "change_points": [...]  // ← Currently generated
}
```

### Solution Options

**Option A**: Extend `createAnnotationJson.py`
- Add `--include-all-silent-segments` flag
- Read original CSV file (has ALL silent segments from CPPreData)
- Mark segments as is_CP=1 if they appear in EDL, is_CP=0 otherwise
- Add silent_segments array to output JSON

**Option B**: Separate workflow
- Keep current script for segment-level annotations
- Create separate script to generate silent segment annotations
- Combine later during training data preparation

**Option C**: Two-pass DaVinci Resolve workflow
- Pass 1: Mark TRUE CPs with red → generate segment annotations
- Pass 2: Review ALL silent segments (from CSV), mark CP vs non-CP → generate silent segment annotations

---

## Future Automation Plan

**Automated Pipeline** (mentioned by user):

```
Claude Workflow Agent
    ↓
1. Fetch videos from Fubo (fubo-scraper)
    ↓
2. Download to local (GetGamesToLocal)
    ↓
3. Auto-trigger: process_videos.py
    ↓
4. Generate EDL + CSV files
    ↓
5. [Manual step or AI-assisted] DaVinci Resolve review
    ↓
6. Auto-trigger: createAnnotationJson.py
    ↓
7. Generate annotation JSON
    ↓
8. Auto-trigger: Training pipeline
```

**Automation Triggers**:
- Video download complete → Start process_videos.py
- EDL generation complete → Notify for manual review
- Adjusted EDL saved → Start createAnnotationJson.py
- Annotation JSON generated → Start training preparation

---

## Current Annotation Locations

### CPPreData Repository

**EDL Files** (manually adjusted):
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/manifests/{video_name}/{video_name}_marker.edl
```

**Annotation JSON** (generated by script):
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/manifests/{video_name}/{video_id}_annotation.json
```

**CSV Files** (reference):
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/manifests/{video_name}/{video_name}.csv
```

### ad-game-explainer Repository

**Copy annotations to** (for training):
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/data/annotations/{video_id}_annotation.json
```

---

## Recommended Next Steps

1. **Test Current Workflow**:
   - Run process_videos.py on 1 game
   - Manually review EDL in DaVinci Resolve
   - Color-code markers (red = ad boundaries)
   - Run createAnnotationJson.py
   - Verify output JSON format

2. **Address Silent Segment Gap**:
   - Decide on Option A, B, or C above
   - Extend createAnnotationJson.py or create new script
   - Generate silent_segments array with is_CP labels

3. **Copy Annotations for Training**:
   - Create script to copy annotation JSONs from CPPreData to ad-game-explainer/data/annotations/
   - Validate format matches training requirements

4. **Setup Automation**:
   - Plan Claude workflow agent triggers
   - Define automation checkpoints
   - Implement error handling and notifications

---

## Master Map Integration

The script supports `master_map.csv` for automatic metadata lookup:

**master_map.csv Format**:
```csv
video_name,video_path,edl_path,duration_seconds
cowboys_eagles_week1,/path/to/video.mp4,manifests/video/video_marker.edl,10800.5
```

**Usage**:
```bash
python createAnnotationJson.py \
  --file_path manifests/video/video_marker.edl \
  --use-master-map \
  --master-map-path master_map.csv
```

This automatically populates:
- `video_id` from video_name
- `duration_seconds` from master_map
- No need for manual --video_id and --duration arguments

---

## Summary

**Status**: ✅ Segment-level annotation workflow complete
**Gap**: ⚠️ Silent segment annotations for CP classifier training needed
**Location**: Annotations saved to `CPPreData/manifests/{video_name}/{video_id}_annotation.json`
**Next**: Address silent segment annotation gap and copy to ad-game-explainer for training

---

**See Also**:
- [DATA_PIPELINE.md](DATA_PIPELINE.md) - Complete 4-repository workflow
- [TRAINING_APPROACH.md](TRAINING_APPROACH.md) - CP classifier training requirements
- CPPreData README.md - process_videos.py documentation
