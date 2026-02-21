#!/bin/bash
#
# Script: 03_process_videos.sh
# Description: Detect silent segments in downloaded NFL games
# Usage: ./scripts/03_process_videos.sh [--input /path/to/videos]
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROCESSOR_DIR="$PROJECT_ROOT/pipeline/03-process-videos"
RAW_VIDEOS_DIR="$PROJECT_ROOT/data/raw_videos"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 3: Processing Videos (Silent Segment Detection)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if processor directory exists
if [ ! -d "$PROCESSOR_DIR" ]; then
    echo -e "${RED}Error: Processor directory not found at $PROCESSOR_DIR${NC}"
    exit 1
fi

# Navigate to processor directory
cd "$PROCESSOR_DIR"

# Check for Python/uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found${NC}"
    echo -e "${YELLOW}Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg not found${NC}"
    echo -e "${YELLOW}Install FFmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)${NC}"
    exit 1
fi

# Sync dependencies
echo -e "${YELLOW}Syncing dependencies with uv...${NC}"
uv sync

# Determine input directory
INPUT_DIR="$RAW_VIDEOS_DIR"
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW}Usage: $0 [--input /path/to/videos]${NC}"
            exit 1
            ;;
    esac
done

# Check if input directory exists and has videos
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

VIDEO_COUNT=$(ls -1 "$INPUT_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No MP4 files found in $INPUT_DIR${NC}"
    echo -e "${YELLOW}Run ./scripts/02_download_games.sh first${NC}"
    exit 1
fi

echo -e "${GREEN}Found $VIDEO_COUNT video(s) to process${NC}"
echo ""

# Create processed directory structure
mkdir -p "$PROCESSED_DIR/audio"
mkdir -p "$PROCESSED_DIR/manifests"
mkdir -p "$PROCESSED_DIR/edl"

# Run processor - write directly to final destination
echo -e "${GREEN}Processing videos...${NC}"
echo -e "${YELLOW}This may take 5-10 minutes per game${NC}"
echo ""

# Use absolute paths to write directly to data/processed/
uv run process_videos.py \
    --input "$INPUT_DIR" \
    --audio-output "$PROCESSED_DIR/audio" \
    --manifest-output "$PROCESSED_DIR/manifests"

# Note: EDL files are created inside manifest directories
# Move them to dedicated edl/ directory for easier access
echo ""
echo -e "${GREEN}Organizing EDL files...${NC}"
if [ -d "$PROCESSED_DIR/manifests" ]; then
    find "$PROCESSED_DIR/manifests" -name "*_marker.edl" -exec cp {} "$PROCESSED_DIR/edl/" \; 2>/dev/null || true
fi

echo -e "${GREEN}✓ Processing complete - files written directly to $PROCESSED_DIR${NC}"

echo ""
echo -e "${GREEN}✓ Processing complete!${NC}"
echo -e "${YELLOW}EDL files: $PROCESSED_DIR/edl/${NC}"
echo -e "${YELLOW}CSV manifests: $PROCESSED_DIR/manifests/${NC}"
echo -e "${YELLOW}WAV audio: $PROCESSED_DIR/audio/${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Import EDL files into DaVinci Resolve"
echo -e "  2. Review and color-code markers (Red = ad boundaries)"
echo -e "  3. Run ./scripts/04_create_annotations.sh"
