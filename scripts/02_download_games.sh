#!/bin/bash
#
# Script: 02_download_games.sh
# Description: Download NFL games from Fubo to local MP4 files
# Usage: ./scripts/02_download_games.sh [path/to/csv]
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
DOWNLOADER_DIR="$PROJECT_ROOT/pipeline/02-download-games"
SCRAPER_OUTPUT="$PROJECT_ROOT/pipeline/01-fubo-scraper/output"
RAW_VIDEOS_DIR="$PROJECT_ROOT/data/raw_videos"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 2: Downloading NFL Games${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if downloader directory exists
if [ ! -d "$DOWNLOADER_DIR" ]; then
    echo -e "${RED}Error: Downloader directory not found at $DOWNLOADER_DIR${NC}"
    exit 1
fi

# Navigate to downloader directory
cd "$DOWNLOADER_DIR"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    echo -e "${YELLOW}Install Node.js: https://nodejs.org/${NC}"
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg not found${NC}"
    echo -e "${YELLOW}Install FFmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)${NC}"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Determine CSV file
if [ -n "$1" ]; then
    CSV_FILE="$1"
else
    # Find most recent CSV from scraper output
    CSV_FILE=$(ls -t "$SCRAPER_OUTPUT"/fubo_recordings_*.csv 2>/dev/null | head -1)
    if [ -z "$CSV_FILE" ]; then
        echo -e "${RED}Error: No CSV file found${NC}"
        echo -e "${YELLOW}Usage: $0 <path/to/csv>${NC}"
        echo -e "${YELLOW}Or run ./scripts/01_scrape_fubo.sh first${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Using most recent CSV: $CSV_FILE${NC}"
fi

# Check if CSV exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: CSV file not found: $CSV_FILE${NC}"
    exit 1
fi

# Create raw_videos directory if it doesn't exist
mkdir -p "$RAW_VIDEOS_DIR"

# Run downloader
echo -e "${GREEN}Downloading games from CSV...${NC}"
echo -e "${YELLOW}This may take a while (each game ~2-3 hours)${NC}"
node download-streams.js "$CSV_FILE"

# Move downloaded videos to data/raw_videos
if [ -d "videos" ] && [ "$(ls -A videos/*.mp4 2>/dev/null)" ]; then
    echo -e "${GREEN}Moving videos to $RAW_VIDEOS_DIR${NC}"
    mv videos/*.mp4 "$RAW_VIDEOS_DIR/" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}✓ Download complete!${NC}"
echo -e "${YELLOW}Videos saved to: $RAW_VIDEOS_DIR${NC}"
echo -e "${YELLOW}Download log: $DOWNLOADER_DIR/master.csv${NC}"
echo ""

# Count downloaded videos
VIDEO_COUNT=$(ls -1 "$RAW_VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
echo -e "${GREEN}Total videos downloaded: $VIDEO_COUNT${NC}"
echo ""
echo -e "${GREEN}Next step: Run ./scripts/03_process_videos.sh${NC}"
