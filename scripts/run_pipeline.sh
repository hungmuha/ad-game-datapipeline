#!/bin/bash
#
# Script: run_pipeline.sh
# Description: Run the complete NFL ad detection data pipeline
# Usage: ./scripts/run_pipeline.sh [--skip-scraper] [--skip-download] [--skip-process]
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse arguments
SKIP_SCRAPER=false
SKIP_DOWNLOAD=false
SKIP_PROCESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-scraper)
            SKIP_SCRAPER=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-process)
            SKIP_PROCESS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-scraper    Skip step 1 (Fubo scraping)"
            echo "  --skip-download   Skip step 2 (game download)"
            echo "  --skip-process    Skip step 3 (video processing)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all steps"
            echo "  $0 --skip-scraper                     # Skip scraping, use existing CSV"
            echo "  $0 --skip-scraper --skip-download     # Process already downloaded videos"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║          NFL Ad Detection - Complete Pipeline             ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}This script will run the complete data acquisition pipeline:${NC}"
echo -e "  ${GREEN}Step 1:${NC} Scrape Fubo TV stream URLs"
echo -e "  ${GREEN}Step 2:${NC} Download NFL games to MP4"
echo -e "  ${GREEN}Step 3:${NC} Process videos (detect silent segments)"
echo -e "  ${YELLOW}Manual:${NC} Review in DaVinci Resolve (you'll be prompted)"
echo -e "  ${GREEN}Step 4:${NC} Generate annotation JSON files"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

MISSING_DEPS=false

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    MISSING_DEPS=true
else
    echo -e "${GREEN}✓ Python found${NC}"
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found${NC}"
    MISSING_DEPS=true
else
    echo -e "${GREEN}✓ Node.js found${NC}"
fi

if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}✗ FFmpeg not found${NC}"
    MISSING_DEPS=true
else
    echo -e "${GREEN}✓ FFmpeg found${NC}"
fi

if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv not found${NC}"
    MISSING_DEPS=true
else
    echo -e "${GREEN}✓ uv found${NC}"
fi

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    echo -e "${RED}Missing dependencies. Please install them first.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"
echo ""
read -p "Press Enter to start the pipeline (or Ctrl+C to cancel)..."
echo ""

# Step 1: Scrape Fubo
if [ "$SKIP_SCRAPER" = false ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Starting Step 1: Scraping Fubo TV${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    "$SCRIPT_DIR/01_scrape_fubo.sh" --headless
    echo ""
    echo -e "${GREEN}Step 1 complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping Step 1: Scraping Fubo TV${NC}"
    echo ""
fi

# Step 2: Download games
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Starting Step 2: Downloading NFL Games${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    "$SCRIPT_DIR/02_download_games.sh"
    echo ""
    echo -e "${GREEN}Step 2 complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping Step 2: Downloading NFL Games${NC}"
    echo ""
fi

# Step 3: Process videos
if [ "$SKIP_PROCESS" = false ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Starting Step 3: Processing Videos${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    "$SCRIPT_DIR/03_process_videos.sh"
    echo ""
    echo -e "${GREEN}Step 3 complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping Step 3: Processing Videos${NC}"
    echo ""
fi

# Manual step reminder
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Manual Review Required (DaVinci Resolve)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The pipeline is paused for manual annotation.${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Open DaVinci Resolve"
echo -e "  2. Import MP4 file from: ${YELLOW}data/raw_videos/${NC}"
echo -e "  3. Import EDL file from: ${YELLOW}data/processed/edl/${NC}"
echo -e "  4. Review markers on timeline:"
echo -e "     ${RED}• Red markers${NC} = Ad boundaries (content→ad or ad→content)"
echo -e "     • Other colors = Non-boundaries (delete or keep for reference)"
echo -e "  5. Export adjusted EDL back to: ${YELLOW}data/processed/manifests/{video_name}/TimelineTest.edl${NC}"
echo ""
echo -e "${YELLOW}See docs/ANNOTATION_WORKFLOW.md for detailed instructions${NC}"
echo ""
read -p "Press Enter when DaVinci Resolve review is complete..."
echo ""

# Step 4: Create annotations
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting Step 4: Creating Annotation JSON${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
"$SCRIPT_DIR/04_create_annotations.sh"
echo ""

# Pipeline complete
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║              Pipeline Complete! 🎉                         ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo -e "  ✓ Stream URLs extracted"
echo -e "  ✓ NFL games downloaded"
echo -e "  ✓ Videos processed (silent segments detected)"
echo -e "  ✓ Manual review completed"
echo -e "  ✓ Annotation JSON files generated"
echo ""
echo -e "${GREEN}Output locations:${NC}"
echo -e "  • Videos: ${YELLOW}data/raw_videos/${NC}"
echo -e "  • Processed: ${YELLOW}data/processed/${NC}"
echo -e "  • Annotations: ${YELLOW}data/annotations/${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Review annotations in data/annotations/"
echo -e "  2. Extract short clips for training"
echo -e "  3. Begin model training"
echo ""
echo -e "${YELLOW}See docs/GETTING_STARTED.md for training instructions${NC}"
echo ""

# Generate pipeline report
echo -e "${BLUE}Generating comprehensive pipeline report...${NC}"
"$SCRIPT_DIR/generate_pipeline_report.sh"
echo ""
