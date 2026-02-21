#!/bin/bash
#
# Script: 04_create_annotations.sh
# Description: Generate annotation JSON from DaVinci Resolve EDL files
# Usage: ./scripts/04_create_annotations.sh [--edl /path/to/file.edl]
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
PROCESSED_DIR="$PROJECT_ROOT/data/processed"
ANNOTATIONS_DIR="$PROJECT_ROOT/data/annotations"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 4: Creating Annotation JSON${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if processor directory exists
if [ ! -d "$PROCESSOR_DIR" ]; then
    echo -e "${RED}Error: Processor directory not found at $PROCESSOR_DIR${NC}"
    exit 1
fi

# Navigate to processor directory
cd "$PROCESSOR_DIR"

# Check for Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Create annotations directory
mkdir -p "$ANNOTATIONS_DIR"

# Parse arguments
EDL_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --edl)
            EDL_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW}Usage: $0 [--edl /path/to/file.edl]${NC}"
            exit 1
            ;;
    esac
done

# If no EDL specified, process all EDL files in processed directory
if [ -z "$EDL_FILE" ]; then
    # Check for manifests directory
    MANIFESTS_DIR="$PROCESSED_DIR/manifests"
    if [ ! -d "$MANIFESTS_DIR" ]; then
        MANIFESTS_DIR="manifests"  # Fallback to local manifests
    fi

    if [ ! -d "$MANIFESTS_DIR" ]; then
        echo -e "${RED}Error: No manifests directory found${NC}"
        echo -e "${YELLOW}Run ./scripts/03_process_videos.sh first${NC}"
        exit 1
    fi

    # Find all EDL files in manifests subdirectories
    EDL_FILES=$(find "$MANIFESTS_DIR" -name "*.edl" 2>/dev/null)

    if [ -z "$EDL_FILES" ]; then
        echo -e "${RED}Error: No EDL files found in $MANIFESTS_DIR${NC}"
        echo -e "${YELLOW}Please review videos in DaVinci Resolve and export adjusted EDL files${NC}"
        exit 1
    fi

    echo -e "${GREEN}Found EDL files to process:${NC}"
    echo "$EDL_FILES" | while read -r edl; do
        echo -e "  ${YELLOW}$edl${NC}"
    done
    echo ""

    # Process each EDL file
    COUNT=0
    echo "$EDL_FILES" | while read -r edl; do
        if [ -f "$edl" ]; then
            echo -e "${GREEN}Processing: $(basename "$edl")${NC}"
            python createAnnotationJson.py --file_path "$edl" --use-master-map
            COUNT=$((COUNT + 1))
        fi
    done

    echo ""
    echo -e "${GREEN}Processed $COUNT EDL file(s)${NC}"
else
    # Process single EDL file
    if [ ! -f "$EDL_FILE" ]; then
        echo -e "${RED}Error: EDL file not found: $EDL_FILE${NC}"
        exit 1
    fi

    echo -e "${GREEN}Processing: $(basename "$EDL_FILE")${NC}"
    python createAnnotationJson.py --file_path "$EDL_FILE" --use-master-map
fi

# Copy annotation JSONs to data/annotations
echo ""
echo -e "${GREEN}Copying annotations to $ANNOTATIONS_DIR${NC}"

# Find and copy all annotation JSON files
if [ -d "$PROCESSED_DIR/manifests" ]; then
    find "$PROCESSED_DIR/manifests" -name "*_annotation.json" -exec cp {} "$ANNOTATIONS_DIR/" \; 2>/dev/null || true
fi

# Also check local manifests directory
if [ -d "manifests" ]; then
    find manifests -name "*_annotation.json" -exec cp {} "$ANNOTATIONS_DIR/" \; 2>/dev/null || true
fi

ANNOTATION_COUNT=$(ls -1 "$ANNOTATIONS_DIR"/*_annotation.json 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo -e "${GREEN}✓ Annotation generation complete!${NC}"
echo -e "${YELLOW}Annotations saved to: $ANNOTATIONS_DIR${NC}"
echo -e "${GREEN}Total annotations: $ANNOTATION_COUNT${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Review annotations in $ANNOTATIONS_DIR"
echo -e "  2. Extract short clips for training"
echo -e "  3. Begin model training (see docs/GETTING_STARTED.md)"
