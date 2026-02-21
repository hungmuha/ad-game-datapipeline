#!/bin/bash
#
# Script: generate_pipeline_report.sh
# Description: Generate comprehensive report of pipeline status
# Usage: ./scripts/generate_pipeline_report.sh [--output report.md]
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default output
OUTPUT_FILE="$PROJECT_ROOT/outputs/pipeline_report_$(date +%Y%m%d_%H%M%S).md"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo -e "${BLUE}Generating pipeline report...${NC}"
echo ""

# Start report
cat > "$OUTPUT_FILE" << 'EOF'
# NFL Ad Detection Pipeline Report

**Generated:** $(date +"%Y-%m-%d %H:%M:%S")

---

## Pipeline Status Overview

EOF

# Add current date
sed -i '' "s/\*\*Generated:\*\* .*/\*\*Generated:\*\* $(date +"%Y-%m-%d %H:%M:%S")/" "$OUTPUT_FILE"

# Step 1: Scraper Status
cat >> "$OUTPUT_FILE" << EOF
### Step 1: Fubo Scraper

**Location:** \`pipeline/01-fubo-scraper/output/\`

EOF

SCRAPER_OUTPUT="$PROJECT_ROOT/pipeline/01-fubo-scraper/output"
if [ -d "$SCRAPER_OUTPUT" ]; then
    CSV_COUNT=$(ls -1 "$SCRAPER_OUTPUT"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$CSV_COUNT" -gt 0 ]; then
        LATEST_CSV=$(ls -t "$SCRAPER_OUTPUT"/*.csv 2>/dev/null | head -1)
        LATEST_CSV_NAME=$(basename "$LATEST_CSV")
        ENTRY_COUNT=$(tail -n +2 "$LATEST_CSV" 2>/dev/null | wc -l | tr -d ' ')

        cat >> "$OUTPUT_FILE" << EOF
- **Status:** ✅ Active
- **CSV files generated:** $CSV_COUNT
- **Latest CSV:** \`$LATEST_CSV_NAME\`
- **Entries in latest CSV:** $ENTRY_COUNT games

**Recent CSVs:**
EOF

        ls -lt "$SCRAPER_OUTPUT"/*.csv 2>/dev/null | head -5 | awk '{print "- `" $9 "` (" $6 " " $7 ", " $8 ")"}' >> "$OUTPUT_FILE"
    else
        echo "- **Status:** ⚠️ No CSV files found" >> "$OUTPUT_FILE"
    fi
else
    echo "- **Status:** ❌ Output directory not found" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# Step 2: Downloaded Videos
cat >> "$OUTPUT_FILE" << EOF
### Step 2: Downloaded Videos

**Location:** \`data/raw_videos/\`

EOF

RAW_VIDEOS="$PROJECT_ROOT/data/raw_videos"
if [ -d "$RAW_VIDEOS" ]; then
    VIDEO_COUNT=$(ls -1 "$RAW_VIDEOS"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        TOTAL_SIZE=$(du -sh "$RAW_VIDEOS" 2>/dev/null | awk '{print $1}')

        cat >> "$OUTPUT_FILE" << EOF
- **Status:** ✅ Active
- **Total videos:** $VIDEO_COUNT games
- **Total size:** $TOTAL_SIZE

**Downloaded games:**
EOF

        ls -lt "$RAW_VIDEOS"/*.mp4 2>/dev/null | awk '{
            size = $5 / 1024 / 1024 / 1024;
            printf "- `%s` (%.2f GB, %s %s)\n", $9, size, $6, $7
        }' | head -20 >> "$OUTPUT_FILE"
    else
        echo "- **Status:** ⚠️ No videos found" >> "$OUTPUT_FILE"
    fi
else
    echo "- **Status:** ❌ Directory not found" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# Check master.csv
MASTER_CSV="$PROJECT_ROOT/pipeline/02-download-games/master.csv"
if [ -f "$MASTER_CSV" ]; then
    TOTAL_ENTRIES=$(tail -n +2 "$MASTER_CSV" | wc -l | tr -d ' ')
    DOWNLOADED=$(grep -c "downloaded" "$MASTER_CSV" 2>/dev/null || echo 0)
    FAILED=$(grep -c "failed" "$MASTER_CSV" 2>/dev/null || echo 0)
    SKIPPED_DRM=$(grep -c "skipped_drm" "$MASTER_CSV" 2>/dev/null || echo 0)

    cat >> "$OUTPUT_FILE" << EOF
**Download Statistics (from master.csv):**
- Total entries: $TOTAL_ENTRIES
- Successfully downloaded: $DOWNLOADED
- Failed: $FAILED
- Skipped (DRM): $SKIPPED_DRM

EOF
fi

# Step 3: Processed Videos
cat >> "$OUTPUT_FILE" << EOF
### Step 3: Processed Videos

**Location:** \`data/processed/\`

EOF

PROCESSED_DIR="$PROJECT_ROOT/data/processed"
if [ -d "$PROCESSED_DIR" ]; then
    AUDIO_COUNT=$(ls -1 "$PROCESSED_DIR/audio"/*.wav 2>/dev/null | wc -l | tr -d ' ')
    MANIFEST_COUNT=$(find "$PROCESSED_DIR/manifests" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
    EDL_COUNT=$(ls -1 "$PROCESSED_DIR/edl"/*.edl 2>/dev/null | wc -l | tr -d ' ')

    if [ "$AUDIO_COUNT" -gt 0 ] || [ "$MANIFEST_COUNT" -gt 0 ]; then
        AUDIO_SIZE=$(du -sh "$PROCESSED_DIR/audio" 2>/dev/null | awk '{print $1}')
        MANIFEST_SIZE=$(du -sh "$PROCESSED_DIR/manifests" 2>/dev/null | awk '{print $1}')

        cat >> "$OUTPUT_FILE" << EOF
- **Status:** ✅ Active
- **Processed games:** $AUDIO_COUNT
- **Audio files:** $AUDIO_COUNT WAV files ($AUDIO_SIZE)
- **Manifests:** $MANIFEST_COUNT CSV files ($MANIFEST_SIZE)
- **EDL files:** $EDL_COUNT files

**Processing coverage:** $AUDIO_COUNT / $VIDEO_COUNT videos ($(( AUDIO_COUNT * 100 / VIDEO_COUNT ))%)

**Processed games:**
EOF

        ls -lt "$PROCESSED_DIR/audio"/*.wav 2>/dev/null | awk '{
            size = $5 / 1024 / 1024;
            printf "- `%s` (%.1f MB, %s %s)\n", $9, size, $6, $7
        }' | head -10 >> "$OUTPUT_FILE"
    else
        echo "- **Status:** ⚠️ No processed files found" >> "$OUTPUT_FILE"
    fi
else
    echo "- **Status:** ❌ Directory not found" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# Step 4: Annotations
cat >> "$OUTPUT_FILE" << EOF
### Step 4: Annotations

**Location:** \`data/annotations/\`

EOF

ANNOTATIONS_DIR="$PROJECT_ROOT/data/annotations"
if [ -d "$ANNOTATIONS_DIR" ]; then
    ANNOTATION_COUNT=$(ls -1 "$ANNOTATIONS_DIR"/*_annotation.json 2>/dev/null | wc -l | tr -d ' ')

    if [ "$ANNOTATION_COUNT" -gt 0 ]; then
        cat >> "$OUTPUT_FILE" << EOF
- **Status:** ✅ Active
- **Annotated games:** $ANNOTATION_COUNT

**Annotation coverage:** $ANNOTATION_COUNT / $AUDIO_COUNT processed ($(( ANNOTATION_COUNT * 100 / AUDIO_COUNT ))%)

**Annotated games:**
EOF

        ls -lt "$ANNOTATIONS_DIR"/*_annotation.json 2>/dev/null | awk '{
            printf "- `%s` (%s %s)\n", $9, $6, $7
        }' | head -10 >> "$OUTPUT_FILE"
    else
        echo "- **Status:** ⚠️ No annotations found" >> "$OUTPUT_FILE"
        echo "- **Action needed:** Run DaVinci Resolve review and \`./scripts/04_create_annotations.sh\`" >> "$OUTPUT_FILE"
    fi
else
    echo "- **Status:** ❌ Directory not found" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# Summary
cat >> "$OUTPUT_FILE" << EOF
---

## Pipeline Progress Summary

| Step | Status | Count | Coverage |
|------|--------|-------|----------|
| 1. Scraped URLs | $([ "$CSV_COUNT" -gt 0 ] && echo "✅" || echo "⚠️") | $CSV_COUNT CSV files | - |
| 2. Downloaded Videos | $([ "$VIDEO_COUNT" -gt 0 ] && echo "✅" || echo "⚠️") | $VIDEO_COUNT games | - |
| 3. Processed Videos | $([ "$AUDIO_COUNT" -gt 0 ] && echo "✅" || echo "⚠️") | $AUDIO_COUNT games | $(( AUDIO_COUNT * 100 / VIDEO_COUNT ))% |
| 4. Annotated Videos | $([ "$ANNOTATION_COUNT" -gt 0 ] && echo "✅" || echo "⚠️") | $ANNOTATION_COUNT games | $([ "$AUDIO_COUNT" -gt 0 ] && echo "$(( ANNOTATION_COUNT * 100 / AUDIO_COUNT ))%" || echo "0%") |

---

## Storage Usage

EOF

# Storage breakdown
cat >> "$OUTPUT_FILE" << EOF
| Directory | Size | Description |
|-----------|------|-------------|
| \`data/raw_videos/\` | $TOTAL_SIZE | Downloaded MP4 files |
| \`data/processed/audio/\` | $AUDIO_SIZE | Extracted WAV files |
| \`data/processed/manifests/\` | $MANIFEST_SIZE | CSV manifests |
| \`data/annotations/\` | $(du -sh "$ANNOTATIONS_DIR" 2>/dev/null | awk '{print $1}') | Annotation JSON files |

**Total pipeline storage:** $(du -sh "$PROJECT_ROOT/data" 2>/dev/null | awk '{print $1}')

---

## Next Steps

EOF

# Determine next steps
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "1. ❌ **Download videos:** Run \`./scripts/02_download_games.sh\`" >> "$OUTPUT_FILE"
elif [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "1. ❌ **Process videos:** Run \`./scripts/03_process_videos.sh\`" >> "$OUTPUT_FILE"
elif [ "$ANNOTATION_COUNT" -eq 0 ]; then
    cat >> "$OUTPUT_FILE" << EOF
1. ⚠️ **Annotate videos:**
   - Import EDL files into DaVinci Resolve
   - Review and color-code markers
   - Export adjusted EDL
   - Run \`./scripts/04_create_annotations.sh\`
EOF
elif [ "$ANNOTATION_COUNT" -lt "$AUDIO_COUNT" ]; then
    cat >> "$OUTPUT_FILE" << EOF
1. ⚠️ **Complete annotations:** $((AUDIO_COUNT - ANNOTATION_COUNT)) more games need annotation
2. ✅ **Ready for training:** $ANNOTATION_COUNT games annotated
EOF
else
    cat >> "$OUTPUT_FILE" << EOF
1. ✅ **All videos annotated!**
2. 🎯 **Ready for training:** Extract clips and begin model training
   - See \`docs/GETTING_STARTED.md\` for training instructions
EOF
fi

echo "" >> "$OUTPUT_FILE"
echo "**Report generated:** $(date +"%Y-%m-%d %H:%M:%S")" >> "$OUTPUT_FILE"

# Display summary
echo -e "${GREEN}✓ Pipeline report generated!${NC}"
echo ""
echo -e "${YELLOW}Report saved to: $OUTPUT_FILE${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "  Videos downloaded: ${GREEN}$VIDEO_COUNT${NC}"
echo -e "  Videos processed: ${GREEN}$AUDIO_COUNT${NC} ($(( AUDIO_COUNT * 100 / VIDEO_COUNT ))%)"
echo -e "  Videos annotated: ${GREEN}$ANNOTATION_COUNT${NC} ($([ "$AUDIO_COUNT" -gt 0 ] && echo "$(( ANNOTATION_COUNT * 100 / AUDIO_COUNT ))%" || echo "0%"))"
echo ""

# Open report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}Opening report in default viewer...${NC}"
    open "$OUTPUT_FILE" 2>/dev/null || true
fi
