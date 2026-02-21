#!/bin/bash
#
# Script: 01_scrape_fubo.sh
# Description: Extract stream URLs from Fubo TV
# Usage: ./scripts/01_scrape_fubo.sh [--headless]
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
SCRAPER_DIR="$PROJECT_ROOT/pipeline/01-fubo-scraper"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 1: Scraping Fubo TV Stream URLs${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if scraper directory exists
if [ ! -d "$SCRAPER_DIR" ]; then
    echo -e "${RED}Error: Scraper directory not found at $SCRAPER_DIR${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$SCRAPER_DIR/.env" ]; then
    echo -e "${RED}Error: .env file not found in $SCRAPER_DIR${NC}"
    echo -e "${YELLOW}Please create .env with your Fubo TV credentials${NC}"
    echo -e "${YELLOW}See $SCRAPER_DIR/README.md for setup instructions${NC}"
    exit 1
fi

# Navigate to scraper directory
cd "$SCRAPER_DIR"

# Check for Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run scraper
echo -e "${GREEN}Running Fubo scraper...${NC}"
python main.py "$@"

echo ""
echo -e "${GREEN}✓ Scraping complete!${NC}"
echo -e "${YELLOW}Output CSV files: $SCRAPER_DIR/output/${NC}"
echo ""
echo -e "${GREEN}Next step: Run ./scripts/02_download_games.sh${NC}"
