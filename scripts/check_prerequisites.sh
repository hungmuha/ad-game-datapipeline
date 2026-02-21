#!/bin/bash
#
# Script: check_prerequisites.sh
# Description: Check if all required dependencies are installed
# Usage: ./scripts/check_prerequisites.sh
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}    Checking Pipeline Prerequisites     ${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

ALL_GOOD=true

# Check Python
echo -n "Checking Python... "
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Found (v$PYTHON_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install: https://www.python.org/downloads/${NC}"
    ALL_GOOD=false
fi

# Check Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Found ($NODE_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install: https://nodejs.org/${NC}"
    ALL_GOOD=false
fi

# Check npm
echo -n "Checking npm... "
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓ Found (v$NPM_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install with Node.js${NC}"
    ALL_GOOD=false
fi

# Check FFmpeg
echo -n "Checking FFmpeg... "
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    echo -e "${GREEN}✓ Found (v$FFMPEG_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install (macOS): brew install ffmpeg${NC}"
    echo -e "  ${YELLOW}Install (Ubuntu): apt-get install ffmpeg${NC}"
    ALL_GOOD=false
fi

# Check uv
echo -n "Checking uv... "
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Found (v$UV_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    ALL_GOOD=false
fi

# Check git
echo -n "Checking git... "
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    echo -e "${GREEN}✓ Found (v$GIT_VERSION)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "  ${YELLOW}Install: https://git-scm.com/downloads${NC}"
    ALL_GOOD=false
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}    Optional Components                 ${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Check Chrome/Chromium (for scraper)
echo -n "Checking Chrome/Chromium... "
if command -v google-chrome &> /dev/null || command -v chromium &> /dev/null || [ -d "/Applications/Google Chrome.app" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${YELLOW}⚠ Not found (required for Fubo scraper)${NC}"
    echo -e "  ${YELLOW}Install: https://www.google.com/chrome/${NC}"
fi

# Check DaVinci Resolve
echo -n "Checking DaVinci Resolve... "
if [ -d "/Applications/DaVinci Resolve.app" ] || [ -d "$HOME/Applications/DaVinci Resolve.app" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${YELLOW}⚠ Not found (required for manual annotation)${NC}"
    echo -e "  ${YELLOW}Download: https://www.blackmagicdesign.com/products/davinciresolve${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"

if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✓ All required dependencies installed!${NC}"
    echo ""
    echo -e "${GREEN}You're ready to run the pipeline:${NC}"
    echo -e "  ${YELLOW}./scripts/run_pipeline.sh${NC}"
else
    echo -e "${RED}✗ Some dependencies are missing${NC}"
    echo ""
    echo -e "${YELLOW}Please install missing dependencies and run this script again${NC}"
    exit 1
fi

echo ""
