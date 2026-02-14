#!/usr/bin/env bash
# Setup script for Git hooks
# Detects uv/venv environments and installs hooks accordingly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}---${NC}"
echo -e "${BLUE}Git Hooks Setup for RAG SDK${NC}"
echo -e "${BLUE}---${NC}\n"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}Not a git repository${NC}"
    echo "Please run this script from the root of your git repository"
    exit 1
fi

# Detect environment
if [ -f ".venv/bin/python" ]; then
    ENV_LABEL="uv/venv (.venv)"
    BIN_DIR=".venv/bin"
elif command -v docker &> /dev/null && [ -f "Dockerfile" ]; then
    ENV_LABEL="Docker"
    BIN_DIR=""
else
    ENV_LABEL="global"
    BIN_DIR=""
fi

echo -e "${YELLOW}Detected environment: ${ENV_LABEL}${NC}\n"

# Create hooks directory if it doesn't exist
HOOKS_DIR=".git/hooks"
mkdir -p "$HOOKS_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${YELLOW}Installing Git hooks...${NC}\n"

# Install pre-commit hook
if [ -f "$SCRIPT_DIR/pre-commit" ]; then
    cp "$SCRIPT_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
    chmod +x "$HOOKS_DIR/pre-commit"
    echo -e "${GREEN}Installed pre-commit hook${NC}"
else
    echo -e "${RED}pre-commit hook file not found${NC}"
fi

# Install pre-push hook
if [ -f "$SCRIPT_DIR/pre-push" ]; then
    cp "$SCRIPT_DIR/pre-push" "$HOOKS_DIR/pre-push"
    chmod +x "$HOOKS_DIR/pre-push"
    echo -e "${GREEN}Installed pre-push hook${NC}"
else
    echo -e "${RED}pre-push hook file not found${NC}"
fi

# Install commit-msg hook
if [ -f "$SCRIPT_DIR/commit-msg" ]; then
    cp "$SCRIPT_DIR/commit-msg" "$HOOKS_DIR/commit-msg"
    chmod +x "$HOOKS_DIR/commit-msg"
    echo -e "${GREEN}Installed commit-msg hook${NC}"
else
    echo -e "${RED}commit-msg hook file not found${NC}"
fi

echo ""

# Check for required dependencies
echo -e "${YELLOW}Checking required dependencies...${NC}\n"

check_tool() {
    local tool=$1
    if [ -n "$BIN_DIR" ] && [ -f "$BIN_DIR/$tool" ]; then
        echo -e "${GREEN}$tool is installed (${BIN_DIR}/${tool})${NC}"
        return 0
    elif command -v "$tool" &> /dev/null; then
        echo -e "${GREEN}$tool is installed ($(which $tool))${NC}"
        return 0
    else
        echo -e "${YELLOW}$tool is not installed${NC}"
        return 1
    fi
}

MISSING_DEPS=0

check_tool "ruff" || MISSING_DEPS=1
check_tool "mypy" || MISSING_DEPS=1
check_tool "pytest" || MISSING_DEPS=1
check_tool "bandit" || echo -e "${BLUE}bandit is optional but recommended${NC}"

echo ""

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${YELLOW}---${NC}"
    echo -e "${YELLOW}Some dependencies are missing${NC}"
    echo -e "${YELLOW}---${NC}\n"

    if command -v uv &> /dev/null; then
        echo -e "${YELLOW}Install missing dependencies with uv:${NC}"
        echo -e "${GREEN}uv venv .venv && uv pip install -e '.[dev]'${NC}"
    else
        echo -e "${YELLOW}Install missing dependencies:${NC}"
        echo -e "${GREEN}python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'${NC}"
    fi
    echo ""
fi

echo -e "${BLUE}---${NC}"
echo -e "${GREEN}Git hooks installation complete!${NC}"
echo -e "${BLUE}---${NC}\n"

echo -e "${YELLOW}Hooks installed:${NC}"
echo "  pre-commit  - Runs linting and basic checks before commit"
echo "  pre-push    - Runs tests and comprehensive checks before push"
echo "  commit-msg  - Enforces conventional commit message format"
echo ""

echo -e "${YELLOW}Environment detection priority:${NC}"
echo "  1. .venv/bin/ (uv venv or python -m venv)"
echo "  2. Docker (if Dockerfile exists)"
echo "  3. Global installs (fallback)"
echo ""

echo -e "${YELLOW}To skip hooks (not recommended):${NC}"
echo "  git commit --no-verify"
echo "  git push --no-verify"
echo ""
