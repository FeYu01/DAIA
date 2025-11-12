#!/usr/bin/env bash
set -e

# fetch_data.sh - helper to ensure dataset exists in /home/codespace/datasets/DAIA
# Usage: bash scripts/fetch_data.sh [--copy-to-workspace]

TARGET_DIR="/home/codespace/datasets/DAIA"
WORKSPACE_DATA_DIR="/workspaces/DAIA/data"

mkdir -p "$TARGET_DIR/real" "$TARGET_DIR/ai_generated"

echo "Checking for dataset in $TARGET_DIR"
if [ -d "$TARGET_DIR/real" ] && [ -n "$(ls -A $TARGET_DIR/real 2>/dev/null)" ]; then
  echo "Real dataset found in $TARGET_DIR/real (size: $(du -sh $TARGET_DIR/real | cut -f1))"
else
  echo "No real dataset found in $TARGET_DIR/real."
  echo "Place your real images in $TARGET_DIR/real or provide a URL and edit this script to download them."
fi

if [ -d "$TARGET_DIR/ai_generated" ] && [ -n "$(ls -A $TARGET_DIR/ai_generated 2>/dev/null)" ]; then
  echo "AI-generated dataset found in $TARGET_DIR/ai_generated (size: $(du -sh $TARGET_DIR/ai_generated | cut -f1))"
else
  echo "No AI dataset found in $TARGET_DIR/ai_generated."
  echo "Place your AI images in $TARGET_DIR/ai_generated or provide a URL and edit this script to download them."
fi

if [ "$1" = "--copy-to-workspace" ]; then
  echo "Copying datasets into workspace data/ (this will create many files in repo workdir but not commit them)"
  mkdir -p "$WORKSPACE_DATA_DIR/real" "$WORKSPACE_DATA_DIR/ai_generated"
  rsync -av --exclude='.gitkeep' "$TARGET_DIR/real/" "$WORKSPACE_DATA_DIR/real/"
  rsync -av --exclude='.gitkeep' "$TARGET_DIR/ai_generated/" "$WORKSPACE_DATA_DIR/ai_generated/"
  echo "Copy complete"
fi

echo "Done."
