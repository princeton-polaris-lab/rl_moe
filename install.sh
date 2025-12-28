#!/bin/bash
# install.sh - Apply custom transformers patches for RL-MoE controller training
#
# This script copies the modified transformers files to your Python environment.
# Run this after installing the requirements.
#
# Usage:
#   ./install.sh                    # Uses current Python environment
#   ./install.sh /path/to/env       # Uses specified environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/transformers_patches"

# Determine site-packages path
if [ -n "$1" ]; then
    # Use provided environment path
    ENV_PATH="$1"
    SITE_PACKAGES=$(find "$ENV_PATH" -type d -name "site-packages" 2>/dev/null | head -1)
else
    # Auto-detect from current Python
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
fi

if [ -z "$SITE_PACKAGES" ] || [ ! -d "$SITE_PACKAGES" ]; then
    echo "Error: Could not find site-packages directory."
    echo "Usage: ./install.sh [/path/to/python/env]"
    exit 1
fi

TRANSFORMERS_DIR="$SITE_PACKAGES/transformers"

if [ ! -d "$TRANSFORMERS_DIR" ]; then
    echo "Error: transformers not found in $SITE_PACKAGES"
    echo "Please install transformers first: pip install transformers==4.57.1"
    exit 1
fi

echo "========================================"
echo "Installing RL-MoE transformers patches"
echo "========================================"
echo "Patches dir:  $PATCHES_DIR"
echo "Target dir:   $TRANSFORMERS_DIR"
echo "========================================"
echo ""

# Create target directories if they don't exist (gpt_oss is a custom model)
if [ ! -d "$TRANSFORMERS_DIR/models/gpt_oss" ]; then
    echo "Creating gpt_oss model directory..."
    mkdir -p "$TRANSFORMERS_DIR/models/gpt_oss"
fi

# Backup original files
BACKUP_DIR="$SCRIPT_DIR/.transformers_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/models/gpt_oss"
mkdir -p "$BACKUP_DIR/integrations"

echo "Backing up original files..."
cp "$TRANSFORMERS_DIR/models/gpt_oss/modeling_gpt_oss.py" "$BACKUP_DIR/models/gpt_oss/" 2>/dev/null && echo "  - Backed up modeling_gpt_oss.py"
cp "$TRANSFORMERS_DIR/models/gpt_oss/configuration_gpt_oss.py" "$BACKUP_DIR/models/gpt_oss/" 2>/dev/null && echo "  - Backed up configuration_gpt_oss.py"
cp "$TRANSFORMERS_DIR/integrations/mxfp4.py" "$BACKUP_DIR/integrations/" 2>/dev/null && echo "  - Backed up mxfp4.py"
echo ""

# Copy patched files
echo "Copying patched files..."
cp "$PATCHES_DIR/models/gpt_oss/modeling_gpt_oss.py" "$TRANSFORMERS_DIR/models/gpt_oss/" && echo "  ✓ modeling_gpt_oss.py"
cp "$PATCHES_DIR/models/gpt_oss/configuration_gpt_oss.py" "$TRANSFORMERS_DIR/models/gpt_oss/" && echo "  ✓ configuration_gpt_oss.py"
cp "$PATCHES_DIR/integrations/mxfp4.py" "$TRANSFORMERS_DIR/integrations/" && echo "  ✓ mxfp4.py"

# Copy __init__.py for gpt_oss if it exists in patches
if [ -f "$PATCHES_DIR/models/gpt_oss/__init__.py" ]; then
    cp "$PATCHES_DIR/models/gpt_oss/__init__.py" "$TRANSFORMERS_DIR/models/gpt_oss/" && echo "  ✓ __init__.py"
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Patched files installed to:"
echo "  $TRANSFORMERS_DIR/models/gpt_oss/"
echo "  $TRANSFORMERS_DIR/integrations/"
echo ""
echo "Backup saved to:"
echo "  $BACKUP_DIR"
echo ""
echo "You can now run controller training with:"
echo "  sbatch launch_grid.sh"
