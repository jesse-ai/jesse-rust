#!/bin/bash

# Jesse Rust - Local Build Script
# This script helps with local development and testing

set -e

echo "🦀 Jesse Rust - Local Build Script"
echo "=================================="

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust is not installed. Please install it from https://rustup.rs/"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not available"
    exit 1
fi

echo "✅ Rust version: $(rustc --version)"
echo "✅ Python version: $(python3 --version)"

# Install/upgrade maturin
echo "📦 Installing/upgrading maturin..."
python3 -m pip install --upgrade maturin

# Install required dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy

# Build development version
echo "🔨 Building development version..."
maturin develop --release

# Test the build
echo "🧪 Testing the build..."
python3 -c "
import jesse_rust
import numpy as np
print('✅ jesse_rust imported successfully!')
print('📋 Available functions:', [name for name in dir(jesse_rust) if not name.startswith('_')])
"

echo "🎉 Build completed successfully!"
echo ""
echo "To build a wheel for distribution:"
echo "  maturin build --release"
echo ""
echo "To publish to PyPI (requires API token):"
echo "  maturin publish" 