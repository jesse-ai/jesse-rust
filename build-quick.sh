#!/bin/bash

# Jesse Rust - Quick Build Script
# Fast builds for the most common platforms

set -e

echo "🦀 Jesse Rust - Quick Build"
echo "============================"

# Create dist directory
mkdir -p dist
rm -rf dist/*

# Build for current platform (always works)
echo "🔨 Building for current platform..."
maturin build --release --out dist

# If on macOS, build for both Intel and Apple Silicon
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Building for macOS architectures..."
    
    # Install targets if not already installed
    rustup target add x86_64-apple-darwin aarch64-apple-darwin
    
    # Build for Intel Macs
    echo "  Building for x86_64 (Intel)..."
    maturin build --release --target x86_64-apple-darwin --out dist
    
    # Build for Apple Silicon Macs
    echo "  Building for aarch64 (Apple Silicon)..."
    maturin build --release --target aarch64-apple-darwin --out dist
fi

# Build source distribution
echo "📦 Building source distribution..."
maturin sdist --out dist

echo ""
echo "📋 Built packages:"
ls -la dist/

echo ""
echo "🎉 Quick build completed!"
echo ""
echo "To install and test locally:"
echo "  pip install --force-reinstall dist/*.whl"
echo ""
echo "To publish to PyPI:"
echo "  maturin publish --skip-existing dist/*" 