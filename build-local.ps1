# Jesse Rust - Local Build Script (PowerShell)
# This script helps with local development and testing on Windows

$ErrorActionPreference = "Stop"

Write-Host "🦀 Jesse Rust - Local Build Script (Windows)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if Rust is installed
try {
    $rustVersion = rustc --version
    Write-Host "✅ Rust version: $rustVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Rust is not installed. Please install it from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "✅ Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not available" -ForegroundColor Red
    exit 1
}

# Install/upgrade maturin
Write-Host "📦 Installing/upgrading maturin..." -ForegroundColor Yellow
python -m pip install --upgrade maturin

# Install required dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install numpy

# Build development version
Write-Host "🔨 Building development version..." -ForegroundColor Yellow
maturin develop --release

# Test the build
Write-Host "🧪 Testing the build..." -ForegroundColor Yellow
python -c @"
import jesse_rust
import numpy as np
print('✅ jesse_rust imported successfully!')
print('📋 Available functions:', [name for name in dir(jesse_rust) if not name.startswith('_')])
"@

Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To build a wheel for distribution:" -ForegroundColor Cyan
Write-Host "  maturin build --release" -ForegroundColor White
Write-Host ""
Write-Host "To publish to PyPI (requires API token):" -ForegroundColor Cyan
Write-Host "  maturin publish" -ForegroundColor White 