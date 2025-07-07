#!/bin/bash

# Jesse Rust - Comprehensive Build Script
# Builds for multiple Python versions and all possible platforms

set -e

echo "🦀 Jesse Rust - Comprehensive Build"
echo "===================================="

# Create dist directory
mkdir -p dist
rm -rf dist/*

echo "🔧 System Information:"
echo "  Rust version: $(rustc --version)"
echo "  Host platform: $OSTYPE ($(uname -m))"
echo "  Maturin version: $(maturin --version)"
echo ""

# Find all available Python versions
echo "🐍 Detecting Python versions..."
PYTHON_VERSIONS=()
PYTHON_PATHS=()

# Common Python version patterns
for py_version in python3.10 python3.11 python3.12 python3.13 python; do
    if command -v "$py_version" &> /dev/null; then
        version_output=$($py_version --version 2>&1)
        if [[ $version_output =~ Python\ ([0-9]+\.[0-9]+) ]]; then
            version="${BASH_REMATCH[1]}"
            if [[ ! " ${PYTHON_VERSIONS[@]} " =~ " ${version} " ]]; then
                PYTHON_VERSIONS+=("$version")
                PYTHON_PATHS+=("$py_version")
                echo "  ✅ Found Python $version at $py_version"
            fi
        fi
    fi
done

if [ ${#PYTHON_VERSIONS[@]} -eq 0 ]; then
    echo "  ❌ No Python versions found!"
    exit 1
fi

echo ""

# Define platform targets based on current OS
echo "🎯 Platform Targets:"
NATIVE_TARGETS=()
CROSS_TARGETS=()

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  📱 macOS Host - Can build for:"
    NATIVE_TARGETS+=("x86_64-apple-darwin")
    NATIVE_TARGETS+=("aarch64-apple-darwin")
    echo "    ✅ macOS Intel (x86_64-apple-darwin)"
    echo "    ✅ macOS Apple Silicon (aarch64-apple-darwin)"
    
    echo "  🚫 Cannot reliably cross-compile to:"
    echo "    ❌ Linux (x86_64-unknown-linux-gnu) - needs Linux Python interpreters"
    echo "    ❌ Windows (x86_64-pc-windows-gnu) - needs Windows Python interpreters"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  🐧 Linux Host - Can build for:"
    NATIVE_TARGETS+=("x86_64-unknown-linux-gnu")
    # ARM64 Linux might work with cross-compilation tools
    if command -v aarch64-linux-gnu-gcc &> /dev/null; then
        NATIVE_TARGETS+=("aarch64-unknown-linux-gnu")
        echo "    ✅ Linux x86_64 (x86_64-unknown-linux-gnu)"
        echo "    ✅ Linux ARM64 (aarch64-unknown-linux-gnu)"
    else
        echo "    ✅ Linux x86_64 (x86_64-unknown-linux-gnu)"
        echo "    ⚠️  Linux ARM64 (need: apt install gcc-aarch64-linux-gnu)"
    fi
    
    echo "  🚫 Cannot reliably cross-compile to:"
    echo "    ❌ macOS - needs macOS SDK and Python interpreters"
    echo "    ❌ Windows - needs Windows Python interpreters"
fi

echo ""

# Install required Rust targets
echo "🔧 Installing Rust targets..."
for target in "${NATIVE_TARGETS[@]}"; do
    echo "  Installing $target..."
    rustup target add "$target"
done

echo ""

# Build tracking
SUCCESSFUL_BUILDS=()
FAILED_BUILDS=()
BUILD_COUNT=0
TOTAL_BUILDS=$((${#NATIVE_TARGETS[@]} * ${#PYTHON_VERSIONS[@]}))

echo "🔨 Building wheels..."
echo "  Total builds planned: $TOTAL_BUILDS (${#NATIVE_TARGETS[@]} targets × ${#PYTHON_VERSIONS[@]} Python versions)"
echo ""

# Build for each target and Python version combination
for target in "${NATIVE_TARGETS[@]}"; do
    for i in "${!PYTHON_VERSIONS[@]}"; do
        BUILD_COUNT=$((BUILD_COUNT + 1))
        py_version="${PYTHON_VERSIONS[$i]}"
        py_path="${PYTHON_PATHS[$i]}"
        
        echo "[$BUILD_COUNT/$TOTAL_BUILDS] Building: $target + Python $py_version"
        
        if maturin build --release --target "$target" --interpreter "$py_path" --out dist 2>/dev/null; then
            echo "  ✅ SUCCESS: $target + Python $py_version"
            SUCCESSFUL_BUILDS+=("$target + Python $py_version")
        else
            echo "  ❌ FAILED: $target + Python $py_version"
            FAILED_BUILDS+=("$target + Python $py_version")
        fi
        echo ""
    done
done

# Build source distribution
echo "📦 Building source distribution..."
if maturin sdist --out dist 2>/dev/null; then
    echo "  ✅ SUCCESS: Source distribution"
else
    echo "  ❌ FAILED: Source distribution"
fi

echo ""
echo "🎯 BUILD SUMMARY"
echo "================"

# Get version info
if ls dist/*.whl &> /dev/null; then
    WHEEL_FILE=$(ls dist/*.whl | head -n1)
    VERSION=$(basename "$WHEEL_FILE" | cut -d'-' -f2)
    echo "📋 Package: jesse_rust v$VERSION"
else
    echo "📋 Package: jesse_rust (version unknown)"
fi

echo "🦀 Rust: $(rustc --version | cut -d' ' -f2)"
echo "🖥️  Host: $(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')"

echo ""
echo "✅ SUCCESSFUL BUILDS (${#SUCCESSFUL_BUILDS[@]}/$TOTAL_BUILDS):"
if [ ${#SUCCESSFUL_BUILDS[@]} -eq 0 ]; then
    echo "  None"
else
    for build in "${SUCCESSFUL_BUILDS[@]}"; do
        echo "  ✓ $build"
    done
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo ""
    echo "❌ FAILED BUILDS (${#FAILED_BUILDS[@]}):"
    for build in "${FAILED_BUILDS[@]}"; do
        echo "  ✗ $build"
    done
fi

echo ""
echo "📦 BUILT PACKAGES:"
if ls dist/* &> /dev/null; then
    for file in dist/*; do
        filename=$(basename "$file")
        size=$(du -h "$file" | cut -f1)
        
        if [[ "$filename" == *.whl ]]; then
            # Extract info from wheel name
            platform=$(echo "$filename" | rev | cut -d'-' -f1 | rev | sed 's/.whl//')
            python_tag=$(echo "$filename" | cut -d'-' -f3)
            echo "  🎯 $filename ($size) - $python_tag, $platform"
        elif [[ "$filename" == *.tar.gz ]]; then
            echo "  📦 $filename ($size) - Source distribution"
        fi
    done
else
    echo "  No packages built"
fi

echo ""
echo "🌍 FOR COMPLETE CROSS-PLATFORM BUILDS:"
echo "======================================="
echo "  Option 1: Use GitHub Actions (recommended)"
echo "    • Builds for Windows, Linux, macOS automatically"
echo "    • Supports Python 3.10-3.13 on all platforms"
echo "    • Your current .github/workflows/build-and-publish.yml does this"
echo ""
echo "  Option 2: Use cibuildwheel with Docker"
echo "    • pip install cibuildwheel"
echo "    • cibuildwheel --platform auto"
echo ""
echo "  Option 3: Manual builds on each platform"
echo "    • Run this script on Linux for Linux wheels"
echo "    • Run on Windows for Windows wheels"
echo "    • Combine all wheels for publishing"

echo ""
echo "🚀 NEXT STEPS:"
echo "  Test locally:    pip install --force-reinstall dist/*.whl"
echo "  Publish current: maturin publish --skip-existing dist/*"

if [ ${#SUCCESSFUL_BUILDS[@]} -eq 0 ]; then
    echo ""
    echo "⚠️  No successful builds!"
    exit 1
fi

echo ""
echo "🎉 Local builds completed successfully!"
echo "   For full cross-platform support, consider using GitHub Actions." 