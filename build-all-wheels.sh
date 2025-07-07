#!/bin/bash

# Jesse Rust - Build All Wheels Script
# Cross-compilation for all supported platforms (no Docker)

set -e

echo "🦀 Jesse Rust - Build All Wheels (Cross-compilation)"
echo "==================================================="

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "📦 Installing maturin..."
    pip install maturin
fi

# Create dist directory
mkdir -p dist
rm -rf dist/*

echo "🔧 System Information:"
echo "  Rust version: $(rustc --version)"
echo "  Python version: $(python3 --version)"
echo "  Maturin version: $(maturin --version)"
echo "  Host platform: $OSTYPE"
echo ""

# Install all cross-compilation targets
echo "🎯 Installing cross-compilation targets..."
TARGETS=()

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS can cross-compile for macOS targets
    TARGETS+=("x86_64-apple-darwin")
    TARGETS+=("aarch64-apple-darwin")
    echo "  Installing macOS targets..."
    rustup target add x86_64-apple-darwin aarch64-apple-darwin
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux can cross-compile for Linux targets
    TARGETS+=("x86_64-unknown-linux-gnu")
    TARGETS+=("aarch64-unknown-linux-gnu")
    echo "  Installing Linux targets..."
    rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
    
    # Check for cross-compilation tools
    if command -v aarch64-linux-gnu-gcc &> /dev/null; then
        echo "  ✅ aarch64 cross-compiler found"
    else
        echo "  ⚠️  aarch64-linux-gnu-gcc not found - ARM64 build may fail"
    fi
fi

# Always try to add the current platform as fallback
CURRENT_TARGET=$(rustc -vV | grep host | cut -d' ' -f2)
TARGETS+=("$CURRENT_TARGET")
echo "  Current platform target: $CURRENT_TARGET"

# Remove duplicates
TARGETS=($(printf "%s\n" "${TARGETS[@]}" | sort -u))

echo ""
echo "🔨 Building wheels for targets: ${TARGETS[*]}"
echo ""

# Build counter and results tracking
SUCCESSFUL_BUILDS=()
FAILED_BUILDS=()
BUILD_COUNT=0

# Build wheels for each target
for target in "${TARGETS[@]}"; do
    BUILD_COUNT=$((BUILD_COUNT + 1))
    echo "[$BUILD_COUNT/${#TARGETS[@]}] Building for $target..."
    
    if maturin build --release --target "$target" --out dist 2>/dev/null; then
        echo "  ✅ SUCCESS: $target"
        SUCCESSFUL_BUILDS+=("$target")
    else
        echo "  ❌ FAILED: $target"
        FAILED_BUILDS+=("$target")
    fi
    echo ""
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

# Get version from built packages
if ls dist/*.whl &> /dev/null; then
    WHEEL_FILE=$(ls dist/*.whl | head -n1)
    VERSION=$(basename "$WHEEL_FILE" | cut -d'-' -f2)
    echo "📋 Package: jesse_rust v$VERSION"
else
    echo "📋 Package: jesse_rust (version unknown)"
fi

echo "🐍 Python: $(python3 --version | cut -d' ' -f2)"
echo "🦀 Rust: $(rustc --version | cut -d' ' -f2)"
echo "��️  Host: $(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')"

echo ""
echo "✅ SUCCESSFUL BUILDS (${#SUCCESSFUL_BUILDS[@]}):"
if [ ${#SUCCESSFUL_BUILDS[@]} -eq 0 ]; then
    echo "  None"
else
    for target in "${SUCCESSFUL_BUILDS[@]}"; do
        echo "  ✓ $target"
    done
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo ""
    echo "❌ FAILED BUILDS (${#FAILED_BUILDS[@]}):"
    for target in "${FAILED_BUILDS[@]}"; do
        echo "  ✗ $target"
    done
fi

echo ""
echo "📦 BUILT PACKAGES:"
if ls dist/* &> /dev/null; then
    for file in dist/*; do
        filename=$(basename "$file")
        size=$(du -h "$file" | cut -f1)
        
        if [[ "$filename" == *.whl ]]; then
            # Extract platform info from wheel name
            platform=$(echo "$filename" | rev | cut -d'-' -f1 | rev | sed 's/.whl//')
            python_tag=$(echo "$filename" | cut -d'-' -f3)
            echo "  🎯 $filename ($size) - Python $python_tag, $platform"
        elif [[ "$filename" == *.tar.gz ]]; then
            echo "  📦 $filename ($size) - Source distribution"
        else
            echo "  📄 $filename ($size)"
        fi
    done
else
    echo "  No packages built"
fi

echo ""
echo "🚀 NEXT STEPS:"
echo "  Test locally:    pip install --force-reinstall dist/*.whl"
echo "  Publish to PyPI: maturin publish --skip-existing dist/*"

# Exit with error if no successful builds
if [ ${#SUCCESSFUL_BUILDS[@]} -eq 0 ]; then
    echo ""
    echo "⚠️  No successful builds! Check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Cross-compilation completed successfully!" 