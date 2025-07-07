# Cross-Compilation Setup for Jesse Rust

This guide explains how to set up cross-compilation for building Python wheels locally instead of using GitHub Actions.

## Quick Start

The easiest way to get started is to use the provided build script:

```bash
./build-all-wheels.sh
```

This will give you options for different build methods.

## Build Methods

### 1. Native Build (Simplest)
Builds only for your current platform:
```bash
maturin build --release --out dist
```

### 2. Cross-Compilation (Fast)
Uses Rust's cross-compilation capabilities. Works best on macOS for multiple targets:

```bash
# Install targets
rustup target add x86_64-apple-darwin aarch64-apple-darwin

# Build for both architectures
maturin build --release --target x86_64-apple-darwin --out dist
maturin build --release --target aarch64-apple-darwin --out dist
```

### 3. cibuildwheel (Most Comprehensive)
Uses Docker containers to build for multiple platforms:

```bash
pip install cibuildwheel
cibuildwheel --platform auto --output-dir dist
```

## Platform-Specific Setup

### macOS (Recommended for cross-compilation)
On macOS, you can easily build for both Intel and Apple Silicon:

```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install targets
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin

# Build
./build-all-wheels.sh
```

### Linux
For Linux, you can build for multiple architectures but need additional setup:

```bash
# Install cross-compilation tools
sudo apt-get install gcc-aarch64-linux-gnu  # For ARM64

# Install targets
rustup target add aarch64-unknown-linux-gnu

# Build
./build-all-wheels.sh
```

### Windows
On Windows, use the PowerShell script or WSL:

```powershell
# Use the existing PowerShell script
./build-local.ps1
```

## Environment Variables

Set these environment variables for PyPI publishing:

```bash
export MATURIN_PYPI_TOKEN="your_pypi_token_here"
```

## Comparison: Local vs GitHub Actions

### Local Cross-Compilation Advantages:
- ✅ **Faster**: No waiting for CI queue
- ✅ **Immediate feedback**: Test builds instantly
- ✅ **Control**: Full control over build environment
- ✅ **Debugging**: Easier to debug build issues
- ✅ **Cost**: No CI minutes usage

### GitHub Actions Advantages:
- ✅ **Comprehensive**: Builds for all platforms reliably
- ✅ **Testing**: Automatic testing on all platforms
- ✅ **Automation**: Triggers on tags/releases
- ✅ **Consistency**: Same environment every time
- ✅ **Windows support**: Better Windows cross-compilation

## Recommended Workflow

1. **Development**: Use local native builds for quick iteration
2. **Testing**: Use cross-compilation for testing on multiple platforms
3. **Release**: Use either local build + manual upload or keep GitHub Actions

## Publishing to PyPI

After building wheels locally:

```bash
# Publish all wheels
maturin publish --skip-existing dist/*

# Or with specific token
MATURIN_PYPI_TOKEN=your_token maturin publish --skip-existing dist/*
```

## Troubleshooting

### Common Issues:
1. **Missing cross-compilation tools**: Install platform-specific toolchains
2. **Target not found**: Run `rustup target add <target>`
3. **Permission denied**: Run `chmod +x build-all-wheels.sh`
4. **Docker not running**: Start Docker for cibuildwheel

### Platform-Specific Issues:
- **macOS**: Make sure Xcode CLI tools are installed
- **Linux**: Install cross-compilation GCC toolchains
- **Windows**: Consider using WSL or keep using GitHub Actions

## Next Steps

1. Try the `./build-all-wheels.sh` script
2. Choose the build method that works best for your setup
3. Consider keeping GitHub Actions for releases but using local builds for development 