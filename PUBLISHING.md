# Jesse Rust - PyPI Publishing Guide

This guide walks you through publishing the `jesse-rust` package to PyPI with multi-platform support.

## Prerequisites

### Required Tools
- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **Python 3.8+**: With pip installed
- **Maturin**: `pip install maturin`
- **Git**: For version control

### PyPI Account
- [PyPI Account](https://pypi.org/account/register/) (for releases)

## Setup Process

### 1. GitHub Repository Setup

Ensure your repository has the following structure:
```
jesse-rust/
├── .github/workflows/build-and-publish.yml
├── src/
│   ├── lib.rs
│   └── indicators.rs
├── Cargo.toml
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
├── build-local.sh
├── build-local.ps1
└── __init__.py
```

### 2. GitHub Secrets Setup

Configure the following secret in your GitHub repository:

1. Go to GitHub Repository → Settings → Secrets and variables → Actions
2. Add new repository secret:
   - **Name**: `PYPI_API_TOKEN`
   - **Value**: Your PyPI API token (get from [PyPI Account Settings](https://pypi.org/manage/account/))

### 3. PyPI API Token Setup

#### Get PyPI Token:
1. Log into [PyPI](https://pypi.org/)
2. Go to Account Settings → API tokens
3. Create new token with scope for your project
4. Copy the token (starts with `pypi-`)

## Local Development

### Quick Start
```bash
# Clone and navigate to directory
cd jesse-rust

# Run build script (Unix/macOS)
./build-local.sh

# Or on Windows
.\build-local.ps1
```

### Manual Build
```bash
# Install maturin
pip install maturin

# Development build (debug)
maturin develop

# Production build (optimized)
maturin develop --release

# Build wheel for distribution
maturin build --release

# Build source distribution
maturin sdist
```

## Publishing Process

### Method 1: Automatic (Recommended)

#### Publishing to PyPI
1. Create and push a version tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
2. GitHub Actions will automatically build and publish to PyPI
3. Install: `pip install jesse-rust`

### Method 2: Manual Local Publishing

#### Build wheels locally
```bash
# Build for current platform
maturin build --release

# Upload to PyPI
maturin publish
```

## Version Management

### Update Version
1. Edit `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"  # Update this
   ```

2. Create git tag:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git tag v0.1.1
   git push origin main
   git push origin v0.1.1
   ```

## Platform Support

The GitHub Actions workflow automatically builds for:

### Linux
- x86_64 (Intel/AMD)
- aarch64 (ARM64)

### macOS
- x86_64 (Intel)
- aarch64 (Apple Silicon M1/M2)

### Windows
- x64 (Intel/AMD)

## Testing

### Automated Testing
GitHub Actions runs tests on:
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Ubuntu, Windows, macOS
- All supported architectures

### Manual Testing
```bash
# Install from PyPI (after publishing)
pip install jesse-rust

# Test import
python -c "import jesse_rust; print('Success!')"

# Run comprehensive test
python -c "
import jesse_rust
import numpy as np
print('Available functions:', dir(jesse_rust))
"
```

## Troubleshooting

### Common Issues

#### Build Fails
- Ensure Rust is installed: `rustc --version`
- Update maturin: `pip install --upgrade maturin`
- Check Rust dependencies in `Cargo.toml`

#### Import Fails
- Verify Python version compatibility (3.8+)
- Check NumPy is installed: `pip install numpy`
- Try rebuilding: `maturin develop --release`

#### Publishing Fails
- Verify API tokens are correct
- Check package name isn't taken on PyPI
- Ensure version number is incremented

### GitHub Actions Debug
- Check Actions tab in GitHub repository
- Look for specific error messages in logs
- Ensure secrets are properly configured

## Maintenance

### Regular Updates
1. Keep dependencies updated in `Cargo.toml`
2. Update Python version support in `pyproject.toml`
3. Monitor security advisories for Rust crates
4. Test new Python releases when available

### Performance Monitoring
- Benchmark critical functions after changes
- Monitor package size (wheels should be <50MB typically)
- Test import time on different platforms

## Resources

- [Maturin Documentation](https://maturin.rs/)
- [PyO3 Guide](https://pyo3.rs/)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) 