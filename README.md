# Jesse Rust

High-performance Rust-based technical indicators for the Jesse trading framework.

## Overview

`jesse-rust` is a Python extension module written in Rust that provides blazing-fast implementations of technical indicators commonly used in algorithmic trading. This package is designed to work seamlessly with the [Jesse](https://jesse.trade) trading framework, offering significant performance improvements over pure Python implementations.

## Features

- **High Performance**: Written in Rust for maximum speed and efficiency
- **Memory Safe**: Leverages Rust's memory safety guarantees
- **Numpy Integration**: Seamless integration with NumPy arrays
- **Cross-Platform**: Pre-built wheels available for Windows, macOS, and Linux
- **Easy Integration**: Drop-in replacement for Python-based indicators

## Installation

Install from PyPI using pip:

```bash
pip install jesse-rust
```

## Requirements

- Python 3.10 or higher
- NumPy 1.26.4 or higher

## Usage

The package is typically imported and used within the Jesse framework:

```python
import jesse_rust

# The module provides various technical indicators
# that can be used directly in your trading strategies
```

## Performance

Rust implementations typically show 5-10x performance improvements over equivalent Python code, especially for computationally intensive indicators with large datasets.

## Building from Source

If you need to build from source:

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Python development headers
- Maturin build tool

### Build Steps

```bash
# Clone the repository
git clone https://github.com/jesse-ai/jesse-rust.git
cd jesse-rust

# Install maturin
pip install maturin

# Build the package
maturin develop --release

# Or build wheel
maturin build --release
```

## Development

For development builds:

```bash
maturin develop --release
```

## Contributing

This package is part of the Jesse trading framework. Please refer to the main [Jesse repository](https://github.com/jesse-ai/jesse) for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- Documentation: [docs.jesse.trade](https://docs.jesse.trade)
- Community: [Jesse Discord](https://jesse.trade/discord)
- Issues: [GitHub Issues](https://github.com/jesse-ai/jesse-rust/issues)

## Acknowledgments

Built with:
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [Maturin](https://github.com/PyO3/maturin) - Build and publish Rust-based Python extensions
- [NumPy](https://numpy.org/) - Numerical computing in Python 