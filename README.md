# Jesse Rust

The native technical-indicator engine that powers the [Jesse](https://jesse.trade)
trading framework, plus its local shared-state coordinator — written in Rust,
exposed to Python via PyO3.

## Why

`jesse_rust` is, to our knowledge, **the fastest technical-indicator library
available for Python.** Indicators are implemented in native Rust, compiled
with full LTO and `opt-level = 3`, and called from Python with zero-copy
NumPy bindings. It is consistently faster than equivalent pure-Python or
pandas-based implementations and is also faster than numba-accelerated
versions of the same algorithms — without the cold-start JIT cost, without a
runtime LLVM dependency, and without numba's per-call overhead.

- **Native speed** — everything runs as compiled machine code, no JIT warm-up.
- **Memory-safe** — no `unsafe` blocks in the indicator code.
- **Zero-copy NumPy** — input arrays are read in place via PyO3's `numpy` bindings.
- **Drop-in for Jesse** — every Jesse indicator that needs an inner loop is
  routed here; you don't have to call this library directly.
- **Cross-platform** — pre-built wheels for Linux, macOS, and Windows.

## Installation

```bash
pip install jesse-rust
```

Requirements: Python 3.10+, NumPy 1.26.4+.

## Usage

You normally don't import `jesse_rust` yourself. Strategies call indicators
through Jesse's standard interface and the Rust backend is used automatically:

```python
from jesse.strategies import Strategy
import jesse.indicators as ta

class MyStrategy(Strategy):
    def update_indicators(self):
        rsi   = ta.rsi(self.candles, period=14)
        ema   = ta.ema(self.candles, period=200)
        macd  = ta.macd(self.candles)
        bands = ta.bollinger_bands(self.candles)
```

Direct use from Python is supported too if you need it:

```python
import numpy as np
import jesse_rust as jr

source = np.asarray(prices, dtype=np.float64)
ema_50 = jr.ema(np.ascontiguousarray(source), 50)
```

## Shared-state coordinator

Starting with version 1.3.1, the same extension includes Jesse's local coordinator.
`jesse run` starts a dedicated Python child that calls `jesse_rust.run_coordinator()`;
it chooses a private loopback port, authenticates workers, and manages shutdown.
Normal users need no separate service, connection settings, or Rust compiler.
Importing this package or Jesse does not start the service.

The coordinator keeps temporary shared values, locks and dashboard events in
memory; PostgreSQL continues to hold persistent application data. See
[coordination/README.md](coordination/README.md) for the protocol, resource limits,
external deployment options and failure boundaries. Dependency notices ship in
`THIRD_PARTY_LICENSES.txt` with every wheel and source archive.

## Supported indicators

99 functions across 9 categories.

### Moving averages and smoothers (29)

`cwma`, `dema`, `ema`, `epma`, `frama`, `hma`, `hwma`, `jma`, `kama`,
`maaq`, `mcginley_dynamic`, `mwdx`, `nma`, `rma`, `sma`, `smma`, `sqwma`,
`srwma`, `t3`, `tema`, `trix`, `vidya`, `vlma_inner`, `vpwma`, `vwap`,
`vwma`, `wilders`, `wma`, `zlema`

### Oscillators / momentum (20)

`aroonosc`, `cci`, `cfo`, `cg`, `cmo`, `dpo`, `dti`, `fisher`, `fosc`,
`lrsi`, `macd`, `mass`, `pfe`, `rsi`, `rsx`, `srsi`, `stoch`, `stochf`,
`willr`, `wt`

### Trend / directional (11)

`adx`, `adxr`, `alligator`, `di`, `dm`, `dx`, `ichimoku_cloud`,
`safezonestop`, `sar`, `supertrend`, `vi`

### Volume (7)

`adosc`, `cvi`, `efi`, `emv`, `nvi`, `pvi`, `wad`

### Bands / channels / volatility (7)

`atr`, `bollinger_bands`, `bollinger_bands_width`, `chande`, `chop`,
`donchian`, `keltner_inner`

### Ehlers filters (14)

`bandpass`, `correlation_cycle`, `edcf`, `gauss`, `high_pass`,
`high_pass_2_pole`, `itrend`, `mama`, `pma`, `reflex`, `supersmoother`,
`supersmoother_3_pole`, `trendflex`, `voss`

### Candle transforms (3)

`emd`, `heikin_ashi_candles`, `qstick`

### Statistical (3)

`damiani_volatmeter`, `hurst_rs`, `linearreg`

### Utilities (5)

`find_order_index`, `moving_std`, `shift`, `subtract_floats`, `sum_floats`

## Building from source

You only need this section if you're contributing.

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Python development headers
- [`maturin`](https://github.com/PyO3/maturin)

### Build

```bash
git clone https://github.com/jesse-ai/jesse-rust.git
cd jesse-rust

pip install maturin

# Develop build (installs into your current Python env)
maturin develop --release

# Wheel build (outputs to ./target/wheels)
maturin build --release
```

## Contributing

This package is part of the Jesse trading framework. Please refer to the main
[Jesse repository](https://github.com/jesse-ai/jesse) for contribution guidelines.

## License

MIT — see the LICENSE file.

## Support

- Documentation: [docs.jesse.trade](https://docs.jesse.trade)
- Community: [Jesse Discord](https://jesse.trade/discord)
- Issues: [GitHub Issues](https://github.com/jesse-ai/jesse-rust/issues)

## Acknowledgments

Built with:

- [PyO3](https://pyo3.rs/) — Rust bindings for Python
- [Maturin](https://github.com/PyO3/maturin) — Build and publish Rust-based Python extensions
- [NumPy](https://numpy.org/) — Numerical computing in Python
