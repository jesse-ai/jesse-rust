# perf-backtests branch — kernel notes

Five commits supporting the jesse `perf-backtests` backtest-performance branch.
Full review guide, benchmarks, per-commit risk notes and shipping order:
**`/home/saleh/dev-jesse/jesse/docs-perf/REPORT.md`** (in the jesse repo, same-named branch).

## New public functions

| function | file | contract |
|---|---|---|
| `ema_last(source, period)` | `src/moving_averages.rs` | bit-for-bit == `ema(...)[-1]`, no full-series allocation |
| `sma_last(source, period)` | `src/moving_averages.rs` | bit-for-bit == `sma(...)[-1]`; NaN-aware; branch-free fast path for NaN-free sources |
| `rsi_last(source, period)` | `src/oscillators.rs` | bit-for-bit == `rsi(...)[-1]`; RSI evaluated once after the Wilder recurrence |
| `atr_last(candles, period)` | `src/bands.rs` | bit-for-bit == `atr(...)[-1]` |
| `bollinger_bands_last(source, period, devup, devdn)` | `src/bands.rs` | `(upper, middle, lower)`, bit-for-bit == last elements of `bollinger_bands(...)` |
| `candle_from_one_minutes(candles)` | `src/candle.rs` | bit-exact HTF candle vs jesse's numpy expression — **only for blocks <= 4320 rows** (replicates numpy 1.26.4 pairwise summation; callers gate + fall back to numpy) |
| `fix_jumped_candles(candles)` | `src/candle.rs` | in-place whole-series pass of jesse's `_get_fixed_jumped_candle`; bit-exact vs the per-candle Python loop |

## Shared contract

- All `*_last` kernels run the **exact same recurrence in the exact same order** as
  their full-series counterparts — bit-for-bit equality verified by randomized sweeps
  (sizes/periods/NaN patterns/strides) with `==` comparison, no tolerance.
- Empty input raises `IndexError` (mirroring numpy's `result[-1]` on an empty array);
  insufficient length (`n < period`, or `n <= period` for RSI) returns NaN.
- All kernels accept **strided (non-contiguous) views** — jesse passes candle column
  slices; nothing requires `as_slice()`.
- `candle_from_one_minutes` is the one **numpy-version-coupled** kernel: its volume sum
  replicates numpy's pairwise blocking, verified on **numpy 1.26.4** for every length
  <= 4320. If numpy changes its pairwise blocking, re-run the equivalence sweep.

## Shipping

This repo must release **before** the jesse branch merges: jesse imports all 7 functions
above, which exist in no released wheel. Bump version (1.1.0 → 1.2.0), build CI wheels,
release, raise jesse's pin — then merge jesse.
