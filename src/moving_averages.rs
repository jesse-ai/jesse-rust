//! Moving averages and smoothers (EMA family, WMA family, adaptive MAs).

use ndarray::{s, Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use crate::helpers::{get_period_from_timestamp, ih_ema, ih_wma};

/// Calculate KAMA (Kaufman Adaptive Moving Average) - Optimized version
#[pyfunction]
pub fn kama(source: PyReadonlyArray1<f64>, period: usize, fast_length: usize, slow_length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::zeros(n);
        
        if n <= period {
            // Fill with source values when we don't have enough data
            for i in 0..n {
                result[i] = source_array[i];
            }
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate the efficiency ratio multiplier
        let fast_alpha = 2.0 / (fast_length as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_length as f64 + 1.0);
        let alpha_diff = fast_alpha - slow_alpha;
        
        // First 'period' values are same as source
        for i in 0..period {
            result[i] = source_array[i];
        }
        
        // Pre-calculate price differences for rolling volatility
        let mut price_diffs = Vec::with_capacity(n - 1);
        for i in 1..n {
            price_diffs.push((source_array[i] - source_array[i - 1]).abs());
        }
        
        // Initialize rolling volatility sum for the first window
        let mut volatility_sum = 0.0;
        for i in 0..(period - 1) {
            volatility_sum += price_diffs[i];
        }
        
        // Start the calculation after the initial period
        for i in period..n {
            // Calculate Efficiency Ratio using rolling volatility
            let change = (source_array[i] - source_array[i - period]).abs();
            
            // Update rolling volatility sum
            if i >= period {
                // Add new difference, remove old difference
                volatility_sum += price_diffs[i - 1]; // Current period's difference
                if i > period {
                    volatility_sum -= price_diffs[i - period - 1]; // Remove oldest difference
                }
            }
            
            let er = if volatility_sum != 0.0 { change / volatility_sum } else { 0.0 };
            
            // Calculate smoothing constant
            let sc = (er * alpha_diff + slow_alpha).powi(2);
            
            // Calculate KAMA
            result[i] = result[i - 1] + sc * (source_array[i] - result[i - 1]);
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate TEMA (Triple Exponential Moving Average)
#[pyfunction]
pub fn tema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::zeros(n);

        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema1 = source_array[0];
        let mut ema2 = ema1;
        let mut ema3 = ema2;

        result[0] = 3.0 * ema1 - 3.0 * ema2 + ema3;

        for i in 1..n {
            ema1 = alpha * source_array[i] + (1.0 - alpha) * ema1;
            ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
            ema3 = alpha * ema2 + (1.0 - alpha) * ema3;
            result[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Simple Moving Average (SMA)
#[pyfunction]
pub fn sma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate first SMA value
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..period {
            if !source_array[i].is_nan() {
                sum += source_array[i];
                count += 1;
            }
        }
        if count > 0 {
            result[period - 1] = sum / count as f64;
        }

        // Calculate subsequent SMA values using sliding window
        for i in period..n {
            if !source_array[i - period].is_nan() {
                sum -= source_array[i - period];
                count -= 1;
            }
            if !source_array[i].is_nan() {
                sum += source_array[i];
                count += 1;
            }
            if count > 0 {
                result[i] = sum / count as f64;
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate SMMA (Smoothed Moving Average)
#[pyfunction]
pub fn smma(source: PyReadonlyArray1<f64>, length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        // Convert PyArray to rust ndarray
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Create output array
        let mut result = Array1::<f64>::zeros(n);
        
        if n < length {
            // Return array of NaNs if we don't have enough data
            for i in 0..n {
                result[i] = f64::NAN;
            }
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate first SMMA value (SMA)
        let alpha = 1.0 / (length as f64);
        let mut total = 0.0;
        for i in 0..length {
            total += source_array[i];
        }
        let init_val = total / (length as f64);
        
        // Set first value
        result[length - 1] = init_val;
        
        // Calculate subsequent SMMA values
        for i in length..n {
            result[i] = alpha * source_array[i] + (1.0 - alpha) * result[i - 1];
        }
        
        // Fill initial values with NaN
        for i in 0..(length - 1) {
            result[i] = f64::NAN;
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Weighted Moving Average - Ultra-optimized version
#[pyfunction]
pub fn wma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period || period == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Pre-calculate weight sum for efficiency
        let weight_sum = (period * (period + 1)) / 2;
        let weight_sum_f64 = weight_sum as f64;
        
        // Calculate WMA for each position
        for i in (period - 1)..n {
            let mut weighted_sum = 0.0;
            
            // Calculate weighted sum with safe indexing
            for j in 0..period {
                let weight = (j + 1) as f64;
                let idx = i.saturating_sub(period - 1) + j;
                if idx < n {
                    let value = source_array[idx];
                    weighted_sum += weight * value;
                }
            }
            
            result[i] = weighted_sum / weight_sum_f64;
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Volume Weighted Moving Average - Ultra-optimized version
#[pyfunction]
pub fn vwma(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract required data
        let close = candles_array.column(2);
        let volume = candles_array.column(5);
        
        // Pre-calculate price * volume for efficiency
        let mut weighted_prices = Array1::<f64>::zeros(n);
        for i in 0..n {
            weighted_prices[i] = close[i] * volume[i];
        }
        
        // Use cumulative sums for ultra-fast rolling calculation
        let mut cumsum_weighted = Array1::<f64>::zeros(n);
        let mut cumsum_volume = Array1::<f64>::zeros(n);
        
        cumsum_weighted[0] = weighted_prices[0];
        cumsum_volume[0] = volume[0];
        
        for i in 1..n {
            cumsum_weighted[i] = cumsum_weighted[i - 1] + weighted_prices[i];
            cumsum_volume[i] = cumsum_volume[i - 1] + volume[i];
        }
        
        // Calculate VWMA using cumulative sums
        for i in 0..n {
            let start_idx = if i >= period { i - period + 1 } else { 0 };
            let end_idx = i;
            
            let sum_weighted = if start_idx == 0 {
                cumsum_weighted[end_idx]
            } else {
                cumsum_weighted[end_idx] - cumsum_weighted[start_idx - 1]
            };
            
            let sum_volume = if start_idx == 0 {
                cumsum_volume[end_idx]
            } else {
                cumsum_volume[end_idx] - cumsum_volume[start_idx - 1]
            };
            
            if sum_volume == 0.0 {
                result[i] = f64::NAN;
            } else {
                result[i] = sum_weighted / sum_volume;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate DEMA (Double Exponential Moving Average) - Ultra-optimized version
#[pyfunction]
pub fn dema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = source_array[0];
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        // Optimized: Conservative optimizations for better performance
        // Pre-calculate constants for better performance
        let two = 2.0;
        
        // Pre-allocate arrays with the exact pattern as Python
        let mut ema1 = Array1::<f64>::zeros(n);
        ema1[0] = source_array[0];
        
        // Step 1: Calculate EMA1 efficiently  
        for i in 1..n {
            ema1[i] = alpha * source_array[i] + one_minus_alpha * ema1[i - 1];
        }
        
        // Step 2: Calculate EMA2 (EMA of EMA1) efficiently
        let mut ema2 = Array1::<f64>::zeros(n);
        ema2[0] = ema1[0];
        
        for i in 1..n {
            ema2[i] = alpha * ema1[i] + one_minus_alpha * ema2[i - 1];
        }
        
        // Step 3: Calculate DEMA = 2*EMA1 - EMA2 efficiently
        for i in 0..n {
            result[i] = two * ema1[i] - ema2[i];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate EMA (Exponential Moving Average) - Optimized version
#[pyfunction]
pub fn ema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Return NaN if period is greater than the data length
        if period > n {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = source_array[0];
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        // Initialize first value
        result[0] = source_array[0];
        
        // Calculate EMA efficiently  
        for i in 1..n {
            result[i] = alpha * source_array[i] + one_minus_alpha * result[i - 1];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Last value of EMA — runs the exact same `prev = alpha*x + (1-alpha)*prev`
/// recurrence as `ema` (same float ops, same order), but skips allocating and
/// filling the full output series. Bit-for-bit equal to `ema(...)[-1]`
/// (verified by randomized bitwise sweeps, `==` comparison, no tolerance).
///
/// Accepts strided (non-contiguous) views, e.g. candle column slices.
/// Edge cases mirror `ema(...)[-1]`: empty input raises IndexError (the same
/// error indexing `[-1]` on an empty numpy result would raise), `period > n`
/// returns NaN, and NaNs in the source propagate through the recurrence
/// exactly as in `ema`.
#[pyfunction]
pub fn ema_last(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
    let source_array = source.as_array();
    let n = source_array.len();

    if n == 0 {
        return Err(PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    }

    if period > n {
        return Ok(f64::NAN);
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let mut prev = source_array[0];
    for i in 1..n {
        prev = alpha * source_array[i] + one_minus_alpha * prev;
    }

    Ok(prev)
}

/// Last value of SMA — identical rolling-sum recurrence to `sma` (NaN-aware
/// windowed mean: sum/count over the window's non-NaN values), but skips
/// allocating the full output series. Bit-for-bit equal to `sma(...)[-1]`
/// (verified across a 400-case randomized sweep incl. NaN prefixes, random
/// NaNs and strided column views; `==` comparison, no tolerance).
///
/// Accepts strided (non-contiguous) views, e.g. candle column slices.
/// Edge cases mirror `sma(...)[-1]`: empty input raises IndexError,
/// `n < period` returns NaN, an all-NaN final window returns NaN.
#[pyfunction]
pub fn sma_last(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
    let source_array = source.as_array();
    let n = source_array.len();

    if n == 0 {
        return Err(PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    }

    if n < period {
        return Ok(f64::NAN);
    }

    // Fast path: when the source has no NaNs (the overwhelmingly common
    // case — raw candle columns never contain NaN), the NaN branches below
    // never fire and `count` stays equal to `period` for every window, so a
    // branch-free loop performs the exact same float operations in the exact
    // same order — bit-for-bit identical result. Sources with NaNs (e.g.
    // indicator-on-indicator series) fall through to the original loop.
    // (works on strided views too — candle columns arrive as non-contiguous
    // column views, so this must not require as_slice())
    if !source_array.iter().any(|v| v.is_nan()) {
        let mut sum = 0.0;
        for i in 0..period {
            sum += source_array[i];
        }
        for i in period..n {
            sum -= source_array[i - period];
            sum += source_array[i];
        }
        return Ok(sum / period as f64);
    }

    let mut sum = 0.0;
    let mut count: i64 = 0;
    for i in 0..period {
        if !source_array[i].is_nan() {
            sum += source_array[i];
            count += 1;
        }
    }

    for i in period..n {
        if !source_array[i - period].is_nan() {
            sum -= source_array[i - period];
            count -= 1;
        }
        if !source_array[i].is_nan() {
            sum += source_array[i];
            count += 1;
        }
    }

    if count > 0 {
        Ok(sum / count as f64)
    } else {
        Ok(f64::NAN)
    }
}

/// Calculate ZLEMA (Zero-Lag Exponential Moving Average)
#[pyfunction]
pub fn zlema(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Return early if we don't have enough data
        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Calculate lag
        let lag = (period - 1) / 2;
        
        // Calculate the smoothing factor
        let alpha = 2.0 / (period as f64 + 1.0);
        
        // Pre-calculate the EMA data = price + (price - price_lag)
        let mut ema_data = Array1::<f64>::from_elem(n, 0.0);
        for i in lag..n {
            ema_data[i] = source_array[i] + (source_array[i] - source_array[i - lag]);
        }
        
        // First value with enough data is just the ema_data at that point
        result[lag] = ema_data[lag];
        
        // Calculate ZLEMA for the rest of the points
        for i in (lag + 1)..n {
            result[i] = alpha * ema_data[i] + (1.0 - alpha) * result[i - 1];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate FRAMA (Fractal Adaptive Moving Average)
#[pyfunction]
pub fn frama(
    candles: PyReadonlyArray2<f64>,
    window: usize,
    fc: usize,
    sc: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = vec![f64::NAN; n];

        // Adjust window to be even if it's not
        let mut n_local = window;
        if n_local % 2 == 1 { n_local += 1; }
        if n < n_local { return Ok(PyArray1::from_vec(py, result).to_owned()); }

        // Extract high/low/close to plain Vec<f64> for fast indexing
        let high: Vec<f64> = candles_array.slice(s![.., 3]).to_vec();
        let low: Vec<f64> = candles_array.slice(s![.., 4]).to_vec();
        let close: Vec<f64> = candles_array.slice(s![.., 2]).to_vec();

        let w = (2.0 / (sc as f64 + 1.0)).ln();
        let half = n_local / 2;
        let inv_half = 1.0 / half as f64;
        let inv_full = 1.0 / n_local as f64;
        let ln2 = 2.0f64.ln();
        let sc_f = sc as f64;
        let fc_f = fc as f64;
        let min_alpha = 2.0 / (sc_f + 1.0);

        let mut d = vec![f64::NAN; n];
        let mut alphas = vec![f64::NAN; n];

        for i in n_local..n {
            // Window is candles[i-n_local..i]. Indices in our extracted arrays: [i-n_local, i).
            let start = i - n_local;
            let mid = start + half;
            // Second half (slice [half..]): indices [mid, i)
            let mut v1_hi = f64::NEG_INFINITY;
            let mut v1_lo = f64::INFINITY;
            for k in mid..i {
                if high[k] > v1_hi { v1_hi = high[k]; }
                if low[k] < v1_lo { v1_lo = low[k]; }
            }
            // First half (slice [..half]): indices [start, mid)
            let mut v2_hi = f64::NEG_INFINITY;
            let mut v2_lo = f64::INFINITY;
            for k in start..mid {
                if high[k] > v2_hi { v2_hi = high[k]; }
                if low[k] < v2_lo { v2_lo = low[k]; }
            }
            // Full window max/min by combining halves
            let hi = v1_hi.max(v2_hi);
            let lo = v1_lo.min(v2_lo);
            let n1 = (v1_hi - v1_lo) * inv_half;
            let n2 = (v2_hi - v2_lo) * inv_half;
            let n3 = (hi - lo) * inv_full;

            if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                d[i] = ((n1 + n2).ln() - n3.ln()) / ln2;
            } else if i > n_local {
                d[i] = d[i - 1];
            } else {
                d[i] = 0.0;
            }

            let old_alpha = (w * (d[i] - 1.0)).exp().clamp(0.1, 1.0);
            let old_n = (2.0 - old_alpha) / old_alpha;
            let n_val = (sc_f - fc_f) * ((old_n - 1.0) / (sc_f - 1.0)) + fc_f;
            let alpha = (2.0 / (n_val + 1.0)).clamp(min_alpha, 1.0);
            alphas[i] = alpha;
        }

        // Calculate FRAMA EMA
        let seed: f64 = close[..n_local].iter().sum::<f64>() / n_local as f64;
        result[n_local - 1] = seed;
        for i in n_local..n {
            result[i] = alphas[i] * close[i] + (1.0 - alphas[i]) * result[i - 1];
        }

        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}

/// Calculate VWAP (Volume Weighted Average Price)
#[pyfunction]
pub fn vwap(
    candles: PyReadonlyArray2<f64>,
    source_type: &str,
    anchor: &str,
    sequential: bool
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract source prices and volumes based on source_type
        let source = match source_type.to_lowercase().as_str() {
            "open" => candles_array.slice(s![.., 1]).to_owned(),
            "high" => candles_array.slice(s![.., 3]).to_owned(),
            "low" => candles_array.slice(s![.., 4]).to_owned(),
            "close" => candles_array.slice(s![.., 2]).to_owned(),
            "hl2" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                (&high + &low) / 2.0
            },
            "hlc3" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            },
            "ohlc4" => {
                let open = candles_array.slice(s![.., 1]);
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&open + &high + &low + &close) / 4.0
            },
            _ => {
                // Default to hlc3
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            }
        };
        
        let volume = candles_array.slice(s![.., 5]).to_owned();
        let timestamps = candles_array.slice(s![.., 0]).to_owned();
        
        // Calculate VWAP with anchoring logic
        let mut cum_vol = 0.0;
        let mut cum_vol_price = 0.0;
        let mut current_period = if n > 0 { get_period_from_timestamp(timestamps[0usize], anchor) } else { 0 };
        
        for i in 0..n {
            let period = get_period_from_timestamp(timestamps[i], anchor);
            
            // Reset if we've moved to a new period
            if period != current_period {
                cum_vol = 0.0;
                cum_vol_price = 0.0;
                current_period = period;
            }
            
            let vol_price = volume[i] * source[i];
            cum_vol_price += vol_price;
            cum_vol += volume[i];
            
            if cum_vol != 0.0 {
                result[i] = cum_vol_price / cum_vol;
            }
        }
        
        if sequential {
            Ok(PyArray1::from_array(py, &result).to_owned())
        } else {
            let last_result = Array1::<f64>::from_elem(1, if n > 0 { result[n-1] } else { f64::NAN });
            Ok(PyArray1::from_array(py, &last_result).to_owned())
        }
    })
}

/// Calculate T3 (Triple Exponential Moving Average)
#[pyfunction]
pub fn t3(
    candles: PyReadonlyArray2<f64>,
    period: usize,
    vfactor: f64,
    source_type: &str,
    sequential: bool
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        
        if n == 0 {
            let result = Array1::<f64>::from_elem(0, f64::NAN);
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract source based on source_type
        let source = match source_type.to_lowercase().as_str() {
            "open" => candles_array.slice(s![.., 1]).to_owned(),
            "high" => candles_array.slice(s![.., 3]).to_owned(),
            "low" => candles_array.slice(s![.., 4]).to_owned(),
            "close" => candles_array.slice(s![.., 2]).to_owned(),
            "hl2" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                (&high + &low) / 2.0
            },
            "hlc3" => {
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&high + &low + &close) / 3.0
            },
            "ohlc4" => {
                let open = candles_array.slice(s![.., 1]);
                let high = candles_array.slice(s![.., 3]);
                let low = candles_array.slice(s![.., 4]);
                let close = candles_array.slice(s![.., 2]);
                (&open + &high + &low + &close) / 4.0
            },
            _ => candles_array.slice(s![.., 2]).to_owned(), // Default to close
        };
        
        let k = 2.0 / (period as f64 + 1.0);
        let k_rev = 1.0 - k;
        
        // Calculate weights based on volume factor
        let w1 = -vfactor.powi(3);
        let w2 = 3.0 * vfactor.powi(2) + 3.0 * vfactor.powi(3);
        let w3 = -6.0 * vfactor.powi(2) - 3.0 * vfactor - 3.0 * vfactor.powi(3);
        let w4 = 1.0 + 3.0 * vfactor + vfactor.powi(3) + 3.0 * vfactor.powi(2);
        
        // Initialize EMAs
        let mut e1 = Array1::<f64>::zeros(n);
        let mut e2 = Array1::<f64>::zeros(n);
        let mut e3 = Array1::<f64>::zeros(n);
        let mut e4 = Array1::<f64>::zeros(n);
        let mut e5 = Array1::<f64>::zeros(n);
        let mut e6 = Array1::<f64>::zeros(n);
        let mut t3_result = Array1::<f64>::zeros(n);
        
        // Initialize first values
        if n > 0 {
            e1[0usize] = source[0usize];
            e2[0usize] = e1[0usize];
            e3[0usize] = e2[0usize];
            e4[0usize] = e3[0usize];
            e5[0usize] = e4[0usize];
            e6[0usize] = e5[0usize];
            t3_result[0usize] = w1 * e6[0usize] + w2 * e5[0usize] + w3 * e4[0usize] + w4 * e3[0usize];
        }
        
        // Calculate all EMAs
        for i in 1..n {
            e1[i] = k * source[i] + k_rev * e1[i-1];
            e2[i] = k * e1[i] + k_rev * e2[i-1];
            e3[i] = k * e2[i] + k_rev * e3[i-1];
            e4[i] = k * e3[i] + k_rev * e4[i-1];
            e5[i] = k * e4[i] + k_rev * e5[i-1];
            e6[i] = k * e5[i] + k_rev * e6[i-1];
            t3_result[i] = w1 * e6[i] + w2 * e5[i] + w3 * e4[i] + w4 * e3[i];
        }
        
        if sequential {
            Ok(PyArray1::from_array(py, &t3_result).to_owned())
        } else {
            let last_result = Array1::<f64>::from_elem(1, if n > 0 { t3_result[n-1] } else { f64::NAN });
            Ok(PyArray1::from_array(py, &last_result).to_owned())
        }
    })
}

/// Cubed Weighted Moving Average
#[pyfunction]
pub fn cwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powi(3)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Squared Weighted Moving Average
#[pyfunction]
pub fn sqwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powi(2)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Square Root Weighted Moving Average
#[pyfunction]
pub fn srwma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).sqrt()).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Variable Power Weighted Moving Average
#[pyfunction]
pub fn vpwma(source: PyReadonlyArray1<f64>, period: usize, power: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Pre-compute weights once
        let weights: Vec<f64> = (0..(period - 1)).map(|i| (period as f64 - i as f64).powf(power)).collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in (period + 1)..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// End Point Moving Average
#[pyfunction]
pub fn epma(source: PyReadonlyArray1<f64>, period: usize, offset: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if period < 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let start = period + offset + 1;
        if n <= start { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let weights: Vec<f64> = (0..(period - 1))
            .map(|i| (period as isize - i as isize - offset as isize) as f64)
            .collect();
        let ws: f64 = weights.iter().sum();
        if ws == 0.0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_ws = 1.0 / ws;
        for j in start..n {
            let mut s = 0.0f64;
            for i in 0..(period - 1) {
                s += src[j - i] * weights[i];
            }
            r[j] = s * inv_ws;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// RMA — Wilder's EMA (alpha = 1/length)
#[pyfunction]
pub fn rma(source: PyReadonlyArray1<f64>, length: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let alpha = 1.0 / length as f64;
        let mut r: Vec<f64> = src.to_vec();
        // Seed with last element to match Jesse's original rma_fast behavior:
        // rma_fast uses newseries[i-1] which wraps to newseries[-1] = source[-1] when i=0
        r[0] = alpha * src[0] + (1.0 - alpha) * src[n - 1];
        for i in 1..n {
            r[i] = alpha * src[i] + (1.0 - alpha) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Wilders Smoothing (same as rma with period param)
#[pyfunction]
pub fn wilders(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r: Vec<f64> = src.to_vec();
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let pf = period as f64;
        let alpha = 1.0 / pf;
        let one_minus_alpha = 1.0 - alpha;
        // r[i] = (r[i-1] * (period - 1) + src[i]) / period = (1-1/period)*r[i-1] + (1/period)*src[i]
        for i in 1..n {
            r[i] = alpha * src[i] + one_minus_alpha * r[i - 1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// McGinley Dynamic
#[pyfunction]
pub fn mcginley_dynamic(source: PyReadonlyArray1<f64>, period: usize, k: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        r[0] = src[0];
        for i in 1..n {
            let prev = r[i-1];
            let ratio = src[i] / prev;
            let denom = (k * period as f64 * ratio.powi(4)).max(1.0);
            r[i] = prev + (src[i] - prev) / denom;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MWDX Average
#[pyfunction]
pub fn mwdx(source: PyReadonlyArray1<f64>, factor: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let val2 = 2.0 / factor - 1.0;
        let fac = 2.0 / (val2 + 1.0);
        let mut r: Vec<f64> = src.to_vec();
        for i in 1..n {
            r[i] = fac * src[i] + (1.0 - fac) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Holt-Winter Moving Average
#[pyfunction]
pub fn hwma(source: PyReadonlyArray1<f64>, na: f64, nb: f64, nc: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![0.0f64; n];
        if n == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let (mut last_a, mut last_v, mut last_f) = (0.0f64, 0.0f64, src[0]);
        for i in 0..n {
            let f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * src[i];
            let v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f);
            let a = (1.0 - nc) * last_a + nc * (v - last_v);
            r[i] = f + v + 0.5 * a;
            last_a = a; last_f = f; last_v = v;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// TRIX — triple EMA of log(price), percent change * 10000
#[pyfunction]
pub fn trix(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let log_src: Vec<f64> = src.iter().map(|&x| x.ln()).collect();
        let e1 = ih_ema(&log_src, period);
        let e2 = ih_ema(&e1, period);
        let e3 = ih_ema(&e2, period);
        let mut r = vec![f64::NAN; n];
        for i in 1..n { r[i] = (e3[i] - e3[i-1]) * 10000.0; }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// HMA — Hull Moving Average
#[pyfunction]
pub fn hma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let half = period / 2;
        let sq = (period as f64).sqrt() as usize;
        let wma_half = ih_wma(&src, half);
        let wma_full = ih_wma(&src, period);
        let raw: Vec<f64> = (0..n).map(|i| 2.0 * wma_half[i] - wma_full[i]).collect();
        let r = ih_wma(&raw, sq);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MAAQ — Moving Average Adaptive Q
#[pyfunction]
pub fn maaq(source: PyReadonlyArray1<f64>, period: usize, fast_period: usize, slow_period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let fast_sc = 2.0 / (fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (slow_period as f64 + 1.0);
        let mut diff = vec![0.0f64; n];
        for i in 1..n { diff[i] = (src[i] - src[i - 1]).abs(); }
        let mut r = src.to_vec();
        if n <= period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Initial rolling-sum window [period+1-period..=period] = [1..=period]
        let mut noise: f64 = diff[1..=period].iter().sum();
        let i0 = period;
        let signal0 = (src[i0] - src[i0 - period]).abs();
        let ratio0 = if noise != 0.0 { signal0 / noise } else { 0.0 };
        let temp0 = (ratio0 * fast_sc + slow_sc).powi(2);
        r[i0] = r[i0 - 1] + temp0 * (src[i0] - r[i0 - 1]);
        for i in (period + 1)..n {
            // Slide: remove diff[i-period], add diff[i]
            noise += diff[i] - diff[i - period];
            let signal = (src[i] - src[i - period]).abs();
            let ratio = if noise != 0.0 { signal / noise } else { 0.0 };
            let temp = (ratio * fast_sc + slow_sc).powi(2);
            r[i] = r[i - 1] + temp * (src[i] - r[i - 1]);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// VIDYA — Variable Index Dynamic Average
#[pyfunction]
pub fn vidya(source: PyReadonlyArray1<f64>, length: usize, fix_cmo: bool, select: bool) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let alpha = 2.0 / (length as f64 + 1.0);
        let cmo_length = if fix_cmo { 9usize } else { length };
        let mut momm = vec![0.0f64; n];
        for i in 1..n { momm[i] = src[i] - src[i-1]; }
        let mut vidya_r = src.to_vec();
        for i in 1..n {
            let start = (i + 1).saturating_sub(cmo_length);
            let (mut sm1, mut sm2) = (0.0f64, 0.0f64);
            for k in start..=i {
                if momm[k] >= 0.0 { sm1 += momm[k]; } else { sm2 -= momm[k]; }
            }
            let k_val = if select {
                let tot = sm1 + sm2;
                if tot != 0.0 { ((sm1 - sm2) / tot * 100.0).abs() / 100.0 } else { 0.0 }
            } else {
                let start2 = (i + 1).saturating_sub(length);
                let slice = &src.as_slice().unwrap()[start2..=i];
                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let var: f64 = slice.iter().map(|&x| (x-mean).powi(2)).sum::<f64>() / slice.len() as f64;
                var.sqrt()
            };
            let eff_alpha = alpha * k_val;
            vidya_r[i] = eff_alpha * src[i] + (1.0 - eff_alpha) * vidya_r[i-1];
        }
        Ok(PyArray1::from_vec(py, vidya_r).to_owned())
    })
}

/// VLMA inner loop (called from Python after computing deviation bands)
#[pyfunction]
pub fn vlma_inner(
    source: PyReadonlyArray1<f64>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>,
    d: PyReadonlyArray1<f64>,
    min_period: usize,
    max_period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let av = a.as_array();
        let bv = b.as_array();
        let cv = c.as_array();
        let dv = d.as_array();
        let n = src.len();
        let mut r = src.to_vec();
        let mut period = max_period as f64;
        for i in 1..n {
            let nz_period = period;
            let next_period = if bv[i] <= src[i] && src[i] <= cv[i] {
                nz_period + 1.0
            } else if src[i] < av[i] || src[i] > dv[i] {
                nz_period - 1.0
            } else { nz_period };
            period = next_period.max(min_period as f64).min(max_period as f64);
            let sc = 2.0 / (period + 1.0);
            r[i] = src[i] * sc + (1.0 - sc) * r[i-1];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// NMA — Natural Moving Average
#[pyfunction]
pub fn nma(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let clipped: Vec<f64> = src.iter().map(|&x| x.max(1e-10)).collect();
        let ln: Vec<f64> = clipped.iter().map(|&x| x.ln() * 1000.0).collect();
        let mut r = vec![f64::NAN; n];
        if period < 1 || n < period + 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Precompute |ln[k] - ln[k-1]| for k >= 1
        let mut abs_diff = vec![0.0f64; n];
        for k in 1..n {
            abs_diff[k] = (ln[k] - ln[k - 1]).abs();
        }
        // Precompute weights: w[i] = sqrt(i+1) - sqrt(i) for i in 0..period
        let weights: Vec<f64> = (0..period).map(|i| (i as f64 + 1.0).sqrt() - (i as f64).sqrt()).collect();
        let last_i = period - 1;
        for j in (period + 1)..n {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..period {
                let oi = abs_diff[j - i];
                num += oi * weights[i];
                den += oi;
            }
            let ratio = if den != 0.0 { num / den } else { 0.0 };
            r[j] = clipped[j - last_i] * ratio + clipped[j - last_i - 1] * (1.0 - ratio);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// JMA — Jurik Moving Average
#[pyfunction]
pub fn jma(source: PyReadonlyArray1<f64>, period: usize, phase: f64, power: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let phase_ratio = if phase < -100.0 { 0.5 } else if phase > 100.0 { 2.5 } else { phase / 100.0 + 1.5 };
        let beta = 0.45 * (period as f64 - 1.0) / (0.45 * (period as f64 - 1.0) + 2.0);
        let alpha = beta.powi(power as i32);
        let mut e0 = vec![0.0f64; n];
        let mut e1 = vec![0.0f64; n];
        let mut e2 = vec![0.0f64; n];
        let mut jma_v = src.to_vec();
        for i in 1..n {
            e0[i] = (1.0 - alpha) * src[i] + alpha * e0[i-1];
            e1[i] = (src[i] - e0[i]) * (1.0 - beta) + beta * e1[i-1];
            e2[i] = (e0[i] + phase_ratio * e1[i] - jma_v[i-1]) * (1.0 - alpha).powi(2) + alpha.powi(2) * e2[i-1];
            jma_v[i] = e2[i] + jma_v[i-1];
        }
        Ok(PyArray1::from_vec(py, jma_v).to_owned())
    })
}
