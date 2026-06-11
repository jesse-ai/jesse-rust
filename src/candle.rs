//! Candle transforms / decompositions (Heikin Ashi, EMD, QStick).

use ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::types::{PyArrTuple3, PyArrTuple4};

use crate::helpers::{ih_sma};

/// numpy `maximum.reduce` fold: keeps the accumulator on ties (and once NaN,
/// stays NaN) — `(m >= v || m.is_nan()) ? m : v`.
#[inline(always)]
fn np_maximum(m: f64, v: f64) -> f64 {
    if m >= v || m.is_nan() {
        m
    } else {
        v
    }
}

/// numpy `minimum.reduce` fold, mirroring `np_maximum`.
#[inline(always)]
fn np_minimum(m: f64, v: f64) -> f64 {
    if m <= v || m.is_nan() {
        m
    } else {
        v
    }
}

/// Bit-exact replica of numpy's scalar pairwise summation (`pairwise_sum_DOUBLE`)
/// as used by `arr.sum()`. Verified bitwise-identical to numpy for every
/// length up to 4320 (the caller falls back to numpy beyond that, where
/// numpy's buffered reduce changes the accumulation order).
fn np_pairwise_sum(a: &ArrayView1<f64>, start: usize, n: usize) -> f64 {
    if n < 8 {
        let mut res = 0.0;
        for i in 0..n {
            res += a[start + i];
        }
        res
    } else if n <= 128 {
        let mut r = [
            a[start],
            a[start + 1],
            a[start + 2],
            a[start + 3],
            a[start + 4],
            a[start + 5],
            a[start + 6],
            a[start + 7],
        ];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[start + i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[start + i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_pairwise_sum(a, start, n2) + np_pairwise_sum(a, start + n2, n - n2)
    }
}

/// Build one bigger-timeframe candle from a block of 1m candles —
/// bit-exact equivalent of jesse's Python:
///   np.array([c[0,0], c[0,1], c[-1,2], c[:,3].max(), c[:,4].min(), c[:,5].sum()])
/// Only call for blocks of <= 4320 rows (see `np_pairwise_sum`).
#[pyfunction]
pub fn candle_from_one_minutes(candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.nrows();
        if n == 0 {
            return Err(PyValueError::new_err("No candles were passed"));
        }

        let high_col = c.column(3);
        let low_col = c.column(4);
        let vol_col = c.column(5);

        let mut high = high_col[0];
        let mut low = low_col[0];
        for i in 1..n {
            high = np_maximum(high, high_col[i]);
            low = np_minimum(low, low_col[i]);
        }

        let volume = np_pairwise_sum(&vol_col, 0, n);

        let out = vec![c[[0, 0]], c[[0, 1]], c[[n - 1, 2]], high, low, volume];
        Ok(PyArray1::from_vec(py, out).to_owned())
    })
}

/// In-place forward pass of jesse's `_get_fixed_jumped_candle` over a whole
/// 1m series: when the open of candle i differs from the close of candle i-1,
/// the open (and high/low when needed) is snapped to the previous close.
/// The fix only ever reads column 2 (close), which it never writes, so one
/// forward pass is bit-exact equivalent to applying it candle-by-candle.
#[pyfunction]
pub fn fix_jumped_candles(mut candles: PyReadwriteArray2<f64>) -> PyResult<()> {
    let mut c = candles.as_array_mut();
    let n = c.nrows();
    for i in 1..n {
        let prev_close = c[[i - 1, 2]];
        let open = c[[i, 1]];
        if prev_close < open {
            c[[i, 1]] = prev_close;
            if prev_close < c[[i, 4]] {
                c[[i, 4]] = prev_close;
            }
        } else if prev_close > open {
            c[[i, 1]] = prev_close;
            if prev_close > c[[i, 3]] {
                c[[i, 3]] = prev_close;
            }
        }
    }
    Ok(())
}

/// QStick — SMA of (close - open)
#[pyfunction]
pub fn qstick(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut r = vec![0.0f64; n];
        if period == 0 || n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let inv_p = 1.0 / period as f64;
        let mut sum: f64 = (0..period).map(|k| c[[k, 2]] - c[[k, 1]]).sum();
        r[period - 1] = sum * inv_p;
        for i in period..n {
            sum += (c[[i, 2]] - c[[i, 1]]) - (c[[i - period, 2]] - c[[i - period, 1]]);
            r[i] = sum * inv_p;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Heikin Ashi Candles → (open, close, high, low)
#[pyfunction]
pub fn heikin_ashi_candles(candles: PyReadonlyArray2<f64>) -> PyArrTuple4 {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut ha_open = vec![f64::NAN; n];
        let mut ha_close = vec![f64::NAN; n];
        let mut ha_high = vec![f64::NAN; n];
        let mut ha_low = vec![f64::NAN; n];
        for i in 1..n {
            ha_open[i] = (c[[i-1,1]] + c[[i-1,2]]) / 2.0;
            ha_close[i] = (c[[i,1]] + c[[i,2]] + c[[i,3]] + c[[i,4]]) / 4.0;
            ha_high[i] = c[[i,3]].max(ha_open[i]).max(ha_close[i]);
            ha_low[i] = c[[i,4]].min(ha_open[i]).min(ha_close[i]);
        }
        Ok((
            PyArray1::from_vec(py, ha_open).to_owned(),
            PyArray1::from_vec(py, ha_close).to_owned(),
            PyArray1::from_vec(py, ha_high).to_owned(),
            PyArray1::from_vec(py, ha_low).to_owned(),
        ))
    })
}

/// EMD — Empirical Mode Decomposition → (upperband, middleband, lowerband)
#[pyfunction]
pub fn emd(candles: PyReadonlyArray2<f64>, period: usize, delta: f64, fraction: f64) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let price: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let pi = std::f64::consts::PI;
        let beta = (2.0 * pi / period as f64).cos();
        let gamma_v = 1.0 / (4.0 * pi * delta / period as f64).cos();
        let alpha = gamma_v - (gamma_v * gamma_v - 1.0).sqrt();
        let mut bp = vec![0.0f64; n];
        for i in 0..n {
            if i > 2 {
                bp[i] = 0.5 * (1.0 - alpha) * (price[i] - price[i-2])
                      + beta * (1.0 + alpha) * bp[i-1]
                      - alpha * bp[i-2];
            } else if i == 2 {
                bp[i] = 0.5 * (1.0 - alpha) * (price[i] - price[i-2]);
            }
        }
        // SMA of bp over 2*period
        let mean = ih_sma(&bp, 2*period);
        let mut peak = bp.clone();
        let mut valley = bp.clone();
        for i in 0..n {
            peak[i] = peak[i-1.min(i)];
            valley[i] = valley[i-1.min(i)];
            if i > 2 {
                if bp[i-1] > bp[i] && bp[i-1] > bp[i-2] { peak[i] = bp[i-1]; }
                if bp[i-1] < bp[i] && bp[i-1] < bp[i-2] { valley[i] = bp[i-1]; }
            }
        }
        let avg_peak: Vec<f64> = {
            let sp = ih_sma(&peak, 50);
            sp.iter().map(|&x| x * fraction).collect()
        };
        let avg_valley: Vec<f64> = {
            let sv = ih_sma(&valley, 50);
            sv.iter().map(|&x| x * fraction).collect()
        };
        Ok((
            PyArray1::from_vec(py, avg_peak).to_owned(),
            PyArray1::from_vec(py, mean).to_owned(),
            PyArray1::from_vec(py, avg_valley).to_owned(),
        ))
    })
}
