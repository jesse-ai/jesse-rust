//! Momentum oscillators (RSI, stoch, CMO, fisher, etc.).

use ndarray::{s, Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use crate::types::{PyArrTuple2, PyArrTuple3};

use crate::helpers::{
    ema_for_wt,
    ih_ema,
    ih_sma,
    sma_array,
    sma_for_wt,
};

/// Calculate RSI (Relative Strength Index)
#[pyfunction]
pub fn rsi(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate initial sum of gains and losses
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        for i in 1..=period {
            let change = source_array[i] - source_array[i - 1];
            if change > 0.0 {
                sum_gain += change;
            } else {
                sum_loss += change.abs();
            }
        }

        let mut avg_gain = sum_gain / period as f64;
        let mut avg_loss = sum_loss / period as f64;

        // Calculate first RSI value
        if avg_loss == 0.0 {
            result[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Calculate subsequent RSI values using Wilder's smoothing
        for i in (period + 1)..n {
            let change = source_array[i] - source_array[i - 1];
            let (current_gain, current_loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, change.abs())
            };

            avg_gain = (avg_gain * (period as f64 - 1.0) + current_gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + current_loss) / period as f64;

            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Last value of RSI — identical Wilder recurrence to `rsi`, but skips
/// allocating the full output series. Bit-for-bit equal to `rsi(...)[-1]`.
#[pyfunction]
pub fn rsi_last(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
    let source_array = source.as_array();
    let n = source_array.len();

    if n == 0 {
        return Err(PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    }

    if n <= period {
        return Ok(f64::NAN);
    }

    let mut sum_gain = 0.0;
    let mut sum_loss = 0.0;
    for i in 1..=period {
        let change = source_array[i] - source_array[i - 1];
        if change > 0.0 {
            sum_gain += change;
        } else {
            sum_loss += change.abs();
        }
    }

    let mut avg_gain = sum_gain / period as f64;
    let mut avg_loss = sum_loss / period as f64;

    // Only the avg_gain/avg_loss recurrences carry state between iterations;
    // the RSI value itself is a pure function of the FINAL averages, so it is
    // computed once after the loop instead of being recomputed (two extra
    // divisions and a branch) on every iteration. Bit-for-bit identical to
    // `rsi(...)[-1]`.
    let period_f = period as f64;
    let period_minus_one = period_f - 1.0;
    for i in (period + 1)..n {
        let change = source_array[i] - source_array[i - 1];
        let (current_gain, current_loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, change.abs())
        };

        avg_gain = (avg_gain * period_minus_one + current_gain) / period_f;
        avg_loss = (avg_loss * period_minus_one + current_loss) / period_f;
    }

    let last = if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    };

    Ok(last)
}

/// Calculate SRSI (Stochastic RSI) - Optimized version
#[pyfunction]
pub fn srsi(source: PyReadonlyArray1<f64>, period: usize, period_stoch: usize, k_period: usize, d_period: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut k_values = Array1::<f64>::from_elem(n, f64::NAN);
        let mut d_values = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }

        // Inline RSI calculation with rolling stochastic
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        
        // Calculate initial RSI values
        for i in 1..=period {
            let change = source_array[i] - source_array[i - 1];
            if change > 0.0 {
                sum_gain += change;
            } else {
                sum_loss += change.abs();
            }
        }
        
        let mut avg_gain = sum_gain / period as f64;
        let mut avg_loss = sum_loss / period as f64;
        
        // Rolling buffers for stochastic calculation
        let mut rsi_buffer = std::collections::VecDeque::with_capacity(period_stoch);
        let mut k_buffer = std::collections::VecDeque::with_capacity(k_period);
        
        // Process each data point
        for i in period..n {
            // Calculate RSI for current point
            let rsi_val = if i == period {
                // First RSI value
                if avg_loss == 0.0 {
                    100.0
                } else {
                    let rs = avg_gain / avg_loss;
                    100.0 - (100.0 / (1.0 + rs))
                }
            } else {
                // Subsequent RSI values using Wilder's smoothing
                let change = source_array[i] - source_array[i - 1];
                let (current_gain, current_loss) = if change > 0.0 {
                    (change, 0.0)
                } else {
                    (0.0, change.abs())
                };
                
                avg_gain = (avg_gain * (period as f64 - 1.0) + current_gain) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + current_loss) / period as f64;
                
                if avg_loss == 0.0 {
                    100.0
                } else {
                    let rs = avg_gain / avg_loss;
                    100.0 - (100.0 / (1.0 + rs))
                }
            };
            
            // Add to RSI buffer
            rsi_buffer.push_back(rsi_val);
            if rsi_buffer.len() > period_stoch {
                rsi_buffer.pop_front();
            }
            
            // Calculate %K when we have enough RSI values
            if rsi_buffer.len() == period_stoch {
                let rsi_min = rsi_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let rsi_max = rsi_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                let k_val = if rsi_max != rsi_min {
                    100.0 * (rsi_val - rsi_min) / (rsi_max - rsi_min)
                } else {
                    f64::NAN
                };
                
                if k_period > 1 {
                    // Smooth %K
                    k_buffer.push_back(k_val);
                    if k_buffer.len() > k_period {
                        k_buffer.pop_front();
                    }
                    
                    if k_buffer.len() == k_period && k_buffer.iter().all(|&x| !x.is_nan()) {
                        let k_smoothed = k_buffer.iter().sum::<f64>() / k_period as f64;
                        k_values[i] = k_smoothed;
                    }
                } else {
                    // No smoothing needed
                    k_values[i] = k_val;
                }
            }
        }
        
        // Calculate %D (SMA of %K) using rolling sum
        if d_period > 0 {
            let mut d_buffer = std::collections::VecDeque::with_capacity(d_period);
            
            for i in 0..n {
                if !k_values[i].is_nan() {
                    d_buffer.push_back(k_values[i]);
                    if d_buffer.len() > d_period {
                        d_buffer.pop_front();
                    }
                    
                    if d_buffer.len() == d_period {
                        d_values[i] = d_buffer.iter().sum::<f64>() / d_period as f64;
                    }
                }
            }
        }
        
        Ok((
            PyArray1::from_array(py, &k_values).to_owned(),
            PyArray1::from_array(py, &d_values).to_owned()
        ))
    })
}

/// Calculate MACD (Moving Average Convergence/Divergence)
#[pyfunction]
pub fn macd(source: PyReadonlyArray1<f64>, fast_period: usize, slow_period: usize, signal_period: usize) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut macd_line_result = Array1::<f64>::zeros(n);
        let mut signal_line_result = Array1::<f64>::zeros(n);
        let mut hist_result = Array1::<f64>::zeros(n);

        if n == 0 {
            return Ok((
                PyArray1::from_array(py, &macd_line_result).to_owned(),
                PyArray1::from_array(py, &signal_line_result).to_owned(),
                PyArray1::from_array(py, &hist_result).to_owned(),
            ));
        }

        let alpha_fast = 2.0 / (fast_period as f64 + 1.0);
        let alpha_slow = 2.0 / (slow_period as f64 + 1.0);
        let alpha_signal = 2.0 / (signal_period as f64 + 1.0);

        let mut ema_fast = source_array[0];
        let mut ema_slow = source_array[0];
        
        let macd_val = ema_fast - ema_slow;
        let macd_val_cleaned = if macd_val.is_nan() { 0.0 } else { macd_val };
        
        let mut signal_ema = macd_val_cleaned;

        macd_line_result[0] = macd_val_cleaned;
        signal_line_result[0] = signal_ema;
        hist_result[0] = macd_val - signal_ema;

        for i in 1..n {
            ema_fast = alpha_fast * source_array[i] + (1.0 - alpha_fast) * ema_fast;
            ema_slow = alpha_slow * source_array[i] + (1.0 - alpha_slow) * ema_slow;

            let macd_val = ema_fast - ema_slow;
            let macd_val_cleaned = if macd_val.is_nan() { 0.0 } else { macd_val };
            
            signal_ema = alpha_signal * macd_val_cleaned + (1.0 - alpha_signal) * signal_ema;
            
            let hist_val = macd_val - signal_ema;

            macd_line_result[i] = macd_val_cleaned;
            signal_line_result[i] = signal_ema;
            hist_result[i] = hist_val;
        }

        Ok((
            PyArray1::from_array(py, &macd_line_result).to_owned(),
            PyArray1::from_array(py, &signal_line_result).to_owned(),
            PyArray1::from_array(py, &hist_result).to_owned(),
        ))
    })
}

/// Calculate Williams' %R - Ultra-optimized version
#[pyfunction]
pub fn willr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract required data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for efficient sliding window
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
        
        // Initialize first window
        for i in 0..period.min(n) {
            let high_val = high[i];
            let low_val = low[i];
            
            // Maintain decreasing order for max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Maintain increasing order for min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
        }
        
        // Set first valid result
        if period <= n {
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            let denom = max_high - min_low;
            if denom == 0.0 {
                result[period - 1] = 0.0;
            } else {
                result[period - 1] = ((max_high - close[period - 1]) / denom) * -100.0;
            }
        }
        
        // Process remaining elements
        for i in period..n {
            let high_val = high[i];
            let low_val = low[i];
            
            // Remove expired elements from max deque
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Remove expired elements from min deque
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element to max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Add new element to min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate Williams' %R
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            let denom = max_high - min_low;
            if denom == 0.0 {
                result[i] = 0.0;
            } else {
                result[i] = ((max_high - close[i]) / denom) * -100.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Stochastic Oscillator - Ultra-optimized version
#[pyfunction]
pub fn stoch(candles: PyReadonlyArray2<f64>, fastk_period: usize, slowk_period: usize, _slowk_matype: usize, slowd_period: usize, _slowd_matype: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        if n < fastk_period {
            let k_values = Array1::<f64>::from_elem(n, f64::NAN);
            let d_values = Array1::<f64>::from_elem(n, f64::NAN);
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }
        
        // Extract price data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate rolling highs and lows using simple approach to match Python exactly
        let mut hh = Array1::<f64>::from_elem(n, f64::NAN);
        let mut ll = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in (fastk_period - 1)..n {
            let start_idx = i + 1 - fastk_period;
            let end_idx = i + 1;
            
            let window_high = high.slice(s![start_idx..end_idx]);
            let window_low = low.slice(s![start_idx..end_idx]);
            
            hh[i] = window_high.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ll[i] = window_low.fold(f64::INFINITY, |a, &b| a.min(b));
        }
        
        // Calculate raw %K values
        let mut raw_k = Array1::<f64>::from_elem(n, f64::NAN);
        for i in (fastk_period - 1)..n {
            let hh_val = hh[i];
            let ll_val = ll[i];
            let close_val = close[i];
            
            if hh_val > ll_val {
                raw_k[i] = 100.0 * (close_val - ll_val) / (hh_val - ll_val);
            }
        }
        
        // Apply smoothing for %K (slow K)
        let smoothed_k = sma_array(&raw_k, slowk_period);
        
        // Apply smoothing for %D
        let smoothed_d = sma_array(&smoothed_k, slowd_period);
        
        Ok((
            PyArray1::from_array(py, &smoothed_k).to_owned(),
            PyArray1::from_array(py, &smoothed_d).to_owned()
        ))
    })
}

/// Calculate Stochastic Fast - Ultra-optimized version
#[pyfunction]
pub fn stochf(candles: PyReadonlyArray2<f64>, fastk_period: usize, fastd_period: usize, fastd_matype: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut k_values = Array1::<f64>::from_elem(n, f64::NAN);
        let mut d_values = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < fastk_period {
            return Ok((
                PyArray1::from_array(py, &k_values).to_owned(),
                PyArray1::from_array(py, &d_values).to_owned()
            ));
        }
        
        // Extract price data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for efficient sliding window max/min
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        
        // Initialize first window
        for i in 0..fastk_period.min(n) {
            let high_val = high[i];
            let low_val = low[i];
            
            // Maintain decreasing order for max deque
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Maintain increasing order for min deque
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
        }
        
        // Calculate fast stochastic values
        // First valid value
        if n >= fastk_period {
            let hh = max_deque.front().unwrap().1;
            let ll = min_deque.front().unwrap().1;
            if hh > ll {
                k_values[fastk_period - 1] = 100.0 * (close[fastk_period - 1] - ll) / (hh - ll);
            } else {
                k_values[fastk_period - 1] = 50.0; // Default when no range
            }
        }
        
        // Sliding window for remaining values
        for i in fastk_period..n {
            // Remove elements outside window
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - fastk_period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - fastk_period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element
            let high_val = high[i];
            let low_val = low[i];
            
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate %K
            let hh = max_deque.front().unwrap().1;
            let ll = min_deque.front().unwrap().1;
            if hh > ll {
                k_values[i] = 100.0 * (close[i] - ll) / (hh - ll);
            } else {
                k_values[i] = 50.0;
            }
        }
        
        // Apply smoothing to get %D
        let smoothed_d = if fastd_matype == 0 {
            // SMA
            sma_array(&k_values, fastd_period)
        } else {
            // Other MA types - simplified, using SMA for now
            sma_array(&k_values, fastd_period)
        };
        
        d_values = smoothed_d;
        
        Ok((
            PyArray1::from_array(py, &k_values).to_owned(),
            PyArray1::from_array(py, &d_values).to_owned()
        ))
    })
}

/// Calculate DTI (Dynamic Trend Index) by William Blau - Optimized version
#[pyfunction]
pub fn dti(candles: PyReadonlyArray2<f64>, r: usize, s: usize, u: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= r || n <= s || n <= u {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract high and low price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Shift high and low by 1 period
        let mut high_1 = Array1::<f64>::from_elem(n, f64::NAN);
        let mut low_1 = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in 1..n {
            high_1[i] = high[i - 1];
            low_1[i] = low[i - 1];
        }
        
        // Compute upward and downward movements
        let mut xhmu = Array1::<f64>::zeros(n);
        let mut xlmd = Array1::<f64>::zeros(n);
        
        for i in 1..n {
            let high_diff = high[i] - high_1[i];
            if high_diff > 0.0 {
                xhmu[i] = high_diff;
            }
            
            let low_diff = low[i] - low_1[i];
            if low_diff < 0.0 {
                xlmd[i] = -low_diff;
            }
        }
        
        // Calculate xPrice and xPriceAbs
        let mut xprice = Array1::<f64>::zeros(n);
        let mut xprice_abs = Array1::<f64>::zeros(n);
        
        for i in 0..n {
            xprice[i] = xhmu[i] - xlmd[i];
            xprice_abs[i] = xprice[i].abs();
        }
        
        // Apply triple EMA to xPrice
        let r_alpha = 2.0 / (r as f64 + 1.0);
        let s_alpha = 2.0 / (s as f64 + 1.0);
        let u_alpha = 2.0 / (u as f64 + 1.0);
        
        let r_one_minus_alpha = 1.0 - r_alpha;
        let s_one_minus_alpha = 1.0 - s_alpha;
        let u_one_minus_alpha = 1.0 - u_alpha;
        
        // First stage EMA on xPrice
        let mut temp = Array1::<f64>::zeros(n);
        temp[0] = xprice[0];
        for i in 1..n {
            temp[i] = r_alpha * xprice[i] + r_one_minus_alpha * temp[i - 1];
        }
        
        // Second stage EMA
        let mut temp2 = Array1::<f64>::zeros(n);
        temp2[0] = temp[0];
        for i in 1..n {
            temp2[i] = s_alpha * temp[i] + s_one_minus_alpha * temp2[i - 1];
        }
        
        // Third stage EMA (xuXA)
        let mut xu_xa = Array1::<f64>::zeros(n);
        xu_xa[0] = temp2[0];
        for i in 1..n {
            xu_xa[i] = u_alpha * temp2[i] + u_one_minus_alpha * xu_xa[i - 1];
        }
        
        // Apply triple EMA to xPriceAbs
        // First stage EMA on xPriceAbs
        let mut temp_abs = Array1::<f64>::zeros(n);
        temp_abs[0] = xprice_abs[0];
        for i in 1..n {
            temp_abs[i] = r_alpha * xprice_abs[i] + r_one_minus_alpha * temp_abs[i - 1];
        }
        
        // Second stage EMA
        let mut temp2_abs = Array1::<f64>::zeros(n);
        temp2_abs[0] = temp_abs[0];
        for i in 1..n {
            temp2_abs[i] = s_alpha * temp_abs[i] + s_one_minus_alpha * temp2_abs[i - 1];
        }
        
        // Third stage EMA (xuXAAbs)
        let mut xu_xa_abs = Array1::<f64>::zeros(n);
        xu_xa_abs[0] = temp2_abs[0];
        for i in 1..n {
            xu_xa_abs[i] = u_alpha * temp2_abs[i] + u_one_minus_alpha * xu_xa_abs[i - 1];
        }
        
        // Calculate Val1 and Val2
        let val1 = xu_xa.mapv(|x| x * 100.0);
        let val2 = xu_xa_abs;
        
        // Calculate DTI value
        for i in 0..n {
            if val2[i] != 0.0 {
                result[i] = val1[i] / val2[i];
            } else {
                result[i] = 0.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Wavetrend indicator
#[pyfunction]
pub fn wt(
    candles: PyReadonlyArray2<f64>,
    wtchannellen: usize,
    wtaveragelen: usize,
    wtmalen: usize,
    oblevel: f64,
    oslevel: f64,
    source_type: &str
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<bool>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        // Extract data from candles based on source_type
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        
        // Get source data based on source_type
        let src = match source_type {
            "open" => {
                let opens = candles_array.slice(s![.., 1]);
                opens.to_owned()
            },
            "high" => {
                let highs = candles_array.slice(s![.., 3]);
                highs.to_owned()
            },
            "low" => {
                let lows = candles_array.slice(s![.., 4]);
                lows.to_owned()
            },
            "close" => {
                let closes = candles_array.slice(s![.., 2]);
                closes.to_owned()
            },
            "hlc3" => {
                let highs = candles_array.slice(s![.., 3]);
                let lows = candles_array.slice(s![.., 4]);
                let closes = candles_array.slice(s![.., 2]);
                
                let mut result = Array1::<f64>::zeros(n);
                for i in 0..n {
                    result[i] = (highs[i] + lows[i] + closes[i]) / 3.0;
                }
                result
            },
            "ohlc4" => {
                let opens = candles_array.slice(s![.., 1]);
                let highs = candles_array.slice(s![.., 3]);
                let lows = candles_array.slice(s![.., 4]);
                let closes = candles_array.slice(s![.., 2]);
                
                let mut result = Array1::<f64>::zeros(n);
                for i in 0..n {
                    result[i] = (opens[i] + highs[i] + lows[i] + closes[i]) / 4.0;
                }
                result
            },
            _ => {
                // Default to close
                let closes = candles_array.slice(s![.., 2]);
                closes.to_owned()
            }
        };
        
        // Calculate Wavetrend components
        let esa = ema_for_wt(&src, wtchannellen);
        
        // Calculate absolute difference
        let mut abs_diff = Array1::<f64>::zeros(n);
        for i in 0..n {
            abs_diff[i] = (src[i] - esa[i]).abs();
        }
        
        let de = ema_for_wt(&abs_diff, wtchannellen);
        
        // Calculate CI (avoid division by zero)
        let mut ci = Array1::<f64>::zeros(n);
        for i in 0..n {
            ci[i] = if de[i] == 0.0 {
                0.0
            } else {
                (src[i] - esa[i]) / (0.015 * de[i])
            };
        }
        
        // Calculate wt1 and wt2
        let wt1 = ema_for_wt(&ci, wtaveragelen);
        let wt2 = sma_for_wt(&wt1, wtmalen);
        
        // Calculate additional components
        let mut wtvwap = Array1::<f64>::zeros(n);
        let mut wtcrossup = Array1::<bool>::from_elem(n, false);
        let mut wtcrossdown = Array1::<bool>::from_elem(n, false);
        let mut wtoversold = Array1::<bool>::from_elem(n, false);
        let mut wtoverbought = Array1::<bool>::from_elem(n, false);
        
        for i in 0..n {
            wtvwap[i] = wt1[i] - wt2[i];
            wtcrossup[i] = wt2[i] - wt1[i] <= 0.0;
            wtcrossdown[i] = wt2[i] - wt1[i] >= 0.0;
            wtoversold[i] = wt2[i] <= oslevel;
            wtoverbought[i] = wt2[i] >= oblevel;
        }
        
        Ok((
            PyArray1::from_array(py, &wt1).to_owned(),
            PyArray1::from_array(py, &wt2).to_owned(),
            PyArray1::from_array(py, &wtcrossup).to_owned(),
            PyArray1::from_array(py, &wtcrossdown).to_owned(),
            PyArray1::from_array(py, &wtoversold).to_owned(),
            PyArray1::from_array(py, &wtoverbought).to_owned(),
            PyArray1::from_array(py, &wtvwap).to_owned()
        ))
    })
}

/// Calculate FOSC (Forecast Oscillator)
#[pyfunction]
pub fn fosc(
    source: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source = source.as_array();
        let n = source.len();
        let mut result = Array1::<f64>::from_elem(n, 0.0);

        if n < period || period == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        let period_f64 = period as f64;
        let sum_x: f64 = (0..period).map(|i| i as f64).sum();
        let mean_x = sum_x / period_f64;
        let sum_xx: f64 = (0..period).map(|i| (i as f64)*(i as f64)).sum();
        let denom = sum_xx - sum_x*mean_x;
        
        // Prefix sums of y and k*y
        let mut prefix_y = Array1::<f64>::zeros(n);
        let mut prefix_xy = Array1::<f64>::zeros(n);
        let mut cum_y = 0.0;
        let mut cum_xy = 0.0;
        for i in 0..n {
            cum_y += source[i];
            cum_xy += (i as f64) * source[i];
            prefix_y[i] = cum_y;
            prefix_xy[i] = cum_xy;
        }
        
        for end in (period-1)..n {
            let start = end + 1 - period;
            let sum_y = if start == 0 { prefix_y[end] } else { prefix_y[end] - prefix_y[start-1] };
            let raw_sum_xy = if start == 0 { prefix_xy[end] } else { prefix_xy[end] - prefix_xy[start-1] };
            // Shift x values so that start becomes 0
            let sum_xy = raw_sum_xy - (start as f64)*sum_y;
            let mean_y = sum_y / period_f64;
            let slope = (sum_xy - sum_x*mean_y)/denom;
            let intercept = mean_y - slope*mean_x;
            let predicted = slope*(period_f64-1.0)+intercept;
            let actual = source[end];
            result[end] = if actual != 0.0 { 100.0*(actual-predicted)/actual } else { 0.0 };
        }
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// DPO — Detrended Price Oscillator
#[pyfunction]
pub fn dpo(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let shift = period / 2 + 1;
        let sma_vals = ih_sma(src.as_slice().unwrap(), period);
        let invalid = period - 1 + shift;
        let mut r = vec![f64::NAN; n];
        for i in invalid..n {
            if i >= shift {
                r[i] = src[i - shift] - sma_vals[i];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CCI — Commodity Channel Index
#[pyfunction]
pub fn cci(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let tp: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]] + c[[i,2]]) / 3.0).collect();
        let mut r = vec![f64::NAN; n];
        for i in (period-1)..n {
            let start = i + 1 - period;
            let sma: f64 = tp[start..=i].iter().sum::<f64>() / period as f64;
            let md: f64 = tp[start..=i].iter().map(|&x| (x - sma).abs()).sum::<f64>() / period as f64;
            r[i] = if md == 0.0 { 0.0 } else { (tp[i] - sma) / (0.015 * md) };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CMO — Chande Momentum Oscillator
#[pyfunction]
pub fn cmo(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n <= period || period == 0 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Pre-split diffs into positive and negative magnitudes
        let mut pos_arr = vec![0.0f64; n - 1];
        let mut neg_arr = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            let d = src[i + 1] - src[i];
            if d > 0.0 { pos_arr[i] = d; } else if d < 0.0 { neg_arr[i] = -d; }
        }
        // Rolling window sum over [i-period, i) which is pos_arr/neg_arr indices [i-period, i-1]
        // First window for i=period: indices [0..period]
        let mut pos: f64 = pos_arr[..period].iter().sum();
        let mut neg: f64 = neg_arr[..period].iter().sum();
        let denom = pos + neg;
        r[period] = if denom == 0.0 { 0.0 } else { 100.0 * (pos - neg) / denom };
        for i in (period + 1)..n {
            // Slide: remove pos_arr[i-period-1], add pos_arr[i-1]
            pos += pos_arr[i - 1] - pos_arr[i - period - 1];
            neg += neg_arr[i - 1] - neg_arr[i - period - 1];
            let denom = pos + neg;
            r[i] = if denom == 0.0 { 0.0 } else { 100.0 * (pos - neg) / denom };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CFO — Chande Forecast Oscillator
#[pyfunction]
pub fn cfo(source: PyReadonlyArray1<f64>, period: usize, scalar: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if period == 0 || n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let pf = period as f64;
        let sx: f64 = (0..period).map(|j| j as f64).sum();
        let sxx: f64 = (0..period).map(|j| (j as f64).powi(2)).sum();
        let denom = pf * sxx - sx * sx;
        // sxy_i = B_i - (i-period+1)*A_i where A_i = sum src[k] and B_i = sum k*src[k] over window
        let mut a: f64 = src.slice(s![0..period]).iter().sum();
        let mut b: f64 = (0..period).map(|k| (k as f64) * src[k]).sum();
        // First valid index: i = period - 1
        let i0 = period - 1;
        {
            let sxy = b - (i0 as f64 - pf + 1.0) * a;
            let slope = (pf * sxy - sx * a) / denom;
            let intercept = (a - slope * sx) / pf;
            let reg_val = intercept + slope * (pf - 1.0);
            r[i0] = if src[i0] != 0.0 { scalar * (src[i0] - reg_val) / src[i0] } else { f64::NAN };
        }
        for i in period..n {
            let kout = i - period;
            a += src[i] - src[kout];
            b += (i as f64) * src[i] - (kout as f64) * src[kout];
            let sxy = b - (i as f64 - pf + 1.0) * a;
            let slope = (pf * sxy - sx * a) / denom;
            let intercept = (a - slope * sx) / pf;
            let reg_val = intercept + slope * (pf - 1.0);
            r[i] = if src[i] != 0.0 { scalar * (src[i] - reg_val) / src[i] } else { f64::NAN };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// CG — Center of Gravity
#[pyfunction]
pub fn cg(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if period < 2 || n <= period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Window of length (period - 1) ending at i, i.e. k in [i - period + 2, i].
        //   denom    = Σ src[k]
        //   numer    = Σ (1 + i - k) · src[k] = (1 + i) · denom - Σ k · src[k]
        // Both sums are maintained as O(1)-update rolling sums.
        let first_i = period + 1;
        let start_k = first_i + 2 - period;
        let mut denom: f64 = src.slice(s![start_k..=first_i]).iter().sum();
        let mut sum_kx: f64 = (start_k..=first_i).map(|k| (k as f64) * src[k]).sum();
        let num0 = (1.0 + first_i as f64) * denom - sum_kx;
        r[first_i] = if denom != 0.0 { -num0 / denom } else { 0.0 };
        for i in (first_i + 1)..n {
            // For i, window is [i - period + 2, i]
            let kin = i;
            let kout = i - period + 1; // (i-1) - (period-1) + 1 = i - period + 1
            denom += src[kin] - src[kout];
            sum_kx += (kin as f64) * src[kin] - (kout as f64) * src[kout];
            let num = (1.0 + i as f64) * denom - sum_kx;
            r[i] = if denom != 0.0 { -num / denom } else { 0.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Aroon Oscillator
#[pyfunction]
pub fn aroonosc(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut r = vec![f64::NAN; n];
        for i in (period-1)..n {
            let start = i + 1 - period;
            let mut best_val = c[[start,3]];
            let mut best_idx = 0usize;
            let mut worst_val = c[[start,4]];
            let mut worst_idx = 0usize;
            for j in 0..period {
                if c[[start+j,3]] > best_val { best_val = c[[start+j,3]]; best_idx = j; }
                if c[[start+j,4]] < worst_val { worst_val = c[[start+j,4]]; worst_idx = j; }
            }
            r[i] = 100.0 * (best_idx as f64 - worst_idx as f64) / period as f64;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// MASS — Mass Index
#[pyfunction]
pub fn mass(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let hl: Vec<f64> = (0..n).map(|i| c[[i,3]] - c[[i,4]]).collect();
        let e1 = ih_ema(&hl, 9);
        let e2 = ih_ema(&e1, 9);
        let ratio: Vec<f64> = (0..n).map(|i| if e2[i] != 0.0 { e1[i] / e2[i] } else { 0.0 }).collect();
        let mut r = vec![0.0f64; n];
        for i in (period-1)..n {
            r[i] = ratio[i+1-period..=i].iter().sum();
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// PFE — Polarized Fractal Efficiency
#[pyfunction]
pub fn pfe(source: PyReadonlyArray1<f64>, period: usize, smoothing: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let ln = period - 1; // lookback = period - 1

        // Compute ln-th order differences (matches Python's np.diff(source, ln))
        // Each iteration reduces length by 1; after ln iterations: length = n - ln
        let mut d: Vec<f64> = src.clone();
        for _ in 0..ln {
            d = d.windows(2).map(|w| w[1] - w[0]).collect();
        }
        // d has length n - ln

        // a[i] = sqrt(d[i]^2 + period^2)
        let a: Vec<f64> = d.iter().map(|&x| (x * x + (period as f64).powi(2)).sqrt()).collect();

        // First differences and their sqrt(1 + dx^2) terms; length = n - 1
        let sqrt_term: Vec<f64> = src.windows(2).map(|w| {
            let dx = w[1] - w[0];
            (1.0 + dx * dx).sqrt()
        }).collect();

        // Rolling sum of sqrt_term with window = ln; matches rolling_sum(sqrt_term, ln)
        // Result: first (ln-1) elements are NaN, then (n-ln) valid sums
        // Total length: n - 1 (same as sqrt_term)
        let mut b: Vec<f64> = vec![f64::NAN; ln - 1];
        if sqrt_term.len() >= ln {
            let init: f64 = sqrt_term[..ln].iter().sum();
            b.push(init);
            for i in ln..sqrt_term.len() {
                let prev = *b.last().unwrap();
                b.push(prev + sqrt_term[i] - sqrt_term[i - ln]);
            }
        }
        // b has length n - 1

        // Align to source length (same_length behavior: prepend NaN to shorter array)
        // a_sl: prepend (n - (n-ln)) = ln NaN → a_sl[i] = a[i-ln] for i >= ln
        // b_sl: prepend (n - (n-1)) = 1 NaN → b_sl[i] = b[i-1] for i >= 1
        // diff_sl: same as a_sl, d[i-ln] for i >= ln
        let mut pfetmp = vec![0.0f64; n];
        let mut sign = vec![-1.0f64; n]; // default: -1 (NaN > 0 is False)
        for i in ln..n {
            let a_val = a[i - ln];
            let b_val = if i >= 1 { b[i - 1] } else { f64::NAN };
            pfetmp[i] = if b_val.is_nan() || b_val == 0.0 {
                0.0
            } else {
                100.0 * a_val / b_val
            };
            if d[i - ln] > 0.0 {
                sign[i] = 1.0;
            }
        }

        let signed: Vec<f64> = sign.iter().zip(pfetmp.iter()).map(|(s, p)| s * p).collect();
        let r = ih_ema(&signed, smoothing);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Laguerre RSI
#[pyfunction]
pub fn lrsi(candles: PyReadonlyArray2<f64>, alpha: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let price: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let gamma = 1.0 - alpha;
        let (mut l0, mut l1, mut l2, mut l3) = (price[0], price[0], price[0], price[0]);
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let new_l0 = alpha * price[i] + gamma * l0;
            let new_l1 = -gamma * new_l0 + l0 + gamma * l1;
            let new_l2 = -gamma * new_l1 + l1 + gamma * l2;
            let new_l3 = -gamma * new_l2 + l2 + gamma * l3;
            l0 = new_l0; l1 = new_l1; l2 = new_l2; l3 = new_l3;
            let mut cu = 0.0f64;
            let mut cd = 0.0f64;
            if l0 >= l1 { cu += l0 - l1; } else { cd += l1 - l0; }
            if l1 >= l2 { cu += l1 - l2; } else { cd += l2 - l1; }
            if l2 >= l3 { cu += l2 - l3; } else { cd += l3 - l2; }
            r[i] = if cu + cd == 0.0 { 0.0 } else { cu / (cu + cd) };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// RSX — Relative Strength Xtra
#[pyfunction]
pub fn rsx(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        let (mut f0, mut f8) = (0.0f64, 0.0f64);
        let (mut f18, mut f20) = (0.0f64, 0.0f64);
        let (mut f28, mut f30) = (0.0f64, 0.0f64);
        let (mut f38, mut f40) = (0.0f64, 0.0f64);
        let (mut f48, mut f50) = (0.0f64, 0.0f64);
        let (mut f58, mut f60) = (0.0f64, 0.0f64);
        let (mut f68, mut f70) = (0.0f64, 0.0f64);
        let (mut f78, mut f80) = (0.0f64, 0.0f64);
        let (mut f88, mut f90) = (0.0f64, 0.0f64);
        let mut v14 = 0.0f64;
        let mut v20 = 0.0f64;
        for i in period..n {
            if f90 == 0.0 {
                f90 = 1.0; f0 = 0.0;
                f88 = if period >= 6 { period as f64 - 1.0 } else { 5.0 };
                f8 = 100.0 * src[i];
                f18 = 3.0 / (period as f64 + 2.0);
                f20 = 1.0 - f18;
            } else {
                f90 = if f88 <= f90 { f88 + 1.0 } else { f90 + 1.0 };
                let f10 = f8;
                f8 = 100.0 * src[i];
                let v8 = f8 - f10;
                f28 = f20 * f28 + f18 * v8;
                f30 = f18 * f28 + f20 * f30;
                let vc = f28 * 1.5 - f30 * 0.5;
                f38 = f20 * f38 + f18 * vc;
                f40 = f18 * f38 + f20 * f40;
                let v10 = f38 * 1.5 - f40 * 0.5;
                f48 = f20 * f48 + f18 * v10;
                f50 = f18 * f48 + f20 * f50;
                v14 = f48 * 1.5 - f50 * 0.5;
                f58 = f20 * f58 + f18 * v8.abs();
                f60 = f18 * f58 + f20 * f60;
                let v18 = f58 * 1.5 - f60 * 0.5;
                f68 = f20 * f68 + f18 * v18;
                f70 = f18 * f68 + f20 * f70;
                let v1c = f68 * 1.5 - f70 * 0.5;
                f78 = f20 * f78 + f18 * v1c;
                f80 = f18 * f78 + f20 * f80;
                v20 = f78 * 1.5 - f80 * 0.5;
                if f88 >= f90 && f8 != f10 { f0 = 1.0; }
                if f88 == f90 && f0 == 0.0 { f90 = 0.0; }
            }
            r[i] = if f88 < f90 && v20 > 1e-10 {
                ((v14 / v20 + 1.0) * 50.0).clamp(0.0, 100.0)
            } else { 50.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Fisher Transform → (fisher, signal)
#[pyfunction]
pub fn fisher(candles: PyReadonlyArray2<f64>, period: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mid: Vec<f64> = (0..n).map(|i| (c[[i,3]] + c[[i,4]]) / 2.0).collect();
        let mut fisher_v = vec![0.0f64; n];
        let mut fisher_sig = vec![0.0f64; n];
        let mut value1 = 0.0f64;
        for i in period..n {
            let max_h = mid[i+1-period..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_l = mid[i+1-period..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let mut value0 = if max_h - min_l == 0.0 { 0.0 } else {
                0.33 * 2.0 * ((mid[i] - min_l) / (max_h - min_l) - 0.5) + 0.67 * value1
            };
            value0 = value0.clamp(-0.999, 0.999);
            fisher_v[i] = 0.5 * ((1.0 + value0) / (1.0 - value0)).ln() + 0.5 * fisher_v[i-1];
            fisher_sig[i] = fisher_v[i-1];
            value1 = value0;
        }
        Ok((
            PyArray1::from_vec(py, fisher_v).to_owned(),
            PyArray1::from_vec(py, fisher_sig).to_owned(),
        ))
    })
}
