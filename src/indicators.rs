use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, s, ArrayView1};
use pyo3::types::PyDict;
use rust_decimal::Decimal;
use std::str::FromStr;

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

/// Calculate Ichimoku Cloud
#[pyfunction]
pub fn ichimoku_cloud(
    candles: PyReadonlyArray2<f64>,
    conversion_line_period: usize,
    base_line_period: usize,
    lagging_line_period: usize,
    displacement: usize
) -> PyResult<(f64, f64, f64, f64)> {
    Python::with_gil(|_py| {
        let candles_array = candles.as_array();
        
        // Get the high and low price columns
        let high_prices = candles_array.slice(s![.., 3]);
        let low_prices = candles_array.slice(s![.., 4]);
        
        // Calculate for earlier period (displaced)
        let earlier_high = high_prices.slice(s![..-((displacement as isize) - 1)]);
        let earlier_low = low_prices.slice(s![..-((displacement as isize) - 1)]);
        
        // Helper function to get period high and low
        let get_period_hl = |highs: ndarray::ArrayView1<f64>, lows: ndarray::ArrayView1<f64>, period: usize| -> (f64, f64) {
            let n = highs.len();
            if n < period {
                return (f64::NAN, f64::NAN);
            }
            
            // Instead of negative slicing, use the last 'period' elements
            let start_idx = n.saturating_sub(period);
            let period_high = highs.slice(s![start_idx..]).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low = lows.slice(s![start_idx..]).fold(f64::INFINITY, |a, &b| a.min(b));
            
            (period_high, period_low)
        };
        
        // Earlier periods calculations
        let (small_ph, small_pl) = get_period_hl(earlier_high, earlier_low, conversion_line_period);
        let (mid_ph, mid_pl) = get_period_hl(earlier_high, earlier_low, base_line_period);
        let (long_ph, long_pl) = get_period_hl(earlier_high, earlier_low, lagging_line_period);
        
        let early_conversion_line = (small_ph + small_pl) / 2.0;
        let early_base_line = (mid_ph + mid_pl) / 2.0;
        let span_a = (early_conversion_line + early_base_line) / 2.0;
        let span_b = (long_ph + long_pl) / 2.0;
        
        // Current period calculations
        let (current_small_ph, current_small_pl) = get_period_hl(high_prices, low_prices, conversion_line_period);
        let (current_mid_ph, current_mid_pl) = get_period_hl(high_prices, low_prices, base_line_period);
        
        let current_conversion_line = (current_small_ph + current_small_pl) / 2.0;
        let current_base_line = (current_mid_ph + current_mid_pl) / 2.0;
        
        Ok((current_conversion_line, current_base_line, span_a, span_b))
    })
}

/// Calculate SRSI (Stochastic RSI) - Optimized version
#[pyfunction]
pub fn srsi(source: PyReadonlyArray1<f64>, period: usize, period_stoch: usize, k_period: usize, d_period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
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

/// Calculate ADX (Average Directional Index)
#[pyfunction]
pub fn adx(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut adx_result = Array1::<f64>::from_elem(n, f64::NAN);

        let required_len = 2 * period;
        if n <= required_len {
            return Ok(PyArray1::from_array(py, &adx_result).to_owned());
        }

        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);

        // State for Wilder smoothing
        let mut tr_smooth: f64 = 0.0;
        let mut plus_dm_smooth: f64 = 0.0;
        let mut minus_dm_smooth: f64 = 0.0;
        
        // Buffer for DX values to calculate the first ADX
        let mut dx_buffer: Vec<f64> = Vec::with_capacity(period);

        // Main calculation loop
        for i in 1..n {
            // 1. Calculate raw TR, +DM, -DM for current step `i`
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);

            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];

            let mut current_plus_dm = 0.0;
            if h_diff > l_diff && h_diff > 0.0 {
                current_plus_dm = h_diff;
            }

            let mut current_minus_dm = 0.0;
            if l_diff > h_diff && l_diff > 0.0 {
                current_minus_dm = l_diff;
            }

            // 2. Update smoothed values
            if i <= period {
                // Accumulate for the first smoothed value
                tr_smooth += current_tr;
                plus_dm_smooth += current_plus_dm;
                minus_dm_smooth += current_minus_dm;
            } else {
                // Apply Wilder's smoothing formula
                tr_smooth = tr_smooth - (tr_smooth / period as f64) + current_tr;
                plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period as f64) + current_plus_dm;
                minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period as f64) + current_minus_dm;
            }
            
            // From index `period` onwards, we can calculate DI and DX
            if i >= period {
                let mut current_dx = 0.0;
                if tr_smooth != 0.0 {
                    let di_plus = 100.0 * plus_dm_smooth / tr_smooth;
                    let di_minus = 100.0 * minus_dm_smooth / tr_smooth;
                    let di_sum = di_plus + di_minus;
                    if di_sum != 0.0 {
                        current_dx = 100.0 * (di_plus - di_minus).abs() / di_sum;
                    }
                }
                
                // Store DX value for initial ADX calculation, or calculate ADX
                if i < required_len {
                    dx_buffer.push(current_dx);
                } else {
                    if i == required_len {
                        // First ADX value is the average of the buffer
                        let dx_sum: f64 = dx_buffer.iter().sum();
                        adx_result[i] = dx_sum / period as f64;
                    } else {
                        // Subsequent ADX values are smoothed
                        if !adx_result[i - 1].is_nan() {
                           adx_result[i] = (adx_result[i - 1] * (period - 1) as f64 + current_dx) / period as f64;
                        }
                    }
                }
            }
        }

        Ok(PyArray1::from_array(py, &adx_result).to_owned())
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

/// Calculate MACD (Moving Average Convergence/Divergence)
#[pyfunction]
pub fn macd(source: PyReadonlyArray1<f64>, fast_period: usize, slow_period: usize, signal_period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
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

/// Calculate Bollinger Bands Width - Optimized version
#[pyfunction]
pub fn bollinger_bands_width(source: PyReadonlyArray1<f64>, period: usize, mult: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Use rolling window for efficient calculation
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        // Initialize first window
        for i in 0..period {
            let val = source_array[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // Calculate first BBW value
        let sma = sum / period as f64;
        let variance = (sum_sq / period as f64) - (sma * sma);
        let std_dev = variance.sqrt();
        
        // Calculate Bollinger Bands Width
        if sma != 0.0 {
            let upper_band = sma + mult * std_dev;
            let lower_band = sma - mult * std_dev;
            result[period - 1] = (upper_band - lower_band) / sma;
        }
        
        // Calculate subsequent values using rolling window
        for i in period..n {
            let old_val = source_array[i - period];
            let new_val = source_array[i];
            
            // Update rolling sums
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
            
            // Calculate SMA and standard deviation
            let sma = sum / period as f64;
            let variance = (sum_sq / period as f64) - (sma * sma);
            let std_dev = variance.sqrt();
            
            // Calculate Bollinger Bands Width
            if sma != 0.0 {
                let upper_band = sma + mult * std_dev;
                let lower_band = sma - mult * std_dev;
                result[i] = (upper_band - lower_band) / sma;
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Bollinger Bands - Optimized version
#[pyfunction]
pub fn bollinger_bands(source: PyReadonlyArray1<f64>, period: usize, devup: f64, devdn: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        let mut upper_band = Array1::<f64>::from_elem(n, f64::NAN);
        let mut middle_band = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lower_band = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok((
                PyArray1::from_array(py, &upper_band).to_owned(),
                PyArray1::from_array(py, &middle_band).to_owned(),
                PyArray1::from_array(py, &lower_band).to_owned()
            ));
        }

        // Use rolling window for efficient calculation
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        // Initialize first window
        for i in 0..period {
            let val = source_array[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // Calculate first Bollinger Bands values
        let sma = sum / period as f64;
        let variance = (sum_sq / period as f64) - (sma * sma);
        let std_dev = variance.sqrt();
        
        middle_band[period - 1] = sma;
        upper_band[period - 1] = sma + devup * std_dev;
        lower_band[period - 1] = sma - devdn * std_dev;
        
        // Calculate subsequent values using rolling window
        for i in period..n {
            let old_val = source_array[i - period];
            let new_val = source_array[i];
            
            // Update rolling sums
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
            
            // Calculate SMA and standard deviation
            let sma = sum / period as f64;
            let variance = (sum_sq / period as f64) - (sma * sma);
            let std_dev = variance.sqrt();
            
            // Calculate bands
            middle_band[i] = sma;
            upper_band[i] = sma + devup * std_dev;
            lower_band[i] = sma - devdn * std_dev;
        }

        Ok((
            PyArray1::from_array(py, &upper_band).to_owned(),
            PyArray1::from_array(py, &middle_band).to_owned(),
            PyArray1::from_array(py, &lower_band).to_owned()
        ))
    })
}

/// Shift array by a given number of periods
#[pyfunction]
pub fn shift(source: PyReadonlyArray1<f64>, periods: isize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if periods == 0 {
            // No shift, return copy of source
            for i in 0..n {
                result[i] = source_array[i];
            }
        } else if periods > 0 {
            // Shift right (positive periods)
            let shift_amount = periods as usize;
            if shift_amount < n {
                for i in shift_amount..n {
                    result[i] = source_array[i - shift_amount];
                }
            }
        } else {
            // Shift left (negative periods)
            let shift_amount = (-periods) as usize;
            if shift_amount < n {
                for i in 0..(n - shift_amount) {
                    result[i] = source_array[i + shift_amount];
                }
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate moving standard deviation
#[pyfunction]
pub fn moving_std(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate moving standard deviation
        for i in (period - 1)..n {
            let start_idx = i + 1 - period;
            let window = source_array.slice(s![start_idx..=i]);
            
            // Calculate mean
            let mean = window.sum() / period as f64;
            
            // Calculate variance
            let variance = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / period as f64;
            
            // Calculate standard deviation
            result[i] = variance.sqrt();
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

/// Calculate Alligator indicator (Jaw, Teeth, Lips) - Optimized version
#[pyfunction]
pub fn alligator(source: PyReadonlyArray1<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Initialize result arrays
        let mut jaw = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 13 {  // Need at least 13 periods for jaw (longest period)
            return Ok((
                PyArray1::from_array(py, &jaw).to_owned(),
                PyArray1::from_array(py, &teeth).to_owned(),
                PyArray1::from_array(py, &lips).to_owned()
            ));
        }
        
        // SMMA parameters
        let jaw_period = 13;
        let teeth_period = 8;
        let lips_period = 5;
        
        let jaw_alpha = 1.0 / jaw_period as f64;
        let teeth_alpha = 1.0 / teeth_period as f64;
        let lips_alpha = 1.0 / lips_period as f64;
        
        // Calculate initial SMA values for each line
        let mut jaw_sum = 0.0;
        let mut teeth_sum = 0.0;
        let mut lips_sum = 0.0;
        
        // Accumulate sums for initial values
        for i in 0..jaw_period {
            jaw_sum += source_array[i];
            if i < teeth_period {
                teeth_sum += source_array[i];
            }
            if i < lips_period {
                lips_sum += source_array[i];
            }
        }
        
        // Initialize SMMA values (without shifts)
        let mut jaw_smma = jaw_sum / jaw_period as f64;
        let mut teeth_smma = teeth_sum / teeth_period as f64;
        let mut lips_smma = lips_sum / lips_period as f64;
        
        // Calculate all SMMA values first, then apply shifts
        let mut jaw_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Set initial SMMA values
        jaw_unshifted[jaw_period - 1] = jaw_smma;
        teeth_unshifted[teeth_period - 1] = teeth_smma;
        lips_unshifted[lips_period - 1] = lips_smma;
        
        // Calculate subsequent SMMA values using single pass
        for i in jaw_period..n {
            // Update jaw (13-period SMMA)
            jaw_smma = jaw_alpha * source_array[i] + (1.0 - jaw_alpha) * jaw_smma;
            jaw_unshifted[i] = jaw_smma;
            
            // Update teeth (8-period SMMA) if we have enough data
            if i >= teeth_period {
                teeth_smma = teeth_alpha * source_array[i] + (1.0 - teeth_alpha) * teeth_smma;
                teeth_unshifted[i] = teeth_smma;
            }
            
            // Update lips (5-period SMMA) if we have enough data
            if i >= lips_period {
                lips_smma = lips_alpha * source_array[i] + (1.0 - lips_alpha) * lips_smma;
                lips_unshifted[i] = lips_smma;
            }
        }
        
        // Apply shifts inline (forward shifts)
        // Jaw: shift by 8 periods forward
        for i in 0..(n - 8) {
            if i + 8 < n && !jaw_unshifted[i].is_nan() {
                jaw[i + 8] = jaw_unshifted[i];
            }
        }
        
        // Teeth: shift by 5 periods forward
        for i in 0..(n - 5) {
            if i + 5 < n && !teeth_unshifted[i].is_nan() {
                teeth[i + 5] = teeth_unshifted[i];
            }
        }
        
        // Lips: shift by 3 periods forward
        for i in 0..(n - 3) {
            if i + 3 < n && !lips_unshifted[i].is_nan() {
                lips[i + 3] = lips_unshifted[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &jaw).to_owned(),
            PyArray1::from_array(py, &teeth).to_owned(),
            PyArray1::from_array(py, &lips).to_owned()
        ))
    })
} 

/// Calculate DI (Directional Indicator) - Optimized version
#[pyfunction]
pub fn di(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result arrays
        let mut plus_di = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_di = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 2 || n < period + 1 {
            return Ok((
                PyArray1::from_array(py, &plus_di).to_owned(),
                PyArray1::from_array(py, &minus_di).to_owned()
            ));
        }
        
        // Extract OHLCV data (assuming standard format: open, high, low, close, volume)
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        
        // Initialize smoothed values for Wilder's smoothing
        let mut tr_smooth = 0.0;
        let mut plus_dm_smooth = 0.0;
        let mut minus_dm_smooth = 0.0;
        
        // Calculate initial sums for the first 'period' values
        for i in 1..=period {
            // Calculate True Range (TR)
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            // Calculate Directional Movements (+DM and -DM)
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Accumulate for initial smoothed values
            tr_smooth += current_tr;
            plus_dm_smooth += current_plus_dm;
            minus_dm_smooth += current_minus_dm;
        }
        
        // Calculate first DI values at index 'period'
        if tr_smooth > 0.0 {
            plus_di[period] = 100.0 * plus_dm_smooth / tr_smooth;
            minus_di[period] = 100.0 * minus_dm_smooth / tr_smooth;
        } else {
            plus_di[period] = 0.0;
            minus_di[period] = 0.0;
        }
        
        // Calculate subsequent DI values using Wilder's smoothing
        for i in (period + 1)..n {
            // Calculate current TR, +DM, -DM
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Apply Wilder's smoothing: smoothed[i] = (smoothed[i-1] * (period - 1) + current) / period
            tr_smooth = (tr_smooth * (period - 1) as f64 + current_tr) / period as f64;
            plus_dm_smooth = (plus_dm_smooth * (period - 1) as f64 + current_plus_dm) / period as f64;
            minus_dm_smooth = (minus_dm_smooth * (period - 1) as f64 + current_minus_dm) / period as f64;
            
            // Calculate DI values
            if tr_smooth > 0.0 {
                plus_di[i] = 100.0 * plus_dm_smooth / tr_smooth;
                minus_di[i] = 100.0 * minus_dm_smooth / tr_smooth;
            } else {
                plus_di[i] = 0.0;
                minus_di[i] = 0.0;
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_di).to_owned(),
            PyArray1::from_array(py, &minus_di).to_owned()
        ))
    })
}

/// Calculate CHOP (Choppiness Index) - Ultra-optimized version
#[pyfunction]
pub fn chop(candles: PyReadonlyArray2<f64>, period: usize, scalar: f64, drift: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate True Range (TR) array
        let mut tr = Array1::<f64>::zeros(n);
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
                          let hc = (high[i] - close[i - 1]).abs();
              let lc = (low[i] - close[i - 1]).abs();
              tr[i] = hl.max(hc).max(lc);
        }
        
        // Calculate ATR using Wilder's smoothing or simple average based on drift
        let mut atr = Array1::<f64>::from_elem(n, f64::NAN);
        
        if drift == 1 {
            // Simple case: ATR = TR
            atr.assign(&tr);
        } else {
            // Wilder's smoothing for ATR
            // Calculate initial ATR as simple average of first 'drift' values
            let mut sum = 0.0;
            for i in 0..drift {
                sum += tr[i];
            }
            atr[drift - 1] = sum / drift as f64;
            
            // Apply Wilder's smoothing for subsequent values
            let alpha = 1.0 / drift as f64;
            for i in drift..n {
                atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1];
            }
        }
        
        // Pre-calculate log10(period) for efficiency
        let log_period = (period as f64).log10();
        
        // Use rolling sum algorithm for ATR sum (O(n) instead of O(n*p))
        let mut atr_sum = 0.0;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        
        // Initialize the first window
        for i in 0..period {
            if !atr[i].is_nan() {
                atr_sum += atr[i];
            }
            if high[i] > highest {
                highest = high[i];
            }
            if low[i] < lowest {
                lowest = low[i];
            }
        }
        
        // Use deque-like structures for efficient rolling max/min
        use std::collections::VecDeque;
        let mut max_deque: VecDeque<(usize, f64)> = VecDeque::new();
        let mut min_deque: VecDeque<(usize, f64)> = VecDeque::new();
        
        // Initialize deques for the first window
        for i in 0..period {
            // Maintain max deque (decreasing order)
            while !max_deque.is_empty() && max_deque.back().unwrap().1 <= high[i] {
                max_deque.pop_back();
            }
            max_deque.push_back((i, high[i]));
            
            // Maintain min deque (increasing order)
            while !min_deque.is_empty() && min_deque.back().unwrap().1 >= low[i] {
                min_deque.pop_back();
            }
            min_deque.push_back((i, low[i]));
        }
        
        // Calculate first CHOP value
        if atr_sum > 0.0 {
            let range = highest - lowest;
            if range > 0.0 {
                result[period - 1] = (scalar * (atr_sum.log10() - range.log10())) / log_period;
            }
        }
        
        // Rolling window calculation for subsequent values
        for i in period..n {
            // Update rolling ATR sum
            if !atr[i].is_nan() {
                atr_sum += atr[i];
            }
            if !atr[i - period].is_nan() {
                atr_sum -= atr[i - period];
            }
            
            // Remove elements outside the window from deques
            while !max_deque.is_empty() && max_deque.front().unwrap().0 <= i - period {
                max_deque.pop_front();
            }
            while !min_deque.is_empty() && min_deque.front().unwrap().0 <= i - period {
                min_deque.pop_front();
            }
            
            // Add new element to deques
            while !max_deque.is_empty() && max_deque.back().unwrap().1 <= high[i] {
                max_deque.pop_back();
            }
            max_deque.push_back((i, high[i]));
            
            while !min_deque.is_empty() && min_deque.back().unwrap().1 >= low[i] {
                min_deque.pop_back();
            }
            min_deque.push_back((i, low[i]));
            
            // Get current max and min
            let current_highest = max_deque.front().unwrap().1;
            let current_lowest = min_deque.front().unwrap().1;
            
            // Calculate CHOP
            if atr_sum > 0.0 {
                let range = current_highest - current_lowest;
                if range > 0.0 {
                    result[i] = (scalar * (atr_sum.log10() - range.log10())) / log_period;
                }
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate ATR (Average True Range) - Ultra-optimized version
#[pyfunction]
pub fn atr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate True Range (TR) inline and ATR simultaneously
        let mut tr_sum = 0.0;
        
        // Calculate first TR value
        let first_tr = high[0] - low[0];
        tr_sum += first_tr;
        
        // Calculate subsequent TR values and accumulate for first period
        for i in 1..period {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        
        // First ATR value is the simple average of the first 'period' true ranges
        result[period - 1] = tr_sum / period as f64;
        
        // Calculate subsequent ATR values using Wilder's smoothing
        // Using the optimized formula: ATR[i] = (ATR[i-1] * (period - 1) + TR[i]) / period
        // Which can be rewritten as: ATR[i] = ATR[i-1] + (TR[i] - ATR[i-1]) / period
        let alpha = 1.0 / period as f64;
        
        for i in period..n {
            // Calculate current True Range
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            
            // Apply Wilder's smoothing: ATR[i] = ATR[i-1] + alpha * (TR[i] - ATR[i-1])
            result[i] = result[i - 1] + alpha * (tr - result[i - 1]);
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Chande (Chandelier Exit) - Ultra-optimized version
#[pyfunction]
pub fn chande(
    candles: PyReadonlyArray2<f64>, 
    period: usize, 
    mult: f64, 
    direction: &str
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result array
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract OHLCV data
        let close = candles_array.column(2);
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Pre-allocate ATR array
        let mut atr = vec![f64::NAN; n];
        
        // Calculate ATR using rolling window - first pass
        let mut tr_sum = 0.0;
        
        // Calculate first TR value
        let first_tr = high[0] - low[0];
        tr_sum += first_tr;
        
        // Calculate subsequent TR values and accumulate for first period
        for i in 1..period {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        
        // First ATR value
        atr[period - 1] = tr_sum / period as f64;
        
        // Calculate subsequent ATR values using Wilder's smoothing
        let alpha = 1.0 / period as f64;
        for i in period..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let tr = hl.max(hc).max(lc);
            atr[i] = atr[i - 1] + alpha * (tr - atr[i - 1]);
        }
        
        // Calculate Chandelier Exit using rolling max/min with deques
        use std::collections::VecDeque;
        
        if direction == "long" {
            // For long: use rolling maximum of high prices
            let mut max_deque: VecDeque<(usize, f64)> = VecDeque::new();
            
            // Initialize the first window
            for i in 0..period.min(n) {
                // Remove elements that are smaller than current (maintain decreasing order)
                while let Some(&(_, val)) = max_deque.back() {
                    if val <= high[i] {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((i, high[i]));
            }
            
            // Calculate chandelier exit for the first valid position
            if period <= n {
                let max_high = max_deque.front().unwrap().1;
                result[period - 1] = max_high - atr[period - 1] * mult;
            }
            
            // Slide the window for remaining positions
            for i in period..n {
                // Remove elements outside the window
                while let Some(&(idx, _)) = max_deque.front() {
                    if idx <= i - period {
                        max_deque.pop_front();
                    } else {
                        break;
                    }
                }
                
                // Add new element maintaining decreasing order
                while let Some(&(_, val)) = max_deque.back() {
                    if val <= high[i] {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((i, high[i]));
                
                // Calculate chandelier exit
                let max_high = max_deque.front().unwrap().1;
                result[i] = max_high - atr[i] * mult;
            }
        } else if direction == "short" {
            // For short: use rolling minimum of low prices
            let mut min_deque: VecDeque<(usize, f64)> = VecDeque::new();
            
            // Initialize the first window
            for i in 0..period.min(n) {
                // Remove elements that are larger than current (maintain increasing order)
                while let Some(&(_, val)) = min_deque.back() {
                    if val >= low[i] {
                        min_deque.pop_back();
                    } else {
                        break;
                    }
                }
                min_deque.push_back((i, low[i]));
            }
            
            // Calculate chandelier exit for the first valid position
            if period <= n {
                let min_low = min_deque.front().unwrap().1;
                result[period - 1] = min_low + atr[period - 1] * mult;
            }
            
            // Slide the window for remaining positions
            for i in period..n {
                // Remove elements outside the window
                while let Some(&(idx, _)) = min_deque.front() {
                    if idx <= i - period {
                        min_deque.pop_front();
                    } else {
                        break;
                    }
                }
                
                // Add new element maintaining increasing order
                while let Some(&(_, val)) = min_deque.back() {
                    if val >= low[i] {
                        min_deque.pop_back();
                    } else {
                        break;
                    }
                }
                min_deque.push_back((i, low[i]));
                
                // Calculate chandelier exit
                let min_low = min_deque.front().unwrap().1;
                result[i] = min_low + atr[i] * mult;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate Donchian Channels - Ultra-optimized version
#[pyfunction]
pub fn donchian(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result arrays
        let mut upperband = Array1::<f64>::from_elem(n, f64::NAN);
        let mut middleband = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lowerband = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < period {
            let result = PyDict::new(py);
            result.set_item("upperband", PyArray1::from_array(py, &upperband).to_owned())?;
            result.set_item("middleband", PyArray1::from_array(py, &middleband).to_owned())?;
            result.set_item("lowerband", PyArray1::from_array(py, &lowerband).to_owned())?;
            return Ok(result.into());
        }
        
        // Extract high and low data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Use VecDeque for O(1) front/back operations
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
            upperband[period - 1] = max_high;
            lowerband[period - 1] = min_low;
            middleband[period - 1] = (max_high + min_low) * 0.5;
        }
        
        // Process remaining elements with optimized sliding window
        for i in period..n {
            let high_val = high[i];
            let low_val = low[i];
            
            // Remove expired elements from front of max deque (O(1) amortized)
            while let Some(&(idx, _)) = max_deque.front() {
                if idx <= i - period {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Remove expired elements from front of min deque (O(1) amortized)
            while let Some(&(idx, _)) = min_deque.front() {
                if idx <= i - period {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }
            
            // Add new element to max deque (maintain decreasing order)
            while let Some(&(_, val)) = max_deque.back() {
                if val <= high_val {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((i, high_val));
            
            // Add new element to min deque (maintain increasing order)
            while let Some(&(_, val)) = min_deque.back() {
                if val >= low_val {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((i, low_val));
            
            // Calculate results
            let max_high = max_deque.front().unwrap().1;
            let min_low = min_deque.front().unwrap().1;
            upperband[i] = max_high;
            lowerband[i] = min_low;
            middleband[i] = (max_high + min_low) * 0.5;
        }
        
        // Return as dictionary
        let result = PyDict::new(py);
        result.set_item("upperband", PyArray1::from_array(py, &upperband).to_owned())?;
        result.set_item("middleband", PyArray1::from_array(py, &middleband).to_owned())?;
        result.set_item("lowerband", PyArray1::from_array(py, &lowerband).to_owned())?;
        Ok(result.into())
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

/// Calculate Stochastic Oscillator - Ultra-optimized version
#[pyfunction]
pub fn stoch(candles: PyReadonlyArray2<f64>, fastk_period: usize, slowk_period: usize, _slowk_matype: usize, slowd_period: usize, _slowd_matype: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
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
pub fn stochf(candles: PyReadonlyArray2<f64>, fastk_period: usize, fastd_period: usize, fastd_matype: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
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

/// Calculate Directional Movement (DM) - Ultra-optimized version
#[pyfunction]
pub fn dm(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut plus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &plus_dm).to_owned(),
                PyArray1::from_array(py, &minus_dm).to_owned()
            ));
        }
        
        // Extract price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate raw directional movements
        let mut raw_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut raw_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];
            
            if up_move > down_move && up_move > 0.0 {
                raw_plus[i] = up_move;
            } else {
                raw_plus[i] = 0.0;
            }
            
            if down_move > up_move && down_move > 0.0 {
                raw_minus[i] = down_move;
            } else {
                raw_minus[i] = 0.0;
            }
        }
        
        // Apply Wilder's smoothing
        if n > period {
            // Calculate initial sum
            let mut sum_plus = 0.0;
            let mut sum_minus = 0.0;
            
            for i in 1..=period {
                sum_plus += raw_plus[i];
                sum_minus += raw_minus[i];
            }
            
            plus_dm[period] = sum_plus;
            minus_dm[period] = sum_minus;
            
            // Apply Wilder's smoothing formula
            for i in (period + 1)..n {
                plus_dm[i] = plus_dm[i - 1] - (plus_dm[i - 1] / period as f64) + raw_plus[i];
                minus_dm[i] = minus_dm[i - 1] - (minus_dm[i - 1] / period as f64) + raw_minus[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_dm).to_owned(),
            PyArray1::from_array(py, &minus_dm).to_owned()
        ))
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

/// Helper function to calculate SMA on an array
fn sma_array(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::from_elem(n, f64::NAN);
    
    if n < period {
        return result;
    }
    
    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        let end_idx = i + 1;
        
        // Calculate SMA for window, handling NaN values
        let window = source.slice(s![start_idx..end_idx]);
        let mut sum = 0.0;
        let mut count = 0;
        
        for &val in window.iter() {
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }
    
    result
}

/// Calculate ADOSC (Chaikin A/D Oscillator) - Ultra-optimized version
#[pyfunction]
pub fn adosc(candles: PyReadonlyArray2<f64>, fast_period: usize, slow_period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = 0.0;
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract price and volume data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        let volume = candles_array.column(5);
        
        // Step 1: Calculate Money Flow Multiplier and Money Flow Volume
        let mut mf_volume = Array1::<f64>::zeros(n);
        for i in 0..n {
            let price_range = high[i] - low[i];
            let multiplier = if price_range != 0.0 {
                ((close[i] - low[i]) - (high[i] - close[i])) / price_range
            } else {
                0.0
            };
            mf_volume[i] = multiplier * volume[i];
        }
        
        // Step 2: Calculate A/D Line (cumulative sum of money flow volume)
        let mut ad_line = Array1::<f64>::zeros(n);
        ad_line[0] = mf_volume[0];
        for i in 1..n {
            ad_line[i] = ad_line[i - 1] + mf_volume[i];
        }
        
        // Step 3: Calculate EMAs of A/D Line
        let fast_alpha = 2.0 / (fast_period as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_period as f64 + 1.0);
        let fast_one_minus_alpha = 1.0 - fast_alpha;
        let slow_one_minus_alpha = 1.0 - slow_alpha;
        
        // Calculate fast EMA
        let mut fast_ema = Array1::<f64>::zeros(n);
        fast_ema[0] = ad_line[0];
        for i in 1..n {
            fast_ema[i] = fast_alpha * ad_line[i] + fast_one_minus_alpha * fast_ema[i - 1];
        }
        
        // Calculate slow EMA
        let mut slow_ema = Array1::<f64>::zeros(n);
        slow_ema[0] = ad_line[0];
        for i in 1..n {
            slow_ema[i] = slow_alpha * ad_line[i] + slow_one_minus_alpha * slow_ema[i - 1];
        }
        
        // Step 4: Calculate ADOSC = Fast EMA - Slow EMA
        for i in 0..n {
            result[i] = fast_ema[i] - slow_ema[i];
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

/// Calculate CVI (Chaikins Volatility Indicator) - Ultra-optimized version
#[pyfunction]
pub fn cvi(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract high and low price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate high-low difference
        let mut hl_diff = Array1::<f64>::zeros(n);
        for i in 0..n {
            hl_diff[i] = high[i] - low[i];
        }
        
        // Calculate EMA of the high-low difference
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        let mut ema_diff = Array1::<f64>::zeros(n);
        ema_diff[0] = hl_diff[0];
        
        for i in 1..n {
            ema_diff[i] = alpha * hl_diff[i] + one_minus_alpha * ema_diff[i - 1];
        }
        
        // Calculate rate of change
        for i in period..n {
            if ema_diff[i - period] != 0.0 {
                result[i] = ((ema_diff[i] - ema_diff[i - period]) / ema_diff[i - period]) * 100.0;
            } else {
                result[i] = 0.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
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

/// Calculate EMA for internal use in Wavetrend indicator
fn ema_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    // Not enough data
    if n == 0 {
        return result;
    }
    
    // Calculate alpha
    let alpha = 2.0 / (period as f64 + 1.0);
    
    // First value is the source
    result[0] = source[0];
    
    // Calculate EMA for the rest
    for i in 1..n {
        result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1];
    }
    
    result
}

/// Calculate SMA for internal use in Wavetrend indicator
fn sma_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    if n == 0 || period == 0 {
        return result;
    }
    
    // Calculate initial values with partial windows
    let mut cumsum = 0.0;
    for i in 0..period.min(n) {
        cumsum += source[i];
        result[i] = cumsum / (i as f64 + 1.0);
    }
    
    if n <= period {
        return result;
    }
    
    // For the remaining windows, use a rolling approach
    for i in period..n {
        result[i] = result[i-1] + (source[i] - source[i-period]) / period as f64;
    }
    
    result
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

/// Calculate RMA (Relative Moving Average) for internal use
fn rma_array(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    
    if n == 0 || period == 0 {
        return result;
    }
    
    let alpha = 1.0 / period as f64;
    result[0] = source[0];
    
    for i in 1..n {
        result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1];
    }
    
    result
}

/// Calculate DX (Directional Movement Index) - Ultra-optimized version
#[pyfunction]
pub fn dx(
    candles: PyReadonlyArray2<f64>,
    di_length: usize,
    adx_smoothing: usize,
    sequential: bool
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        
        if n < 2 || di_length == 0 || adx_smoothing == 0 {
            let empty = Array1::<f64>::from_elem(if sequential { n } else { 1 }, f64::NAN);
            return Ok((
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
            ));
        }
        
        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);
        
        // Ultra-fast path for non-sequential mode
        if !sequential {
            let required_len = di_length + adx_smoothing;
            if n < required_len {
                let result = Array1::<f64>::from_elem(1, f64::NAN);
                return Ok((
                    PyArray1::from_array(py, &result).to_owned(),
                    PyArray1::from_array(py, &result).to_owned(),
                    PyArray1::from_array(py, &result).to_owned(),
                ));
            }
            
            // Wilder's smoothing factor
            let alpha = 1.0 / di_length as f64;
            let one_minus_alpha = 1.0 - alpha;
            
            // Initialize accumulators with first period sum
            let mut tr_sum = 0.0;
            let mut plus_dm_sum = 0.0;
            let mut minus_dm_sum = 0.0;
            
            // First TR value
            tr_sum = high[0usize] - low[0usize];
            
            // Accumulate initial sums
            for i in 1..di_length {
                let up = high[i] - high[i - 1];
                let down = low[i - 1] - low[i];
                
                if up > down && up > 0.0 {
                    plus_dm_sum += up;
                }
                if down > up && down > 0.0 {
                    minus_dm_sum += down;
                }
                
                let tr1 = high[i] - low[i];
                let tr2 = (high[i] - close[i - 1]).abs();
                let tr3 = (low[i] - close[i - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            
            // Initial smoothed values
            let mut tr_smooth = tr_sum;
            let mut plus_dm_smooth = plus_dm_sum;
            let mut minus_dm_smooth = minus_dm_sum;
            
            // Buffer for DX values to calculate ADX
            let mut dx_values = Vec::with_capacity(adx_smoothing);
            
            // Calculate DX values for ADX smoothing period
            for i in di_length..(di_length + adx_smoothing) {
                let up = high[i] - high[i - 1];
                let down = low[i - 1] - low[i];
                
                let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
                let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };
                
                let tr1 = high[i] - low[i];
                let tr2 = (high[i] - close[i - 1]).abs();
                let tr3 = (low[i] - close[i - 1]).abs();
                let tr = tr1.max(tr2).max(tr3);
                
                // Wilder's smoothing
                tr_smooth = one_minus_alpha * tr_smooth + alpha * tr;
                plus_dm_smooth = one_minus_alpha * plus_dm_smooth + alpha * plus_dm;
                minus_dm_smooth = one_minus_alpha * minus_dm_smooth + alpha * minus_dm;
                
                // Calculate DX
                if tr_smooth > 0.0 {
                    let plus_di = plus_dm_smooth / tr_smooth;
                    let minus_di = minus_dm_smooth / tr_smooth;
                    let di_sum = plus_di + minus_di;
                    
                    if di_sum > 0.0 {
                        dx_values.push(100.0 * (plus_di - minus_di).abs() / di_sum);
                    } else {
                        dx_values.push(0.0);
                    }
                } else {
                    dx_values.push(0.0);
                }
            }
            
            // Initial ADX as average
            let mut adx = dx_values.iter().sum::<f64>() / adx_smoothing as f64;
            
            // Continue for remaining periods if any
            for i in (di_length + adx_smoothing)..n {
                let up = high[i] - high[i - 1];
                let down = low[i - 1] - low[i];
                
                let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
                let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };
                
                let tr1 = high[i] - low[i];
                let tr2 = (high[i] - close[i - 1]).abs();
                let tr3 = (low[i] - close[i - 1]).abs();
                let tr = tr1.max(tr2).max(tr3);
                
                // Wilder's smoothing
                tr_smooth = one_minus_alpha * tr_smooth + alpha * tr;
                plus_dm_smooth = one_minus_alpha * plus_dm_smooth + alpha * plus_dm;
                minus_dm_smooth = one_minus_alpha * minus_dm_smooth + alpha * minus_dm;
                
                // Calculate current DX
                let dx_val = if tr_smooth > 0.0 {
                    let plus_di = plus_dm_smooth / tr_smooth;
                    let minus_di = minus_dm_smooth / tr_smooth;
                    let di_sum = plus_di + minus_di;
                    if di_sum > 0.0 { 100.0 * (plus_di - minus_di).abs() / di_sum } else { 0.0 }
                } else { 0.0 };
                
                // Smooth ADX
                adx = (one_minus_alpha * adx * adx_smoothing as f64 + alpha * dx_val * adx_smoothing as f64) / adx_smoothing as f64;
            }
            
            // Final DI values
            let plus_di = if tr_smooth > 0.0 { 100.0 * plus_dm_smooth / tr_smooth } else { 0.0 };
            let minus_di = if tr_smooth > 0.0 { 100.0 * minus_dm_smooth / tr_smooth } else { 0.0 };
            
            let adx_result = Array1::from_elem(1, adx);
            let plus_di_result = Array1::from_elem(1, plus_di);
            let minus_di_result = Array1::from_elem(1, minus_di);
            
            return Ok((
                PyArray1::from_array(py, &adx_result).to_owned(),
                PyArray1::from_array(py, &plus_di_result).to_owned(),
                PyArray1::from_array(py, &minus_di_result).to_owned(),
            ));
        }
        
        // Sequential mode - optimized full array calculation
        let mut adx_result = Array1::<f64>::from_elem(n, f64::NAN);
        let mut plus_di_result = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_di_result = Array1::<f64>::from_elem(n, f64::NAN);
        
        let alpha = 1.0 / di_length as f64;
        let one_minus_alpha = 1.0 - alpha;
        
        let mut tr_smooth = 0.0;
        let mut plus_dm_smooth = 0.0;
        let mut minus_dm_smooth = 0.0;
        
        // Initialize with first period sum
        tr_smooth = high[0usize] - low[0usize];
        
        for i in 1..n.min(di_length) {
            let up = high[i] - high[i - 1];
            let down = low[i - 1] - low[i];
            
            if up > down && up > 0.0 {
                plus_dm_smooth += up;
            }
            if down > up && down > 0.0 {
                minus_dm_smooth += down;
            }
            
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();
            tr_smooth += tr1.max(tr2).max(tr3);
        }
        
        // Calculate for remaining periods
        let mut dx_buffer = Vec::new();
        
        for i in di_length..n {
            let up = high[i] - high[i - 1];
            let down = low[i - 1] - low[i];
            
            let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
            let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };
            
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();
            let tr = tr1.max(tr2).max(tr3);
            
            // Wilder's smoothing
            tr_smooth = one_minus_alpha * tr_smooth + alpha * tr;
            plus_dm_smooth = one_minus_alpha * plus_dm_smooth + alpha * plus_dm;
            minus_dm_smooth = one_minus_alpha * minus_dm_smooth + alpha * minus_dm;
            
            if tr_smooth > 0.0 {
                plus_di_result[i] = 100.0 * plus_dm_smooth / tr_smooth;
                minus_di_result[i] = 100.0 * minus_dm_smooth / tr_smooth;
                
                let di_sum = plus_di_result[i] + minus_di_result[i];
                let dx_val = if di_sum > 0.0 {
                    100.0 * (plus_di_result[i] - minus_di_result[i]).abs() / di_sum
                } else { 0.0 };
                
                dx_buffer.push(dx_val);
                
                // Calculate ADX when we have enough DX values
                if dx_buffer.len() == adx_smoothing {
                    adx_result[i] = dx_buffer.iter().sum::<f64>() / adx_smoothing as f64;
                    dx_buffer.remove(0);
                } else if dx_buffer.len() > adx_smoothing {
                    let prev_adx = adx_result[i - 1];
                    adx_result[i] = (prev_adx * (adx_smoothing as f64 - 1.0) + dx_val) / adx_smoothing as f64;
                }
            }
        }
        
        Ok((
            PyArray1::from_array(py, &adx_result).to_owned(),
            PyArray1::from_array(py, &plus_di_result).to_owned(),
            PyArray1::from_array(py, &minus_di_result).to_owned(),
        ))
    })
}

/// Calculate FOSC (Forecast Oscillator)
#[pyfunction]
pub fn fosc(
    candles: PyReadonlyArray2<f64>, 
    period: usize, 
    source_type: &str
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = Array1::<f64>::from_elem(n, 0.0);
        
        if n < period || period == 0 {
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

/// Helper function for linear regression calculation for FOSC
fn linear_regression_line(x: &Array1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len() as f64;
    let sum_x: f64 = x.sum();
    let sum_y: f64 = y.sum();
    
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    
    for i in 0..x.len() {
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    let mut result = Array1::<f64>::zeros(x.len());
    for i in 0..x.len() {
        result[i] = slope * x[i] + intercept;
    }
    
    result
}

/// Calculate FRAMA (Fractal Adaptive Moving Average)
#[pyfunction]
pub fn frama(
    candles: PyReadonlyArray2<f64>,
    window: usize,
    fc: usize,
    sc: usize
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Adjust window to be even if it's not
        let mut n_local = window;
        if n_local % 2 == 1 {
            n_local += 1;
        }
        
        if n < n_local {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract close prices
        let close = candles_array.slice(s![.., 2]).to_owned();
        
        // Calculate w (log of 2/(SC+1))
        let w = (2.0 / (sc as f64 + 1.0)).ln();
        
        // Initialize arrays
        let mut d = Array1::<f64>::from_elem(n, f64::NAN);
        let mut alphas = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Calculate dimension and alphas
        for i in n_local..n {
            let period_slice = candles_array.slice(s![i-n_local..i, ..]);
            
            // Extract period high and low
            let period_high = period_slice.slice(s![.., 3]);
            let period_low = period_slice.slice(s![.., 4]);
            
            // Split into two halves
            let half = n_local / 2;
            
            // First half
            let v1_high = period_high.slice(s![half..]);
            let v1_low = period_low.slice(s![half..]);
            let v1_high_max = v1_high.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let v1_low_min = v1_low.fold(f64::INFINITY, |a, &b| a.min(b));
            let n1 = (v1_high_max - v1_low_min) / (half as f64);
            
            // Second half
            let v2_high = period_high.slice(s![..half]);
            let v2_low = period_low.slice(s![..half]);
            let v2_high_max = v2_high.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let v2_low_min = v2_low.fold(f64::INFINITY, |a, &b| a.min(b));
            let n2 = (v2_high_max - v2_low_min) / (half as f64);
            
            // Entire period
            let period_high_max = period_high.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low_min = period_low.fold(f64::INFINITY, |a, &b| a.min(b));
            let n3 = (period_high_max - period_low_min) / (n_local as f64);
            
            // Calculate dimension
            if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                d[i] = ((n1 + n2).ln() - n3.ln()) / 2.0f64.ln();
            } else {
                // Use previous value if calculation is not possible
                if i > n_local {
                    d[i] = d[i - 1];
                } else {
                    d[i] = 0.0;
                }
            }
            
            // Calculate alpha
            let mut old_alpha = (w * (d[i] - 1.0)).exp();
            old_alpha = old_alpha.max(0.1);
            old_alpha = old_alpha.min(1.0);
            
            let old_n = (2.0 - old_alpha) / old_alpha;
            let n_val = ((sc as f64 - fc as f64) * ((old_n - 1.0) / (sc as f64 - 1.0))) + fc as f64;
            let mut alpha = 2.0 / (n_val + 1.0);
            
            // Ensure alpha is within bounds
            if alpha < 2.0 / (sc as f64 + 1.0) {
                alpha = 2.0 / (sc as f64 + 1.0);
            } else if alpha > 1.0 {
                alpha = 1.0;
            }
            
            alphas[i] = alpha;
        }
        
        // Calculate FRAMA
        result[n_local - 1] = close.slice(s![..n_local]).mean().unwrap_or(0.0);
        
        for i in n_local..n {
            result[i] = (alphas[i] * close[i]) + ((1.0 - alphas[i]) * result[i - 1]);
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Get period identifier from timestamp based on anchor
fn get_period_from_timestamp(timestamp: f64, anchor: &str) -> i64 {
    // Convert timestamp from milliseconds to seconds
    let seconds = (timestamp / 1000.0) as i64;
    
    match anchor.to_uppercase().as_str() {
        "D" => seconds / 86400,        // Daily: 24 * 60 * 60 seconds
        "H" => seconds / 3600,         // Hourly: 60 * 60 seconds  
        "M" => seconds / 60,           // Minute: 60 seconds
        "4H" => seconds / 14400,       // 4 Hour: 4 * 60 * 60 seconds
        "12H" => seconds / 43200,      // 12 Hour: 12 * 60 * 60 seconds
        "W" => seconds / 604800,       // Weekly: 7 * 24 * 60 * 60 seconds
        "MN" => {
            // Monthly - approximate as 30 days for simplicity
            seconds / 2592000  // 30 * 24 * 60 * 60 seconds
        },
        _ => seconds / 86400,          // Default to daily
    }
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

/// Calculate VI (Vortex Indicator)
#[pyfunction] 
pub fn vi(
    candles: PyReadonlyArray2<f64>,
    period: usize,
    sequential: bool
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut vi_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut vi_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ));
        }
        
        let close = candles_array.slice(s![.., 2]).to_owned();
        let high = candles_array.slice(s![.., 3]).to_owned();
        let low = candles_array.slice(s![.., 4]).to_owned();
        
        // Calculate True Range, VP and VM for each period
        let mut tr = Array1::<f64>::zeros(n);
        let mut vp = Array1::<f64>::zeros(n);
        let mut vm = Array1::<f64>::zeros(n);
        
        // First candle
        if n > 0 {
            tr[0usize] = high[0usize] - low[0usize];
        }
        
        // Calculate TR, VP, VM for each candle
        for i in 1..n {
            let hl = high[i] - low[i];
            let hpc = (high[i] - close[i - 1]).abs();
            let lpc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hpc).max(lpc);
            
            vp[i] = (high[i] - low[i - 1]).abs();
            vm[i] = (low[i] - high[i - 1]).abs();
        }
        
        // Calculate rolling sums and VI values
        for i in period..n {
            let start_idx = i + 1 - period;
            
            let sum_tr: f64 = tr.slice(s![start_idx..=i]).sum();
            let sum_vp: f64 = vp.slice(s![start_idx..=i]).sum();
            let sum_vm: f64 = vm.slice(s![start_idx..=i]).sum();
            
            if sum_tr != 0.0 {
                vi_plus[i] = sum_vp / sum_tr;
                vi_minus[i] = sum_vm / sum_tr;
            }
        }
        
        if sequential {
            Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ))
        } else {
            let plus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_plus[n-1] } else { f64::NAN });
            let minus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_minus[n-1] } else { f64::NAN });
            Ok((
                PyArray1::from_array(py, &plus_result).to_owned(),
                PyArray1::from_array(py, &minus_result).to_owned()
            ))
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

/// Sums two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn sum_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise addition
    let result_decimal = decimal1 + decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}

/// Subtracts two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn subtract_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise subtraction
    let result_decimal = decimal1 - decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}