//! Bands, channels, and range-based indicators (Bollinger, Donchian, Keltner, Chop, Chande).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::types::{PyArrTuple3};

use crate::helpers::{ih_atr_wilder};

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
pub fn bollinger_bands(source: PyReadonlyArray1<f64>, period: usize, devup: f64, devdn: f64) -> PyArrTuple3 {
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


/// Last values of Bollinger Bands — identical rolling-sum recurrence to
/// `bollinger_bands`, but skips allocating the full output series.
/// Bit-for-bit equal to the last elements of `bollinger_bands(...)`.
#[pyfunction]
pub fn bollinger_bands_last(source: PyReadonlyArray1<f64>, period: usize, devup: f64, devdn: f64) -> PyResult<(f64, f64, f64)> {
    let source_array = source.as_array();
    let n = source_array.len();

    if n == 0 {
        return Err(PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    }

    if n < period {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..period {
        let val = source_array[i];
        sum += val;
        sum_sq += val * val;
    }

    for i in period..n {
        let old_val = source_array[i - period];
        let new_val = source_array[i];

        sum = sum - old_val + new_val;
        sum_sq = sum_sq - (old_val * old_val) + (new_val * new_val);
    }

    let sma = sum / period as f64;
    let variance = (sum_sq / period as f64) - (sma * sma);
    let std_dev = variance.sqrt();

    Ok((sma + devup * std_dev, sma, sma - devdn * std_dev))
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

/// Last value of ATR — identical Wilder recurrence to `atr`, but skips
/// allocating the full output series. Bit-for-bit equal to `atr(...)[-1]`.
#[pyfunction]
pub fn atr_last(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<f64> {
    let candles_array = candles.as_array();
    let n = candles_array.nrows();

    if n == 0 {
        return Err(PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ));
    }

    if n < period {
        return Ok(f64::NAN);
    }

    let close = candles_array.column(2);
    let high = candles_array.column(3);
    let low = candles_array.column(4);

    let mut tr_sum = 0.0;

    let first_tr = high[0] - low[0];
    tr_sum += first_tr;

    for i in 1..period {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr = hl.max(hc).max(lc);
        tr_sum += tr;
    }

    let mut prev = tr_sum / period as f64;

    let alpha = 1.0 / period as f64;

    for i in period..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr = hl.max(hc).max(lc);

        prev = prev + alpha * (tr - prev);
    }

    Ok(prev)
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

/// Keltner Channel inner (takes pre-computed MA) → (upper, middle, lower)
#[pyfunction]
pub fn keltner_inner(
    ma_values: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
    multiplier: f64,
) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let ma = ma_values.as_array();
        let h = high.as_array();
        let l = low.as_array();
        let cl = close.as_array();
        let n = ma.len();
        let h_sl = h.as_slice().unwrap();
        let l_sl = l.as_slice().unwrap();
        let cl_sl = cl.as_slice().unwrap();
        let atr_vals = ih_atr_wilder(h_sl, l_sl, cl_sl, period);
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        for i in 0..n {
            if !atr_vals[i].is_nan() {
                upper[i] = ma[i] + atr_vals[i] * multiplier;
                lower[i] = ma[i] - atr_vals[i] * multiplier;
            }
        }
        Ok((
            PyArray1::from_vec(py, upper).to_owned(),
            PyArray1::from_vec(py, ma.to_vec()).to_owned(),
            PyArray1::from_vec(py, lower).to_owned(),
        ))
    })
}
