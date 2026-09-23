//! Match candle timestamps against UTC intervals compiled by Python's zoneinfo calendar.
//! Keeping calendar conversion in Python preserves its DST and date-override semantics.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reuse the current interval for neighboring candles, with binary searches on jumps.
/// `upper` is the number of interval starts <= the last timestamp. Searching backwards
/// as well as forwards preserves arbitrary input order, duplicates and reversed views.
struct IntervalCursor<'a> {
    starts: &'a [i64],
    ends: &'a [i64],
    upper: usize,
}

impl<'a> IntervalCursor<'a> {
    fn new(starts: &'a [i64], ends: &'a [i64]) -> PyResult<Self> {
        if starts.len() != ends.len() || starts.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(PyValueError::new_err(
                "interval starts and ends must have equal lengths and starts must be sorted",
            ));
        }
        Ok(Self {
            starts,
            ends,
            upper: 0,
        })
    }

    #[inline]
    fn contains(&mut self, timestamp: i64) -> bool {
        if self.upper > 0 && timestamp < self.starts[self.upper - 1] {
            self.upper = self.starts.partition_point(|&start| start <= timestamp);
        } else if self.upper < self.starts.len() && timestamp >= self.starts[self.upper] {
            self.upper += self.starts[self.upper..].partition_point(|&start| start <= timestamp);
        }
        // Match searchsorted(side="right") exactly, including zero-length DST windows.
        self.upper > 0 && timestamp < self.ends[self.upper - 1]
    }
}

/// Boolean membership for integer UTC milliseconds and sorted, merged UTC intervals.
/// Timestamps may be strided or unordered; interval arrays must be contiguous.
#[pyfunction]
pub fn trading_hours_mask<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
) -> PyResult<&'py PyArray1<bool>> {
    let mut cursor = IntervalCursor::new(starts.as_slice()?, ends.as_slice()?)?;
    let result: Vec<bool> = timestamps
        .as_array()
        .iter()
        .map(|&t| cursor.contains(t))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

/// Filter float64 candle rows without allocating a boolean mask or per-row search indexes.
/// Preconverted timestamps preserve NumPy's integer conversion; all candle columns and
/// their bit patterns are copied unchanged. The result never aliases a nonempty input.
#[pyfunction]
pub fn filter_candles_by_intervals<'py>(
    py: Python<'py>,
    candles: PyReadonlyArray2<f64>,
    timestamps: PyReadonlyArray1<i64>,
    starts: PyReadonlyArray1<i64>,
    ends: PyReadonlyArray1<i64>,
) -> PyResult<&'py PyArray2<f64>> {
    let candles = candles.as_array();
    let timestamps = timestamps.as_array();
    if candles.nrows() != timestamps.len() {
        return Err(PyValueError::new_err(
            "one timestamp is required per candle row",
        ));
    }
    let mut cursor = IntervalCursor::new(starts.as_slice()?, ends.as_slice()?)?;

    // Session rows usually occur in long runs. Record only their boundaries, then
    // allocate exactly the retained size and copy whole runs from contiguous arrays.
    let mut runs = Vec::new();
    let mut run_start = None;
    let mut kept = 0;
    for (row, &timestamp) in timestamps.iter().enumerate() {
        if cursor.contains(timestamp) {
            run_start.get_or_insert(row);
            kept += 1;
        } else if let Some(start) = run_start.take() {
            runs.push(start..row);
        }
    }
    if let Some(start) = run_start {
        runs.push(start..candles.nrows());
    }

    let columns = candles.ncols();
    let mut result = Vec::with_capacity(kept * columns);
    if let Some(contiguous) = candles.as_slice() {
        for run in runs {
            result.extend_from_slice(&contiguous[run.start * columns..run.end * columns]);
        }
    } else {
        // Jesse also passes row/column slices and Fortran-order research arrays.
        for run in runs {
            for row in run {
                result.extend(candles.row(row).iter().copied());
            }
        }
    }
    let result = Array2::from_shape_vec((kept, columns), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}
