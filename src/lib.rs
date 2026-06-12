//! `jesse_rust` — native implementations of Jesse's indicators and a few
//! supporting utilities, exposed to Python through PyO3.
//!
//! Modules are grouped by indicator category; see each submodule's docstring
//! for what lives where. Private numerical helpers live in `helpers.rs`; the
//! shared PyO3 return-type aliases live in `types.rs`.

use pyo3::prelude::*;

mod helpers;
mod types;

mod bands;
mod candle;
mod ehlers;
mod moving_averages;
mod oscillators;
mod statistical;
mod trend;
mod util;
mod volume;

#[pymodule]
fn jesse_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // ---- moving averages ----------------------------------------------------
    m.add_function(wrap_pyfunction!(moving_averages::cwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::dema, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::ema, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::ema_last, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::epma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::frama, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::hma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::hwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::jma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::kama, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::maaq, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::mcginley_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::mwdx, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::nma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::rma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::sma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::sma_last, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::smma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::sqwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::srwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::t3, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::tema, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::trix, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::vidya, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::vlma_inner, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::vpwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::vwap, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::vwma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::wilders, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::wma, m)?)?;
    m.add_function(wrap_pyfunction!(moving_averages::zlema, m)?)?;

    // ---- oscillators --------------------------------------------------------
    m.add_function(wrap_pyfunction!(oscillators::aroonosc, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::cci, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::cfo, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::cg, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::cmo, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::dpo, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::dti, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::fisher, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::fosc, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::lrsi, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::macd, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::mass, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::pfe, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::rsi_last, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::rsx, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::srsi, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::stoch, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::stochf, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::willr, m)?)?;
    m.add_function(wrap_pyfunction!(oscillators::wt, m)?)?;

    // ---- trend / directional ------------------------------------------------
    m.add_function(wrap_pyfunction!(trend::adx, m)?)?;
    m.add_function(wrap_pyfunction!(trend::adxr, m)?)?;
    m.add_function(wrap_pyfunction!(trend::alligator, m)?)?;
    m.add_function(wrap_pyfunction!(trend::di, m)?)?;
    m.add_function(wrap_pyfunction!(trend::dm, m)?)?;
    m.add_function(wrap_pyfunction!(trend::dx, m)?)?;
    m.add_function(wrap_pyfunction!(trend::ichimoku_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(trend::safezonestop, m)?)?;
    m.add_function(wrap_pyfunction!(trend::sar, m)?)?;
    m.add_function(wrap_pyfunction!(trend::supertrend, m)?)?;
    m.add_function(wrap_pyfunction!(trend::vi, m)?)?;

    // ---- volume -------------------------------------------------------------
    m.add_function(wrap_pyfunction!(volume::adosc, m)?)?;
    m.add_function(wrap_pyfunction!(volume::cvi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::efi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::emv, m)?)?;
    m.add_function(wrap_pyfunction!(volume::nvi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::pvi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::wad, m)?)?;

    // ---- bands / channels / volatility --------------------------------------
    m.add_function(wrap_pyfunction!(bands::atr, m)?)?;
    m.add_function(wrap_pyfunction!(bands::atr_last, m)?)?;
    m.add_function(wrap_pyfunction!(bands::bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(bands::bollinger_bands_last, m)?)?;
    m.add_function(wrap_pyfunction!(bands::bollinger_bands_width, m)?)?;
    m.add_function(wrap_pyfunction!(bands::chande, m)?)?;
    m.add_function(wrap_pyfunction!(bands::chop, m)?)?;
    m.add_function(wrap_pyfunction!(bands::donchian, m)?)?;
    m.add_function(wrap_pyfunction!(bands::keltner_inner, m)?)?;

    // ---- Ehlers filters -----------------------------------------------------
    m.add_function(wrap_pyfunction!(ehlers::bandpass, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::correlation_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::edcf, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::gauss, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::high_pass, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::high_pass_2_pole, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::itrend, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::mama, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::pma, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::reflex, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::supersmoother, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::supersmoother_3_pole, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::trendflex, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers::voss, m)?)?;

    // ---- candle transforms --------------------------------------------------
    m.add_function(wrap_pyfunction!(candle::candle_from_one_minutes, m)?)?;
    m.add_function(wrap_pyfunction!(candle::fix_jumped_candles, m)?)?;
    m.add_function(wrap_pyfunction!(candle::emd, m)?)?;
    m.add_function(wrap_pyfunction!(candle::heikin_ashi_candles, m)?)?;
    m.add_function(wrap_pyfunction!(candle::qstick, m)?)?;

    // ---- statistical --------------------------------------------------------
    m.add_function(wrap_pyfunction!(statistical::damiani_volatmeter, m)?)?;
    m.add_function(wrap_pyfunction!(statistical::hurst_rs, m)?)?;
    m.add_function(wrap_pyfunction!(statistical::linearreg, m)?)?;

    // ---- utilities ----------------------------------------------------------
    m.add_function(wrap_pyfunction!(util::find_order_index, m)?)?;
    m.add_function(wrap_pyfunction!(util::moving_std, m)?)?;
    m.add_function(wrap_pyfunction!(util::shift, m)?)?;
    m.add_function(wrap_pyfunction!(util::subtract_floats, m)?)?;
    m.add_function(wrap_pyfunction!(util::sum_floats, m)?)?;

    Ok(())
}
