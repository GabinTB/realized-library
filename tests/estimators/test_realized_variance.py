import numpy as np
import pytest
from realized_library.estimators.realized_variance import (
    compute as rv
)
from realized_library.utils.resampling import ( 
    compute as resample,
)

@pytest.fixture
def dummy_data():
    np.random.seed(42)

    trade_times = np.arange(0, 3600, 1)
    trade_prices = np.cumprod(1 + 0.0001 * np.random.randn(len(trade_times))) * 100

    midquote_times = trade_times.copy()
    midquote_prices = trade_prices + 0.01 * np.random.randn(len(trade_times))

    ohlcv_1s_times = trade_times.copy()
    ohlcv_1s_prices = trade_prices.copy()

    ohlcv_1m_times = np.arange(0, 3600, 60)
    ohlcv_1m_prices = trade_prices[::60]

    invalid_prices = np.array([100, 101, 0, -5, 103])

    return {
        "trade": (trade_prices, trade_times),
        "midquote": (midquote_prices, midquote_times),
        "ohlcv_1s": (ohlcv_1s_prices, ohlcv_1s_times),
        "ohlcv_1m": (ohlcv_1m_prices, ohlcv_1m_times),
        "invalid": (invalid_prices, np.arange(len(invalid_prices))),
    }


def test_realized_variance_no_resampling(dummy_data):
    prices, _ = dummy_data["trade"]
    rv_value = rv(prices)
    assert rv_value > 0


def test_realized_variance_resample_5s(dummy_data):
    prices, times = dummy_data["trade"]
    resampled_prices = resample(times, prices, resample_freq="5s")
    rv_value = rv(resampled_prices)
    assert rv_value > 0


def test_realized_variance_resample_1m(dummy_data):
    prices, times = dummy_data["trade"]
    resampled_prices = resample(times, prices, resample_freq="1m")
    rv_value = rv(resampled_prices)
    assert rv_value > 0


def test_realized_variance_midquote_resample(dummy_data):
    prices, times = dummy_data["midquote"]
    resampled_prices = resample(times, prices, resample_freq="10s")
    rv_value = rv(resampled_prices)
    assert rv_value > 0


def test_realized_variance_invalid_prices(dummy_data):
    prices, _ = dummy_data["invalid"]
    with pytest.raises(ValueError):
        rv(prices)