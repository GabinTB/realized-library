import numpy as np
import pytest
from realized_library.utils.resampling import (
    _parse_resample_freq,
    compute
)

# ----------------------
# Tests for _parse_resample_freq
# ----------------------

@pytest.mark.parametrize("freq_str, expected", [
    ("1s", 1),
    ("30s", 30),
    ("2m", 120),
    ("500ms", 0.5),
    ("200us", 0.0002),
    ("100ns", 1e-7),
    (" 10S ", 10),
    ("0.5m", 30),
])
def test_parse_resample_freq_valid(freq_str, expected):
    result = _parse_resample_freq(freq_str)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("freq_str", [
    "10h", "abc", "10", "m10", "", None, " ", "10x", 10, 12.0
])
def test_parse_resample_freq_invalid(freq_str):
    with pytest.raises(ValueError):
        _parse_resample_freq(freq_str)

# ----------------------
# Tests for compute (resampling logic)
# ----------------------

def test_compute_basic_resampling():
    timestamps = np.array([1, 2, 3, 4, 5, 6])
    prices = np.array([100, 101, 102, 103, 104, 105])
    resample_freq = "3s"  # Bins: [1-4), [4-7)

    bin_timestamps, bin_prices = compute(timestamps, prices, resample_freq)

    expected_prices = np.array([102, 105])
    expected_timestamps = np.array([3, 6])

    np.testing.assert_array_equal(bin_prices, expected_prices)
    np.testing.assert_array_equal(bin_timestamps, expected_timestamps)


def test_compute_empty_bins():
    timestamps = np.array([1, 5, 10])
    prices = np.array([100, 105, 110])
    resample_freq = "3s"  # Bins: [1-4), [4-7), [7-10), [10-13)

    bin_timestamps, bin_prices = compute(timestamps, prices, resample_freq)

    expected_prices = np.array([100, 105, 110])
    expected_timestamps = np.array([1, 5, 10])

    np.testing.assert_array_equal(bin_prices, expected_prices)
    np.testing.assert_array_equal(bin_timestamps, expected_timestamps)


def test_compute_invalid_lengths():
    timestamps = np.array([1, 2, 3])
    prices = np.array([100, 101])  # Length mismatch
    with pytest.raises(ValueError):
        compute(timestamps, prices, "1s")


def test_compute_non_numpy_inputs():
    timestamps = [1, 2, 3]
    prices = [100, 101, 102]
    with pytest.raises(ValueError):
        compute(timestamps, prices, "1s")
