# Realized Library

**Realized Library** is an open-source Python package for the computation of realized measures of volatility and other high-frequency econometrics estimators. It is designed as a modern, efficient, and production-ready continuation of the Oxford-Man Institute's original Realized Library.

The package focuses on performance and scalability, supporting large-scale high-frequency financial datasets up to the nanoseconds precision. The primary goal is to provide accurate, fast, and customizable estimators suitable for both academic research and industrial quantitative analysis.

---

## Background and Academic Context

This library is inspired by the Oxford-Man Institute's original Realized Library (no longer maintained). It is designed for researchers, practitioners, and quantitative analysts who require fast and reliable computation of realized measures in financial markets.

For reference to the underlying econometric methods, please consult the academic literature included in the original OMI library or related publications in high-frequency financial econometrics.

---

## Key Features

- Efficient realized variance estimation from high-frequency price data
- Flexible resampling utility with customizable frequency and timestamp handling
- Numba-accelerated core computation for speed and scalability
- Designed for research, production pipelines, and large HFT datasets
- Modular architecture allowing the addition of other realized estimators (e.g., bipower variation, realized kernels, etc.)

---

## Installation

```bash
pip install realized-library
```

Or, for development use:
```bash
git clone https://github.com/yourusername/realized-library.git
cd realized-library
pip install -e .
```

The library requires Python 3.12.
