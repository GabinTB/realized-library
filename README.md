# Realized Library

**Realized Library** is an open-source Python package for the computation of high-frequency econometrics estimators, such as variance, covariance and jump detection. It is designed as a modern, efficient, and production-ready continuation of the Oxford-Man Institute's original Realized Library.

The package focuses on performance and scalability, supporting large-scale high-frequency financial datasets (with trade, quote and time-sampled prices) up to the nanoseconds precision. The primary goal is to provide accurate, fast, and customizable estimators suitable for both academic research and quantitative analysis.

---

## Background

This library is inspired by the Oxford-Man Institute of Quantitative Finance's original "Realized Library" (no longer maintained). It is designed for researchers, practitioners, and quantitative analysts who require fast and reliable estimations of variance, co-variance, jump and co-jumps in financial markets.

For reference to the underlying econometric methods, please consult the academic literature included in the original library or related publications in high-frequency financial econometrics.
Since the website is no longer maintained, key information could be found on Wayback Machine trough this [URL](https://web.archive.org/web/20220903214048/https://realized.oxford-man.ox.ac.uk/).

---

## Key Features

- Efficient realized variance estimation from high-frequency price data
- Flexible resampling, subsampling and preaveraging utility with customizable parameters
- Designed for research, production pipelines, and large nanoseconds HFT datasets

---

## Installation

```bash
pip install realized-library
```

Or, for development use:
```bash
git clone https://github.com/GabinTB/realized-library
cd realized-library
pip install -e .
```

The library requires Python 3.10.