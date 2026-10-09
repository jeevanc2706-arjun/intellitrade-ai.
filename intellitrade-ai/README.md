# IntelliTrade AI

## Explainable Hybrid Learning Framework for Cryptocurrency Trading

## Overview

IntelliTrade AI is a portfolio implementation of an AI-driven framework for cryptocurrency market-trend analysis, trading-signal generation, explainability, and basic risk-management analysis.

> **Reconstructed portfolio implementation:** The original source files and original cryptocurrency dataset were no longer available. This repository reconstructs a transparent, educational implementation from the project description in my resume. It is not presented as the original source code or original trading results.

## Architecture

Market Data → Feature Engineering → Hybrid Signal Model → Buy/Sell/Hold Signal → Explainability → Risk Analysis

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Explainable AI concepts

## Hybrid Approach

The demonstration combines:
1. A trend classifier based on engineered market indicators.
2. A rule-based risk layer for interpreting signal strength and volatility.

The implementation is designed for learning and portfolio demonstration, not financial advice or live trading.

## Explainability

The project exposes the feature values contributing to the generated signal and presents a simple feature-importance view for the classifier.

## Risk Management

The demonstration calculates volatility and assigns a simple risk category to the generated signal.

## How to Run

```bash
pip install -r requirements.txt
python src/intellitrade.py
```

## Data

Because the original cryptocurrency dataset was unavailable, the demonstration creates a synthetic time series. Replace it with the original project dataset before claiming original experimental results.

## Project Structure

```text
intellitrade-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── intellitrade.py
│   ├── data_processing.py
│   └── explainability.py
├── notebooks/
│   └── analysis.ipynb
└── results/
    ├── market_signals.png
    └── feature_importance.png
```

## Author

Jeevan C
