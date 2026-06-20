# Monte Carlo Project S8

This repository contains a Monte Carlo pricing project for European and basket options, together with a Streamlit application and a notebook for numerical experiments.

## Project Overview

- `pricer_app.py`: interactive Streamlit pricing application.
- `main_1.ipynb`: notebook used to run experiments and generate benchmark results.
- `requirements.txt`: Python dependencies.
- `.streamlit/config.toml`: local Streamlit configuration.

## Features

- European option pricing with a Black-Scholes benchmark.
- Arithmetic basket option pricing in higher dimensions.
- Comparison of multiple variance reduction techniques.
- Sensitivity analysis on key Black-Scholes parameters.
- Interactive controls for testing pricing methods and simulation settings.

## Setup

This project targets Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Application

```powershell
streamlit run pricer_app.py
```

## Notebook Workflow

Open `main_1.ipynb` in VS Code or Jupyter to reproduce the numerical experiments.

## Notes

The notebook generates results locally when it is executed. Those outputs are intentionally not part of the source code.