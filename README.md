# Systematic Equity Alpha: Factor Models & Statistical Arbitrage

Python tooling for equity factor research: CAPM and Fama–French three-factor (FF3) regressions on daily data, cross-sectional summaries, and a small Streamlit dashboard for exploration.

## What’s in the repo

| Area | Role |
|------|------|
| `utils/fama_french.py` | End-to-end pipeline: load Ken French daily factors, align with Yahoo Finance stock returns, OLS for CAPM vs FF3, plots, multi-ticker CSV. |
| `utils/beta.py` | Simple rolling-style beta via sklearn `LinearRegression` (stock vs index returns). |
| `utils/stat_functions.py` | CAPM-style expected return helper using beta, market return from Yahoo, and risk-free rate from FRED. |
| `dashboard/app.py` | Streamlit UI to explore `outputs/cross_section.csv` (scatter plots, scatter matrix, table, download). |

Data sources:

- **Fama–French factors**: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (daily CSV zip, downloaded at runtime).
- **Prices**: [yfinance](https://github.com/ranaroussi/yfinance).
- **Risk-free proxy** (in `stat_functions.py`): [FRED](https://fred.stlouisfed.org/) Treasury series.

## Setup

Requires Python 3.10+ (the code uses union types like `X | Y`).

```bash
cd "Systematic Equity Alpha Factor Models and Statistical Arbitrage"
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `yfinance`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `streamlit`, `plotly`.

## Running the factor pipeline (`fama_french.py`)

Run commands from the **repository root** so default paths like `outputs/` resolve correctly.

**Step 1 — Single ticker, CAPM vs FF3 summary (stdout)**

```bash
python utils/fama_french.py AAPL
python utils/fama_french.py MSFT --period 5y
```

**Step 2 — Model comparison table + PNG plots**

Writes a comparison table and saves figures under `--out` (default `outputs/`):

- `{TICKER}_pred_vs_actual.png` — predicted vs actual excess returns (CAPM & FF3).
- `{TICKER}_residuals_time.png` — residuals over time.

```bash
python utils/fama_french.py AAPL --step2
python utils/fama_french.py AAPL --step2 --period 5y --out outputs
```

**Step 3 — Cross-section (many tickers → CSV)**

Runs CAPM/FF3 for each symbol and writes `outputs/cross_section.csv` by default (or `--csv`).

```bash
python utils/fama_french.py --step3
python utils/fama_french.py --step3 --tickers AAPL,MSFT,GOOGL --period 5y
python utils/fama_french.py --step3 --csv outputs/cross_section_smoke.csv
```

Rows with download or alignment errors include an `error` column; the dashboard drops those when present.

## Streamlit dashboard

After generating cross-section data (e.g. `--step3`):

```bash
streamlit run dashboard/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). By default the app loads `outputs/cross_section.csv`; you can change the path in the sidebar or upload a CSV.

## Other utilities

From `utils/` (you may need `cd utils` or adjust `PYTHONPATH` for relative imports in `stat_functions.py`):

```bash
cd utils
python beta.py          # example AAPL vs SPY beta
python stat_functions.py   # example CAPM expected return using FRED + Yahoo
```

## Outputs

Generated artifacts are intended to live under `outputs/` (gitignored as needed): PNG charts, `cross_section.csv`, etc.

