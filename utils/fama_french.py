import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

FF_DAILY_ZIP = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

DEFAULT_CROSS_SECTION_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "UNH",
    "JNJ",
    "XOM",
    "JPM",
    "V",
    "PG",
    "MA",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
]


def _load_ff_factors_daily() -> pd.DataFrame:
    with urllib.request.urlopen(FF_DAILY_ZIP, timeout=120) as response:
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
        name = zf.namelist()[0]
        body = zf.read(name)
    raw = body.decode("utf-8", errors="replace")
    lines = raw.splitlines()
    header_idx = next(
        i for i, line in enumerate(lines) if line.strip().startswith(",Mkt-RF")
    )
    head = "Date," + lines[header_idx].lstrip(",")
    data_block = "\n".join([head] + lines[header_idx + 1 :])
    df = pd.read_csv(io.StringIO(data_block))
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    for col in ("MKT_RF", "SMB", "HML", "RF"):
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    df = df.dropna(subset=["MKT_RF", "SMB", "HML", "RF"])
    return df


def _period_years(period: str) -> int:
    return int(period[:-1])


def factors_for_period(period: str) -> pd.DataFrame:
    factors = _load_ff_factors_daily()
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=_period_years(period))
    out = factors.loc[(factors.index >= start) & (factors.index <= end)]
    if out.empty:
        raise ValueError("No Fama-French factor rows in the requested window.")
    return out


def get_stock_daily_returns(ticker: str, period: str = "5y") -> pd.Series:
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if hist.empty or "Adj Close" not in hist.columns:
        raise ValueError(f"No adjusted close data returned for ticker '{ticker}'.")
    adj = hist["Adj Close"].dropna()
    if adj.empty:
        raise ValueError(f"No valid adjusted close observations for ticker '{ticker}'.")
    idx = pd.DatetimeIndex(adj.index).tz_localize(None).normalize()
    adj.index = idx
    return adj.pct_change().dropna().rename(ticker)


def align_stock_and_factors(
    ticker: str,
    period: str = "5y",
    factors_window: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if factors_window is None:
        factors = factors_for_period(period)
    else:
        factors = factors_window
    if factors.empty:
        raise ValueError("No Fama-French factor rows in the requested window.")

    rets = get_stock_daily_returns(ticker, period=period)
    merged = pd.merge(
        rets.rename("stock_ret"),
        factors,
        left_index=True,
        right_index=True,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping dates between stock returns and FF factors.")
    excess = merged["stock_ret"] - merged["RF"]
    excess.name = "excess_ret"
    X_cols = merged[["MKT_RF", "SMB", "HML"]]
    return excess, X_cols


def run_capm_ols(y: pd.Series, factors: pd.DataFrame):
    X = sm.add_constant(factors[["MKT_RF"]])
    return sm.OLS(y, X).fit()


def run_ff3_ols(y: pd.Series, factors: pd.DataFrame):
    X = sm.add_constant(factors[["MKT_RF", "SMB", "HML"]])
    return sm.OLS(y, X).fit()


def compare_capm_ff3(
    ticker: str,
    period: str = "5y",
    factors_window: pd.DataFrame | None = None,
) -> dict:
    y, X = align_stock_and_factors(
        ticker, period=period, factors_window=factors_window
    )
    capm = run_capm_ols(y, X)
    ff3 = run_ff3_ols(y, X)

    capm_var = float(np.var(capm.resid, ddof=1))
    ff3_var = float(np.var(ff3.resid, ddof=1))

    return {
        "ticker": ticker,
        "period": period,
        "n_obs": int(len(y)),
        "y": y,
        "capm": capm,
        "ff3": ff3,
        "capm_r2": float(capm.rsquared),
        "ff3_r2": float(ff3.rsquared),
        "capm_adj_r2": float(capm.rsquared_adj),
        "ff3_adj_r2": float(ff3.rsquared_adj),
        "capm_alpha": float(capm.params["const"]),
        "ff3_alpha": float(ff3.params["const"]),
        "capm_alpha_p": float(capm.pvalues["const"]),
        "ff3_alpha_p": float(ff3.pvalues["const"]),
        "smb_coef": float(ff3.params["SMB"]),
        "smb_p": float(ff3.pvalues["SMB"]),
        "hml_coef": float(ff3.params["HML"]),
        "hml_p": float(ff3.pvalues["HML"]),
        "capm_resid_var": capm_var,
        "ff3_resid_var": ff3_var,
    }


def model_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for label, m in (("CAPM", results["capm"]), ("FF3", results["ff3"])):
        rows.append(
            {
                "model": label,
                "R2": float(m.rsquared),
                "Adj_R2": float(m.rsquared_adj),
                "Alpha": float(m.params["const"]),
                "Alpha_p": float(m.pvalues["const"]),
                "Residual_variance": float(np.var(m.resid, ddof=1)),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def save_comparison_plots(results: dict, output_dir: str = "outputs") -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ticker = results["ticker"]
    y = results["y"]
    capm = results["capm"]
    ff3 = results["ff3"]
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model, title in zip(
        axes,
        (capm, ff3),
        ("CAPM", "Fama-French 3-factor"),
    ):
        fitted = model.fittedvalues.reindex(y.index)
        ax.scatter(y.values, fitted.values, s=4, alpha=0.35, edgecolors="none")
        lo = float(min(y.min(), fitted.min()))
        hi = float(max(y.max(), fitted.max()))
        pad = (hi - lo) * 0.05 + 1e-8
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1, alpha=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual excess return")
        ax.set_ylabel("Predicted excess return")
        ax.set_title(f"{title} ({ticker})")
    fig.suptitle("Predicted vs actual daily excess returns", y=1.02)
    fig.tight_layout()
    p1 = out / f"{ticker}_pred_vs_actual.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p1.resolve()))

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    capm_r = pd.Series(capm.resid, index=y.index)
    ff3_r = pd.Series(ff3.resid, index=y.index)
    axes2[0].plot(capm_r.index, capm_r.values, lw=0.7, color="C0")
    axes2[0].axhline(0.0, color="gray", lw=0.6)
    axes2[0].set_ylabel("CAPM residual")
    axes2[1].plot(ff3_r.index, ff3_r.values, lw=0.7, color="C1")
    axes2[1].axhline(0.0, color="gray", lw=0.6)
    axes2[1].set_ylabel("FF3 residual")
    axes2[1].set_xlabel("Date")
    fig2.suptitle(f"Residuals over time ({ticker})", y=1.0)
    fig2.tight_layout()
    p2 = out / f"{ticker}_residuals_time.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    paths.append(str(p2.resolve()))

    return paths


def run_step2(
    ticker: str, period: str = "5y", output_dir: str = "outputs"
) -> tuple[pd.DataFrame, list[str]]:
    results = compare_capm_ff3(ticker, period=period)
    table = model_comparison_table(results)
    paths = save_comparison_plots(results, output_dir=output_dir)
    return table, paths


def cross_section_row(results: dict) -> dict:
    capm = results["capm"]
    ff3 = results["ff3"]
    return {
        "ticker": results["ticker"],
        "n_obs": results["n_obs"],
        "capm_alpha": results["capm_alpha"],
        "capm_alpha_p": results["capm_alpha_p"],
        "capm_beta_mkt": float(capm.params["MKT_RF"]),
        "capm_r2": results["capm_r2"],
        "capm_adj_r2": results["capm_adj_r2"],
        "ff3_alpha": results["ff3_alpha"],
        "ff3_alpha_p": results["ff3_alpha_p"],
        "ff3_mkt": float(ff3.params["MKT_RF"]),
        "ff3_smb": float(ff3.params["SMB"]),
        "ff3_hml": float(ff3.params["HML"]),
        "ff3_r2": results["ff3_r2"],
        "ff3_adj_r2": results["ff3_adj_r2"],
    }


def run_cross_section(
    tickers: list[str] | None = None,
    period: str = "5y",
    csv_path: str | None = "outputs/cross_section.csv",
) -> pd.DataFrame:
    universe = list(tickers) if tickers is not None else list(DEFAULT_CROSS_SECTION_TICKERS)
    fw = factors_for_period(period)
    rows: list[dict] = []
    for sym in universe:
        sym = sym.strip().upper()
        try:
            res = compare_capm_ff3(sym, period=period, factors_window=fw)
            rows.append(cross_section_row(res))
        except Exception as e:
            rows.append({"ticker": sym, "error": str(e)})
    df = pd.DataFrame(rows)
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


def _print_step1_summary(out: dict) -> None:
    print("ticker:", out["ticker"], "n:", out["n_obs"], "window:", out["period"])
    print("CAPM R2 / adj:", round(out["capm_r2"], 4), round(out["capm_adj_r2"], 4))
    print("FF3  R2 / adj:", round(out["ff3_r2"], 4), round(out["ff3_adj_r2"], 4))
    print("Alpha CAPM / p:", round(out["capm_alpha"], 6), round(out["capm_alpha_p"], 4))
    print("Alpha FF3  / p:", round(out["ff3_alpha"], 6), round(out["ff3_alpha_p"], 4))
    print("SMB coef / p:", round(out["smb_coef"], 6), round(out["smb_p"], 4))
    print("HML coef / p:", round(out["hml_coef"], 6), round(out["hml_p"], 4))
    print("Resid var CAPM / FF3:", round(out["capm_resid_var"], 8), round(out["ff3_resid_var"], 8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAPM / Fama-French utilities")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--step2", action="store_true", help="comparison table + plots")
    parser.add_argument(
        "--step3",
        action="store_true",
        help="cross-section CAPM/FF3 for many tickers + CSV",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="comma-separated symbols for --step3 (default: built-in list of 20)",
    )
    parser.add_argument("--out", default="outputs", help="directory for Step 2 PNGs")
    parser.add_argument(
        "--csv",
        default="outputs/cross_section.csv",
        help="output CSV path for --step3",
    )
    args = parser.parse_args()

    if args.step3:
        tick_list = (
            [t.strip() for t in args.tickers.split(",") if t.strip()]
            if args.tickers.strip()
            else None
        )
        df_cs = run_cross_section(
            tickers=tick_list, period=args.period, csv_path=args.csv
        )
        print(df_cs.to_string())
        print()
        print(args.csv)
    elif args.step2:
        tbl, plot_paths = run_step2(args.ticker, period=args.period, output_dir=args.out)
        print(tbl.to_string(float_format=lambda x: f"{x:.8g}"))
        print()
        for path in plot_paths:
            print(path)
    else:
        out = compare_capm_ff3(args.ticker, period=args.period)
        _print_step1_summary(out)
