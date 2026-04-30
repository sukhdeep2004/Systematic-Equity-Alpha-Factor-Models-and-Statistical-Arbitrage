import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_daily_data(ticker, period="3y", interval="1d"):
    # Use adjusted close so beta is estimated from total-return series.
    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if data.empty or "Adj Close" not in data.columns:
        raise ValueError(f"No adjusted close data returned for ticker '{ticker}'.")
    close = data["Adj Close"].dropna()
    if close.empty:
        raise ValueError(f"No valid adjusted close observations for ticker '{ticker}'.")
    return close.rename(ticker)


def get_beta(ticker1, ticker2, period="3y", interval="1d", just_beta=False):
    data1 = get_daily_data(ticker1, period, interval)
    data2 = get_daily_data(ticker2, period, interval)
    data = pd.merge(data1, data2, left_index=True, right_index=True)

    data[f"{ticker1}_return"] = data[ticker1].pct_change()
    data[f"{ticker2}_return"] = data[ticker2].pct_change()
    data = data.dropna()

    if data.empty:
        raise ValueError("Insufficient overlapping return history to estimate beta.")

    x = data[f"{ticker2}_return"].values.reshape(-1, 1)
    y = data[f"{ticker1}_return"].values
    model = LinearRegression()
    model.fit(x, y)

    if just_beta:
        return model.coef_[0]
    return model.coef_[0], model


if __name__ == "__main__":
    ticker1 = "AAPL"
    index = "SPY"
    beta, model = get_beta(ticker1, index, just_beta=False)
    print(beta)
    print(model.intercept_)
    print(model.coef_)