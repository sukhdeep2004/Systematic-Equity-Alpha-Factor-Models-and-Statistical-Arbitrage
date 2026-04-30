import pandas as pd
import yfinance as yf

from beta import get_beta
def get_rf_rate(treasury_type="3m"):
    type_to_series = {
        "4wk": "DTB4WK",
        "3m": "DGS3MO",
        "6m": "DGS6MO",
        "1y": "DGS1",
        "3y": "DGS3",
        "5y": "DGS5",
        "7y": "DGS7",
        "10y": "DGS10",
    }
    series_id = type_to_series.get(treasury_type.lower())
    if not series_id:
        valid_options = ", ".join(f'"{option}"' for option in type_to_series.keys())
        raise ValueError(
            f"Treasury type '{treasury_type}' is not recognized. Valid options are: {valid_options}."
        )
    # FRED public CSV 
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    rate_data = pd.read_csv(url, parse_dates=["observation_date"])
    series = rate_data[series_id].dropna()
    if series.empty:
        raise ValueError(f"No observations returned for FRED series '{series_id}'.")
    most_recent_rate = float(series.iloc[-1])
    most_recent_rate_pct = most_recent_rate / 100
    return most_recent_rate_pct

def get_market_return(ticker, period="3y"):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=int(period[:-1]))
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
    if hist.empty or "Adj Close" not in hist.columns:
        raise ValueError(f"No adjusted close data returned for ticker '{ticker}'.")
    adj_close = hist["Adj Close"].dropna()
    if adj_close.empty:
        raise ValueError(f"No valid adjusted close observations for ticker '{ticker}'.")
    start_price = float(adj_close.iloc[0])
    end_price = float(adj_close.iloc[-1])
    if start_price <= 0:
        raise ValueError(f"Invalid start price for ticker '{ticker}'.")

    elapsed_years = (adj_close.index[-1] - adj_close.index[0]).days / 365.25
    if elapsed_years <= 0:
        raise ValueError(f"Insufficient elapsed time to annualize returns for ticker '{ticker}'.")

    annualized_return = (end_price / start_price) ** (1 / elapsed_years) - 1
    return float(annualized_return)

def calculate_capm(stock="AAPL", index="SPY", beta_period="5y", beta_interval="1wk", market_period="10y", treasury_type="3m"):
    index_rate = get_market_return(index, market_period)
    beta = get_beta(stock, index, beta_period, beta_interval, just_beta=True)
    rf_rate = get_rf_rate(treasury_type)
    return rf_rate + beta * (index_rate - rf_rate)
if __name__ == "__main__":
    print(calculate_capm("AAPL", "SPY", "5y", "1wk", "10y", "3m"))