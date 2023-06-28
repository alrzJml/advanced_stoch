import pandas as pd
import numpy as np
import pytse_client as tse
from collections import namedtuple

Option = namedtuple(
    "Option", ["stock_symbol", "option_symbol", "strike", "maturity_date", "call"]
)


def fetch_data(option: Option) -> pd.DataFrame:
    """Fetch historical data for a given option along with underlying stock
    Args:
        option (Option): option to fetch data for
    Returns:
        pd.DataFrame: historical data
    """
    # set start_date 1 year before now
    start_date = pd.Timestamp.now() - pd.Timedelta(days=365 + 90)
    # format start_date as 'YYYY-MM-DD'
    start_date = start_date.strftime("%Y-%m-%d")

    # Fetch historical data
    stock_data = tse.download(symbols=option.stock_symbol, write_to_csv=False)[
        option.stock_symbol
    ]
    option_data = tse.download(symbols=option.option_symbol, write_to_csv=False)[
        option.option_symbol
    ]

    # filter stock data by start_date
    stock_data = stock_data[stock_data["date"] >= start_date]

    # Merge dataframes
    data = pd.merge(
        stock_data[["date", "adjClose"]],
        option_data[["date", "adjClose"]],
        on="date",
        suffixes=["Stock", "Option"],
        how="outer",
    )
    return data


def add_std(data: pd.DataFrame, rolling_window: int = 90) -> pd.DataFrame:
    """Add rolling standard deviation to dataframe
    Args:
        data (pd.DataFrame): dataframe to add std to
        rolling_window (int, optional): rolling window for std. Defaults to 90.
    Returns:
        pd.DataFrame: dataframe with std column
    """
    # Calculate number of trading days in a year
    end_date = data["date"].max()
    start_date = end_date - pd.Timedelta(days=365)
    std_period = data[(data["date"] >= start_date) & (data["date"] <= end_date)].shape[
        0
    ]
    # Calculate rolling std
    data["std"] = data["adjCloseStock"].pct_change().rolling(rolling_window).std()
    # Annualize std
    data["std"] = data["std"] * np.sqrt(std_period)

    return data


def add_rf(data: pd.DataFrame) -> pd.DataFrame:
    """Add risk-free rate to dataframe
    Args:
        data (pd.DataFrame): dataframe to add rf to
    Returns:
        pd.DataFrame: dataframe with rf column
    """
    # read risk-free rate data
    df_rf = pd.read_excel("data/freeRisk.xlsx")
    # convert dayKey to date
    df_rf["date"] = pd.to_datetime(df_rf["dayKey"], format="%Y%m%d")
    df_rf = df_rf[["date", "riskFreeRate-90days"]]
    df_rf.columns = ["date", "rf"]
    # merge dataframes
    data = pd.merge(data, df_rf, on="date", how="left")
    data["rf"] = data["rf"].ffill()

    return data