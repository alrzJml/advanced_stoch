import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_errs(results: list[pd.DataFrame], option_tags: list[str]) -> pd.DataFrame:
    # Initialize a DataFrame to hold the error metrics for each option
    errors = pd.DataFrame(columns=["Option", "Method", "MAE", "RMSE"])

    # Iterate over the list of DataFrames (and assume 'options' is a list of option names)
    for i, df in enumerate(results):
        # Drop rows where actual_price is NaN
        df = df.dropna(subset=["actual_option"])

        # Calculate error metrics for each method and add them to the errors DataFrame
        methods = ["binPrice", "triPrice", "blsPrice", "mcPrice"]
        for method in methods:
            mae = mean_absolute_error(df["actual_option"], df[method])
            rmse = np.sqrt(mean_squared_error(df["actual_option"], df[method]))
            err_df = pd.DataFrame(
                {"Option": option_tags[i], "Method": method, "MAE": mae, "RMSE": rmse},
                index=[i],
            )
            errors = pd.concat([errors, err_df], ignore_index=True)

    return errors


def get_coint_df(results: list[pd.DataFrame], option_tags: list[str]) -> pd.DataFrame:
    from statsmodels.tsa.stattools import coint

    cointegration_results = pd.DataFrame(
        columns=["Option", "Method", "p-value", "is-coint"]
    )

    methods = ["binPrice", "triPrice", "blsPrice", "mcPrice"]
    for i, df in enumerate(results):
        df = df.dropna(
            subset=["actual_option"]
        )  # remove rows with nan in 'actual_price'

        for method in methods:
            score, pvalue, _ = coint(df["actual_option"], df[method])
            method_result = pd.DataFrame(
                {
                    "Option": option_tags[i],
                    "Method": method,
                    "p-value": pvalue,
                    "is-coint": pvalue < 0.05,
                },
                index=[i],
            )
            cointegration_results = pd.concat(
                [cointegration_results, method_result], ignore_index=True
            )

    return cointegration_results


def get_is_diff_correlated(
    results: list[pd.DataFrame], option_tags: list[str]
) -> pd.DataFrame:
    from scipy.stats import pearsonr

    # Initialize a DataFrame to hold the correlation results
    correlation_results = pd.DataFrame(
        columns=["Option", "Method", "Correlation", "p-value", "is-signf"]
    )
    methods = ["binPrice", "triPrice", "blsPrice", "mcPrice"]
    for i, df in enumerate(results):
        df = df.dropna(
            subset=["actual_option"]
        )  # remove rows with nan in 'actual_price'

        for method in methods:
            difference = np.abs(
                df["actual_option"] - df[method]
            )  # calculate the difference
            corr, pvalue = pearsonr(
                difference, df["T"]
            )  # calculate the correlation with 'T'
            per_result = pd.DataFrame(
                {
                    "Option": option_tags[i],
                    "Method": method,
                    "Correlation": corr,
                    "p-value": pvalue,
                    "is-signf": corr > 0 and pvalue < 0.05,
                },
                index=[i],
            )
            correlation_results = pd.concat(
                [correlation_results, per_result], ignore_index=True
            )

    return correlation_results
