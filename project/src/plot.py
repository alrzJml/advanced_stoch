import pandas as pd
import matplotlib.pyplot as plt


def plot_methods(results: list[pd.DataFrame], option_tags: list[str]) -> None:
    methods = ["binPrice", "triPrice", "blsPrice", "mcPrice"]

    _, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 20))

    for i, df in enumerate(results):
        # Drop rows where actual_price is NaN
        df = df.dropna(subset=["actual_option"])

        for j, method in enumerate(methods):
            ax = axes[i, j]
            df[["actual_option", method]].plot(ax=ax)
            df[["S0"]].plot(ax=ax, secondary_y=True)
            ax.set_title(f"{option_tags[i]} - {method}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

    plt.tight_layout()
    plt.show()


def plot_errs(errors: pd.DataFrame, option_tags: list[str]):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    for i, option in enumerate(option_tags):
        ax = axes[i // 2, i % 2]  # Determine the correct subplot
        errors_option = errors[errors["Option"] == option]
        ax.bar(errors_option["Method"], errors_option["MAE"])
        ax.set_title(f"MAE for {option}")
        ax.set_ylabel("Mean Absolute Error")

    # Remove any empty subplots
    if len(option_tags) % 2 != 0:
        fig.delaxes(axes.flatten()[-1])

    plt.tight_layout()
    plt.show()
