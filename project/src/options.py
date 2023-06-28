import pandas as pd
import numpy as np

from src.data import Option


class Bionomial:
    """Binomial pricing model for European options"""

    @staticmethod
    def price(
        s0: float, K: float, T: float, r: float, v: float, N: int, call: bool = True
    ) -> float:
        """Price an option using the binomial pricing model
        Args:
            s0 (float): initial stock price
            K (float): strike price
            T (float): time to maturity as a fraction of one year
            r (float): risk-free interest rate
            v (float): annualized volatility
            N (int): number of time steps
            call (bool, optional): True for call option, False for put option. Defaults to True.
        Returns:
            float: option price
        """
        dt = T / N  # time step
        u = np.exp(v * np.sqrt(dt))  # up factor
        d = 1 / u  # down factor

        # Risk-neutral probability
        p = (np.exp(r * dt) - d) / (u - d)

        # Price tree
        price_tree = np.zeros([N + 1, N + 1])
        for i in range(N + 1):
            for j in range(i + 1):
                price_tree[j, i] = s0 * (d**j) * (u ** (i - j))

        option_tree = np.zeros([N + 1, N + 1])
        if call:
            # Option value at each final node is max(S - K, 0)
            option_tree[:, N] = np.maximum(np.zeros(N + 1), price_tree[:, N] - K)
        else:
            # Option value at each final node is max(K - S, 0)
            option_tree[:, N] = np.maximum(np.zeros(N + 1), K - price_tree[:, N])

        # Calculate option price at t = 0
        for i in np.arange(N - 1, -1, -1):
            for j in np.arange(0, i + 1):
                option_tree[j, i] = np.exp(-r * dt) * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )

        return option_tree[0, 0]

    @staticmethod
    def add_price(option: Option, data: pd.DataFrame) -> pd.DataFrame:
        """Add option price to dataframe
        Args:
            data (pd.Dataframe): dataframe to add option price to
        Returns:
            pd.DataFrame: dataframe with option price
        """
        data["binPrice"] = data.apply(
            lambda x: Bionomial.price(
                s0=x["S0"],
                K=option.strike,
                T=x["T"],
                r=x["rf"],
                v=x["std"],
                N=90,
                call=option.call,
            ),
            axis=1,
        )
        return data
