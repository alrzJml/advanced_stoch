import pandas as pd
import numpy as np
import jdatetime
import datetime
from scipy.stats import norm

from src.data import Option


class Bionomial:
    """Binomial pricing model for European options"""

    @staticmethod
    def price(
        S0: float, K: float, T: float, r: float, v: float, N: int, call: bool = True
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
                price_tree[j, i] = S0 * (d**j) * (u ** (i - j))

        option_tree = np.zeros([N + 1, N + 1])
        if call:
            # Option value at each final node is max(S - K, 0)
            option_tree[:, N] = np.maximum(
                np.zeros(N + 1), price_tree[:, N] - K)
        else:
            # Option value at each final node is max(K - S, 0)
            option_tree[:, N] = np.maximum(
                np.zeros(N + 1), K - price_tree[:, N])

        # Calculate option price at t = 0
        for i in np.arange(N - 1, -1, -1):
            for j in np.arange(0, i + 1):
                option_tree[j, i] = np.exp(-r * dt) * (
                    p * option_tree[j, i + 1] +
                    (1 - p) * option_tree[j + 1, i + 1]
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
                S0=x["S0"],
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


class Trinomial:
    """Trinomial pricing model for European options"""

    @staticmethod
    def price(
        S0: float, K: float, T: float, r: float, v: float, N: int, call: bool = True
    ) -> float:
        dt = T / N  # time step
        u = np.exp(v * np.sqrt(2 * dt))  # up factor
        d = 1 / u  # down factor
        m = 1  # no change

        # Risk-neutral probabilities
        pu = (
            (np.exp(r * dt / 2) - np.exp(-v * np.sqrt(dt / 2)))
            / (np.exp(v * np.sqrt(dt / 2)) - np.exp(-v * np.sqrt(dt / 2)))
        ) ** 2
        pd = (
            (np.exp(v * np.sqrt(dt / 2)) - np.exp(r * dt / 2))
            / (np.exp(v * np.sqrt(dt / 2)) - np.exp(-v * np.sqrt(dt / 2)))
        ) ** 2
        pm = 1 - pu - pd

        # Price tree
        price_tree = np.zeros([2 * N + 1, N + 1])
        price_tree[N, 0] = S0
        for i in range(1, N + 1):
            price_tree[N - i, i] = u * price_tree[N - i + 1, i - 1]
            for j in range(N - i + 1, N + i + 1):
                price_tree[j, i] = d * price_tree[j - 1, i - 1]

        # Option value at each final node
        if call:
            option_tree = np.maximum(price_tree - K, 0)
        else:
            option_tree = np.maximum(K - price_tree, 0)

        # Calculate option price at t = 0
        for i in np.arange(N - 1, -1, -1):
            for j in np.arange(N - i, N + i + 1):
                option_tree[j, i] = np.exp(-r * dt) * (
                    pu * option_tree[j - 1, i + 1]
                    + pm * option_tree[j, i + 1]
                    + pd * option_tree[j + 1, i + 1]
                )

        return option_tree[N, 0]

    @staticmethod
    def add_price(option: Option, data: pd.DataFrame) -> pd.DataFrame:
        """Add option price to dataframe
        Args:
            data (pd.Dataframe): dataframe to add option price to
        Returns:
            pd.DataFrame: dataframe with option price
        """
        data["triPrice"] = data.apply(
            lambda x: Trinomial.price(
                S0=x["S0"],
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


class BLS:
    """Black-Scholes pricing model for European options"""

    @staticmethod
    def price(
        S0: float, K: float, T: float, r: float, v: float, call: bool = True
    ) -> float:
        """Price an option using the Black-Scholes pricing model

        Args:
            S0 (float): initial stock price
            K (float): strike price
            T (float): time to maturity as a fraction of one year
            r (float): risk-free interest rate
            v (float): annualized volatility
            call (bool, optional): True for call option, False for put option. Defaults to True.

        Returns:
            float: option price
        """

        # Calculate d1 and d2 parameters
        d1 = (np.log(S0 / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)

        # Calculate option price
        if call:
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

        return price

    @staticmethod
    def add_price(option: Option, data: pd.DataFrame) -> pd.DataFrame:
        """Add option price to dataframe
        Args:
            data (pd.Dataframe): dataframe to add option price to
        Returns:
            pd.DataFrame: dataframe with option price
        """
        data["blsPrice"] = data.apply(
            lambda x: BLS.price(
                S0=x["S0"],
                K=option.strike,
                T=x["T"],
                r=x["rf"],
                v=x["std"],
                call=option.call,
            ),
            axis=1,
        )
        return data


class MontCarlo:
    """Monte Carlo pricing model for European options"""

    @staticmethod
    def price(
        S0: float,
        K: float,
        T: float,
        r: float,
        v: float,
        N: int,
        num_simulations: int = 10000,
    ) -> float:
        """Price an option using the Monte Carlo pricing model

        Args:
            S0 (float): initial stock price
            K (float): strike price
            T (float): time to maturity as a fraction of one year
            r (float): risk-free interest rate
            v (float): annualized volatility
            num_simulations (int, optional): number of simulations. Defaults to 10000.

        Returns:
            float: option price
        """
        dt = T / N
        # Creating an array to store payoffs
        payoffs = np.zeros(num_simulations)

        # Simulate price paths
        for i in range(num_simulations):
            price_path = S0 * np.cumprod(
                np.exp(
                    (r - 0.5 * v**2) * dt
                    + v * np.sqrt(dt) * np.random.standard_normal(N)
                )
            )
            # Compute the payoff for each path
            payoffs[i] = max(price_path[-1] - K, 0)

        # Average payoffs and discount back to today
        return np.exp(-r * T) * np.mean(payoffs)

    @staticmethod
    def add_price(option: Option, data: pd.DataFrame) -> pd.DataFrame:
        """Add option price to dataframe
        Args:
            data (pd.Dataframe): dataframe to add option price to
        Returns:
            pd.DataFrame: dataframe with option price
        """
        data["mcPrice"] = data.apply(
            lambda x: MontCarlo.price(
                S0=x["S0"],
                K=option.strike,
                T=x["T"],
                r=x["rf"],
                v=x["std"],
                N=90,
                num_simulations=10000,
            ),
            axis=1,
        )
        return data


class Strategy:

    @staticmethod
    def add_greeks(K: float, mat_date: str, is_call: bool, today: str, S0: float, rf: float, sigma: float) -> tuple[float, float, float, float]:
        """Calculate option greeks
        Args:
            option (Option): option to calculate greeks for
            date (str): date to calculate greeks for
            S0 (float): initial stock price
            rf (float): risk-free interest rate
            sigma (float): annualized volatility
        Returns:
            tuple[float, float, float, float]: delta, gamma, vega, theta
        """
        end_date = mat_date
        if end_date[:2] == "14":
            yr, mn, dy = end_date.split("-")
            end_date = jdatetime.date(int(yr), int(mn), int(dy)).togregorian()
        today = datetime.datetime.strptime(
            today.split(' ')[0], "%Y-%m-%d").date()
        T = (end_date - today).days / 365

        # Calculate Greeks
        d1 = (np.log(S0 / K) + (rf + 0.5 * sigma ** 2) * T) / \
            (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate Delta, Gamma, Vega, Theta, Rho
        delta = norm.cdf(d1) * (1 if is_call else -1)
        gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        if is_call:
            theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - \
                rf * K * np.exp(-rf * T) * norm.cdf(d2)
        else:
            theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + \
                rf * K * np.exp(-rf * T) * norm.cdf(-d2)

        rho = K * T * np.exp(-rf * T) * norm.cdf(d2)

        return delta, gamma, vega, theta, rho

    @staticmethod
    def delta_neutral_strategy(delta, num_shares_owned, contract_size=1000):
        """
        This function calculates the number of options to sell to make your portfolio delta neutral.
        """
        num_options_to_sell = num_shares_owned / (delta * contract_size)
        return num_options_to_sell

    def vega_neutral_strategy(vega, num_shares_owned, contract_size=1000):
        """
        This function calculates the number of options to sell to make your portfolio vega neutral.
        """
        num_options_to_sell = num_shares_owned / (vega * contract_size)
        return num_options_to_sell

    def theta_neutral_strategy(theta, num_shares_owned, contract_size=1000):
        """
        This function calculates the number of options to sell to make your portfolio theta neutral.
        """
        num_options_to_sell = num_shares_owned / (theta * contract_size)
        return num_options_to_sell
