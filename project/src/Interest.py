"""Interest rate models"""

import numpy as np
import matplotlib.pyplot as plt


class HullWhite:
    """Hull-White model class"""

    @staticmethod
    def objective(params, *args) -> float:
        """Objective function for calibration of Hull-White model to market

        Parameters
        ----------
        params : tuple(float, float)
            a, sigma
        args : tuple(array_like, array_like, array_like)
            prices, maturities, ytm

        Returns
        -------
        float
            Sum of squared errors between model prices and market prices
        """

        a, sigma = params
        prices, maturities, ytm = args

        # calculate model prices
        model_prices = np.zeros(len(prices))
        for i in range(len(maturities)):
            B = (1 - np.exp(-a * maturities[i])) / a
            A = np.exp((B - maturities[i]) * (a**2 * ytm[i] -
                       sigma**2 / (2 * a**2)) - sigma**2 * B**2 / (4 * a))
            model_prices[i] = A * np.exp(-B * ytm[i])

        # calculate error
        error = prices - model_prices

        return np.sum(error**2)

    @staticmethod
    def generate_paths(mean_reversion, volatility, initial_rate, maturity, num_paths, num_steps):
        """Generate paths using the Hull-White model

        Parameters
        ----------
        mean_reversion : float
            Mean reversion parameter
        volatility : float
            Volatility parameter
        initial_rate : float
            Initial short rate
        maturity : float
            Maturity of the short rate
        num_paths : int
            Number of paths to generate
        num_steps : int
            Number of time steps per path

        Returns
        -------
        array_like
            Array of paths
        """
        dt = maturity / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        # Set the initial value of the paths to the initial short rate
        paths[:, 0] = initial_rate

        for i in range(num_paths):
            for j in range(1, num_steps + 1):
                # Generate a normally distributed random number
                dW = np.random.normal(0, np.sqrt(dt))

                # Calculate the short rate using the Hull-White model formula
                paths[i, j] = paths[i, j - 1] + mean_reversion * \
                    (initial_rate - paths[i, j - 1]) * dt + volatility * dW

        return paths

    @staticmethod
    def plot_paths(paths, num_steps, maturity):
        """Plot the paths generated by the Hull-White model

        Parameters
        ----------
        paths : array_like
            Array of paths
        num_steps : int
            Number of time steps per path
        maturity : float
            Maturity of the short rate

            Returns
            -------
            None
        """
        time_steps = np.linspace(0, maturity, num=num_steps + 1)
        plt.figure(figsize=(10, 6))

        # Extract the first 60 days of time steps and paths
        time_steps_60 = time_steps[:61]
        paths_60 = paths[:, :61]

        plt.plot(time_steps_60, paths_60.T, lw=1)
        plt.xlabel('Time')
        plt.ylabel('Short Rate')
        plt.title('Hull-White Model Random Paths')
        plt.grid(True)
        plt.show()