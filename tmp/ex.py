"""This module does blah blah."""

from datetime import datetime
import pandas as pd
import numpy as np
import pydtmc
import yfinance as yf


# define ticker
TICKER = "TSLA"
# today
today = datetime.today().strftime("%Y-%m-%d")
# 1000 days ego
start = (datetime.today() - pd.Timedelta(days=1000)).strftime("%Y-%m-%d")
# get TSLA data from start to today
data_raw = yf.download(TICKER, start=start, end=today, progress=False)

# filter Adj Close column and compute daily return
data = data_raw[["Adj Close"]].copy()
data.loc[:, "Return"] = data["Adj Close"].pct_change()
data = data[1:]

# define 3 states for the `Return` column: Bull, Bear and Cons
data.loc[:, "State"] = pd.cut(
    data["Return"],
    bins=[-np.inf, -0.001, 0.001, np.inf],
    labels=["Bear", "Cons", "Bull"],
)
# add `priorState` column
data.loc[:, "priorState"] = data["State"].shift(1)
# drop nan values
data = data.dropna()
# define states_matrix based on `State` and `priorState` columns
states_matrix = pd.crosstab(data["priorState"], data["State"], normalize="index")

# create a markov chain based on states_matrix
mc = pydtmc.MarkovChain(states_matrix.values, states_matrix.index.tolist())
# print mc attributes
print("model absorbing states:", mc.absorbing_states)
print("is model ergodic?", mc.is_ergodic)
print("model recurrent states:", mc.recurrent_states)
print("model transient states:", mc.transient_states)
print("model steady state:", mc.steady_states)

# plots
pydtmc.plot_graph(mc)
pydtmc.plot_eigenvalues(mc)
pydtmc.plot_sequence(mc, 10, plot_type="histogram")
pydtmc.plot_sequence(mc, 10, plot_type="heatmap")
pydtmc.plot_sequence(mc, 10, plot_type="matrix")
pydtmc.plot_redistributions(mc, 10, plot_type="heatmap")
pydtmc.plot_redistributions(mc, 10, plot_type="projection")
