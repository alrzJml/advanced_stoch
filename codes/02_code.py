# Import required libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import fsolve

# Define initial parameters for the simulation
current_price = 100.0   # Current asset price
expected_return = 0.20  # Expected annual return on the asset
σ = 0.30                # Annual volatility
p = 0.5                 # Probability for going up
# Time period in years for which the simulation is to be run (1 month in this case)
τ = 1.0/12

# Define the function to be solved for the expected return


def func(µ): return (0.5 * (np.exp(µ*τ + σ*np.sqrt(τ)) +
                            np.exp(µ*τ - σ*np.sqrt(τ)))) ** 12 - (1+expected_return)


# Solve for the expected return
init_point = 0.12
µ = fsolve(func, init_point)[0]

# Calculate the up and down factors
h1 = np.exp(µ*τ + σ * np.sqrt(τ))
h2 = np.exp(µ*τ - σ * np.sqrt(τ))

# Define the function to generate the random walk


def random_pick(p):
    if random.random() < p:
        return 1
    else:
        return -1


# Initialize an empty DataFrame to store the random walk results
df = pd.DataFrame()

# Set the figure size for the plot
plt.figure(figsize=(15, 7))

num_steps = 36  # Number of steps in each random walk
sim_size = 10000  # Number of random walks to simulate

# Run the simulation for 10000 random walks
for i in range(sim_size):

    # Generate random walk steps using the defined probability
    rw_steps = np.array([random_pick(p)
                        for _ in range(num_steps)], dtype=np.float64)
    rw_steps[rw_steps == 1] = h1
    rw_steps[rw_steps == -1] = h2

    # Calculate the random walk values based on the initial price and steps
    rw = current_price * np.cumprod(rw_steps)
    rw = np.insert(rw, 0, current_price)

    # Append the random walk values to the DataFrame
    df = pd.concat([df, pd.DataFrame(rw).T], axis=0)

    # Plot a sample of the smoothed random walks
    if i % 200 == 0:
        # Smooth the random walk values using a Gaussian filter
        ysmoothed = gaussian_filter1d(rw, sigma=2)
        plt.plot(ysmoothed)

# Add a title to the plot and display it
plt.title(f"A Sample of {sim_size} Random Walks")
plt.show()

# Round the values in the DataFrame to 4 decimal places
df_rounded = df.round(4)

# Define the number of divisions for the probability distribution
num_divisions = 25

# Initialize DataFrames to store the results and probabilities
df_res = pd.DataFrame(np.zeros((num_divisions, int((num_divisions + 1) / 2))))
df_prob = pd.DataFrame(np.zeros((num_divisions, int((num_divisions + 1) / 2))))

# Function to get the index range for the DataFrame based on the division and index value


def get_index(num_divisions, index_val):
    start = (num_divisions - 1) / 2 - index_val
    end = (num_divisions - 1) / 2 + index_val + 1
    return range(int(start), int(end), 2)


# Loop through the divisions and populate the results and probability DataFrames
for i in range(int((num_divisions + 1) / 2)):
    index_range = get_index(num_divisions, i)
    df_res.iloc[index_range, i] = df_rounded[i].value_counts(
    ).index.sort_values(ascending=False)
    df_prob.iloc[index_range, i] = df_rounded[i].value_counts(
    ).sort_index(ascending=False).values

# Replace zeros with NaN values in the DataFrames
df_res.replace(0.0, np.nan, inplace=True)
df_prob.replace(0.0, np.nan, inplace=True)

# Round the values in the results DataFrame to 2 decimal places
df_res_rounded = df_res.round(2)

# Function to color the cells in the DataFrame based on intensity values


def color_cells_by_intensity(data, intensity_data, cmap='coolwarm', low=0, high=1):
    def cell_color(row):
        row_index = row.name  # Get the current row index
        cell_colors = []  # Initialize an empty list to store cell colors

        # Loop through the columns and calculate the background color for each cell
        for col_index in row.index:
            intensity = intensity_data.loc[row_index, col_index]
            normalized_intensity = (intensity - intensity_data.min().min()) / (
                intensity_data.max().max() - intensity_data.min().min())
            color = plt.cm.get_cmap(cmap)(normalized_intensity)

            # Append the calculated background color to the cell_colors list
            if color[0] == 0 and color[1] == 0:
                cell_colors.append('')
            else:
                cell_colors.append(
                    f'background-color: rgba({100+color[0]*200.0:.0f}, {100+color[1]*100.0:.0f}, {100+color[2]*100.0:.0f}, {0.5+color[3]*0.5:.2f})')

        return cell_colors

    # Apply the cell_color function to each row in the DataFrame
    return data.round(2).style.apply(cell_color, axis=1)


# Fill NaN values with empty strings in the DataFrame
filled_df_res2 = df_res_rounded.fillna('')

# Color the cells in the DataFrame based on the intensity data (probabilities)
colored_df1 = color_cells_by_intensity(filled_df_res2, df_prob)

# Round the values in the colored DataFrame to 5 significant digits
colored_df1_rounded = colored_df1.format("{:.5}")
# Display the colored DataFrame (Open this in Jupyter Notebook)
colored_df1_rounded
