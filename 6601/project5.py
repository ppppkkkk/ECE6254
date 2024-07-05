import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Updated parameters for a 1-year simulation
T = 1           # time horizon in years
mu = 0.05       # expected return
sigma = 0.2     # volatility
dt = 1/252      # time increment (assuming 252 trading days in a year)
N = round(T/dt) # number of time steps
n_scenarios = 10 # number of scenarios to simulate

# Update the array for stock prices, initial prices set to S0
stock_prices = np.zeros((N + 1, n_scenarios))
stock_prices[0] = S0

# Simulate the scenarios for 1 year
for t in range(1, N + 1):
    z = np.random.standard_normal(n_scenarios) # random numbers for stochastic process
    stock_prices[t] = stock_prices[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

# Plot the simulations for 1 year
plt.figure(figsize=(10,6))
for i in range(n_scenarios):
    plt.plot(stock_prices[:, i], lw=2)

plt.title("GBM Model 1-Year Stock Price Simulation")
plt.xlabel("Time (days)")
plt.ylabel("Stock Price ($)")
plt.show()
