import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from src.curve_amm import Curve, get_y, stableswap_y, stableswap_x, _xp
from src.uniswap_amm import Uniswap, uniswap_y, uniswap_x, linear_y
from src.tax_functions import quadratic_tax, linear_tax, no_tax
from src.random import generate_trade
from src.time_series_data import create_time_series_data_store



if __name__=="__main__":
    print("DSD DIP-14 Simulations!")



# Create data structures to hold simulation time series data
data_stores = create_time_series_data_store()
avg_prices = data_stores['avg_prices']
avg_burns = data_stores['avg_burns']
avg_treasury_balances = data_stores['avg_treasury_balances']
colors = data_stores['colors']


mu = -1000
sigma = 8000
nobs = 8000
plot_variate = 'prices'
# plot_variate = 'treasury_balances'
# plot_variate = 'burns'

# DSD initial price: $0.2
lp_initial_usdc = 1000000
lp_initial_dsd  = 5000000
colors = dict({
    "quadratic_tax_uni": "dodgerblue",
    "linear_tax": "mediumorchid",
    "no_tax": "black",
    "slippage_tax_uni": "crimson",
    "slippage_tax_curve": "green",
    "quadratic_tax_curve": "orange",
})
alpha_opacity = 0.05
num_iterations = 5



def average_over_timeseries(tax_style, num_iterations, ax):
    """
    averages over all simulation timeseries to produce
    an average timeseries plotline
    """
    # last loop, average over all iterations
    avg_prices[tax_style] = np.divide(
        avg_prices[tax_style],
        num_iterations
    )
    avg_burns[tax_style] = np.divide(
        avg_burns[tax_style],
        num_iterations
    )
    avg_treasury_balances[tax_style] = np.divide(
        avg_treasury_balances[tax_style],
        num_iterations
    )
    # Then plot mean price line with alpha=1
    ax.plot(
        np.linspace(0,nobs,nobs+1),
        avg_prices[tax_style],
        color=colors[tax_style],
        alpha=1,
        linewidth=2,
        linestyle="dotted",
    )



fig, ax = plt.subplots()

########## START UNISWAP ##################
# quadratic_tax
for i in range(num_iterations):
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    # _ = [u.swap(x, tax_function=quadratic_tax) for x in trades]
    _ = [u.swap(x, tax_function=quadratic_tax) for x in trades]

    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller
    tax_style = 'quadratic_tax_uni'

    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color=colors[tax_style],
        alpha=alpha_opacity,
    )

    if i == 0:
        avg_prices[tax_style] = u.history['prices']
        avg_burns[tax_style] = u.history['burns']
        avg_treasury_balances[tax_style] = u.history['treasury_balances']
    else:
        # accumulate prices, divide by num_iterations after
        avg_prices[tax_style] = np.add(
            avg_prices[tax_style],
            u.history['prices']
        )
        avg_burns[tax_style] = np.add(
            avg_burns[tax_style],
            u.history['burns']
        )
        avg_treasury_balances[tax_style] = np.add(
            avg_treasury_balances[tax_style],
            u.history['treasury_balances']
        )

    # last loop, average over all iterations
    if i == (num_iterations - 1):
        average_over_timeseries(
            tax_style,
            num_iterations,
            ax
        )






plt.title("Simulating sales taxes on Uniswap vs Curve AMMs")
plt.xlabel("number of trades")
plt.ylabel("Price: DSD/USDC")
# place a text box in upper left in axes coords
ax.text(
    1550, .22,
    r'''
    {runs} runs of {nobs} trades sampled from a
    $X \sim N(\mu=${mu},$\sigma$={sigma}) distribution.

    Initial price: 0.2 DSD/USDC
    Initial LP: 100,000 USDC / 500,000 DSD
    '''.format(runs=4*num_iterations, nobs=nobs, mu=mu, sigma=sigma),
    {'color': 'black', 'fontsize': 8},
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)


legend_elements = [
    Line2D([0], [0], color=colors['quadratic_tax_uni'], lw=2,
           label=r'$(1-price)^2 \times DSD_{sold}$'),
    # Line2D([0], [0], color=colors['linear_tax'], lw=2,
    #        label=r'$(1-price) \times DSD_{sold}$'),
    # Line2D([0], [0], color=colors['slippage_tax_uni'], lw=2,
    #        label=r'$(1 - slippage) \times DSD_{sold}$'),
    # Line2D([0], [0], color=colors["no_tax"], lw=2,
    #        label=r'$0 \times DSD_{sold}$'),
    Line2D([0], [0], color=colors['quadratic_tax_curve'], lw=2,
           label=r'Curve quadratic tax'),
    Line2D([0], [0], color=colors['slippage_tax_curve'], lw=2,
           label=r'Curve slippage tax'),
]
ax.legend(handles=legend_elements, loc='upper left')



## The random samples are sampling with replacement, reality with the burns is....sampling without replacement (since it gets burnt away)















# import pymc3 as pm
# # True parameter values
# alpha, sigma = 1, 1
# beta = [1, 2.5]
#
# # Size of dataset
# size = 4000
#
# # Simulate outcome variable
# Y = alpha + beta[0] * x1 + beta[1] * x2 + np.random.randn(size) * sigma
#
# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
# axes[0].scatter(x1, Y, alpha=0.6)
# axes[1].scatter(x2, Y, alpha=0.6)
# axes[0].set_ylabel("Y")
# axes[0].set_xlabel("x1")
# axes[1].set_xlabel("k2");
#
# basic_model = pm.Model()
#
# with basic_model:
#
#     # Priors for unknown model parameters
#     alpha = pm.Normal("alpha", mu=0, sigma=10)
#     beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
#     sigma = pm.HalfNormal("sigma", sigma=1)
#
#     # Expected value of outcome
#     mu = alpha + beta[0] * x1 + beta[1] * x2
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
#
# map_estimate = pm.find_MAP(model=basic_model)
#
# with basic_model:
#     # draw 500 posterior samples
#     trace = pm.sample(500, return_inferencedata=False)
#

