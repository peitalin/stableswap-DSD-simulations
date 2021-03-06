import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from src.curve_amm import Curve, get_y, stableswap_y, stableswap_x, _xp
from src.uniswap_amm import Uniswap, uniswap_y, uniswap_x, linear_y
from src.tax_functions import quadratic_tax, linear_tax, no_tax, logistic_tax, linear_logistic_tax
from src.random import generate_trade, bayes_update_normal
from src.time_series_data import create_time_series_data_store


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



if __name__=="__main__":
    print("DSD DIP-14 Simulations!")



# Create data structures to hold simulation time series data
data_stores = create_time_series_data_store()
avg_prices = data_stores['avg_prices']
avg_burns = data_stores['avg_burns']
avg_treasury_balances = data_stores['avg_treasury_balances']
colors = data_stores['colors']


mu = 0
sigma = 1000
nobs = 5000
plot_variate = 'prices'

# DSD initial price: $0.2
lp_initial_usdc = 1000000
lp_initial_dsd  = 5000000
alpha_opacity = 0.1
num_iterations = 50






fig, ax = plt.subplots()
mu_1 = mu # initial value of mean
sigma_1 = sigma # initial value of sigma

########## START UNISWAP ##################
# quadratic_tax
for i in range(num_iterations):

    print('Quadratic tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)

    # divide each time series into 10 lots,
    # 10 updates to the trade generating distribution
    for j in range(100):
        # print('bucket: ', j)
        print('mu_1: ', mu_1)
        # print('sigma_1: ', sigma_1)
        # generate 1/10 of trades with this mu & sigma
        trades = [generate_trade(mu_1, sigma_1) for x in range(nobs//100)]
        trades2 = [t['amount'] for t in trades]
        # generate prices for 1/10 of trades
        j_prices = [u.swap(x, tax_function=quadratic_tax) for x in trades]

        # calculate mean, stdev for recent rades
        mu_2 = np.mean(j_prices) * 1000
        # sigma_2 = np.std(trades_std)

        # update trade Normal distribution with
        # new mean and sigma from recent trades
        # posterior_distribution = bayes_update_normal(
        #     sigma_1,
        #     sigma_2,
        #     mu_1,
        #     mu_2,
        # )
        # mu_1 = posterior_distribution['mu']

        # prevent mu_2 from exploding upwards
        if mu_2 > 1000:
            mu_1 = 1000
        else:
            mu_1 = mu_2



    tax_style = 'quadratic_tax_uni_bayesian'

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

    if i == (num_iterations - 1):
        # last loop, average over all iterations
        average_over_timeseries(
            tax_style,
            num_iterations,
            ax
        )


























for i in range(num_iterations):
    print('Logistic tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=quadratic_tax) for x in trades]

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






plt.title("Sales taxes on Uniswap with dynamic traders")
plt.xlabel("number of trades")
plt.ylabel("Price: DSD/USDC")
# place a text box in upper left in axes coords
ax.text(
    100, .3,
    r'''
    {runs} runs of {nobs} trades sampled from a
    $X \sim N(\mu=${mu},$\sigma$={sigma}) distribution.

    where $\mu = price_j$ * 1000, a function of price

    Sales tax is quadratic_tax
    Initial price: 0.2 DSD/USDC
    Initial LP: 1,000,000 USDC / 5,000,000 DSD
    '''.format(runs=4*num_iterations, nobs=nobs, mu=mu, sigma=sigma),
    {'color': 'black', 'fontsize': 8},
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)


legend_elements = [
    Line2D([0], [0], color=colors['quadratic_tax_uni_bayesian'], lw=2,
           label=r'Bayesian traders - dynamic $\mu$'),
    Line2D([0], [0], color=colors['quadratic_tax_uni'], lw=2,
           label=r'Naive traders - static $\mu = 0$'),
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

