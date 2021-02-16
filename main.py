import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from src.curve_amm import Curve, get_y, stableswap_y, stableswap_x
from src.uniswap_amm import Uniswap, uniswap_y, uniswap_x, linear_y
from src.tax_functions import quadratic_tax, linear_tax, no_tax
from src.random import generate_trade


from src.curve_amm import Curve, _xp, stableswap_y, stableswap_x
# c = Curve(100000, 500000, A=20)
# stableswap_x( 100000 + 1, [100000, 100000], 2000)
# (100001-100000)/(99999.04761917747 - 100000)

# 0.9895494479
# 0.9894878188482094
# 0.9898556552223209

c = Curve(100000, 100000, A=20)
c.swap(({ 'type': "sell", "amount": 10000 }), tax_function=no_tax)
c.swap(({ 'type': "buy", "amount": 10000 }), tax_function=no_tax)


avg_prices = dict({
    "quadratic_tax_uni": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax_uni": [],
    "slippage_tax_curve": [],
    "quadratic_tax_curve": [],
})
avg_burns = dict({
    "quadratic_tax_uni": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax_uni": [],
    "slippage_tax_curve": [],
    "quadratic_tax_curve": [],
})
avg_treasury_balances = dict({
    "quadratic_tax_uni": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax_uni": [],
    "slippage_tax_curve": [],
    "quadratic_tax_curve": [],
})


if __name__=="__main__":
    print("DSD DIP-14 Simulations!")





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
num_iterations = 50



########## CURVE ##################

fig, ax = plt.subplots()

# Curve quadratic tax
for i in range(num_iterations):
    c = Curve(lp_initial_usdc, lp_initial_dsd, A=100)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    # trades = [x for x in _trades if x['type'] == 'sell']
    _ = [c.swap(x, tax_function=no_tax) for x in trades]
    # _ = [c.swap(x, tax_function=quadratic_tax) for x in trades]

    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller
    tax_style = 'quadratic_tax_curve'

    ax.plot(
        np.linspace(0,len(trades),len(trades)+1),
        c.history[plot_variate],
        color=colors[tax_style],
        alpha=alpha_opacity,
    )

    if i == 0:
        avg_prices[tax_style] = c.history['prices']
        avg_burns[tax_style] = c.history['burns']
        avg_treasury_balances[tax_style] = c.history['treasury_balances']
    else:
        # accumulate prices, divide by num_iterations after
        avg_prices[tax_style] = np.add(
            avg_prices[tax_style],
            c.history['prices']
        )
        avg_burns[tax_style] = np.add(
            avg_burns[tax_style],
            c.history['burns']
        )
        avg_treasury_balances[tax_style] = np.add(
            avg_treasury_balances[tax_style],
            c.history['treasury_balances']
        )

    if i == (num_iterations - 1):
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


# # Curve slippage tax
# for i in range(num_iterations):
#     c = Curve(lp_initial_usdc, lp_initial_dsd)
#     trades = [generate_trade(mu, sigma) for x in range(nobs)]
#     _ = [c.swap(x, tax_function='slippage') for x in trades]
#
#     # notes: Average trade is -1000, but DSD prices generally
#     # end up increasing because of the burn + slippage working against a seller
#     tax_style = 'slippage_tax_curve'
#
#     ax.plot(
#         np.linspace(0,nobs,nobs+1),
#         c.history[plot_variate],
#         color=colors[tax_style],
#         alpha=alpha_opacity,
#     )
#
#     if i == 0:
#         avg_prices[tax_style] = c.history['prices']
#         avg_burns[tax_style] = c.history['burns']
#         avg_treasury_balances[tax_style] = c.history['treasury_balances']
#     else:
#         # accumulate prices, divide by num_iterations after
#         avg_prices[tax_style] = np.add(
#             avg_prices[tax_style],
#             c.history['prices']
#         )
#         avg_burns[tax_style] = np.add(
#             avg_burns[tax_style],
#             c.history['burns']
#         )
#         avg_treasury_balances[tax_style] = np.add(
#             avg_treasury_balances[tax_style],
#             c.history['treasury_balances']
#         )
#
#     if i == (num_iterations - 1):
#         # last loop, average over all iterations
#         avg_prices[tax_style] = np.divide(
#             avg_prices[tax_style],
#             num_iterations
#         )
#         avg_burns[tax_style] = np.divide(
#             avg_burns[tax_style],
#             num_iterations
#         )
#         avg_treasury_balances[tax_style] = np.divide(
#             avg_treasury_balances[tax_style],
#             num_iterations
#         )
#         # Then plot mean price line with alpha=1
#         ax.plot(
#             np.linspace(0,nobs,nobs+1),
#             avg_prices[tax_style],
#             color=colors[tax_style],
#             alpha=1,
#             linewidth=2,
#             linestyle="dotted",
#         )
# ########## END CURVE ##################







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

    if i == (num_iterations - 1):
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

## Even with a slight negative bias, mean = -100, the burns push the price upward slowly over time

## The random samples are sampling with replacement, reality with the burns is....sampling without replacement (since it gets burnt away)

## For plotting candlestick charts of the last simulation
# u.ohlc_plot(100)














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

