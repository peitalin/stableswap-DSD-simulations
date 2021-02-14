

import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from curve_amm import get_D, get_y, stableswap_y
from uniswap_amm import Uniswap, uniswap_y, uniswap_x, linear_y
from tax_functions import quadratic_tax, linear_tax, no_tax
# curve stableswap plots
from stableswap_plots import plot_fig1_fig2



def generate_trade(mu, sigma):
    rv = np.random.normal(mu, sigma)
    if rv >= 0:
        return dict({ 'type': "buy", 'amount': rv })
    else:
        return dict({ 'type': "sell", 'amount': rv })



if __name__=="__main__":
    print("DSD DIP-14 Simulations!")





mu = -1000
sigma = 4000
nobs = 2000
plot_variate = 'prices'
# plot_variate = 'treasury_balances'
# plot_variate = 'burns'

# DSD initial price: $0.2
lp_initial_usdc = 1000000
lp_initial_dsd  = 5000000
colors = [
    "dodgerblue",
    "mediumorchid",
    "crimson",
    "black",
]
alpha_opacity = 0.05
num_iterations = 200



avg_prices = dict({
    "quadratic_tax": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax": [],
})
avg_burns = dict({
    "quadratic_tax": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax": [],
})
avg_treasury_balances = dict({
    "quadratic_tax": [],
    "linear_tax": [],
    "no_tax": [],
    "slippage_tax": [],
})


fig, ax = plt.subplots()

# quadratic_tax
for i in range(num_iterations):
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=quadratic_tax) for x in trades]

    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller

    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color='mediumorchid',
        alpha=alpha_opacity,
    )

    tax_style = 'quadratic_tax';

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
            color='mediumorchid',
            alpha=1,
            linewidth=2,
            linestyle="dotted",
        )



# linear_tax
for i in range(num_iterations):
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=linear_tax) for x in trades]

    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color='dodgerblue',
        alpha=alpha_opacity,
    )

    tax_style = 'linear_tax';

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
            color='dodgerblue',
            alpha=1,
            linewidth=2,
            linestyle="dotted",
        )


# no_atx
for i in range(num_iterations):
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=no_tax) for x in trades]


    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color = "black",
        alpha=alpha_opacity
    )

    tax_style = 'no_tax';

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
            color="black",
            alpha=1,
            linewidth=2,
            linestyle="dotted",
        )


# slippage_tax
for i in range(num_iterations):
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function="slippage") for x in trades]
    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller

    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color='crimson',
        alpha=alpha_opacity,
    )

    tax_style = 'slippage_tax';

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
            color='crimson',
            alpha=1,
            linewidth=2,
            linestyle="dotted",
        )



plt.title("Simulating sales taxes on Uniswap AMMs")
plt.xlabel("number of trades")
plt.ylabel("Price: DSD/USDC")
# place a text box in upper left in axes coords
ax.text(
    1550, .22,
    r'''
    800 runs of {nobs} trades sampled from a
    $X \sim N(\mu=${mu},$\sigma$={sigma}) distribution.

    Initial price: 0.2 DSD/USDC
    Initial LP: 100,000 USDC / 500,000 DSD
    '''.format(runs=4*num_iterations, nobs=nobs, mu=mu, sigma=sigma),
    {'color': 'black', 'fontsize': 8},
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)


legend_elements = [
    Line2D([0], [0], color='purple', lw=2,
           label=r'$(1-price)^2 \times DSD_{sold}$'),
    Line2D([0], [0], color='blue', lw=2,
           label=r'$(1-price) \times DSD_{sold}$'),
    Line2D([0], [0], color='black', lw=2,
           label=r'$0 \times DSD_{sold}$'),
    Line2D([0], [0], color='red', lw=2,
           label=r'$(1 - slippage) \times DSD_{sold}$'),
]
ax.legend(handles=legend_elements, loc='upper left')

## Even with a slight negative bias, mean = -100, the burns push the price upward slowly over time

## The random samples are sampling with replacement, reality with the burns is....sampling without replacement (since it gets burnt away)

# u.ohlc_plot(100)









# not all scalping is bad
# For example, scalping/arbitrage between liqduity pools is good
# what is bad is scalpers contributing to slippage, causing out dollar to go off-peg...that is what should be taxed

# This sales tax is really a slippage tax, levied on sellers who sell at the wrong times. Since the goal of a algostablecoin is to remain on-peg, any actions individuals take to destabilize the peg should be tax/punished. Selling while on peg is fine.

# e.g. with this mechanic, arbitrage between an off-peg sushi pool at $0.9 and an on-peg curve pool at $1 would not incur slippage, the arbitrague is not taxed much. He has fulfilled a positive role in price discovery by bringing prices in the sushi pool closer to $1, without imparting slippage into the curve pool

# what these dis-coordination motives bring is......an equilibrium level of disequilibrium

# "We need some form of supply control that handles excess supply"
# excess supply is a problem, when it is sold and contributes to destabilization, order flow is negative
# excess supply is not a problem when it is locked, or used in transactions where order flow is random and neutral (sideways action)....this is what adoption looks like



###### Bonded LP
# Basically acts as semi-permanently locked liquidity, similar to CORE. CORE's permanent liquidity allows them to calculate a price floor, guaranteeing the price will never go below floor (how?)
# as CORE liquidity grows, fees accrue which form part of the price floow
# In practice, CORE has done remarkably well as a $4000 stable coin
# The key mechanics are taxing 1% all transfers, and redistributign this as APY to locked liquidity holders

## The DSD analogue would be redistributing sales taxes to LPs,






## On the role of expansion
# Expansion shouldn't be somethign we achieve by gathering a few whales together and pumping prices artificially
# ALl this does it create temporary APY and yields that later needs to be dumped on a market not ready to take that supply

# Expansions should be the natural consequnce of adoption....through collateral used on other platforms, or trasacting.
# Expansions should be attained as pent up demand for the coin slowly starts to push the AMM far along to the top regions where positive slippage starts to occur, forcing the protocol to expand...an actual shortage of coins



#
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
