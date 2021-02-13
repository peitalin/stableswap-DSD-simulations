

import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from curve_amm import get_D, get_y
from tax_functions import quadratic_tax, linear_tax, no_tax
from uniswap_amm import Uniswap
import stableswap_plots


def generate_trade(mu, sigma):
    rv = np.random.normal(mu, sigma)
    if rv >= 0:
        return dict({ 'type': "buy", 'amount': rv })
    else:
        return dict({ 'type': "sell", 'amount': rv })



if __name__=="__main__":
    print("DSD DIP-14 Simulations!")



fig, ax = plt.subplots()

mu = -500
sigma = 500
nobs = 1000

# DSD initial price: $0.2
# quadratic_tax
for i in range(200):
    u = Uniswap(100000,500000)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=quadratic_tax) for x in trades]
    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller
    ax.plot(
        np.linspace(0,1000,nobs+1),
        u.history['prices'],
        color="purple",
        alpha=0.06
    )


# linear_tax
for i in range(200):
    u = Uniswap(100000,500000)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=linear_tax) for x in trades]

    ax.plot(
        np.linspace(0,1000,nobs+1),
        u.history['prices'],
        color="blue",
        alpha=0.06
    )


# no_atx
for i in range(200):
    u = Uniswap(100000,500000)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=no_tax) for x in trades]

    ax.plot(
        np.linspace(0,1000,nobs+1),
        u.history['prices'],
        color="black",
        alpha=0.06
    )



plt.title("Simulating sales taxes on Uniswap AMMs")
plt.xlabel("number of trades")
plt.ylabel("Price: DSD/USDC")
# place a text box in upper left in axes coords
ax.text(
    700, .22,
    r'''
    500 runs of 1000 trades sampled from a
    $X \sim N(\mu=${mu},$\sigma$={sigma}) distribution.

    Initial price: 0.2 DSD/USDC
    Initial LP: 100,000 USDC / 500,000 DSD
    '''.format(mu=mu, sigma=sigma),
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
]
ax.legend(handles=legend_elements, loc='upper left')


## The random samples are sampling with replacement, reality with the burns is....sampling without replacement (since it gets burnt away)










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





### Curve Swap
## call "exchange" function

