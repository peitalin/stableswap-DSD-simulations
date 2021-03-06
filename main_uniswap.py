import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt

from src.curve_amm import Curve, get_y, stableswap_y, stableswap_x
from src.uniswap_amm import Uniswap, uniswap_y, uniswap_x, linear_y
from src.tax_functions import quadratic_tax, linear_tax, no_tax, logistic_tax, linear_logistic_tax
from src.random import generate_trade
from src.time_series_data import create_time_series_data_store





if __name__=="__main__":
    print("DSD DIP-14 Uniswap Simulations!")


# Create data structures to hold simulation time series data
data_stores = create_time_series_data_store()
avg_prices = data_stores['avg_prices']
avg_burns = data_stores['avg_burns']
avg_treasury_balances = data_stores['avg_treasury_balances']
colors = data_stores['colors']


mu = 0
sigma = 5000
nobs = 10000
plot_variate = 'prices'
# plot_variate = 'treasury_balances'
# plot_variate = 'burns'

# DSD initial price: $0.1
lp_initial_usdc = 1_000_000
lp_initial_dsd  = 10_000_000
alpha_opacity = 0.05
num_iterations = 50


########## START UNISWAP PLOTS ############

fig, ax = plt.subplots()

# quadratic_tax
for i in range(num_iterations):
    print('Quadratic tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]

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



# linear_tax
for i in range(num_iterations):
    print('Linear tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=linear_tax) for x in trades]

    tax_style = 'linear_tax_uni';

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


# no_atx
for i in range(num_iterations):
    print('No tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=no_tax) for x in trades]

    tax_style = 'no_tax_uni';

    ax.plot(
        np.linspace(0,nobs,nobs+1),
        u.history[plot_variate],
        color=colors[tax_style],
        alpha=alpha_opacity
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


# logistic_tax
for i in range(num_iterations):
    print('Logistic tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=logistic_tax) for x in trades]
    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller

    tax_style = 'logistic_tax_uni';

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



# linear_logistic_tax
for i in range(num_iterations):
    print('Linear Logistic tax iteration: ', i)
    u = Uniswap(lp_initial_usdc, lp_initial_dsd)
    trades = [generate_trade(mu, sigma) for x in range(nobs)]
    _ = [u.swap(x, tax_function=linear_logistic_tax) for x in trades]
    # notes: Average trade is -1000, but DSD prices generally
    # end up increasing because of the burn + slippage working against a seller

    tax_style = 'linear_logistic_tax_uni';

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

# slippage_tax
# for i in range(num_iterations):
#     print('Slippage tax iteration: ', i)
#     u = Uniswap(lp_initial_usdc, lp_initial_dsd)
#     trades = [generate_trade(mu, sigma) for x in range(nobs)]
#     _ = [u.swap(x, tax_function="slippage") for x in trades]
#     # notes: Average trade is -1000, but DSD prices generally
#     # end up increasing because of the burn + slippage working against a seller
#
#     tax_style = 'slippage_tax_uni';
#
#     ax.plot(
#         np.linspace(0,nobs,nobs+1),
#         u.history[plot_variate],
#         color=colors[tax_style],
#         alpha=alpha_opacity,
#     )
#
#     if i == 0:
#         avg_prices[tax_style] = u.history['prices']
#         avg_burns[tax_style] = u.history['burns']
#         avg_treasury_balances[tax_style] = u.history['treasury_balances']
#     else:
#         # accumulate prices, divide by num_iterations after
#         avg_prices[tax_style] = np.add(
#             avg_prices[tax_style],
#             u.history['prices']
#         )
#         avg_burns[tax_style] = np.add(
#             avg_burns[tax_style],
#             u.history['burns']
#         )
#         avg_treasury_balances[tax_style] = np.add(
#             avg_treasury_balances[tax_style],
#             u.history['treasury_balances']
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



plt.title("Simulating sales taxes on Uniswap AMMs")
plt.xlabel("number of trades")
plt.ylabel("Price: DSD/USDC")
# place a text box in upper left in axes coords
ax.text(
    6000, .15,
    r'''
    {runs} runs of {nobs} trades sampled from a
    $X \sim N(\mu=${mu},$\sigma$={sigma}) distribution.

    Initial price: 0.1 DSD/USDC
    Initial LP: 1,000,000 USDC / 10,000,000 DSD
    '''.format(runs=4*num_iterations, nobs=nobs, mu=mu, sigma=sigma),
    {'color': 'black', 'fontsize': 8},
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)


legend_elements = [
    Line2D([0], [0], color=colors['quadratic_tax_uni'], lw=2,
           label=r'$(1-price)^2 \times DSD_{sold}$ (quadratic_tax)'),
    Line2D([0], [0], color=colors['linear_tax_uni'], lw=2,
           label=r'$(1-price) \times DSD_{sold}$ (linear_tax)'),
    Line2D([0], [0], color=colors['slippage_tax_uni'], lw=2,
           label=r'$1/(1 - e^{price-0.5}) \times DSD_{sold}$ (logistic_tax)'),
    Line2D([0], [0], color=colors["no_tax_uni"], lw=2,
           label=r'$0 \times DSD_{sold}$ (no_tax)'),
    # Line2D([0], [0], color=colors['quadratic_tax_curve'], lw=2,
    #        label=r'Curve quadratic tax'),
    # Line2D([0], [0], color=colors['slippage_tax_curve'], lw=2,
    #        label=r'Curve slippage tax'),
    Line2D([0], [0], color=colors['linear_logistic_tax_uni'], lw=2, label=r'$(1 - price)*logistic + (price)*linear$ (linear_logistic_tax)'),
]
ax.legend(handles=legend_elements, loc='upper left')

## Even with a slight negative bias, mean = -100, the burns push the price upward slowly over time

## The random samples are sampling with replacement, reality with the burns is....sampling without replacement (since it gets burnt away)


## Generate candlestick chart of last simulation timeseries
## lets you see what price action might look like
# u.ohlc_plot(100)



def plot_treasury_balances():

    fig, ax = plt.subplots()

    for tax_style in avg_treasury_balances.keys():
        if tax_style not in [
            "quadratic_tax_uni",
            "linear_tax_uni",
            "no_tax_uni",
            "logistic_tax_uni"
        ]:
            continue

        ax.plot(
            np.linspace(0,nobs,nobs+1),
            avg_treasury_balances[tax_style],
            color=colors[tax_style],
            alpha=1,
            linewidth=2,
            linestyle="dotted",
        )

    plt.title("Treasury funds from sales taxes")
    plt.xlabel("number of trades")
    plt.ylabel("Treasury balance (millions DSD)")
    legend_elements = [
        Line2D([0], [0], color=colors['quadratic_tax_uni'], lw=2,
               label=r'$(1-price)^2 \times DSD_{sold}$ (quadratic_tax)'),
        Line2D([0], [0], color=colors['linear_tax_uni'], lw=2,
               label=r'$(1-price) \times DSD_{sold}$ (linear_tax)'),
        Line2D([0], [0], color=colors['logistic_tax_uni'], lw=2,
               label=r'$(1 - e^{price-0.5}) \times DSD_{sold}$ (logistic_tax)'),
        Line2D([0], [0], color=colors["no_tax"], lw=2,
               label=r'$0 \times DSD_{sold}$ (no_tax)'),
        # Line2D([0], [0], color=colors['quadratic_tax_curve'], lw=2,
        #        label=r'Curve quadratic tax'),
        # Line2D([0], [0], color=colors['slippage_tax_curve'], lw=2,
        #        label=r'Curve slippage tax'),
        # Line2D([0], [0], color=colors['linear_logistic_tax'], lw=2,
        #        label=r'$(1 - price)*logistic + (price)*linear$ - linear_logistic_tax'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')


