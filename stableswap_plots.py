import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.curve_amm import get_y, stableswap_y, get_D, stableswap_x
from src.uniswap_amm import uniswap_y, uniswap_x, linear_y

#######################################
## Curve's Stableswap Whitepaper plots
#######################################

## Whitepaper:
## https://www.curve.fi/stableswap-paper.pdf

## Code implementation:
## https://github.com/curvefi/curve-contract/blob/295e7daaad0654a6c7a233f77e82a01fb78d85b4/contracts/pools/usdt/StableSwapUSDT.vy#L183

## Derivation of code implementation from whitepaper formula can be found here:
## https://docs.google.com/document/d/1ujHQk4c9SgotgEkIZFS0oYrOYd7_eexRYh1dg9emzbo/edit




def find_peg_point(x1: list[float], y1: list[float]) -> int:
    """find index where x - y is smallest, which is where balances of x-coins and y-coins in the LP pool are equal"""
    return np.argmin([np.abs(x - y) for x,y in zip(x1, y1)])


def dydx_array(yy: list[float], xx: list[float], absolute=True) -> list[float]:
    """calculates an array of derivatives"""
    assert len(xx) == len(yy)
    dydx_array: list[float] = np.abs(np.diff(yy)/np.diff(xx))
    return np.abs(dydx_array) if absolute else dydx_array



def plot_fig1():

    NUM_OBS = 8000

    ##### Fig. 1 In Stableswap whitepaper
    x1 = np.linspace(0.01, 30, NUM_OBS)
    y1 = [uniswap_y(x, 25) for x in x1]

    x2 = np.linspace(0.01, 10, NUM_OBS)
    y2 = [linear_y(x, 10) for x in x2]

    # x3 = np.linspace(0.01, 1000000, NUM_OBS)
    # xp = [100000,500000]
    x3 = np.linspace(0.01, 30, NUM_OBS)
    xp = [5,5]
    y3 = [stableswap_y(x, xp, 20) for x in x3]

    plt.figure(figsize=[4.75,3])
    # plt.figure(figsize=[4,4])
    plt.plot(x1, y1, color='purple', linestyle='dashed')
    plt.plot(x2, y2, color='red', linestyle='dotted')
    plt.plot(x3, y3, color='blue')
    # plt.axis([0, 1000000, 0, 1000000])
    plt.axis([-0.05, 30, -0.05, 25])






def plot_fig2():

    ##### Fig. 2 In Stableswap whitepaper
    NUM_OBS = 4000

    ## Uniswap plot
    x1 = np.linspace(0.01, 30, NUM_OBS)
    y1 = [uniswap_y(x, 196) for x in x1]
    # Get derivatives of the curve to plot
    dydx1 = dydx_array(y1, x1)
    peg_index1 = find_peg_point(x1, y1)
    # shifts graph to past peg point
    dx1 = [x-14 for x in x1]


    ## Curve plot

    fig, ax = plt.subplots(figsize=[6,4])

    peg_point: int = 240

    x3 = np.linspace(0.001, 2000+peg_point, NUM_OBS*10)
    dx3 = [x-peg_point for x in x3]
    # balances = [5,5]
    # xp = _xp([1, 1]) # -> [5,5]
    y3 = [stableswap_y(x, [700,400], 100) for x in x3]
    peg_index3: int = find_peg_point(x3, y3)
    # dydx3 = dydx_array(y3, x3)
    dydx3 = [get_D([y,x], 100) / (x+y) for x,y in zip(x3, y3)]


    ax.plot(
        dx1[peg_index1+1:],
        dydx1[peg_index1:],
        color='purple',
        linestyle='dashed',
    )
    ax.plot(
        np.divide(dx3, 200), # rescale results
        dydx3,
        color="blue",
    )

    # plt.axis([0, 10, 0.3, 1.05])


    ###### Sales Taxes on Curve AMMs

    # # proportional slippage sales tax
    # slippageDx2 = [(1-x)  for x in dydx3]

    # quadratic slippage sales tax, 50% max sales tax
    # slippageDx3 = [(1 + (1-x)**2)**1/3 - 1/3  for x in dydx3]

    percentage_bonded = 0.12
    # sales tax based on percentage DAO bonded
    # i.e a "global" coordination game
    slippageDx3 = [(1-x)*(1-percentage_bonded)  for x in dydx3]

    # experiment with coordination-games driven yields
    # the more people stay bonded, the greate the yield
    buyerBonusDx3 = [x*(1-percentage_bonded) for x in slippageDx3]

    ## Dynamic Sales Tax, slippage scaled
    ax.plot(
        np.divide(dx3, 200), # rescale results
        slippageDx3,
        color="green",
    )
    ## Dynamic Buyers bonus, slippage scaled, 50% of sales
    ax.plot(
        np.divide(dx3, 200), # rescale results
        buyerBonusDx3,
        color="green",
        lineStyle="dashed"
    )

    ax.axis([0, 10, -0.1, 1.1])
    # plt.axis([0, 10, 0, 30])

    plt.title("Curve slippage and Sales taxes")
    plt.ylabel("Price DSD/USDC")
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2,
               label=r'Curve slippage'),
        Line2D([0], [0], color='purple', lw=2,
               label=r'Uniswap slippage', lineStyle="dotted"),
        Line2D([0], [0], color='green', lw=2,
               label=r'Curve sales tax'),
        Line2D([0], [0], color='green', lw=2,
               lineStyle="dotted",
               label=r'tax diverted to buy/bonding rewards'),
    ]
    ax.legend(handles=legend_elements, loc='center left')



def plot_fig3():
    ### For testing purposes only

    ##### Fig. 2 In Stableswap whitepaper
    ##### using small numbers
    NUM_OBS = 4000

    ## Uniswap plot
    x1 = np.linspace(0.01, 30, NUM_OBS)
    y1 = [uniswap_y(x, 25) for x in x1]
    # Get derivatives of the curve to plot
    dydx1 = dydx_array(y1, x1)
    peg_index1 = find_peg_point(x1, y1)
    # shifts graph to past peg point
    dx1 = [x-5 for x in x1]


    ## Curve plot


    peg_point: int = 0

    x3 = np.linspace(0.01, 30, NUM_OBS)
    xp = [5,5]
    y3 = [stableswap_y(x, xp, 90) for x in x3]
    dx3 = [x-peg_point for x in x3]

    peg_index3: int = find_peg_point(x3, y3)
    dydx3 = dydx_array(y3, x3)
    dydx3_ = [get_D([y,x], 100) / (x+y) for x,y in zip(x3, y3)]

    fig, ax = plt.subplots(figsize=[6,4])
    ax.plot(
        dx1[peg_index1+1:],
        dydx1[peg_index1:],
        color='purple',
        linestyle='dashed',
    )
    ax.plot(
        dx3[1:], # rescale results
        dydx3,
        color="blue",
    )
    ax.plot(
        np.divide(dx3, 2), # rescale results
        dydx3_,
        color="green",
    )

    plt.axis([0, 10, 0, 1.2])





if __name__=="__main__":
    print("Curve Stableswap plots")
    plot_fig1()
    plot_fig2()
