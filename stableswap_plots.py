import numpy as np
import matplotlib.pyplot as plt
from curve_amm import get_y, stableswap_y
from uniswap_amm import uniswap_y, uniswap_x, linear_y

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


def dydx(yy: list[float], xx: list[float], absolute=True) -> list[float]:
    """calculates an array of derivatives"""
    assert len(xx) == len(yy)
    dydx_array: list[float] = np.abs(np.diff(yy)/np.diff(xx))
    return np.abs(dydx_array) if absolute else dydx_array



def plot_fig1_fig2():
    NUM_OBS = 4000

    ##### Fig. 1 In Stableswap whitepaper
    x1 = np.linspace(0.01, 30, NUM_OBS)
    y1 = [uniswap_y(x, 25) for x in x1]

    x2 = np.linspace(0.01, 10, NUM_OBS)
    y2 = [linear_y(x, 10) for x in x2]

    x3 = np.linspace(0.01, 30, NUM_OBS)
    xp = [5,5]
    y3 = [stableswap_y(x, [5,5], 20) for x in x3]

    plt.figure(figsize=[4.75,3])
    plt.plot(x1, y1, color='purple', linestyle='dashed')
    plt.plot(x2, y2, color='red', linestyle='dotted')
    plt.plot(x3, y3, color='blue')
    plt.axis([0, 30, 0, 20])


    ##### Fig. 2 In Stableswap whitepaper
    peg_point: int = 5
    peg_index1: int = find_peg_point(x1, y1)
    peg_index3: int = find_peg_point(x3, y3)


    dx1 = [x-peg_point for x in x1] # shifts graph to past peg point
    dydx1 = dydx(y1, x1)

    dx3 = [x-peg_point for x in x3]
    # balances = [5,5]
    # xp = _xp([1, 1]) # -> [5,5]
    y3 = [stableswap_y(x, [5,5], 100) for x in x3]
    dydx3 = dydx(y3, x3)

    plt.figure(figsize=[5,3])
    plt.plot(
        dx1[peg_index1+1:],
        dydx1[peg_index1:],
        color='purple',
        linestyle='dashed',
    )
    plt.plot(
        dx3[peg_index3+1:],
        dydx3[peg_index3:],
        color="blue",
    )
    plt.axis([0, 10, 0, 1.05])


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

    ## Dynamic dales Tax, slippage scaled
    plt.plot(
        dx3[peg_index3+1:],
        slippageDx3[peg_index3:],
        color="green",
        )
    ## Dynamic Buyers bonus, slippage scaled, 50% of sales
    plt.plot(
        dx3[peg_index3+1:],
        buyerBonusDx3[peg_index3:],
        color="green",
        lineStyle="dashed"
    )

    plt.axis([0, 10, 0, 1.05])
    # plt.axis([0, 10, 0, 30])



if __name__=="__main__":
    print("Curve Stableswap plots")
    plot_fig1_fig2()
