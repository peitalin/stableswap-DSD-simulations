
import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt


# rates: uint256[N_COINS] -> uint256[N_COINS];
PRECISION = 1
# PRECISION2 = 1
PRECISION2 = 0.001


# https://github.com/curvefi/curve-contract/blob/295e7daaad0654a6c7a233f77e82a01fb78d85b4/contracts/pools/usdt/StableSwapUSDT.vy#L183
def get_D(xp, A=85):

    N_COINS = len(xp)
    S = 0
    for _x in xp:
        S += _x
    if S == 0:
        return 0

    Dprev = 0
    D = S
    Ann = A * N_COINS
    # Ann = A * N_COINS ** 2

    for _i in range(255):
        D_P = D
        for _x in xp:
            D_P = D_P * D / (_x * N_COINS + 1)  # +1 is to prevent /0
        Dprev = D
        D = (Ann * S + D_P * N_COINS) * D / ((Ann - 1) * D + (N_COINS + 1) * D_P)
        # Equality with the precision of 1
        if D > Dprev:
            if D - Dprev <= PRECISION2:
                break
        else:
            if Dprev - D <= PRECISION2:
                break
    return D




# def get_y(i: int128, j: int128, x: uint256, _xp: uint256[N_COINS]) -> uint256:
# https://github.com/curvefi/curve-contract/blob/295e7daaad0654a6c7a233f77e82a01fb78d85b4/contracts/pools/usdt/StableSwapUSDT.vy#L331
def get_y(i, j, x, _xp, A=85):
    # x in the input is converted to the same price/precision
    N_COINS = len(_xp)

    assert (i != j) and (i >= 0) and (j >= 0) and (i < N_COINS) and (j < N_COINS)

    D = get_D(_xp, A)
    c = D
    S_ = 0
    Ann = A * N_COINS
    # Ann = A * N_COINS ** 2

    _x = 0
    for _i in range(N_COINS):
        if _i == i:
            _x = x
        elif _i != j:
            _x = _xp[_i]
        else:
            continue
        S_ += _x
        c = c * D / (_x * N_COINS)

    c = c * D / (Ann * N_COINS)
    b = S_ + D / Ann  # - D
    y_prev = 0
    y = D
    for _i in range(255):
        y_prev = y
        y = (y*y + c) / (2 * y + b - D)
        # Equality with the precision of 1
        if y > y_prev:
            if y - y_prev <= PRECISION2:
                break
        else:
            if y_prev - y <= PRECISION2:
                break
    return y



def _xp(rates):
    result = rates
    for i in range(N_COINS):
        result[i] = result[i] * balances[i] / PRECISION
        print(result[i])
    return result


#### Invariants ####
## get the number of token y in a pool, given x
## holding the Stableswap invariant constant

def stableswap_y(x, xp=[50,50], A=85):
    i = 0 # position 0 for first coin
    j = 1 # position 1 for second coin
    amp = A
    y = get_y(i, j, x, xp, amp)
    assert not np.isnan(y)
    assert y >= 0
    return y



