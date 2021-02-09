
import matplotlib.pyplot as plt
import numpy as np
# x: uint256[N_COINS]


xp = [5, 5]
N_COINS = len(xp)

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
    for _i in range(255):
        D_P = D
        for _x in xp:
            D_P = D_P * D / (_x * N_COINS + 1)  # +1 is to prevent /0
        Dprev = D
        D = (Ann * S + D_P * N_COINS) * D / ((Ann - 1) * D + (N_COINS + 1) * D_P)
        # Equality with the precision of 1
        if D > Dprev:
            if D - Dprev <= 1:
                break
        else:
            if Dprev - D <= 1:
                break
    return D


# rates: uint256[N_COINS] -> uint256[N_COINS];
PRECISION = 0.1


def _xp(rates):
    result = rates
    for i in range(N_COINS):
        result[i] = result[i] * self.balances[i] / PRECISION
    return result



# def get_y(i: int128, j: int128, x: uint256, _xp: uint256[N_COINS]) -> uint256:
def get_y(i, j, x, _xp, A=85):
    # x in the input is converted to the same price/precision

    # assert (i != j) and (i >= 0) and (j >= 0) and (i < N_COINS) and (j < N_COINS)

    D = get_D(_xp, A)
    c = D
    S_ = 0
    N_COINS = len(xp)
    Ann = A * N_COINS

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
            if y - y_prev <= 1:
                break
        else:
            if y_prev - y <= 1:
                break
    return y




def uniswap_curve(x, k=250):
    y = k/x
    return y

def linear_curve(x, k=250):
    y = k - x
    return y

def curve_curve(x, k=250):
    return

NUM_OBS = 1000

x1 = np.linspace(0, 2500, NUM_OBS)
y1 = [uniswap_curve(x, 2500) for x in x1]

x2 = np.linspace(0,100, NUM_OBS)
y2 = [linear_curve(x, 100) for x in x2]

x3 = np.linspace(0, 300, NUM_OBS)
xp = [3000,3000]
y3 = [get_y(0, 1, x, xp, 0.1) for x in x3]

plt.figure(figsize=[6,6])
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.axis([0, 300, 0, 300])





