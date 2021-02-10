
import matplotlib.pyplot as plt
import numpy as np

# x: uint256[N_COINS]
xp = [5, 5]
N_COINS = len(xp)
# rates: uint256[N_COINS] -> uint256[N_COINS];
PRECISION = 0.1
# PRECISION2 = 1
PRECISION2 = 0.001


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




def uniswap_y(x, k=250):
    y = k/x
    return y

def linear_y(x, k=250):
    y = k - x
    return y

def curve_y(x, xp=[50,50], A=85):
    i = 0 # position 0 for first coin
    j = 1 # position 1 for second coin
    amp = A
    y = get_y(i, j, x, xp, amp)
    return y




NUM_OBS = 4000

x1 = np.linspace(0, 30, NUM_OBS)
y1 = [uniswap_y(x, 25) for x in x1]

x2 = np.linspace(0, 10, NUM_OBS)
y2 = [linear_y(x, 10) for x in x2]

x3 = np.linspace(0, 30, NUM_OBS)
xp = [5,5]
y3 = [curve_y(x, xp, 20) for x in x3]

plt.figure(figsize=[4.75,3])
plt.plot(x1, y1, color='purple', linestyle='dashed')
plt.plot(x2, y2, color='red', linestyle='dotted')
plt.plot(x3, y3, color='blue')
plt.axis([0, 30, 0, 20])


def find_peg_point(x1, y1):
    return np.argmin([np.abs(x - y) for x,y in zip(x1, y1)])


def dx(xx):
    dx_array = []
    for i in range(len(xx)):
        if i == 0:
            continue

        dx = xx[i-1] - xx[i]
        dx_array.append( np.abs(dx) )

    return dx_array

def dy(yy):
    return dx(yy)


def dydx(xx, yy):
    assert len(xx) == len(yy)
    dydx_array = np.abs(np.diff(yy)/np.diff(xx))
    # dydx_array = np.diff(yy)/np.diff(xx)
    return dydx_array


peg_point = 5
peg_index1 = find_peg_point(x1, y1)
peg_index3 = find_peg_point(x3, y3)
# peg_index=0

dx1 = [x-peg_point for x in x1]
dydx1 = dydx(x1, y1)

dx3 = [x-peg_point for x in x3]
xp = [5,5]
y3 = [curve_y(x, xp, 100) for x in x3]
dydx3 = dydx(x3, y3)

# proportional slippage sales tax
slippageDx2 = [(1-x)  for x in dydx3]
# quadratic slippage sales tax, 50% max sales tax
slippageDx3 = [(1 + (1-x)**2)**1/2 - 1/2  for x in dydx3]

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

## Dynamic Sales Tax, slippage scaled
plt.plot(
    dx3[peg_index3+1:],
    slippageDx3[peg_index3:],
    color="green",
)

plt.axis([0, 10, 0, 1.05])
# plt.axis([0, 10, 0, 30])



# plt.axis([0, 10, 0, 30])


# dydx1 = dydx(x1, y1)
# dydx3 = dydx(x3, y3)
# # dydx1 = [y/x for x,y in zip(x1,y1)]
# plt.figure(figsize=[6,6])
# plt.plot(x3[:999], dydx3[:999])
# plt.plot(x1[:999], dydx1[:999])
#
# plt.axis([0, 10, 0, 1])
#
# # plt.axis([0, 10, 0, 30])

