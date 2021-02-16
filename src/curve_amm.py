
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
# on Curve smart contract, this is 1 for integer precision
# I set it ast 0.001 here for smoother plots

# "rates" on Curve' smart contract. needs investigation
# I suspect it's Compoun/Aave lending rates
RATES = [ 0.9999983333, 0.9999983333 ]


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



# Buggy for now, needs investigation + help from Curve
def _xp(balances: list[float], rates: list[float]):
    # N_COINS = len(balances)
    # # result = rates
    # result = RATES
    # for i in range(N_COINS):
    #     print("before: result[i]", result[i])
    #     print("before balances[i]", result[i])
    #     result[i] = result[i] * balances[i] / PRECISION
    #     print("result[i]", result[i])
    #
    # print("results", result)
    # return result
    return balances





class Curve:
    "The Curve Stableswap AMM"

    def __init__(self,
        x=1200,
        y=400,
        x_name="USDC",
        y_name="DSD",
        treasury_tax_rate=0.5,
        A=100
    ):
        # x, y are initial balances
        self.balance_x = x
        self.balance_y = y
        self.x_name = x_name
        self.y_name = y_name

        # uses _xp(rates) function, but need to find out
        # what "rates" is from Curve smart contract
        # [x, y] even seems to give results that look like
        # the plots in the Stableswap whitepaper
        # self.xp = [x, y]
        # self.xp = _xp([x, y], RATES)
        self.A = A # amplification parameter
        # lower values make the curve behave close to
        # Uniswap constant product invariant,
        # higher values make it behave
        # more closely to a sum invariant

        self.history = dict({
            # history of treasury balances over time
            'treasury_balances': [0],
            # history of prices
            'prices': [self.price_oracle([x, y])], # initial price
            # history of burns over time
            'burns': [0],
        })
        self.ohlc = None
        self.treasury_tax_rate=0.5


    def __repr__(self):
        treasury_balance = self.history['treasury_balances'][-1]
        return """
        Liquidity Pool:
        {x_name} balance:\t{balance_x:>12.4f}
        {y_name} balance:\t{balance_y:>12.4f}
        {y_name}/{x_name} price: {price:.10f}

        DAO Treasury from sales taxes:
        {y_name} balance: {treasury_balance:>12.4f}
        """.format(
            x_name = self.x_name,
            balance_x = self.balance_x,
            y_name = self.y_name,
            balance_y = self.balance_y,
            price = self.price_oracle([
                self.balance_x,
                self.balance_y,
            ]),
            treasury_balance = treasury_balance
        )


    # def ohlc_generate_prices(self, num_sections=10):
    #     # split history['prices'] into arrays [[], []]
    #     ts_prices = np.array_split(self.history['prices'], num_sections)
    #     dates = pd.date_range(start='1/1/2021', periods=len(ts_prices))
    #     ohlc_timeseries = []
    #
    #     for i, ts in enumerate(ts_prices):
    #         ohlc_timeseries.append(dict({
    #             'Date': dates[i],
    #             'Open': ts[0],
    #             'High': np.max(ts),
    #             'Low': np.min(ts),
    #             'Close': ts[-1],
    #         }))
    #     # dates = pd.date_range(start='1/1/2021', end='1/2/2021', periods=1000)
    #     ohlc_df = pd.DataFrame(ohlc_timeseries)
    #     self.ohlc = ohlc_df.set_index("Date")
    #     return self.ohlc
    #
    #
    # def ohlc_plot(self, num_sections=10):
    #     self.ohlc_generate_prices(num_sections)
    #     fplt.plot(
    #         self.ohlc,
    #         type='candle',
    #         style='yahoo',
    #         title='DSD, simulated trades',
    #         ylabel='Price DSD/USDC'
    #     )


    def price_oracle(self, xp):
        prior_balance_x = self.balance_x
        prior_balance_y = self.balance_y

        # print("xp: ", xp)
        adjusted_xp = _xp([xp[0], xp[1]], RATES)
        # print("adjusted_xp: ", adjusted_xp)

        after_balance_x = self.balance_x + 1
        after_balance_y = stableswap_y(
            self.balance_x + 1,
            adjusted_xp,
            self.A,
        )

        slippage = dxdy_once(
            y2 = after_balance_y,
            y1 = prior_balance_y,
            x2 = after_balance_x,
            x1 = prior_balance_x,
        )
        price = np.abs(slippage)
        return price



    def swap(self, trade, tax_function):
        """
        trade: dict({ 'type': 'sell'|'buy', amount: float })
        """

        if trade['type'] == 'buy':
            price_after = self.buy_dsd(trade['amount'])
        else:
            if tax_function == "slippage":
                price_after = self.sell_dsd_slippage_tax(trade['amount'])
            else:
                price_after = self.sell_dsd(trade['amount'], tax_function)

        return price_after


    def buy_dsd(self, usdc_amount):
        """Buys usdc_amount worth of DSD
        no taxes for buys"""

        xp = _xp([ self.balance_x, self.balance_y ], RATES)

        y = stableswap_y(
            self.balance_x + usdc_amount,
            xp,
            self.A,
        )
        self.balance_x += usdc_amount
        self.balance_y = y
        after_price = self.price_oracle([
            self.balance_x,
            self.balance_y,
        ])


        self.history['treasury_balances'].append(
            self.history['treasury_balances'][-1]
        ) # no change to treasury on buys
        self.history['prices'].append(after_price)
        self.history['burns'].append(0)
        return after_price


    def sell_dsd(self,
             dsd_amount,
             tax_function=lambda *args, **kwargs: 0
         ):
        """Sells dsd_amount worth of DSD"""
        # plt.plot(np.linspace(0,1,100), [(1 - x**2) for x in np.linspace(1,0,100)])

        prior_balance_x = self.balance_x
        prior_balance_y = self.balance_y
        prior_price = self.price_oracle([
            prior_balance_x,
            prior_balance_y,
        ])
        # print("curve prior price: ", prior_price)
        # print("prior_balance_x: ", prior_balance_x)
        # print("prior_balance_y: ", prior_balance_y)

        # Calculate DSD burn before updating balances
        burn = tax_function(
            price=prior_price,
            dsd_amount=dsd_amount
        )
        print("burn:", burn)

        # actual amount sold into LP pool after burn
        leftover_dsd = np.abs(dsd_amount) - burn

        after_balance_y = self.balance_y + leftover_dsd

        after_balance_x = stableswap_x(
            after_balance_y,
            [self.balance_x, after_balance_y],
            self.A,
        )
        # print("after_balance_x: ", after_balance_x)
        # print("after_balance_y: ", after_balance_y)

        # calculate burn first, update balances
        self.balance_y = after_balance_y
        self.balance_x = after_balance_x
        after_price = self.price_oracle([
            after_balance_x,
            after_balance_y,
        ])
        print("after_price: ", after_price)

        # # Update xp balances
        # self.xp = [after_balance_x, after_balance_y]

        # fraction of burnt dsd, to treasury, say 50%
        burn_to_treasury = self.treasury_tax_rate * burn
        actual_burn = (1 - self.treasury_tax_rate) * burn

        self.history['treasury_balances'].append(
            self.history['treasury_balances'][-1] + burn_to_treasury
        )
        self.history['prices'].append(after_price)
        self.history['burns'].append(actual_burn)

        return after_price



    def sell_dsd_slippage_tax(self, dsd_amount):
        """Sells dsd_amount worth of DSD
        sales taxes are scaled by slippage imparted to AMM curve
        """
        # this needs its own function as you need to calculate slipage first, before calculating burn and updating pool balances
        # unlike the other simpe price-based sales taxes
        dsd = np.abs(dsd_amount)
        prior_balance_x = self.balance_x
        prior_balance_y = self.balance_y
        prior_price = self.price_oracle([
            prior_balance_x,
            prior_balance_y,
        ])

        xp = _xp([ self.balance_x, self.balance_y ], RATES)
        # balance_y (DSD) increases when DSD is sold to pool
        _after_balance_x = stableswap_x(
            prior_balance_y + dsd,
            xp,
            self.A,
        )
        _after_balance_y = self.balance_y + dsd

        # calculate slippage + burn first, before swap
        slippage = dxdy_once(
            y2 = _after_balance_y,
            y1 = prior_balance_y,
            x2 = _after_balance_x,
            x1 = prior_balance_x,
        )
        burn =  (1 - np.abs(slippage)) * dsd if (np.abs(slippage) < 1) else 0
        # print("burn: {}".format(1 - slippage))
        # print("price: {}".format(prior_price))
        print("slippage: {}".format(slippage))
        print("burn: {}".format(burn))

        # actual amount sold into LP pool after burn
        leftover_dsd = dsd - burn

        # now update calculate burn-adjusted balance for x
        after_balance_x = stableswap_x(
            prior_balance_y + leftover_dsd,
            xp,
            self.A,
        )
        after_balance_y = prior_balance_x + leftover_dsd

        # now update balances adjusting for burn
        self.balance_y = after_balance_y
        self.balance_x = after_balance_x
        after_price = self.price_oracle([
            after_balance_x,
            after_balance_y,
        ])


        # fraction of sales taxes paid to treasury
        burn_to_treasury = self.treasury_tax_rate * burn
        actual_burn = (1 - self.treasury_tax_rate) * burn

        self.history['treasury_balances'].append(
            self.history['treasury_balances'][-1] + burn_to_treasury
        )
        self.history['prices'].append(after_price)
        self.history['burns'].append(actual_burn)

        return after_price








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

def stableswap_x(y, xp=[50,50], A=85):
    i = 0 # position 0 for first coin
    j = 1 # position 1 for second coin
    amp = A
    # swap coins i and j around
    x = get_y(j, i, y, xp, amp)
    assert not np.isnan(x)
    assert x >= 0
    return x


def dydx_once(y2, y1, x2, x1):
    """calculates derivative for dy relative to dx"""
    # # Needed to figure out dUSDC/dDSD slippage/price impact
    # print("y2: ", y2)
    # print("y1: ", y1)
    # print("x2: ", x2)
    # print("x1: ", x1)
    return np.diff([y2, y1])[0] / np.diff([x2, x1])[0]


def dxdy_once(y2, y1, x2, x1):
    """calculates derivative for dx relative to dy"""
    # Needed to figure out dUSDC/dDSD slippage/price impact
    # print("y2: ", y2)
    # print("y1: ", y1)
    # print("x2: ", x2)
    # print("x1: ", x1)
    return np.diff([x2, x1])[0] / np.diff([y2, y1])[0]




def price_curve(
    prior_balance_x,
    prior_balance_y,
    xp,
    A,
):

    after_balance_x = prior_balance_x + 1
    after_balance_y = stableswap_y(
        prior_balance_x + 1,
        xp,
        A,
    )
    # calculate slippage + burn first, before swap
    slippage = dxdy_once(
        y2 = after_balance_y,
        y1 = prior_balance_y,
        x2 = after_balance_x,
        x1 = prior_balance_x,
    )
    price = np.abs(slippage)
    return price


# from curve_amm import price_curve, dxdy_once
#
# c = Curve(lp_initial_usdc, lp_initial_dsd, A=100)
# # c = Curve(999999.07, 5000001, A=100)
# c.swap({'type': 'sell', 'amount': -1},
#     tax_function=no_tax)
#
# # prior
# price_curve(
#     1000000,
#     5000000,
#     [1000000, 5000000],
#     100
# )
# # after
# price_curve(
#     999999.0709302969,
#     5000001,
#     [999999.0709302969, 5000001],
#     100
# )
#
# # after
# price_curve(
#     1000000.03039,
#     5000001,
#     [1000000, 5000000],
#     100
# )
#



