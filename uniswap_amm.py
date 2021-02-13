
import numpy as np
import pandas as pd
# plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as fplt




class Uniswap:
    "This is a Uniswap AMM"

    def __init__(self,
        x=1200,
        y=400,
        x_name="USDC",
        y_name="DSD"
    ):
        # x, y are initial balances
        self.balance_x = x
        self.balance_y = y
        self.x_name = x_name
        self.y_name = y_name
        self.k = x * y # invariant
        # for more on how AMMs work:
        # https://uniswap.org/docs/v2/protocol-overview/how-uniswap-works/
        self.history = dict({
            # history of treasury balances over time
            'treasury_balances': [0],
            # history of prices
            'prices': [x/y], # initial price
            # history of burns over time
            'burns': [0],
        })
        self.ohlc = None


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
            price = self.price_oracle(),
            treasury_balance = treasury_balance
        )


    def ohlc_generate_prices(self, num_sections=10):
        # split history['prices'] into arrays [[], []]
        ts_prices = np.array_split(self.history['prices'], num_sections)
        dates = pd.date_range(start='1/1/2021', periods=len(ts_prices))
        ohlc_timeseries = []

        for i, ts in enumerate(ts_prices):
            ohlc_timeseries.append(dict({
                'Date': dates[i],
                'Open': ts[0],
                'High': np.max(ts),
                'Low': np.min(ts),
                'Close': ts[-1],
            }))
        # dates = pd.date_range(start='1/1/2021', end='1/2/2021', periods=1000)
        ohlc_df = pd.DataFrame(ohlc_timeseries)
        self.ohlc = ohlc_df.set_index("Date")
        return self.ohlc

    def ohlc_plot(self, num_sections=10):
        self.ohlc_generate_prices(num_sections)
        fplt.plot(
            self.ohlc,
            type='candle',
            style='yahoo',
            title='DSD, simulated trades',
            ylabel='Price DSD/USDC'
        )


    def show_balances(self):
        # print("{} balance:\t{}".format(x_name, self.balance_x))
        print("{} balance:\t{}".format(y_name, self.balance_y))

    def show_price(self):
        print("{y_name}/{x_name} price: {price}".format(
            x_name = self.x_name,
            y_name = self.y_name,
            price = self.price_oracle(),
        ))

    def dxdy_once(self, y2, y1, x2, x1):
        """calculates derivative for dx relative to dy"""
        # warning: this is an inverted derivative: dxdy
        # run over rise, as we need to figure out dUSDC/dDSD
        # when we sell DSD
        return np.diff([x2, x1])[0] / np.diff([y2, y1])[0]

    def price_oracle(self):
        return self.balance_x / self.balance_y

    def swap(self, trade, tax_function):
        """
        trade: dict({ 'type': 'sell'|'buy', amount: float })
        """

        if trade['type'] == 'buy':
            price_after = self.buy_dsd(trade['amount'])
        else:
            price_after = self.sell_dsd(trade['amount'], tax_function)

        self.show_balances()
        # self.show_price()
        return price_after


    def buy_dsd(self, usdc_amount):
        """Buys usdc_amount worth of DSD
        no taxes for buys"""

        y = uniswap_y(self.balance_x + usdc_amount, self.k)
        self.balance_x += usdc_amount
        self.balance_y = y

        self.history['treasury_balances'].append(
            self.history['treasury_balances'][-1]
        ) # no change to treasury on buys
        self.history['prices'].append(self.price_oracle())
        self.history['burns'].append(0)
        return self.price_oracle()


    def sell_dsd(self,
             dsd_amount,
             tax_function=lambda *args, **kwargs: 0
         ):
        """Sells dsd_amount worth of DSD
        Sales tax style: quadratic with distance from peg
        ($1 - price) * DSD
        """
        # plt.plot(np.linspace(0,1,100), [(1 - x**2) for x in np.linspace(1,0,100)])

        prior_balance_x = self.balance_x
        prior_balance_y = self.balance_y
        prior_price = self.price_oracle()

        # Calculate DSD burn before updating balances
        # Or after? After might be better as it takes into account the size of the sell order (slippage)
        burn = tax_function(
            price=prior_price,
            dsd_amount=dsd_amount
        )

        # actual amount sold into LP pool after burn
        leftover_dsd = np.abs(dsd_amount) - burn

        x = uniswap_x(self.balance_y + leftover_dsd, self.k)
        after_balance_x = x
        after_balance_y = self.balance_y + leftover_dsd

        # calculate burn first, update balances
        self.balance_y = after_balance_y
        self.balance_x = after_balance_x
        after_price = self.price_oracle()

        # fraction of burnt dsd, to treasury, say 50%
        burn_to_treasury = 0.5 * burn
        self.history['treasury_balances'].append(
            self.history['treasury_balances'][-1] + burn_to_treasury
        )
        self.history['prices'].append(after_price)
        self.history['burns'].append(burn_to_treasury)

        return self.price_oracle()


    def sell_dsd_slippage_tax(self, dsd_amount):
        """Sells dsd_amount worth of DSD
        sales taxes are scaled by slippage imparted to AMM curve
        """
        # this needs its own function as you need to calculate slipage first, before calculating burn and updating pool balances
        # unlike the other simpe price-based sales taxes

        dsd = np.abs(dsd_amount)

        x = uniswap_x(self.balance_y + dsd, self.k)

        prior_balance_x = self.balance_x
        prior_balance_y = self.balance_y

        after_balance_x = x
        after_balance_y = self.balance_y + dsd

        # calculate slippage + burn first, before swap
        slippage = self.dxdy_once(
            after_balance_y,
            prior_balance_y,
            after_balance_x,
            prior_balance_x,
        )
        print("slippage: {}".format(slippage))

        self.balance_y = after_balance_y
        self.balance_x = after_balance_x

        # if (slippage > )
        # dsd_burn = np.abs(slippage * dsd)

        # fraction of sales are burnt, fraction to treasury
        # scaled by slippage
        self.treasury_balance += (1 - slippage) * dsd

        return self.price_oracle()






