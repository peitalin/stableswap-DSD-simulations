

def create_time_series_data_store():
    """
    Creates data strcutures to store timeseries data
    for simulations
    """
    avg_prices = dict({
        "quadratic_tax_uni": [],
        "linear_tax_uni": [],
        "no_tax_uni": [],
        "logistic_tax_uni": [],
        "linear_logistic_tax_uni": [],
        "no_tax_curve": [],
        "slippage_tax_uni": [],
        "slippage_tax_curve": [],
        "quadratic_tax_curve": [],
    })
    avg_burns = dict({
        "quadratic_tax_uni": [],
        "linear_tax_uni": [],
        "no_tax_uni": [],
        "logistic_tax_uni": [],
        "linear_logistic_tax_uni": [],
        "no_tax_curve": [],
        "slippage_tax_uni": [],
        "slippage_tax_curve": [],
        "quadratic_tax_curve": [],
    })
    avg_treasury_balances = dict({
        "quadratic_tax_uni": [],
        "linear_tax_uni": [],
        "no_tax_uni": [],
        "logistic_tax_uni": [],
        "linear_logistic_tax_uni": [],
        "no_tax_curve": [],
        "slippage_tax_uni": [],
        "slippage_tax_curve": [],
        "quadratic_tax_curve": [],
    })
    colors = dict({
        "quadratic_tax_uni": "dodgerblue",
        "linear_tax_uni": "mediumorchid",
        "no_tax_uni": "black",
        "no_tax_curve": "black",
        "logistic_tax_uni": "red",
        "linear_logistic_tax_uni": "yellow",
        "slippage_tax_uni": "crimson",
        "slippage_tax_curve": "green",
        "quadratic_tax_curve": "orange",
    })

    return dict({
        'avg_prices': avg_prices,
        'avg_burns': avg_burns,
        'avg_treasury_balances': avg_treasury_balances,
        'colors': colors,
    })
