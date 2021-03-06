import numpy as np

def generate_trade(mu, sigma):
    rv = np.random.normal(mu, sigma)
    if rv >= 0:
        return dict({ 'type': "buy", 'amount': rv })
    else:
        return dict({ 'type': "sell", 'amount': rv })


def bayes_update_normal(
    sigma_1,
    sigma_2,
    mu_1,
    mu_2,
):
    """
    Precision-weighted average of mu_1 and mu_2.
    A Bayes update for normal distributions.
    """
    sigma_12 = sigma_1 + sigma_2

    new_mean = (sigma_1 * mu_1 + sigma_2 * mu_2) / sigma_12
    new_variance = (sigma_1 * sigma_2) / sigma_12

    return dict({
        "mu": new_mean,
        "sigma": np.sqrt(new_variance),
        "variance": new_variance,
    })
