import numpy as np

def generate_trade(mu, sigma):
    rv = np.random.normal(mu, sigma)
    if rv >= 0:
        return dict({ 'type': "buy", 'amount': rv })
    else:
        return dict({ 'type': "sell", 'amount': rv })
