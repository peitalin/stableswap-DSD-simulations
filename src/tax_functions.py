import numpy as np

def quadratic_tax(price, dsd_amount):
    return (1 - price)**2 * np.abs(dsd_amount)

def linear_tax(price, dsd_amount):
    return (1 - price) * np.abs(dsd_amount)

def no_tax(price, dsd_amount):
    return 0

