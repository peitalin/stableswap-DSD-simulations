import numpy as np

def quadratic_tax(price, dsd_amount):
    return (1 - price)**2 * np.abs(dsd_amount)

def linear_tax(price, dsd_amount):
    return (1 - price) * np.abs(dsd_amount)

def no_tax(price, dsd_amount):
    return 0

def log_tax(price, dsd_amount):
    return np.log(1 + (1 - price)) * np.abs(dsd_amount)

def cubic_tax(price, dsd_amount):
    return ((1 + (1 - price)**2)**1/3 - 1/3) * np.abs(dsd_amount)

