import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def quadratic_tax(price, dsd_amount):
    return (1 - price)**2 * np.abs(dsd_amount)

def logistic_tax(price, dsd_amount):
    return 1/(1 + np.exp((price-0.5)*10)) * np.abs(dsd_amount)

def linear_logistic_tax(price, dsd_amount):
    logistic_tax_amount = 1/(1 + np.exp((price-0.5)*10)) * np.abs(dsd_amount)
    linear_tax_amount = (1 - price) * np.abs(dsd_amount)
    return (price) * logistic_tax_amount + (1 - price) * linear_tax_amount


def linear_tax(price, dsd_amount):
    return (1 - price) * np.abs(dsd_amount)

def no_tax(price, dsd_amount):
    return 0

def log_tax(price, dsd_amount):
    return np.log(1 + (1 - price)) * np.abs(dsd_amount)

def cubic_tax(price, dsd_amount):
    return ((1 + (1 - price)**2)**1/3 - 1/3) * np.abs(dsd_amount)


if __name__=="__main__":

    fig, ax = plt.subplots()

    xx = np.linspace(0, 1, 1000)

    plt.plot(xx, [linear_tax(x, 1) * 100 for x in xx],
             color="mediumorchid")
    plt.plot(xx, [quadratic_tax(x, 1) * 100 for x in xx],
             color='dodgerblue')
    plt.plot(xx, [logistic_tax(x, 1) * 100 for x in xx],
             color='red')
    # plt.plot(xx, [linear_logistic_tax(x, 1) * 100 for x in xx],
    #          color='orange')

    # # bad, log_tax is not 0 at $1 peg
    # plt.plot(xx, [log_tax(x, 1) for x in xx])
    # # bad, cubic_tax is not 0 at $1 peg
    # plt.plot(xx, [cubic_tax(x, 1) for x in xx])


    plt.title("Tax functions")
    plt.xlabel("Price from \$0 to \$1")
    plt.ylabel("% Tax Paid")

    legend_elements = [
        Line2D([0], [0], color="mediumorchid", lw=2,
               label=r'$(1-price)$ - linear_tax'),
        Line2D([0], [0], color='dodgerblue', lw=2,
               label=r'$(1-price)^2$ - quadratic_tax'),
        Line2D([0], [0], color="red", lw=2,
               label=r'$(1 - e^{(price-0.5) * 10})$ - logistic_tax'),
        # Line2D([0], [0], color="orange", lw=2,
        #        label=r'$(1 - price)*logistic + (price)*linear$ - linear_logistic_tax'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')


