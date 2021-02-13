import setuptools

setuptools.setup(
    name="DSD-dip14",
    version="0.1",
    author="Peita Lin",
    author_email="n6378056@gmail.com",
    description="Dynamic Set Dollar DIP 14 Simulations",
    packages=[
        "curve_amm",
        "uniswap_amm",
    ],
    install_requires=[
        "ipython",
        "numpy",
        "pandas",
        "matplotlib",
        "mplfinance",
    ],
)
