   
Contents
========

.. toctree::
    :maxdepth: 2

    Crossval
    Outliers

.. toctree::
    :maxdepth: 1
    :caption: Other information

    About
    
Project principles and design decisions
=======================================

- It should be easy to swap out individual components of the optimization process
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimization unless it can be practically
  applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this,
  I have deferred **all** formatting decisions to `Black
  <https://github.com/ambv/black>`_.


Advantages over existing implementations
========================================

- Includes both classical methods (Markowitz 1952 and Black-Litterman), suggested best practices
  (e.g covariance shrinkage), along with many recent developments and novel
  features, like L2 regularisation, exponential covariance, hierarchical risk parity.
- Native support for pandas dataframes: easily input your daily prices data.
- Extensive practical tests, which use real-life data.
- Easy to combine with your proprietary strategies and models.
- Robust to missing data, and price-series of different lengths (e.g FB data
  only goes back to 2012 whereas AAPL data goes back to 1980).
