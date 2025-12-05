"""A library of goodness of fit metrics that measure hydrologic skill.

Each metric is contained in a function, and every function has the parameters to treat missing
values as well as remove zero and negative values from the timeseries data.

Each function contains two properties, name and abbr. These can be used in the Hydrostats package
when creating tables and adding metrics to the plots. Link to the hydrostats package:
https://github.com/BYU-Hydroinformatics/Hydrostats.

An example of this functionality is shown below.

>>> import HydroErr as he
>>> he.acc.name
'Anomaly Correlation Coefficient'
>>> he.acc.abbr
'ACC'
"""

from .HydroErr import *  # noqa: F403

__version__ = "2.0.0"
