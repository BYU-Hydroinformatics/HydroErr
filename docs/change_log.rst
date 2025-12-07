Release Notes
=============

This is the list of changes to HydroErr between each release. For full details, see the commit logs at
https://github.com/BYU-Hydroinformatics/HydroErr.

Version 2.0.0rc1
^^^^^^^^^^^^^^^^

Breaking Changes:

- Drop support for Python 2.7, 3.6 and 3.7. Minimum Python version is now 3.10.
- Raise TypeError instead of AssertionError in the kge_2009 and kge_2012 metrics for invalid
  "return_all" parameter.
- When the kge_2009 or kge_2012 metrics are not able to be computed, return np.float64(np.nan)
  values instead of just np.nan.

Other Changes:

- Add type hints throughout the codebase for better developer experience.
- Minor refactoring of code structure to improve maintainability and modularity.
- More modern documentation theme (Furo) for better readability.
- Use MathJax for rendering mathematical expressions in the documentation via the sphinx.ext.mathjax
  extension.
- Add ruff as a linter to the development workflow for improved code quality.
- Use uv for managing development environment.
- Use ty for checking type hints in the codebase.
- Use pytest as the testing framework.

Version 1.24
^^^^^^^^^^^^
- Added .name and .abbr properties to each metric to give a little bit more info to the users, and also for ease of use
  in the Hydrostats package.
- Allowed users to import all metrics by simply using :code:`import HydroErr` instead of
  :code:`import HydroErr.HydroErr`.
