import warnings

import numpy as np
import pytest
import pytest_check as check

import HydroErr.HydroErr as he


class TestHelperFunctions:
    def test_treat_values_remove(self) -> None:
        rng = np.random.default_rng()
        a = rng.integers(low=1, high=101, size=(30, 2))
        a = a.astype(np.float16)
        a[0, 0] = np.nan
        a[1, 1] = np.nan
        a[2, 0] = np.inf
        a[3, 1] = np.inf
        a[4, 0] = 0
        a[5, 1] = 0
        a[6, 0] = -1
        a[7, 1] = -1

        sim = a[:, 0]
        obs = a[:, 1]

        # Tests
        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            sim_treated, obs_treated = he.treat_values(sim, obs, remove_zero=True, remove_neg=True)

        # Verify some things
        check.is_true(len(w) == 4)
        check.is_true(issubclass(w[0].category, UserWarning))
        check.is_true(issubclass(w[1].category, UserWarning))
        check.is_true(issubclass(w[2].category, UserWarning))
        check.is_true(issubclass(w[3].category, UserWarning))

        print(w[1].message)
        check.is_true(
            "Row(s) [0 1] contained NaN values and the row(s) have been removed "
            "(Rows are zero indexed)." in str(w[0].message)
        )
        check.is_true(
            "Row(s) [2 3] contained Inf or -Inf values and the row(s) have been "
            "removed (Rows are zero indexed)." in str(w[1].message)
        )
        check.is_true(
            "Row(s) [4 5] contained zero values and the row(s) have been removed "
            "(Rows are zero indexed)." in str(w[2].message)
        )
        check.is_true(
            "Row(s) [6 7] contained negative values and the row(s) have been "
            "removed (Rows are zero indexed)." in str(w[3].message)
        )

        check.is_none(
            np.testing.assert_equal(sim_treated, a[8:, 0]),
            "Treat values function did not work properly when removing values from "
            "the simulated data.",
        )
        check.is_none(
            np.testing.assert_equal(obs_treated, a[8:, 1]),
            "Treat values function did not work properly when removing values from "
            "the observed data.",
        )

    def test_treat_values_replace(self) -> None:
        sim = np.array([np.nan, np.inf, 9, 2, 4.5, 6.7])
        obs = np.array([4.7, 6, np.nan, np.inf, 4, 7])

        sim_new = np.array([32.0, 1000.0, 9, 2, 4.5, 6.7])
        obs_new = np.array([4.7, 6, 32.0, 1000.0, 4.0, 7.0])

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            sim_treated, obs_treated = he.treat_values(sim, obs, replace_nan=32, replace_inf=1000)
            # Verify some things
            check.is_true(len(w) == 2)
            check.is_true(issubclass(w[0].category, UserWarning))
            check.is_true(issubclass(w[1].category, UserWarning))
            check.is_true(
                "Elements(s) [0] contained NaN values in the simulated array and "
                "elements(s) [2] contained NaN values in the observed array and have "
                "been replaced (Elements are zero indexed)." in str(w[0].message)
            )
            check.is_true(
                "Elements(s) [1] contained Inf values in the simulated array and "
                "elements(s) [3] contained Inf values in the observed array and have "
                "been replaced (Elements are zero indexed)." in str(w[1].message)
            )

            # Check if arrays match
            check.is_none(
                np.testing.assert_equal(sim_treated, sim_new),
                "Treat values function did not work properly when replacing values from "
                "the simulated data.",
            )
            check.is_none(
                np.testing.assert_equal(obs_treated, obs_new),
                "Treat values function did not work properly when replacing values from "
                "the observed data.",
            )

    def test_treat_values_unequal_length(self) -> None:
        sim = np.array([1, 2, 3, 4])
        obs = np.array([1, 2, 3])

        with pytest.raises(Exception, match=r"^The two ndarrays are not the same size."):
            he.treat_values(sim, obs)
            # If it matches the regex, it passes
