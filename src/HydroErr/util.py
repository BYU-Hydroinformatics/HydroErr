"""Utility functions."""

import warnings

import numpy as np

from .typing_aliases import FloatArray, InputArray


def treat_values(  # noqa: C901
    simulated_array: InputArray,
    observed_array: InputArray,
    replace_nan: float | None = None,
    replace_inf: float | None = None,
    remove_neg: bool = False,
    remove_zero: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Remove the nan, negative, and inf values in two numpy arrays."""
    sim_copy: FloatArray = np.copy(simulated_array)
    obs_copy: FloatArray = np.copy(observed_array)

    # Checking to see if the vectors are the same length
    if not sim_copy.ndim == 1:
        raise ValueError("The simulated array is not one dimensional.")
    if not obs_copy.ndim == 1:
        raise ValueError("The observed array is not one dimensional.")

    if sim_copy.size != obs_copy.size:
        raise ValueError("The two ndarrays are not the same size.")

    # Treat missing data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain nan values
    all_treatment_array = np.ones(obs_copy.size, dtype=bool)

    if np.any(np.isnan(obs_copy)) or np.any(np.isnan(sim_copy)):
        if replace_nan is not None:
            # Finding the NaNs
            sim_nan = np.isnan(sim_copy)
            obs_nan = np.isnan(obs_copy)
            # Replacing the NaNs with the input
            sim_copy[sim_nan] = replace_nan
            obs_copy[obs_nan] = replace_nan

            warnings.warn(
                f"Elements(s) {np.where(sim_nan)[0]} contained NaN values in the simulated array"
                f" and elements(s) {np.where(obs_nan)[0]} contained NaN values in the observed"
                " array and have been replaced (Elements are zero indexed).",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Getting the indices of the nan values, combining them, and informing user.
            nan_indices_fcst = ~np.isnan(sim_copy)
            nan_indices_obs = ~np.isnan(obs_copy)
            all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_nan_indices)

            warnings.warn(
                f"Row(s) {np.where(~all_nan_indices)[0]} contained NaN values and the row(s) have"
                f" been removed (Rows are zero indexed).",
                UserWarning,
                stacklevel=2,
            )

    if np.any(np.isinf(obs_copy)) or np.any(np.isinf(sim_copy)):
        if replace_nan is not None:
            # Finding the NaNs
            sim_inf = np.isinf(sim_copy)
            obs_inf = np.isinf(obs_copy)
            # Replacing the NaNs with the input
            sim_copy[sim_inf] = replace_inf
            obs_copy[obs_inf] = replace_inf

            warnings.warn(
                f"Elements(s) {np.where(sim_inf)[0]} contained Inf values in the simulated array"
                f" and elements(s) {np.where(obs_inf)[0]} contained Inf values in the observed"
                " array and have been replaced (Elements are zero indexed).",
                UserWarning,
                stacklevel=2,
            )
        else:
            inf_indices_fcst = ~(np.isinf(sim_copy))
            inf_indices_obs = ~np.isinf(obs_copy)
            all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_inf_indices)

            warnings.warn(
                f"Row(s) {np.where(~all_inf_indices)[0]} contained Inf or -Inf values and"
                " the row(s) have been removed (Rows are zero indexed).",
                UserWarning,
                stacklevel=2,
            )

    # Treat zero data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain zero values
    if remove_zero and ((obs_copy == 0).any() or (sim_copy == 0).any()):
        zero_indices_fcst = ~(sim_copy == 0)
        zero_indices_obs = ~(obs_copy == 0)
        all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
        all_treatment_array = np.logical_and(all_treatment_array, all_zero_indices)

        warnings.warn(
            f"Row(s) {np.where(~all_zero_indices)[0]} contained zero values and the row(s)"
            " have been removed (Rows are zero indexed).",
            UserWarning,
            stacklevel=2,
        )

    # Treat negative data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain negative values

    # Ignore runtime warnings from comparing
    if remove_neg:
        with np.errstate(invalid="ignore"):
            obs_copy_bool = obs_copy < 0
            sim_copy_bool = sim_copy < 0

        if obs_copy_bool.any() or sim_copy_bool.any():
            neg_indices_fcst = ~sim_copy_bool
            neg_indices_obs = ~obs_copy_bool
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

            warnings.warn(
                f"Row(s) {np.where(~all_neg_indices)[0]} contained negative values and the row(s)"
                f" have been removed (Rows are zero indexed).",
                UserWarning,
                stacklevel=2,
            )

    obs_copy = obs_copy[all_treatment_array]
    sim_copy = sim_copy[all_treatment_array]

    return sim_copy, obs_copy
