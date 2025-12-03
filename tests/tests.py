import sys
import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if package_path not in sys.path:
    sys.path.insert(0, package_path)

import HydroErr.HydroErr as he
import unittest
import numpy as np
import warnings


class HydroErrTests(unittest.TestCase):

    def setUp(self):
        self.sim = np.array([5, 7, 9, 2, 4.5, 6.7])
        self.obs = np.array([4.7, 6, 10, 2.5, 4, 6.8])
        self.sim_bad_data = np.array([6, np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 5, 7, 9, 2, 4.5, 6.7])
        self.obs_bad_data = np.array([np.nan, 100, np.inf, 200, -np.inf, 300, 0, 400, -0.1, 500, 4.7, 6, 10, 2.5, 4, 6.8])

    def test_me(self):
        expected_value = 0.03333333333333336
        test_value = he.me(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.me(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mae(self):
        expected_value = 0.5666666666666665
        test_value = he.mae(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mae(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mse(self):
        expected_value = 0.4333333333333333
        test_value = he.mse(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mle(self):
        expected_value = 0.002961767058151136
        test_value = he.mle(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_male(self):
        expected_value = 0.09041652188064823
        test_value = he.male(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.male(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_msle(self):
        expected_value = 0.010426437593600514
        test_value = he.msle(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.msle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mde(self):
        expected_value = 0.10000000000000009
        test_value = he.mde(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mde(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mdae(self):
        expected_value = 0.5
        test_value = he.mdae(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mdae(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mdse(self):
        expected_value = 0.25
        test_value = he.mdse(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mdse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_ed(self):
        expected_value = 1.6124515496597098
        test_value = he.ed(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.ed(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_ned(self):
        expected_value = 0.28491828688422466
        test_value = he.ned(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.ned(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_rmse(self):
        expected_value = 0.6582805886043833
        test_value = he.rmse(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.rmse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_rmsle(self):
        expected_value = 0.10210992896677833
        test_value = he.rmsle(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.rmsle(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nrmse_range(self):
        expected_value = 0.08777074514725111
        test_value = he.nrmse_range(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nrmse_range(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nrmse_mean(self):
        expected_value = 0.11616716269489116
        test_value = he.nrmse_mean(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nrmse_mean(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nrmse_iqr(self):
        expected_value = 0.27145591282654985
        test_value = he.nrmse_iqr(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nrmse_iqr(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_irmse(self):
        expected_value = 0.14438269394140332
        test_value = he.irmse(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.irmse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mase(self):
        expected_value = 0.1656920077972709
        test_value = he.mase(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mase(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_r_squared(self):
        expected_value = 0.9246652089263256
        test_value = he.r_squared(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.r_squared(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_pearson_r(self):
        expected_value = 0.9615951377405804
        test_value = he.pearson_r(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.pearson_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_spearman_r(self):
        expected_value = 0.942857142857143
        test_value = he.spearman_r(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.spearman_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_acc(self):
        expected_value = 0.8013292814504837
        test_value = he.acc(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.acc(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mape(self):
        expected_value = 11.170038937560838
        test_value = he.mape(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mape(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mapd(self):
        expected_value = 0.09999999999999998
        test_value = he.mapd(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mapd(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_maape(self):
        expected_value = 0.11083600320216158
        test_value = he.maape(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.maape(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_smape1(self):
        expected_value = 5.630408980871249
        test_value = he.smape1(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.smape1(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_smape2(self):
        expected_value = 11.260817961742498
        test_value = he.smape2(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.smape2(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_d(self):
        expected_value = 0.9789712067292139
        test_value = he.d(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.d(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_d1(self):
        expected_value = 0.8508771929824561
        test_value = he.d1(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.d1(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_dmod(self):
        expected_value = 0.8508771929824561
        test_value = he.dmod(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.dmod(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_drel(self):
        expected_value = 0.9746276298212023
        test_value = he.drel(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.drel(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_dr(self):
        expected_value = 0.853448275862069
        test_value = he.dr(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.dr(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

        # Test for the case when a > b in the function
        simulated_array = np.array([6.46798627, 7.29596011, 8.84220973, 4.29514505, 0.28713612,
                                    6.72170644, 0.73659359, 0.88821022, 8.54288031, 8.46199717])
        observed_array = np.array([6.61975021, 0.66489119, 5.54279687, 8.66670447, 5.79587539,
                                   5.52870883, 7.83817005, 9.03424271, 5.87438289, 0.40828201])
        expected_value = -0.13041791707510286

        self.assertTrue(np.isclose(he.dr(simulated_array, observed_array), expected_value))

    def test_watt_m(self):
        expected_value = 0.832713182570339
        test_value = he.watt_m(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.watt_m(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mb_r(self):
        expected_value = 0.7843551797040169
        test_value = he.mb_r(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mb_r(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nse(self):
        expected_value = 0.923333988598388
        test_value = he.nse(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nse(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nse_mod(self):
        expected_value = 0.706896551724138
        test_value = he.nse_mod(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nse_mod(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_nse_rel(self):
        expected_value = 0.9074983335293921
        test_value = he.nse_rel(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.nse_rel(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_kge_2009(self):
        expected_value = 0.9181073779138655
        expected_tuple = (0.9615951377405804, 0.927910707932087, 1.0058823529411764, 0.9181073779138655)

        test_value = he.kge_2009(self.sim, self.obs)
        test_tuple = he.kge_2009(self.sim, self.obs, return_all=True)

        self.assertTrue(np.isclose(expected_value, test_value))
        self.assertTrue(np.all(np.isclose(np.array(expected_tuple), np.array(test_tuple))))

        test_value_bad_data = he.kge_2009(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

        # Testing warnings in the function
        sim = np.array([1, 2, 3, 4, 5])
        obs = np.array([-1, 1, 0, -2, 2])  # Making the mean 0

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            test_val_mean_0 = he.kge_2009(sim, obs)
            self.assertTrue(len(w) == 1)
            self.assertTrue('Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE '
                            'value cannot be computed.' in str(w[0].message))
            self.assertTrue(np.isnan(test_val_mean_0))

        sim = np.array([1, 2, 3, 4, 5])
        obs = np.array([1, 1, 1, 1, 1])  # Making the standard deviation 0

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            test_val_std_0 = he.kge_2009(sim, obs)
            self.assertTrue(len(w) == 2)  # There is also a warning for divide by zero
            self.assertTrue('Warning: The observed data standard deviation is 0. Therefore, Alpha is infinite '
                            'and the KGE value cannot be computed.' in str(w[1].message))
            self.assertTrue(np.isnan(test_val_std_0))

    def test_kge_2012(self):
        expected_value = 0.9132923608280753
        expected_tuple = (0.9615951377405804, 0.9224843295231272, 1.0058823529411764, 0.9132923608280753)

        test_value = he.kge_2012(self.sim, self.obs)
        test_tuple = he.kge_2012(self.sim, self.obs, return_all=True)

        self.assertTrue(np.isclose(expected_value, test_value))
        self.assertTrue(np.all(np.isclose(np.array(expected_tuple), np.array(test_tuple))))

        test_value_bad_data = he.kge_2012(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

        # Testing warnings in the function
        sim = np.array([1, 2, 3, 4, 5])
        obs = np.array([-1, 1, 0, -2, 2])  # Making the mean 0

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            test_val_mean_0 = he.kge_2012(sim, obs)
            self.assertTrue(len(w) == 3)
            self.assertTrue('Warning: The observed data mean is 0. Therefore, Beta is infinite and the KGE '
                            'value cannot be computed.' in str(w[2].message))
            self.assertTrue(np.isnan(test_val_mean_0))

        sim = np.array([1, 2, 3, 4, 5])
        obs = np.array([1, 1, 1, 1, 1])  # Making the standard deviation 0

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            test_val_std_0 = he.kge_2012(sim, obs)
            self.assertTrue(len(w) == 3)  # There is also a warning for divide by zero
            self.assertTrue('Warning: The observed data standard deviation is 0. Therefore, Gamma is infinite '
                            'and the KGE value cannot be computed.' in str(w[2].message))
            self.assertTrue(np.isnan(test_val_std_0))

        sim = np.array([-1, 1, 0, -2, 2])  # Making the mean 0
        obs = np.array([1, 2, 3, 4, 5])

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            test_val_mean_0_sim = he.kge_2012(sim, obs)
            self.assertTrue(len(w) == 2)  # There is also a warning for divide by zero
            self.assertTrue('Warning: The simulated data mean is 0. Therefore, Gamma is infinite '
                            'and the KGE value cannot be computed.' in str(w[1].message))
            self.assertTrue(np.isnan(test_val_mean_0_sim))

    def test_lm_index(self):
        expected_value = 0.706896551724138
        test_value = he.lm_index(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.lm_index(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

        # Testing with obs_bar_p argument
        expected_value_obs_bar_p_param = 0.706896551724138
        test_val_obs_bar_p_param = he.lm_index(self.sim, self.obs, obs_bar_p=5)
        self.assertTrue(np.isclose(expected_value_obs_bar_p_param, test_val_obs_bar_p_param))

    def test_d1_p(self):
        expected_value = 0.8508771929824561
        test_value = he.d1_p(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.d1_p(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

        # Testing with obs_bar_p argument
        expected_value_obs_bar_p_param = 0.8508771929824561
        test_val_obs_bar_p_param = he.d1_p(self.sim, self.obs, obs_bar_p=5)
        self.assertTrue(np.isclose(expected_value_obs_bar_p_param, test_val_obs_bar_p_param))

    def test_ve(self):
        expected_value = 0.9
        test_value = he.ve(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.ve(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_sa(self):
        expected_value = 0.10732665576112205
        test_value = he.sa(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.sa(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_sc(self):
        expected_value = 0.27804040550591774
        test_value = he.sc(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.sc(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_sid(self):
        expected_value = 0.03429918932223696
        test_value = he.sid(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.sid(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_sga(self):
        expected_value = 0.2645366651790464
        test_value = he.sga(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.sga(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h1_mhe(self):
        expected_value = 0.006798428591294671
        test_value = he.h1_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h1_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h1_mahe(self):
        expected_value = 0.11170038937560837
        test_value = he.h1_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h1_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h1_rmshe(self):
        expected_value = 0.1276017779995636
        test_value = he.h1_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h1_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h2_mhe(self):
        expected_value = -0.010344705046197581
        test_value = he.h2_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h2_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h2_mahe(self):
        expected_value = 0.11500078970228221
        test_value = he.h2_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h2_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h2_rmshe(self):
        expected_value = 0.13627318643885672
        test_value = he.h2_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h2_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h3_mhe(self):
        expected_value = -0.001491885359832964
        test_value = he.h3_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h3_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h3_mahe(self):
        expected_value = 0.11260817961742497
        test_value = he.h3_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h3_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h3_rmshe(self):
        expected_value = 0.13039562916009131
        test_value = he.h3_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h3_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h4_mhe(self):
        expected_value = -0.0016319199045327479
        test_value = he.h4_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h4_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h4_mahe(self):
        expected_value = 0.11297850488299188
        test_value = he.h4_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h4_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h4_rmshe(self):
        expected_value = 0.1309317900186668
        test_value = he.h4_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h4_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h5_mhe(self):
        expected_value = -0.0017731382274514507
        test_value = he.h5_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h5_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h5_mahe(self):
        expected_value = 0.11335058953894532
        test_value = he.h5_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h5_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h5_rmshe(self):
        expected_value = 0.13147134893754783
        test_value = he.h5_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h5_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h6_mhe(self):
        expected_value = -0.001491885359832948
        test_value = he.h6_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h6_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h6_mahe(self):
        expected_value = 0.11260817961742496
        test_value = he.h6_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h6_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h6_rmshe(self):
        expected_value = 0.1303956291600913
        test_value = he.h6_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h6_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h7_mhe(self):
        expected_value = 0.008498035739118379
        test_value = he.h7_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h7_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h7_mahe(self):
        expected_value = 0.13962548671951047
        test_value = he.h7_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h7_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h7_rmshe(self):
        expected_value = 0.1595022224994545
        test_value = he.h7_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h7_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h8_mhe(self):
        expected_value = 0.00582722450682403
        test_value = he.h8_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h8_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h8_mahe(self):
        expected_value = 0.09574319089337861
        test_value = he.h8_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h8_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h8_rmshe(self):
        expected_value = 0.1093729525710545
        test_value = he.h8_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h8_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h10_mhe(self):
        expected_value = 0.002961767058151136
        test_value = he.h10_mhe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h10_mhe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h10_mahe(self):
        expected_value = 0.09041652188064823
        test_value = he.h10_mahe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h10_mahe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_h10_rmshe(self):
        expected_value = 0.10210992896677833
        test_value = he.h10_rmshe(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.h10_rmshe(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_g_mean_diff(self):
        expected_value = 0.9924930879953174
        test_value = he.g_mean_diff(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.g_mean_diff(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def test_mean_var(self):
        expected_value = 0.010417665529493766
        test_value = he.mean_var(self.sim, self.obs)
        self.assertTrue(np.isclose(expected_value, test_value))

        test_value_bad_data = he.mean_var(self.sim_bad_data, self.obs_bad_data, remove_neg=True, remove_zero=True)
        self.assertTrue(np.isclose(expected_value, test_value_bad_data))

    def tearDown(self):
        del self.sim
        del self.obs


class HelperFunctionsTests(unittest.TestCase):

    def test_treat_values_remove(self):
        a = np.random.random_integers(low=100, size=(30, 2))
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
        self.assertTrue(len(w) == 4)
        self.assertTrue(issubclass(w[0].category, UserWarning))
        self.assertTrue(issubclass(w[1].category, UserWarning))
        self.assertTrue(issubclass(w[2].category, UserWarning))
        self.assertTrue(issubclass(w[3].category, UserWarning))

        self.assertTrue("Row(s) [0 1] contained NaN values and the row(s) have been removed "
                        "(Rows are zero indexed)." in str(w[0].message))
        self.assertTrue("Row(s) [2 3] contained Inf or -Inf values and the row(s) have been "
                        "removed (Rows are zero indexed)." in str(w[1].message))
        self.assertTrue("Row(s) [4 5] contained zero values and the row(s) have been removed "
                        "(Rows are zero indexed)." in str(w[2].message))
        self.assertTrue("Row(s) [6 7] contained negative values and the row(s) have been "
                        "removed (Rows are zero indexed)." in str(w[3].message))

        self.assertIsNone(np.testing.assert_equal(sim_treated, a[8:, 0]),
                          "Treat values function did not work properly when removing values from "
                          "the simulated data.")
        self.assertIsNone(np.testing.assert_equal(obs_treated, a[8:, 1]),
                          "Treat values function did not work properly when removing values from "
                          "the observed data.")

    def test_treat_values_replace(self):
        sim = np.array([np.nan, np.inf, 9, 2, 4.5, 6.7])
        obs = np.array([4.7, 6, np.nan, np.inf, 4, 7])

        sim_new = np.array([32.0, 1000.0, 9, 2, 4.5, 6.7])
        obs_new = np.array([4.7, 6, 32.0, 1000.0, 4., 7.])

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            sim_treated, obs_treated = he.treat_values(sim, obs, replace_nan=32, replace_inf=1000)
            # Verify some things
            self.assertTrue(len(w) == 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertTrue("Elements(s) [0] contained NaN values in the simulated array and "
                            "elements(s) [2] contained NaN values in the observed array and have "
                            "been replaced (Elements are zero indexed)." in str(w[0].message))
            self.assertTrue("Elements(s) [1] contained Inf values in the simulated array and "
                            "elements(s) [3] contained Inf values in the observed array and have "
                            "been replaced (Elements are zero indexed)." in str(w[1].message))

            # Check if arrays match
            self.assertIsNone(np.testing.assert_equal(sim_treated, sim_new),
                              "Treat values function did not work properly when replacing values from "
                              "the simulated data.")
            self.assertIsNone(np.testing.assert_equal(obs_treated, obs_new),
                              "Treat values function did not work properly when replacing values from "
                              "the observed data.")

    def test_treat_values_unequal_length(self):
        sim = np.array([1, 2, 3, 4])
        obs = np.array([1, 2, 3])

        with self.assertRaises(Exception) as context:
            he.treat_values(sim, obs)

        self.assertTrue("The two ndarrays are not the same size." in context.exception.args[0])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    unittest.TextTestRunner(verbosity=2).run(suite)
