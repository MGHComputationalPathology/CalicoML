# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

import os
import pkg_resources
import pandas as pd
import numpy as np
import nose

from calicoml.core.lifestats.survival_stats import Survival
from tests.test_reporting import with_temporary_directory
from lifelines.datasets import load_rossi
from lifelines.utils import concordance_index


def sample_data():
    """ read sample problem copied from http://stats.idre.ucla.edu/r/examples,
        that page used first 5 records, here we use all records hence numbers are different """
    data_path = pkg_resources.resource_filename('tests.data', 'test_lifestats_data')
    test_csv_path = os.path.join(data_path, 'hmohiv.csv')
    df = pd.read_csv(test_csv_path)
    return df


def test_hr():
    """ test hazard ratios for  ComparativeSurvival """
    df = sample_data()
    example = Survival(df, 'time', 'censor').stratify('drug', ['no drug', 'drug'])
    cf_value = example.cox_model_fitted.summary['p']
    nose.tools.assert_less(cf_value[0], 0.01)
    hr_value = example.hazard_ratio()
    nose.tools.assert_less(hr_value.pval, 0.01)

    np.testing.assert_allclose(hr_value.estimate, 2.3, atol=0.15)
    cm_values = example.hazards
    np.testing.assert_allclose(cm_values[0].estimate, 2.74273380219, atol=0.05)
    np.testing.assert_allclose(cm_values[0].low, 1.78384243895, atol=0.05)
    nose.tools.assert_equal(str(cm_values[0].pval), 'nan')
    np.testing.assert_allclose(cm_values[1].high, 4.22659770806, atol=0.05)


@with_temporary_directory
def test_plot_file_exists(output_dir):
    """ test that plot files are generated for ComparativeSurvival """
    df = sample_data()
    example = Survival(df, 'time', 'censor').stratify('drug', ['no drug', 'drug'])
    path_km = os.path.join(output_dir, 'file_with_KM.png')
    example.plot_kaplan_meier(path_km)
    path_na = os.path.join(output_dir, 'file_with_NA.png')
    example.plot_nelson_aalen(path_na)

    nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'file_with_KM.png')))
    nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'file_with_NA.png')))


def test_cox_concordance():
    """ test based on lifelines doc example from lifelines.readthedocs.io
    Warning: doc example has multivariable result, here is one variable result """
    rossi_dataset = load_rossi()
    example = Survival(rossi_dataset, 'week', 'arrest').stratify('fin', ['no fin', 'fin'])
    cox_model = example.cox_model_fitted
    concordance_value = concordance_index(cox_model.durations,
                                          -cox_model.predict_partial_hazard(cox_model.data).values.ravel(),
                                          cox_model.event_observed)

    np.testing.assert_allclose(concordance_value, 0.545735287211, atol=0.005)


def test_overall_survival():
    """ test hazard ratios for overall survival"""
    df = sample_data()
    example = Survival(df, 'time', 'censor')
    cf_value_all = example.hazards
    np.testing.assert_allclose(cf_value_all.estimate, 3.05316751791, atol=0.000001)

    comp_example = example.stratify('drug', ['no drug', 'drug'])
    cf_value = comp_example.cox_model_fitted.summary['p']
    nose.tools.assert_less(cf_value[0], 0.01)
    hr_value = comp_example.hazard_ratio()
    nose.tools.assert_less(cf_value[0], 0.01)
    np.testing.assert_allclose(hr_value.estimate, 2.3, atol=0.15)


@with_temporary_directory
def test_overall_survival_plot_file_exists(output_dir):
    """ test that plot files are generated """
    df = sample_data()
    example = Survival(df, 'time', 'censor')
    path_km = os.path.join(output_dir, 'file_with_KM.png')
    example.plot_kaplan_meier(path_km)
    path_na = os.path.join(output_dir, 'file_with_NA.png')
    example.plot_nelson_aalen(path_na)

    nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'file_with_KM.png')))
    nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'file_with_NA.png')))
