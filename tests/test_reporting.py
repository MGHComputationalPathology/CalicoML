# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import unicode_literals
from __future__ import print_function
from functools import wraps
import shutil
from tempfile import mkdtemp
from calicoml.core.reporting import ReportRenderer, ComparativeClassificationReport, ClassificationReport
from calicoml.core.algo.learners import SelectAndClassify

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
import nose
from calicoml.core.problem import Problem
import numpy as np
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import seaborn as sns


def with_temporary_directory(func):
    """Calls the function with a temporary directory as the first argument, then removes the
    directory and all contents when done"""

    @wraps(func)
    def inner(*args, **kwargs):
        """Closure over the temporary directory"""
        try:
            directory_path = mkdtemp()
            return func(directory_path, *args, **kwargs)
        finally:
            shutil.rmtree(directory_path)

    return inner


@with_temporary_directory
def test_renderer(output_path):
    """Verifies that the base report renderer works as expected"""
    renderer = ReportRenderer(output_path)

    # Tables
    df = pd.DataFrame(columns=['A', 'B'], data=[['a', 0], ['b', 1]])
    renderer.add_table('table1', df)
    renderer.add_table('table2', df)

    for name in ['table1.txt', 'table2.txt']:
        nose.tools.ok_(os.path.exists(os.path.join(output_path, name)))
        df_from_file = pd.read_csv(os.path.join(output_path, name), sep='\t')

        # pylint thinks df_from_file is a tuple (it's not). Hence disable below.
        nose.tools.assert_list_equal(list(df.columns), list(df_from_file.columns))  # pylint: disable=E1101
        np.testing.assert_array_equal(df.values, df_from_file.values)  # pylint: disable=E1101

    # Plots
    df_for_plotting = pd.DataFrame(columns=['input'], data=np.arange(100).reshape((100, 1)))
    df_for_plotting['response'] = 2.0 * df_for_plotting['input']

    sample_plt = sns.regplot(df_for_plotting['input'], df_for_plotting['response']).get_figure()
    renderer.add_plot('sample_plot', sample_plt)

    # Not much we can check automatically here beyond the file existing
    nose.tools.ok_(os.path.exists(os.path.join(output_path, 'sample_plot.pdf')))


class LoggingRenderer(ReportRenderer):
    """Stores tables and plots internally. For testing only."""
    def __init__(self):
        super(LoggingRenderer, self).__init__(None, create_directory=False)
        self.tables = {}
        self.plots = {}

    def add_plot(self, name, plots):
        self.plots[name] = plots

    def add_table(self, name, df, index=False, index_col=None):
        self.tables[name] = df


def assert_lengths_equal(*args):
    """Throws a value error if input lists have different lengths"""
    if len({len(arg) for arg in args}) > 1:
        raise ValueError("Input lists must be same length")


def mock_cv_results_list(features, learning_approach_name, selected_features, train_scores, test_scores,
                         train_aucs, test_aucs, p_values):
    """
    Creates and returns list of mock CV results for report testing

    :param features: List of strings, with each string being a feature
    :param learning_approach_name: Name of learning approach. Each result in the list will have this name
    :param selected_features: List of sublists. Length of the list is the number of mock results in the
                              mock results list; each sublist is a list of selected features for a mock
                              result in the list
    :param train_scores: List of sublists representing training scores, similar to selected_features, except that
                         scores are float/double between 0 and 1 inclusive. One sublist per mock result in output list.
                   Currently must be list of exactly 100 floats.
    :param test_scores: Same as train_scores, except for test instead of train.
    :param train_aucs: List of training AUCs as float/double. One AUC per mock result in output list.
    :param test_aucs: List of testing AUCs as float/double. One AUC per mock result in output list. Testing aucs
                      usually lower than training AUCs.
    :param p_values: List of p_values for each mock result.

    :return: List of mock results

    """
    assert_lengths_equal(selected_features, train_scores, test_scores, train_aucs, test_aucs, p_values)

    cvr = []
    for cv_idx, _ in enumerate(selected_features):
        cvr.append({'cv_index': cv_idx,
                    'approach': {'type': 'test_approach', 'name': learning_approach_name,
                                 'selected_features': selected_features[cv_idx],
                                 'feature_p_values': {feat: p_values[cv_idx] for feat in features}},
                    'train': {'n_features': len(features),
                              'sample': ['train{}'.format(idx) for idx in range(100)],
                              'scores': train_scores[cv_idx],
                              'truth': [0, 1] * 50,
                              'outcome': ['N', 'Y'] * 50,
                              'positive_outcome': ['Y'] * 100,
                              'metrics': {'auc': train_aucs[cv_idx]}},
                    'test': {'n_features': len(features),
                             'sample': ['test{}'.format(idx) for idx in range(100)],
                             'scores': test_scores[cv_idx],
                             'truth': [0, 1] * 50,
                             'outcome': ['N', 'Y'] * 50,
                             'positive_outcome': ['Y'] * 100,
                             'metrics': {'auc': test_aucs[cv_idx]}}})
    return cvr


def test_comparative_report():
    """Verifies that we correctly generate a classification report"""
    features = ['f1', 'f2', 'f3', 'f4', 'f5']
    cv_results_1 = mock_cv_results_list(features, 'my test approach 1',
                                        [['f1', 'f2'], ['f2', 'f5']],
                                        [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                        [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                        [0.9, 1.0], [0.66, 0.76], [0.05, 0.01])
    cv_results_2 = mock_cv_results_list(features, 'my test approach 1',
                                        [['f2', 'f3'], ['f3', 'f4']],
                                        [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                        [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                        [0.95, 0.97], [0.85, 0.78], [0.05, 0.01])
    approach1 = SelectAndClassify(SelectKBest(k=2), LogisticRegression(), name="logit")
    approach2 = SelectAndClassify(SelectKBest(k=3), GaussianNB(), name="nb")
    cv_results = {approach1: cv_results_1, approach2: cv_results_2}
    renderer = LoggingRenderer()
    report = ComparativeClassificationReport(renderer)
    report.generate(cv_results)

    for expected_plot in ['score_plots']:
        nose.tools.assert_true(expected_plot in renderer.plots)

    for expected_table in ['mean_scores', 'mean_metrics']:
        nose.tools.assert_true(expected_table in renderer.tables)

        # Should have the exact same number of entries for each approach
        groups = dict(list(renderer.tables[expected_table].groupby('approach')))
        nose.tools.eq_(set(groups.keys()), {'logit', 'nb'})
        assert_lengths_equal(*list(groups.values()))

    metrics_df = renderer.tables["mean_metrics"]
    nose.tools.assert_equal(dict(list(zip(metrics_df["approach"], metrics_df["train_auc"]))),
                            {'logit': 0.95, 'nb': 0.96})
    nose.tools.assert_equal(dict(list(zip(metrics_df["approach"], metrics_df["test_auc"]))),
                            {'logit': 0.71, 'nb': 0.815})


def test_classification_report():
    """Verifies that we correctly generate a classification report"""
    features = ['f1', 'f2', 'f3', 'f4', 'f5']
    cv_results = mock_cv_results_list(features, 'my test approach',
                                      [['f1', 'f2'], ['f2', 'f5']],
                                      [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                      [[0.1, 0.2, 0.8, 1.0] * 25, [0.0, 0.1, 0.7, 0.9] * 25],
                                      [0.9, 1.0], [0.66, 0.76],
                                      [0.05, 0.01])

    renderer = LoggingRenderer()
    report = ClassificationReport(renderer, output_train_scores=True)
    report.generate(cv_results)

    # Check features
    nose.tools.eq_(3, len(renderer.tables['selected_features']))
    for _, row in renderer.tables['selected_features'].iterrows():
        nose.tools.eq_(row['times_selected'], 2 if row['feature'] == 'f2' else 1)
        nose.tools.eq_(row['frequency'], 1.0 if row['feature'] == 'f2' else 0.5)
        nose.tools.eq_(row['n_cv_splits'], 2)
        nose.tools.assert_almost_equal(row['median_p_value'], 0.03)

    # Check metrics
    nose.tools.eq_(2, len(renderer.tables['cv_metrics']))
    for _, row in renderer.tables['cv_metrics'].iterrows():
        nose.tools.eq_(row['approach_type'], 'test_approach')
        nose.tools.eq_(row['train_auc'], 0.9 if row['cv_index'] == 0 else 1.0)
        nose.tools.eq_(row['test_auc'], 0.66 if row['cv_index'] == 0 else 0.76)

    nose.tools.eq_(1, len(renderer.tables['mean_metrics']))
    for _, row in renderer.tables['mean_metrics'].iterrows():
        nose.tools.eq_(row['approach_type'], 'test_approach')
        nose.tools.eq_(row['train_auc'], 0.95)
        nose.tools.eq_(row['test_auc'], 0.71)

    # Check scores
    scores = renderer.tables['mean_sample_scores']
    nose.tools.eq_(set(scores['subset'].unique()), {'train', 'test'})

    for _, sdf in scores.groupby(by='subset'):
        nose.tools.eq_(100, len(sdf))
        for _, row in sdf.iterrows():
            idx = int(row['sample'].replace('train', '').replace('test', ''))  # train10 -> 10
            nose.tools.assert_almost_equal([0.05, 0.15, 0.75, 0.95][idx % 4], row['score'])
            nose.tools.eq_('Y' if row['truth'] == 1 else 'N', row['outcome'])
            nose.tools.eq_('Y', row['positive_outcome'])


def mock_coords_data(data, index_class, class_value, data2=None, append_missed=True):
    """ Utility to generate test data"""
    for index_coord in range(4):
        delta0 = 0.1 if index_coord < 2 else 0.0
        coord0 = 1.0 - delta0 if index_class < 2 else delta0
        delta1 = 0.1 if index_coord % 2 == 1 else 0.0
        coord1 = 1.0 - delta1 if index_class % 2 == 0 else delta1
        data.append([coord0, coord1, class_value])
        if data2 is not None and index_class != 2:
            data2.append([coord0, coord1, class_value])
    if append_missed:
        coord0_miss = 0.3 if index_class < 2 else 0.7
        coord1_miss = 0.3 if index_class % 2 == 0 else 0.7
        data.append([coord0_miss, coord1_miss, class_value])
        if data2 is not None and index_class != 2:
            data2.append([coord0_miss, coord1_miss, class_value])
    return data, data2


def test_binary_report_with_score_vector():
    " Test that in binary case score as vector contains same data as with positive outcome only"
    data = []
    class_values = ['A', 'B']
    for index_class in range(4):
        data = mock_coords_data(data, index_class, class_values[index_class % 2], data2=None, append_missed=False)[0]

    df = pd.DataFrame(columns=['coord0', 'coord1', 'class'], data=data)
    prob = Problem(df, ['coord0', 'coord1'], 'class', 'B')

    classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(),
                                   name='test binary with score vector').fit(prob)
    y_score_positive = classifier.apply(prob)
    y_score_all = classifier.apply(prob, False)
    nose.tools.ok_(np.allclose(y_score_positive, y_score_all[:, 1]))
