# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from calicoml.core.algo.qq import QQ, QQPlot
from calicoml.core.problem import Problem
from calicoml.core.reporting import ReportRenderer
from calicoml.core.utils import with_numpy_arrays
import nose

import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from tests.test_reporting import with_temporary_directory


@with_numpy_arrays
def is_non_decreasing(lst):
    """True iff a list is non-decreasing"""
    return np.all(lst[1:] - lst[:-1] >= 0)


def mock_problem(n_samples, n_features, n_informative, theta):
    """Mocks up a problem for testing"""
    rand = np.random.RandomState(0xC0FFEE)
    X = rand.normal(size=(n_samples, n_features))
    y = rand.choice([0, 1], size=n_samples)

    informative_idx = rand.choice(list(range(n_informative)), size=n_informative, replace=False)
    for idx in informative_idx:
        X[y == 1, idx] += theta

    features = ['true-{}'.format(idx) if idx in informative_idx else 'null-{}'.format(idx)
                for idx in range(n_features)]

    df = pd.DataFrame(data=X, columns=features)
    df['y'] = y
    return Problem(df, features, 'y', 1)


def test_qq():
    """Verifies that the QQ plot computation works as expected"""
    def checkme(n_samples, n_features, n_informative, null_or_signal):
        """Utility"""
        assert null_or_signal in {'null', 'signal'}
        prob = mock_problem(n_samples, n_features, n_informative, 2.0 if null_or_signal == 'signal' else 0.0)

        qq_df = QQ(prob).dataframe
        nose.tools.eq_(n_features, len(qq_df))
        nose.tools.assert_list_equal(list(qq_df['rank']), list(range(1, n_features + 1)))
        is_non_decreasing(qq_df['observed_p_value'])
        is_non_decreasing(qq_df['expected_p_value'])
        nose.tools.ok_(np.all(qq_df['n_features'].values == n_features))
        nose.tools.ok_(np.all(qq_df['n_samples'].values == n_samples))
        nose.tools.ok_(all([0.0 <= p_val <= 1.0 for p_val in qq_df['observed_p_value']]))
        nose.tools.ok_(all([0.0 <= p_val <= 1.0 for p_val in qq_df['expected_p_value']]))

        if null_or_signal == 'signal':
            # Theta is big enough for us to pick up all true features
            nose.tools.ok_(all([feat.startswith('true') for feat in qq_df.head(n_informative)['feature']]))

        # All non-informative points should be roughly on the line
        null_qq_df = qq_df if null_or_signal == 'null' else qq_df.tail(len(qq_df) - n_informative)
        deltas = null_qq_df['observed_p_value'] - null_qq_df['expected_p_value']
        np.testing.assert_allclose(np.mean(deltas), 0.0, atol=0.05)

        line_fit = LinearRegression().fit(null_qq_df['expected_p_value'].values.reshape((len(null_qq_df), 1)),
                                          null_qq_df['observed_p_value'])
        np.testing.assert_allclose(line_fit.intercept_, 0.0, atol=0.05)
        np.testing.assert_allclose(line_fit.coef_[0], 1.0, atol=0.05)

    for n_samples in [100, 500, 1000, 2000]:
        for n_features in [100, 200]:
            for n_informative in [5, 10, 50]:
                yield checkme, n_samples, 1000, n_informative, 'null'
                yield checkme, n_features, 1000, n_informative, 'signal'


def test_qq_plot():
    """Validates QQ plotting"""
    @with_temporary_directory
    def checkme(output_dir, n_samples, n_features, n_informative, theta, show_labels):
        """Utility"""
        prob = mock_problem(n_samples, n_features, n_informative, theta)
        qq = QQ(prob)
        qq_plot = QQPlot(qq, ReportRenderer(output_dir), significance_threshold=0.01, min_p_value=0.01,
                         show_labels=show_labels)
        plot_df = qq_plot.get_plotting_dataframe()
        nose.tools.eq_(len(plot_df), len(qq.dataframe))
        nose.tools.ok_(all(p_val >= qq.get_bonferroni_threshold(0.01) for p_val in plot_df['observed_p_value']))
        if not show_labels:
            nose.tools.eq_(set(plot_df['label'].unique()), {''})

        # Not much we can automatically check in the actual plot beyond it having been written
        qq_plot.generate()
        nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'qq.pdf')))
        nose.tools.ok_(os.path.exists(os.path.join(output_dir, 'qq.txt')))

    for n_samples in [50, 100]:
        for n_features in [50, 100]:
            for theta in [0.0, 1.0]:
                yield checkme, n_samples, n_features, 10, theta, True
                yield checkme, n_samples, n_features, 10, theta, False
