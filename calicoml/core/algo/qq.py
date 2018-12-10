# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import unicode_literals
from __future__ import print_function
from calicoml.core.utils import partially_reorder_columns

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns


class QQ(object):
    """Computes the quantile-quantile statistics for feature significance"""
    def __init__(self, problem, significance_test=None):
        """\

        :param problem: a Problem instance
        :param significance_test: significance test to use. Must have the signature X, y -> scores, pvalues.
        Default: f_classif.

        """
        self.problem = problem
        self.significance_test = significance_test or f_classif
        self._frame = None
        self._compute()

    def _compute(self):
        """Computes the QQ statistics, returning them as a pandas DataFrame"""
        X, y = self.problem.X, self.problem.y

        _, observed_p_vals = self.significance_test(X, y)

        df = pd.DataFrame({'feature': self.problem.features,
                           'observed_p_value': observed_p_vals,
                           'n_samples': self.problem.n_samples,
                           'n_features': self.problem.n_features})
        df.sort_values('observed_p_value', ascending=True, inplace=True)
        df['rank'] = np.arange(len(df)) + 1
        df['expected_p_value'] = df['rank'] / X.shape[1]
        return partially_reorder_columns(df, ['rank', 'feature', 'observed_p_value', 'expected_p_value'])

    def get_bonferroni_threshold(self, alpha):
        """Applies Bonferroni correction to the given significance threshold"""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Alpha has to be between 0 and 1')
        return alpha / self.problem.X.shape[1]

    @property
    def dataframe(self):
        """Returns the pandas DataFrame with QQ statistics"""
        if self._frame is None:
            self._frame = self._compute()
        return self._frame


class QQPlot(object):
    """Generates a QQ plot"""
    def __init__(self, qq, renderer, significance_threshold=0.05, min_p_value=1e-10, show_labels=True,
                 title='QQ plot for features'):
        """\

        :param qq: a QQ object
        :param renderer: renderer to use for the output
        :param significance_threshold: where to draw the significance line (will be Bonferroni adjusted)
        :param min_p_value: p values smaller than this threshold will be clipped to the threshold. Useful
        when you have one or two massively significant features which then dwarf the rest of the plot.
        :param show_labels: whether to show labels for the significant features
        :param title: title of the plot

        """
        self.renderer = renderer
        self.qq = qq
        self.significance_threshold = significance_threshold
        self.min_p_value = min_p_value
        self.show_labels = show_labels
        self.title = title

    def get_plotting_dataframe(self):
        """\
        Returns the dataframe containing post-processed statistics for the plot

        :return: pandas DataFrame

        """
        bonferroni_threshold = self.qq.get_bonferroni_threshold(self.significance_threshold)
        df = pd.DataFrame(self.qq.dataframe, copy=True)
        if self.min_p_value is not None:
            df['observed_p_value'] = [max(p_val, self.qq.get_bonferroni_threshold(self.min_p_value))
                                      for p_val in df['observed_p_value']]

        if self.show_labels:
            df['label'] = [row['feature'] if row['observed_p_value'] <= bonferroni_threshold else ''
                           for _, row in df.iterrows()]
        else:
            df['label'] = ''
        return df

    def generate(self):
        """Generates the plot and sends it to the renderer"""
        qq_df = self.get_plotting_dataframe()
        bonf = self.qq.get_bonferroni_threshold(0.05)
        qq = plt.figure()
        qq = sns.regplot(-np.log10(qq_df['expected_p_value']), -np.log10(qq_df['observed_p_value']), fit_reg=False,
                         scatter_kws={'s': 4})
        qq.set(xlabel='-log10(expected pval)', ylabel='-log10(observed pval)', title=self.title)
        ylim = max(-np.log10(qq_df['expected_p_value'])) + .2
        xlim = max(-np.log10(qq_df['observed_p_value'])) + .2
        lim = max(xlim, ylim)
        qq.plot([0, lim], [0, lim], '-', color='grey')
        qq.hlines(y=-np.log10(bonf), xmin=0, xmax=lim, linewidth=1, linestyles='dashed', color='g')
        for _, row in qq_df.iterrows():
            qq.text(-np.log10(row['expected_p_value']), -np.log10(row['observed_p_value']), row['label'], fontsize=6)
        qq_plot = qq.get_figure()
        self.renderer.add_plot('qq', qq_plot)
        self.renderer.add_table('qq', self.qq.dataframe)
