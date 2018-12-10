# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from numpy.random.mtrand import RandomState

from calicoml.core.utils import with_numpy_arrays, format_p_value

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr

from sklearn.metrics import roc_curve, roc_auc_score


@with_numpy_arrays
def ppv(y_true, y_pred):
    """\
    Calculates the positive predictive value

    :param y_true: truth labels (0 or 1)
    :param y_pred: predicted labels (0 or 1)
    :return: the PPV

    """
    assert len(y_true) == len(y_pred)
    assert set(y_true).issubset({0, 1})
    assert set(y_pred).issubset({0, 1})

    return float(np.sum(y_true[y_pred == 1] == 1)) / np.sum(y_pred == 1)


def npv(y_true, y_pred):
    """\
    Calculates the negative predictive value

    :param y_true: truth labels (0 or 1)
    :param y_pred: predicted labels (0 or 1)
    :return: the NPV

    """
    return ppv(np.ones(len(y_true)) - y_true, np.ones(len(y_pred)) - y_pred)


def threshold_at_median(metric, y_true, y_pred):
    """\
    Binarizes scores at the median, and then applies the given metric

    :param metric: the metric to apply
    :param y_true: truth labels (0 or 1)
    :param y_pred: predicted scores
    :return: metric applied to the binarized scores

    """
    cutoff = np.median(y_pred)
    return metric(y_true, [1 if score >= cutoff else 0 for score in y_pred])


def f_pearson(X, y):
    """ Computes pearson correlation for columns of X vs. y\
        TODO: for y with categorical values replace column y with expected means from each column of X"""
    rs_pearson = []
    ps_pearson = []

    for feat_vals in X.T:
        r_column, p_column = pearsonr(feat_vals, y)
        rs_pearson.append(r_column)
        ps_pearson.append(p_column)

    return rs_pearson, ps_pearson


class ConditionalMeansSelector(object):
    """ wrapper to be enable feature selection for multiclass"""
    def __init__(self, selector, column_pairwise_selector=False):
        """\
        :param selector: matrix or vector base correlation-like function for feature selection
        :param column_pairwise_selector: True if selector is vector based, False if matrix based
        """
        self.selector = selector
        self.column_pairwise_selector = column_pairwise_selector

    @staticmethod
    def _conditional_map(y, feat_vals):
        """ method to build conditional means map for given feature column"""
        y_to_cond_mean = {}
        y_to_cond_count = {}
        for index, y_value in enumerate(y):
            if y_value not in y_to_cond_mean:
                y_to_cond_mean[y_value] = 0.0
                y_to_cond_count[y_value] = 0
            y_to_cond_mean[y_value] += feat_vals[index]
            y_to_cond_count[y_value] += 1

        for y_value in y_to_cond_mean:
            y_to_cond_mean[y_value] /= y_to_cond_count[y_value]
        return y_to_cond_mean

    def selector_function(self, X, y):
        """ apply selection function after replacing target column with conditional means"""
        rs_result = []
        ps_result = []
        for feat_vals in X.T:
            map_y_to_conditional_mean = ConditionalMeansSelector._conditional_map(y, feat_vals)
            y_mapped = np.asarray([map_y_to_conditional_mean[y_value] for y_value in y])
            if self.column_pairwise_selector:
                rs_column, ps_column = self.selector(feat_vals, y_mapped)
                rs_result.append(rs_column)
                ps_result.append(ps_column)
            else:
                feat_arr2d = np.asarray(feat_vals).reshape((feat_vals.shape[0], 1))
                rs_column, ps_column = self.selector(feat_arr2d, y_mapped)
                rs_result.append(rs_column[0])
                ps_result.append(ps_column[0])

        return rs_result, ps_result


def compute_averaged_metrics(y_truth, y_score, compute_metric):
    """ Compute metrics by averaging over target class values for multiclass
    y_score should contain score vector for binary case and scores matrix for multiclass case"""
    y_values = np.unique(y_truth)
    sum_results = 0.0
    count_results = 0
    for y_value in y_values:
        y_one_vs_all = [1 if y == y_value else 0 for y in y_truth]
        has_0 = 0 in y_one_vs_all
        has_1 = 1 in y_one_vs_all
        if not (has_0 and has_1):
            print("Warning: while computing metric ignored unknown class")
            continue
        scores_y_one_vs_all = [score[y_value] for score in y_score]
        metrics_value = compute_metric(y_one_vs_all, scores_y_one_vs_all)
        sum_results += metrics_value
        count_results += 1
    return 0.0 if count_results == 0 else sum_results / count_results


def accuracy_from_confusion_matrix(y_truth, y_score, confusion_matrix):
    """ computes accuracy count from confusion matrix"""
    sample_count = len(y_truth)
    if sample_count < 1:
        return 0.0
    if len(y_score) != sample_count:
        raise ValueError("Score size is different from sample size")
    accurate_count = 0
    for index in range(confusion_matrix.shape[0]):
        accurate_count += confusion_matrix[index, index]

    return float(accurate_count) / float(sample_count)


class ConfidenceInterval(object):
    """Represents a confidence interval for an estimate, including (optionally) a p value"""
    def __init__(self, estimate, low, high, pval=None):
        """

        :param estimate: point estimate
        :param low: lower bound
        :param high: upper bound
        :param pval: p value (default: None)

        """
        assert low <= estimate <= high
        if pval is not None and not np.isnan(pval):
            assert 0.0 <= pval <= 1.0
        self.estimate, self.low, self.high, self.pval = estimate, low, high, pval

    def __str__(self):
        if self.pval is not None and not np.isnan(self.pval):
            pval_str = ' p={}'.format(format_p_value(self.pval, True))
        else:
            pval_str = ''

        return '{:.3f} ({:.3f} - {:.3f}){}'.format(self.estimate, self.low, self.high, pval_str)


class ROC(object):
    """\
    Container for a receiver operating characteristic (ROC) curve.

    """
    def __init__(self, fpr, tpr, thresholds, y_true=None, y_pred=None, ci_width=95):
        """Creates ROC from false positive rate, true positive rate, and scores"""
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.y_true, self.y_pred = np.asarray(y_true), np.asarray(y_pred)
        self.ci_width = float(ci_width)

        self._auc = sklearn.metrics.auc(self.fpr, self.tpr, reorder=True)
        self._ci = None

    @property
    def auc_ci(self):
        """The AUC confidence interval computed with 10k rounds of bootstrapping"""
        if self._ci is not None:
            return self._ci
        elif self.y_true is None or self.y_pred is None:
            raise ValueError("Cannot compute confidence interval without y_true and y_pred")

        rnd = RandomState(seed=0xC0FFEE)
        aucs = []
        for _ in range(10000):
            idx = rnd.randint(0, len(self.y_true), len(self.y_true))
            if len(set(self.y_true[idx])) < 2:
                continue  # skip sets without both labels
            aucs.append(roc_auc_score(self.y_true[idx], self.y_pred[idx]))

        delta = (100.0 - self.ci_width) / 2

        return ConfidenceInterval(self.auc, np.percentile(aucs, delta), np.percentile(aucs, 100.0 - delta))

    @staticmethod
    def from_scores(y_true, y_pred):
        """Creates a ROC from true/predicted scores"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return ROC(fpr, tpr, thresholds, y_true, y_pred)

    @property
    def auc(self):
        """Computes and returns the area under the ROC curve"""
        return self._auc

    @property
    def dataframe(self):
        """Builds a Pandas dataframe from the ROC"""
        return pd.DataFrame({'fpr': self.fpr, 'tpr': self.tpr, 'thresholds': self.thresholds})
