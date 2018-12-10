# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

import pandas as pd

from calicoml.core.utils import clone_random_state, majority_label
import numpy as np

from numpy.random.mtrand import RandomState
from sklearn.cross_validation import StratifiedKFold


class RepeatedStratifiedKFold(object):
    """\
    Wrapper around scikit's StratifiedKFold which repeats the cross-validaton many times on re-shuffled data

    """
    def __init__(self, y, n_folds=10, n_repartitions=10, random_state=None, check_binary=True):
        """\

        :param y: the truth labels. Need to be an array with one label per sample, where 1=positive and 0=negative
        :param n_folds: number of folds to run for each shuffle
        :param n_repartitions: total number of shuffles to run
        :param random_state: numpy RandomState or None (default: None)
        :param check_binary: if True allow only binary problem, set to False for multiclass
        :return:

        """
        if n_folds < 2:
            raise ValueError('Number of folds should be at least 2')
        elif n_repartitions < 1:
            raise ValueError('Number of repartitions should be 1 or more')
        elif check_binary and len(set(y)) != 2:
            raise ValueError('Not a binary problem. Unique labels: {}'.format(', '.join(sorted(set(y)))))

        self.random_state = random_state or RandomState()
        self.n_folds = n_folds
        self.n_repartitions = n_repartitions
        self.y = y

    def __len__(self):
        return self.n_folds * self.n_repartitions

    def __iter__(self):
        """Iterator over train/test splits"""
        for _ in range(self.n_repartitions):
            order = self.random_state.permutation(len(self.y))
            base_cv = StratifiedKFold(y=self.y[order], n_folds=self.n_folds, random_state=self.random_state)
            for train, test in base_cv:
                yield order[train], order[test]


class CVSplitGenerator(object):
    """CV generator which operates on Problem instances. For now it's just a thin wrapper around
    RepeatedStratifiedKFold, but it will grow in the future as we add functionality that requires Problem
    metadata (e.g. slicing along gender & age, different cohorts in train and test etc)."""
    def __init__(self, problem, n_folds=10, n_repartitions=10, random_state=None, train_filter=None, test_filter=None):
        """\

        :param problem: a Problem instance
        :param n_folds: number of folds to run
        :param n_repartitions: total number of shuffles to run
        :param random_state: numpy RandomState or None (default: None)

        """
        self.problem = problem
        self.n_folds = n_folds
        self.n_repartitions = n_repartitions
        self.random_state = random_state or RandomState()
        self.train_filter = train_filter if train_filter is not None else lambda _: True
        self.test_filter = test_filter if test_filter is not None else lambda _: True

    def clone(self):
        """Clones this CVSplitGenerator"""
        return CVSplitGenerator(self.problem, self.n_folds, self.n_repartitions, clone_random_state(self.random_state),
                                self.train_filter, self.test_filter)

    @property
    def n_total_splits(self):
        """Returns the total number of train/test splits"""
        return self.n_folds * self.n_repartitions

    def __iter__(self):
        def get_set_name(in_train, in_test):
            """Utility: gets a string identifier of the set a sample is in"""
            return {(True, True): 'train_and_test',
                    (True, False): 'train_only',
                    (False, True): 'test_only',
                    (False, False): 'ignore'}[(in_train, in_test)]

        sample_sets = np.asarray([get_set_name(self.train_filter(meta), self.test_filter(meta))
                                  for meta in self.problem])
        cv_problem = self.problem[sample_sets == 'train_and_test']
        train_only_problem = self.problem[sample_sets == 'train_only']
        test_only_problem = self.problem[sample_sets == 'test_only']

        if len(cv_problem) == 0:  # the degenerate case: no overlap between train and test sets
            yield train_only_problem, test_only_problem
            return

        if cv_problem.should_be_binary and\
                (cv_problem.n_positives < self.n_folds or cv_problem.n_negatives < self.n_folds):
            raise ValueError('Requested CV with {} folds, but there are only {} positives and {} negatives in '
                             'the CV frame'.format(self.n_folds, cv_problem.n_positives, cv_problem.n_negatives))

        base_cv = RepeatedStratifiedKFold(cv_problem.y, n_folds=self.n_folds, n_repartitions=self.n_repartitions,
                                          random_state=self.random_state, check_binary=cv_problem.should_be_binary)

        for train_idx, test_idx in base_cv:
            yield cv_problem.iloc(train_idx) + train_only_problem, cv_problem.iloc(test_idx) + test_only_problem


class LearningCurveCVGenerator(object):
    """\
    CV generator for learning curves. Takes a fraction and a CV generator, then subsamples it

    """

    def __init__(self, fraction, cv, random_state=None):
        """Builds a LearningCurveCVGenerator around a CV Split Generator
        :param fraction: Float in (0,1) - fraction of training test to randomly forward.
        :param cv: CVSplitGenerator to subsample
        :param random_state: np.random.RandomState object. Default: None
        """
        if fraction < 0 or fraction > 1:
            raise ValueError("LearningCurveCVGenerator: fraction must be in range (0,1)")
        self.cv = cv
        self.fraction = fraction
        self.random_state = random_state or np.random.RandomState()

    def clone(self):
        """Clones this LearningCurveCVGenerator"""
        return LearningCurveCVGenerator(self.fraction, self.cv.clone(), clone_random_state(self.random_state))

    @property
    def n_total_splits(self):
        """Returns the total number of train/test splits"""
        return self.cv.n_total_splits

    def __iter__(self):
        for train, test in self.cv:
            train_idx_to_take = self.random_state.permutation(len(train))
            train_idx_to_take = train_idx_to_take[:int(len(train) * self.fraction)]
            yield train.iloc(train_idx_to_take), test


class GroupingCVSplitGenerator(object):
    """\
    Generates cross validation splits such that samples within the same 'group' are never
    split between train and test. For example, you might want to test the predictive performance
    of an email classifier on individual messages, but you don't want messages from the same author
    to appear in both train and test.

    GroupingCVSplitGenerator stratifies on groups, using the majority label for each group.

    """

    def __init__(self, problem, group_by, n_folds=10, n_repartitions=10, *args, **kwargs):
        """\

        :param problem: Problem instance
        :param group_by: name of the column on which to group samples
        :param n_folds: number of folds
        :param n_repartitions: number of random repartitions

        """
        self.problem = problem
        self.group_by = group_by
        self.n_folds = n_folds
        self.n_repartitions = n_repartitions

        self._cv_args = args
        self._cv_kwargs = kwargs

    @property
    def n_total_splits(self):
        """Total number of CV splits"""
        return self.n_folds * self.n_repartitions

    def __iter__(self):
        """Iterator over train/test splits"""
        def get_groups(df, groups_to_keep):
            """Returns a DataFrame subset containing rows belonging to the given set of groups"""
            return df[[grp in groups_to_keep for grp in df[self.group_by]]]

        df = self.problem.dataframe

        # Compute the majority label for each group
        majority_labels = {group: majority_label(sdf[self.problem.outcome_column])
                           for group, sdf in df.groupby(by=[self.group_by])}

        # Group data by the key, then generate splits based on the key
        grouped_df = pd.DataFrame(df.drop_duplicates(self.group_by))
        grouped_df[self.problem.outcome_column] = [majority_labels[group] for group in grouped_df[self.group_by]]
        grouped_prob = self.problem.set_data(grouped_df)

        grouped_cv = CVSplitGenerator(grouped_prob, n_folds=self.n_folds, n_repartitions=self.n_repartitions,
                                      *self._cv_args, **self._cv_kwargs)

        for train, test in grouped_cv:
            ungrouped_train_df = get_groups(df, set(train.dataframe[self.group_by].unique()))
            ungrouped_test_df = get_groups(df, set(test.dataframe[self.group_by].unique()))
            yield train.set_data(ungrouped_train_df), test.set_data(ungrouped_test_df)
