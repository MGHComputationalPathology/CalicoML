# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

import numpy as np
import pandas as pd

import nose
from collections import Counter

from calicoml.core.algo.cross_validation import RepeatedStratifiedKFold, CVSplitGenerator, LearningCurveCVGenerator, \
    GroupingCVSplitGenerator
from calicoml.core.problem import Problem
from calicoml.core.utils import prevalence


def test_repeated_stratified_kfold():
    """Tests that RepeatedStratifiedKFold works correctly"""
    def checkme(n_folds, n_repartitions):
        """Utility"""
        y = np.asarray([0, 0, 0, 1] * 100)
        cv = RepeatedStratifiedKFold(y, n_folds=n_folds, n_repartitions=n_repartitions)
        splits = list(cv)

        # Total number of train/test splits should match
        nose.tools.eq_(len(splits), n_folds * n_repartitions)
        nose.tools.eq_(len(cv), n_folds * n_repartitions)

        train_occurrences = Counter()
        test_occurrences = Counter()

        seen_train_sets = set()
        seen_test_sets = set()
        for train, test in splits:
            # There should be no overlap between train/test
            nose.tools.eq_(set(train) & set(test), set([]))

            # Train and test should be perfectly complementary and with no duplicates
            nose.tools.eq_(len(set(train)) + len(set(test)), len(y))
            nose.tools.eq_(sorted(set(train) | set(test)), list(range(len(y))))

            # Prevalence should be approximately equal to the global prevalence
            np.testing.assert_allclose(prevalence(y[test]), prevalence(y), atol=0.01)

            # Record which samples we saw
            train_occurrences.update(train)
            test_occurrences.update(test)

            # Sets should be unique
            nose.tools.ok_(tuple(sorted(train)) not in seen_train_sets)
            nose.tools.ok_(tuple(sorted(test)) not in seen_test_sets)
            seen_train_sets.add(tuple(sorted(train)))
            seen_test_sets.add(tuple(sorted(test)))

        for sample_idx in range(len(y)):
            # Should have seen each sample in test exactly once per shuffle
            nose.tools.eq_(test_occurrences[sample_idx], n_repartitions)

            # Should have seen each sample in train in all folds except the one where it was in test
            nose.tools.eq_(train_occurrences[sample_idx], n_repartitions * (n_folds - 1))

    for n_folds in 2, 4, 10, 20:
        for n_repartitions in [1, 2, 10]:
            yield checkme, n_folds, n_repartitions


class DataFrameWrapper(pd.DataFrame):
    """Overwrites str and repr to make pd.DataFrame play nicely with TeamCity. This is necessary because when DataFrames
    are used with yield, str(DataFrame) becomes a part of the test's name, and exceedingly long names cause problems."""
    def __init__(self, name, *args, **kwargs):
        super(DataFrameWrapper, self).__init__(*args, **kwargs)
        self._df_name = name

    def __str__(self):
        return self._df_name

    def __repr__(self):
        return self._df_name


def mock_frame(id_prefix, n_samples):
    """Mocks a trivial DataFrame with the given prefix for sample IDs and the given number of samples"""
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even")

    sample_names = ['{}-{}'.format(id_prefix, idx) for idx in range(n_samples)]
    return DataFrameWrapper(name='DataFrame {} with {} samples'.format(id_prefix, n_samples),
                            data={'name': sample_names, 'f1': range(n_samples), 'f2': range(n_samples),
                                  'y': [0, 1] * int(n_samples / 2)},
                            index=sample_names, columns=['name', 'y', 'f1', 'f2'])


def test_filtered_cross_validation():
    """Validates that we correctly generate CV splits with custom train and test filters"""

    def empty_frame():
        """Mocks an empty frame"""
        return mock_frame('empty', 0)

    def checkme(cv_df, train_df, test_df, ignore_df):
        """Test utility: validates CV split properties for the given CV/train-only/test-only/ignored data frames"""
        for setname, df in [('cv', cv_df), ('train', train_df), ('test', test_df), ('ignore', ignore_df)]:
            df['set'] = setname

        prob = Problem(pd.concat([pd.DataFrame(df) for df in (cv_df, train_df, test_df, ignore_df)]),
                       ['f1', 'f2'], 'y', 1)
        cv_gen = CVSplitGenerator(prob, 10, 2, random_state=np.random.RandomState(0xC0FFEE),
                                  train_filter=lambda meta: meta['set'] == 'cv' or meta['set'] == 'train',
                                  test_filter=lambda meta: meta['set'] == 'cv' or meta['set'] == 'test')
        cv_gen = list(cv_gen)  # so that we can check the length
        nose.tools.eq_(len(cv_gen), 20 if len(cv_df) > 0 else 1)

        for cv_train, cv_test in cv_gen:
            np.testing.assert_allclose(len(cv_train), 0.9 * len(cv_df) + len(train_df), atol=1.0)  # 90% CV + train-only
            np.testing.assert_allclose(len(cv_test), 0.1 * len(cv_df) + len(test_df), atol=1.0)  # 10% CV + test-only

            # Sanity check: no train/test overlap
            nose.tools.eq_(set(cv_train.sample_ids) & set(cv_test.sample_ids), set())

            # Train samples: must be from either CV or train-only set
            # Test samples: must be from either CV or test-only set
            nose.tools.ok_(all([sample in cv_df.index or sample in train_df.index for sample in cv_train.sample_ids]))
            nose.tools.ok_(all([sample in cv_df.index or sample in test_df.index for sample in cv_test.sample_ids]))

            # All train-only and all test-only samples should be present
            nose.tools.ok_(all([sample in cv_train.sample_ids for sample in train_df.index]))
            nose.tools.ok_(all([sample in cv_test.sample_ids for sample in test_df.index]))

            # Samples in ignore_df should never be emitted
            nose.tools.ok_(not any([sample in ignore_df.index for sample in cv_train.sample_ids + cv_test.sample_ids]))

    # Standard CV: all samples participate in CV
    yield checkme, mock_frame('A', 1000), empty_frame(), empty_frame(), empty_frame()

    # CV + train-only samples
    yield checkme, mock_frame('A', 1000), mock_frame('B', 100), empty_frame(), empty_frame()

    # CV + test-only samples
    yield checkme, mock_frame('A', 1000), empty_frame(), mock_frame('B', 100), empty_frame()

    # CV + train-only + test-only samples
    yield checkme, mock_frame('A', 1000), mock_frame('B', 100), mock_frame('C', 100), empty_frame()

    # CV + train-only + test-only + Ignored samples
    yield checkme, mock_frame('A', 1000), mock_frame('B', 100), mock_frame('C', 100), mock_frame('D', 100)

    # Degenerate case: no CV samples
    yield checkme, empty_frame(), mock_frame('B', 100), mock_frame('C', 100), mock_frame('D', 100)


def sanity_check_cv_generator(prob, cv_gen):
    """\
    Runs sanity checks that should work for any valid CV generator:

    1. No duplicates in train
    2. No duplicates in test
    3. No train/test overlap
    4. All samples appear in test, and they appear the exact same number of times
    5. The reported n_total_splits matches reality

    """
    splits_seen = 0

    test_occurrences = Counter()
    train_occurrences = Counter()
    for train, test in cv_gen:
        nose.tools.eq_(sorted(set(train.sample_ids)), sorted(train.sample_ids))  # no duplicates in training set
        nose.tools.eq_(sorted(set(test.sample_ids)), sorted(test.sample_ids))  # no duplicates in test set
        nose.tools.eq_(set(train.sample_ids) & set(test.sample_ids), set())  # no samples overlap

        # Count the samples we've seen
        train_occurrences.update(train.sample_ids)
        test_occurrences.update(test.sample_ids)

        splits_seen += 1

    nose.tools.eq_(cv_gen.n_total_splits, splits_seen)
    nose.tools.eq_(set(test_occurrences.keys()), set(prob.sample_ids))  # must have seen all samples in test

    # each sample must have appeared the same number of times in train
    nose.tools.eq_(len(set(test_occurrences.values())), 1)


def test_not_enough_examples():
    """Validates that we fail if we don't have enough positive and/or negative examples"""
    def checkme(n_pos, n_neg, n_folds, fail_or_pass):
        """Utility"""
        assert fail_or_pass in {'fail', 'pass'}
        y = np.asarray([1] * n_pos + [0] * n_neg)
        X = np.zeros((y.shape[0], 2))
        df = pd.DataFrame(data=X, columns=['f1', 'f2'])
        df['y'] = y
        prob = Problem(df, ['f1', 'f2'], 'y', 1)

        if fail_or_pass == 'pass':
            runner = lambda thunk: thunk()
        else:
            runner = lambda thunk: nose.tools.assert_raises(ValueError, thunk)

        cv = CVSplitGenerator(prob, n_folds, 2, random_state=np.random.RandomState(0xC0FFEE))
        runner(lambda: next(cv.__iter__()))

    yield checkme, 50, 50, 10, 'pass'
    yield checkme, 10, 50, 10, 'pass'
    yield checkme, 50, 10, 10, 'pass'

    yield checkme, 50, 50, 51, 'fail'  # not enough positives and negatives
    yield checkme, 10, 50, 51, 'fail'  # not enough positives and negatives
    yield checkme, 50, 10, 51, 'fail'  # not enough positives and negatives

    yield checkme, 50, 50, 50, 'pass'  # just enough positives and negatives
    yield checkme, 10, 50, 50, 'fail'  # not enough positives
    yield checkme, 50, 10, 50, 'fail'  # not enough negatives


def test_learning_curve_cross_validation():
    """Tests learning curve subsampling fractions from 0.0 to 1.0"""

    def checkme(fraction):
        """Tests learning curve CV downsampling
        :param fraction: Float/double in [0,1] (inclusive on both ends) - sampling rate for CV generation
        """
        problem_size = 1000
        prob = Problem(mock_frame('A', problem_size), ['f1', 'f2'], 'y', 1)
        cv_gen = CVSplitGenerator(prob, 10, 2, random_state=np.random.RandomState(0xC0FFEE))
        sanity_check_cv_generator(prob, cv_gen)
        learning_curve = LearningCurveCVGenerator(fraction, cv_gen, random_state=cv_gen.random_state)

        train_occurrences = Counter()
        for train, test in learning_curve:
            nose.tools.eq_(len(train), int(problem_size * 0.9 * fraction))  # subsampled 900/100 split (10 folds)
            nose.tools.eq_(len(test), int(problem_size * 0.1))  # verify that test set is 1/10 of the problem size

            train_occurrences.update(train.sample_ids)

        # This is a fairly weak test, but in general it's difficult to predict how many unique train samples we'll see,
        # especially when both the subsampling fraction and the total number of splits are small. It does protect
        # us against truly terrible bugs though, e.g. if we accidentally return the same training set over and over.
        nose.tools.assert_greater_equal(len(set(train_occurrences.keys())), problem_size * fraction)

    # actual function
    for fraction in np.linspace(0.0, 1.0, 10):
        yield checkme, fraction


def test_grouping_cross_validation():
    """Validates that the grouping CV generator works as expected"""
    df = mock_frame('A', 100)
    df['group'] = ['group{}'.format(idx) for idx in range(20)] * 5  # 20 groups repeated 5 times
    prob = Problem(df, ['f1', 'f2'], 'y', 1)

    cv = GroupingCVSplitGenerator(prob, group_by='group', n_folds=10, n_repartitions=10)
    sanity_check_cv_generator(prob, cv)

    for train, test in cv:
        nose.tools.eq_(train.n_samples, 18 * 5)  # 18/20 groups, 5 samples per group
        nose.tools.eq_(test.n_samples, 2 * 5)  # 2/20 groups, 5 samples per group

        nose.tools.eq_(set(train.dataframe['group']) & set(test.dataframe['group']), set())  # no groups overlap
