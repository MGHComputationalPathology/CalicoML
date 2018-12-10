# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from calicoml.core.problem import Problem, ProblemVectorizer
import nose

import numpy as np
import pandas as pd


def mock_problem():
    """Mocks DataFrames for a simple learning problem"""
    index = ['S-{}'.format(idx) for idx in range(5)]
    feat_df = pd.DataFrame(columns=['gene1', 'gene2'], data=0.1 * np.arange(10).reshape((5, 2)),
                           index=index)
    clinical_df = pd.DataFrame(columns=['gender', 'disease'],
                               data=[['male', 'yes'],
                                     ['male', 'no'],
                                     ['female', 'no'],
                                     ['female', 'yes'],
                                     ['female', 'yes']],
                               index=index)
    combined_df = pd.concat([feat_df, clinical_df], axis=1)
    return feat_df, clinical_df, combined_df


def mock_badvector_problem():
    """Mocks noisy DataFrames for testing vectorization"""
    feat_df, _, combined_df = mock_problem()

    # Feature columns are numeric, but we want to assign bad non-numeric values to test our pre-processing.
    # To be able to do this, we need to set the datatype to np.object.
    for feat_name in feat_df.columns:
        feat_df[feat_name] = pd.Series(feat_df[feat_name], dtype=np.object, copy=True)

    feat_df.loc['S-1']['gene1'] = 'invalid'
    feat_df.loc['S-2']['gene1'] = 'nul'
    feat_df.loc['S-0']['gene2'] = 'a'
    feat_df.loc['S-1']['gene2'] = 'b'
    feat_df.loc['S-3']['gene2'] = 0.5
    feat_df.loc['S-2']['gene2'] = 'c'
    feat_df.loc['S-4']['gene2'] = 'd'
    feat_df['disease'] = combined_df['disease']
    return Problem(feat_df, ['gene1', 'gene2'], 'disease', 'yes')


def test_problem_creation():
    """Validates that Problem instances behave as expected"""
    feat_df, _, combined_df = mock_problem()
    prob = Problem(combined_df, feat_df.columns, 'disease', 'yes')
    nose.tools.eq_(prob.n_features, 2)
    nose.tools.eq_(prob.n_samples, 5)
    nose.tools.assert_list_equal(prob.sample_ids, ['S-0', 'S-1', 'S-2', 'S-3', 'S-4'])
    np.testing.assert_array_equal(prob.y, [1, 0, 0, 1, 1])
    np.testing.assert_array_equal(prob.dataframe.values, combined_df.values)
    nose.tools.eq_(prob.X.shape[0], prob.n_samples)
    nose.tools.eq_(prob.X.shape[1], prob.n_features)
    np.testing.assert_array_equal(prob.X, feat_df.values)

    # Try subsetting features and a different outcome variable
    sub_prob = Problem(combined_df, ['gene2'], 'gender', 'male')
    nose.tools.eq_(sub_prob.n_features, 1)
    nose.tools.eq_(sub_prob.n_samples, 5)
    nose.tools.assert_list_equal(sub_prob.sample_ids, ['S-0', 'S-1', 'S-2', 'S-3', 'S-4'])
    np.testing.assert_array_equal(sub_prob.y, [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(sub_prob.dataframe.values, combined_df.values)
    nose.tools.eq_(sub_prob.X.shape[0], sub_prob.n_samples)
    nose.tools.eq_(sub_prob.X.shape[1], sub_prob.n_features)
    np.testing.assert_array_equal(sub_prob.X.ravel(), feat_df.values[:, 1])


def test_bad_problems():
    """Validates that we can't create Problems with bad data and/or parameters"""
    def assert_fails(*args, **kwargs):
        """Utility: calls the Problem ctor with the given arguments and expects it to raise an error"""
        nose.tools.assert_raises(ValueError, lambda: Problem(*args, **kwargs))

    _, _, df = mock_problem()
    assert_fails(None, ['gene1'], 'disease', 'yes')  # No data
    assert_fails(df, [], 'disease', 'yes')  # No features
    assert_fails(df, ['gene1', 'gene2', 'disease'], 'disease', 'yes')  # outcome overlaps with features
    assert_fails(df, ['gene1', 'gene2', 'gene3'], 'disease', 'yes')  # gene3 doesn't exist
    assert_fails(df, ['gene1', 'gene2'], 'other_disease', 'yes')  # outcome doesn't exist
    assert_fails(df, ['gene1', 'gene2', 'gene2'], 'disease', 'yes')  # duplicate feature


def assert_metadata_eq(prob_a, prob_b):
    """Utility: asserts that the features, outcome and other metadata are the same for two Problems"""
    nose.tools.assert_list_equal(prob_a.features, prob_b.features)
    nose.tools.eq_(prob_a.outcome_column, prob_b.outcome_column)
    nose.tools.eq_(prob_a.positive_outcome, prob_b.positive_outcome)


def test_problem_slicing():
    """Validates that we can slice problems along the sample axis"""
    _, _, df = mock_problem()
    prob = Problem(df, ['gene1', 'gene2'], 'disease', 'yes')

    male_prob = prob[prob.dataframe['gender'] == 'male']
    assert_metadata_eq(prob, male_prob)
    nose.tools.eq_(male_prob.n_samples, 2)
    nose.tools.eq_(male_prob.n_features, 2)
    np.testing.assert_array_equal(male_prob.y, [1, 0])
    np.testing.assert_array_equal(male_prob.X, prob.X[:2])

    custom_prob = prob.iloc([0, 2, 3])
    assert_metadata_eq(prob, custom_prob)
    nose.tools.eq_(custom_prob.n_samples, 3)
    nose.tools.eq_(custom_prob.n_features, 2)
    np.testing.assert_array_equal(custom_prob.y, [1, 0, 1])
    np.testing.assert_array_equal(custom_prob.X, prob.X[[0, 2, 3]])


def test_problem_vectorize():
    """Tests that we can vectorize Problem instances"""
    def checkme(keep_discrete_columns):
        """Utility"""
        _, _, df = mock_problem()
        prob = Problem(df, ['gene1', 'gender'], 'disease', 'yes')
        vectorized_prob = prob.vectorize(keep_discrete_columns=keep_discrete_columns)
        print(vectorized_prob.dataframe)
        nose.tools.eq_(vectorized_prob.outcome_column, prob.outcome_column)
        nose.tools.eq_(vectorized_prob.positive_outcome, prob.positive_outcome)
        np.testing.assert_array_equal(vectorized_prob.y, prob.y)

        if keep_discrete_columns:
            expected_columns = ['gene1', 'gene2', 'disease', 'gender', 'gender=male', 'gender=female']
            nose.tools.assert_list_equal(list(vectorized_prob.dataframe['gender']), list(prob.dataframe['gender']))
        else:
            expected_columns = ['gene1', 'gene2', 'disease', 'gender=male', 'gender=female']

        nose.tools.assert_list_equal(sorted(list(vectorized_prob.dataframe.columns)), sorted(expected_columns))
        nose.tools.assert_list_equal(vectorized_prob.features, ['gender=female', 'gender=male', 'gene1'])
        np.testing.assert_almost_equal(vectorized_prob.X,
                                       np.asarray([[0, 1, 0.0],
                                                   [0, 1, 0.2],
                                                   [1, 0, 0.4],
                                                   [1, 0, 0.6],
                                                   [1, 0, 0.8]]), decimal=10)

    yield checkme, True
    yield checkme, False


def test_problem_concatenation():
    """Validates that we can concatenate Problem instances"""
    _, _, df = mock_problem()
    df = df.sort_values('gender')  # need to sort so that we can reverse slicing by simple concatenation

    prob = Problem(df, ['gene1', 'gene2'], 'disease', 'yes')
    sub_prob_male = prob[prob.dataframe['gender'] == 'male']
    sub_prob_female = prob[prob.dataframe['gender'] == 'female']
    reconstituted_prob = sub_prob_female + sub_prob_male  # here's where the sort matters

    np.testing.assert_array_equal(reconstituted_prob.dataframe.values, prob.dataframe.values)
    np.testing.assert_array_equal(reconstituted_prob.outcome_column, prob.outcome_column)
    np.testing.assert_array_equal(reconstituted_prob.positive_outcome, prob.positive_outcome)
    nose.tools.assert_list_equal(reconstituted_prob.features, prob.features)
    nose.tools.assert_list_equal(reconstituted_prob.sample_ids, prob.sample_ids)

    # Incompatible outcome columns
    nose.tools.assert_raises(ValueError, lambda: sub_prob_male + Problem(sub_prob_female.dataframe, ['gene1', 'gene2'],
                                                                         'gender', 'male'))

    # Incompatible positive outcome
    nose.tools.assert_raises(ValueError, lambda: sub_prob_male + Problem(sub_prob_female.dataframe, ['gene1', 'gene2'],
                                                                         'disease', 'no'))

    # Incompatible features
    nose.tools.assert_raises(ValueError, lambda: sub_prob_male + Problem(sub_prob_female.dataframe, ['f1'],
                                                                         'disease', 'yes'))


def isnan2(x):
    """\
    Wrapper for numpy.isnan that checks that input is a float.

    :param x: Input
    :return: True if x is a NaN float and false otherwise
    """
    try:
        return np.isnan(x)
    except TypeError:
        return False


def test_vectorize():
    """\
    Tests ProblemVectorizer in problem.py

    """
    def assert_equal_with_nans(lst_a, lst_b):
        """\
        Checks that lists a and b are equal with nose.tools.eq_
        Lists must be of equal length.
        NaNs are handled specially with np.isNaN

        :param lst_a: List 1
        :param lst_b: List 2
        """
        nose.tools.eq_(len(lst_a), len(lst_b))
        for x, y in zip(lst_a, lst_b):
            # Special handling for NaNs
            if isnan2(x) and isnan2(y):
                continue
            if isinstance(x, float) and isinstance(y, float):
                nose.tools.assert_almost_equal(x, y)
            else:
                nose.tools.eq_(x, y)

    vec = ProblemVectorizer(['gene1'], ['gene2'])

    assert_equal_with_nans(vec.preprocess_numeric([1.0, 100, 2.3, 'missing', -8, 'nul']),
                           [1.0, 100, 2.3, float('nan'), -8, float('nan')])
    assert_equal_with_nans(vec.preprocess_discrete(['gene1', 'disease1', 100, -2.1, 'missing', 'vector', 'attribute'],
                                                   'unknown'),
                           ['gene1', 'disease1', 'unknown=100', 'unknown=-2.1', 'missing', 'vector', 'attribute'])

    # test vectorize end-to-end
    prob = mock_badvector_problem()
    vec_prob = vec.fit_apply(prob)

    # the vectorizer pipeline converts NaN to column average: 1.4 / 3.0 in our case
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene1'].values), [0.0, 1.4 / 3.0, 1.4 / 3.0, 0.6, 0.8])
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene2=a'].values), [1, 0, 0, 0, 0])
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene2=b'].values), [0, 1, 0, 0, 0])
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene2=c'].values), [0, 0, 1, 0, 0])
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene2=d'].values), [0, 0, 0, 0, 1])
    assert_equal_with_nans(list(vec_prob.data.dataframe['gene2=discrete=0.5'].values), [0, 0, 0, 1, 0])


def test_permissive_vectorize():
    """Validates that the vectorizer's permissive switch works as expected"""
    def checkme(permissive_or_not, fail_or_pass, expected_numeric, expected_discrete, df_columns):
        """Utility"""
        assert permissive_or_not in {'permissive', 'strict'}
        assert fail_or_pass in {'fail', 'pass'}
        df = pd.DataFrame({col: list(range(10)) for col in df_columns})
        df['y'] = [0, 1] * 5
        prob = Problem(df, df_columns, 'y', 1)
        vec = ProblemVectorizer(expected_numeric=expected_numeric, expected_discrete=expected_discrete,
                                permissive=(permissive_or_not == 'permissive'))
        if fail_or_pass == 'pass':
            vec.fit_apply(prob)
        else:
            nose.tools.assert_raises(ValueError, lambda: vec.fit_apply(prob))

    yield checkme, 'permissive', 'pass', ['a'], ['b'], ['a', 'b']
    yield checkme, 'permissive', 'pass', ['a'], ['b'], ['a']
    yield checkme, 'permissive', 'pass', ['a', 'b', 'c'], ['d'], ['a', 'b', 'c']
    yield checkme, 'permissive', 'pass', ['a', 'b', 'c'], ['d'], ['d']

    yield checkme, 'strict', 'pass', ['a'], ['b'], ['a', 'b']
    yield checkme, 'strict', 'fail', ['a'], ['b'], ['a']
    yield checkme, 'strict', 'fail', ['a', 'b', 'c'], ['d'], ['a', 'b', 'c']
    yield checkme, 'strict', 'fail', ['a', 'b', 'c'], ['d'], ['d']


def test_no_column_overwrite():
    """Validates that we don't overwrite input values if the input contains NaNs in discrete columns"""
    df = pd.DataFrame({'A': ['a', 'aa', float('nan')],
                       'B': ['b', 'bb', 'bbb'],
                       'y': [0, 1, 1]})
    prob = Problem(df, ['A', 'B'], 'y', 1)
    vec = ProblemVectorizer()

    vec_prob = vec.fit_apply(prob, keep_discrete_columns=True)
    vec_df = vec_prob.dataframe

    nose.tools.assert_list_equal(sorted(vec_prob.features), ['A=a', 'A=aa', 'B=b', 'B=bb', 'B=bbb'])

    nose.tools.assert_list_equal(list(vec_df['A=a']), [1, 0, 0])
    nose.tools.assert_list_equal(list(vec_df['A=aa']), [0, 1, 0])

    nose.tools.assert_list_equal(list(vec_df['B=b']), [1, 0, 0])
    nose.tools.assert_list_equal(list(vec_df['B=bb']), [0, 1, 0])
    nose.tools.assert_list_equal(list(vec_df['B=bbb']), [0, 0, 1])

    # Original input columns shouldn't have changed.
    #
    # In the initial implementation, this test failed for column 'A'. This happened
    # because scikit's vectorizer creates an all-zero column with the exact same name if the input is
    # discrete and contains NaNs, which causes the original values to be overwritten.
    nose.tools.assert_list_equal(list(vec_df['A']), list(df['A']))
    nose.tools.assert_list_equal(list(vec_df['B']), list(df['B']))

    nose.tools.assert_list_equal(sorted(vec_df.columns),
                                 sorted(['A', 'A=a', 'A=aa', 'B', 'B=b', 'B=bb', 'B=bbb', 'y']))


def test_y_for_multiclass_slicing():
    """ Testing y method for multiclass"""
    df = pd.DataFrame(columns=['gene', 'number'],
                      data=[['gene1', 'one'],
                            ['gene2', 'two'],
                            ['gene3', 'three'],
                            ['gene4', 'four'],
                            ['gene5', 'five']])
    prob = Problem(df, ['gene'], 'number', None)
    y = prob.y
    nose.tools.assert_list_equal(list(y), [2, 4, 3, 1, 0])

    subset_prob = prob[prob.dataframe['gene'] != 'gene3']
    y_subset = subset_prob.y
    nose.tools.assert_list_equal(list(y_subset), [2, 4, 1, 0])

    subset_df = df[df['gene'] != 'gene3']
    prob_subset_df = Problem(subset_df, ['gene'], 'number', None)
    y_subset_df = prob_subset_df.y
    nose.tools.assert_list_equal(list(y_subset_df), [2, 3, 1, 0])

    prob_subset_df_with_list = Problem(subset_df, ['gene'], 'number', None, prob.label_list)
    y_subset_df_with_list = prob_subset_df_with_list.y
    nose.tools.assert_list_equal(list(y_subset_df_with_list), list(y_subset))

    custom_prob = prob.iloc([0, 2, 3])
    y_custom = custom_prob.y
    nose.tools.assert_list_equal(list(y_custom), [2, 3, 1])

    custom_df = df.iloc[[0, 2, 3]]
    prob_custom_df_with_list = Problem(custom_df, ['gene'], 'number', None)
    y_custom_df_with_list = prob_custom_df_with_list.y
    nose.tools.assert_list_equal(list(y_custom_df_with_list), [1, 2, 0], None)

    prob_custom_df = Problem(custom_df, ['gene'], 'number', None, prob.label_list)
    y_custom_df = prob_custom_df.y
    nose.tools.assert_list_equal(list(y_custom_df), list(y_custom))
