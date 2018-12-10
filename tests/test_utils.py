# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

import traceback

import six

from calicoml.core.utils import binarize_seq, format_p_value, prevalence, \
    assert_rows_are_concordant, partially_reorder_columns, is_numeric, majority_label, get_filename_without_extension, \
    ReplaceStdStreams, pretty_print_dataframe
import nose

import numpy as np
import pandas as pd


def test_pretty_print_dataframe():
    """Validates that we can print data frames"""
    df = pd.DataFrame(columns=['A', 'B', 'C'], data=[[0, 1.0 / 3, 'foo'], [10, 20, 'bar']])
    nose.tools.eq_(pretty_print_dataframe(df), """+----+---------+-----+
| A  |    B    |  C  |
+----+---------+-----+
| 0  |  0.3333 | foo |
| 10 | 20.0000 | bar |
+----+---------+-----+""")


def test_binarize_seq():
    """Tests binarize_seq"""
    def checkme(pos_value, xs, expected):
        """Utility"""
        nose.tools.assert_list_equal(list(binarize_seq(xs, pos_value)), expected)

    for pos_val in [1, 'Y', 'yes', True, 'maybe']:
        yield checkme, pos_val, [], []

    yield checkme, 1, [1, 0, 1, 0], [1, 0, 1, 0]
    yield checkme, 0, [1, 0, 1, 0], [0, 1, 0, 1]
    yield checkme, 'Y', ['Y', 'Y', 'N', 'N', 'Y'], [1, 1, 0, 0, 1]
    yield checkme, 'Y', ['Y', 'Y', 'maybe', 'missing', 'Y'], [1, 1, 0, 0, 1]


def test_is_numeric():
    """ Tests is_numeric """
    def checkme(number, expected_output):
        """Utility"""
        nose.tools.assert_equal(is_numeric(number), expected_output)

    yield checkme, 5.0, True
    yield checkme, 2, True
    yield checkme, 'a', False
    yield checkme, '%', False
    yield checkme, ['a', 2, 5.6789, '%'], False


def test_format_p_value():
    """Tests p value formatting"""
    nose.tools.eq_(format_p_value(0.0), '0.0')
    nose.tools.eq_(format_p_value(1.0), '1.0')
    nose.tools.eq_(format_p_value(0.2), '0.20')
    nose.tools.eq_(format_p_value(0.22), '0.22')
    nose.tools.eq_(format_p_value(0.223), '0.22')
    nose.tools.eq_(format_p_value(0.226), '0.23')
    nose.tools.eq_(format_p_value(0.049), '0.05')
    nose.tools.eq_(format_p_value(0.0221), '0.02')
    nose.tools.eq_(format_p_value(0.01), '0.01')
    nose.tools.eq_(format_p_value(0.0151), '0.02')
    nose.tools.eq_(format_p_value(0.0002), '2.00e-04')
    nose.tools.assert_raises(ValueError, lambda: format_p_value(np.inf))
    nose.tools.assert_raises(ValueError, lambda: format_p_value(-np.inf))
    nose.tools.assert_raises(ValueError, lambda: format_p_value(2.0))
    nose.tools.assert_raises(ValueError, lambda: format_p_value(-0.001))


def test_prevalence():
    """Tests the prevalence computation"""
    def checkme(n_pos, n_neg, pos_val, neg_val):
        """Utility"""
        y = np.asarray(n_pos * [pos_val] + n_neg * [neg_val])
        for _ in range(10):
            y_shuffled = np.random.permutation(y)
            nose.tools.eq_(prevalence(y_shuffled, pos_val), float(n_pos) / (n_pos + n_neg))

    for n_pos in [0, 1, 10, 100]:
        for n_neg in [0, 1, 10, 100]:
            if n_pos + n_neg == 0:
                continue

            for pos_val, neg_val in [(1, 0), ('Y', 'N')]:
                yield checkme, n_pos, n_neg, pos_val, neg_val


def test_rows_are_concordant():
    """Verifies that assert_rows_are_concordant works as expected"""
    df = pd.DataFrame(columns=['A', 'B', 'C'],
                      data=[['foo', 0, 1],
                            ['foo', 2, 1],
                            ['bar', 0, 1],
                            ['baz', 0, 1]])

    # Things that should work
    nose.tools.eq_({'C': 1}, assert_rows_are_concordant(df, ignore_columns=['A', 'B']))
    nose.tools.eq_({'A': 'foo', 'C': 1}, assert_rows_are_concordant(df[df['A'] == 'foo'], ignore_columns=['B']))
    nose.tools.eq_({'A': 'bar', 'B': 0, 'C': 1}, assert_rows_are_concordant(df[df['A'] == 'bar']))

    # Things that should raise a ValueError
    nose.tools.assert_raises(ValueError, lambda: assert_rows_are_concordant(df))
    nose.tools.assert_raises(ValueError, lambda: assert_rows_are_concordant(df[df['A'] == 'foo']))
    nose.tools.assert_raises(ValueError, lambda: assert_rows_are_concordant(df[df['A'] == 'foo'], ignore_columns=['C']))


def test_partially_reorder_columns():
    """Verifies that partially_reorder_corlumns works as expected"""
    def checkme(df, partial_order, expected_order):
        """Utility"""
        rearranged_df = partially_reorder_columns(df, partial_order)
        nose.tools.assert_list_equal(expected_order, list(rearranged_df.columns))

        for col in df.columns:
            np.testing.assert_array_equal(df[col], rearranged_df[col])

    df = pd.DataFrame(data=np.arange(100).reshape(10, 10), columns=['C{}'.format(c) for c in range(10)])
    yield checkme, df, [], ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    yield checkme, df, ['C9'], ['C9', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    yield checkme, df, ['C9', 'C8'], ['C9', 'C8', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    yield checkme, df, ['C9', 'C8', 'C0', 'C5'], ['C9', 'C8', 'C0', 'C5', 'C1', 'C2', 'C3', 'C4', 'C6', 'C7']


def test_majority_label():
    """Verifies that we correctly select the most common element"""
    def checkme(expected_val, values):
        """Utility"""
        if not hasattr(expected_val, '__iter__'):
            expected_val = {expected_val}

        nose.tools.assert_in(majority_label(values), expected_val)

    yield checkme, 0, [0] * 30 + [1] * 10 + [2] * 20
    yield checkme, 1, [0] * 10 + [1] * 30 + [2] * 20
    yield checkme, 2, [0] * 10 + [1] * 20 + [2] * 30

    # Multiple most frequent values
    yield checkme, {0, 2}, [0] * 30 + [1] * 20 + [2] * 30
    yield checkme, {0, 1, 2}, [0] * 30 + [1] * 30 + [2] * 30


def test_get_filename_without_extension():
    """Verifies that we correctly get the filename without the extension"""
    nose.tools.eq_(get_filename_without_extension('bar'), 'bar')
    nose.tools.eq_(get_filename_without_extension('bar.txt'), 'bar')
    nose.tools.eq_(get_filename_without_extension('bar.txt.gzip'), 'bar')
    nose.tools.eq_(get_filename_without_extension('/a/b/c/bar.txt.gzip'), 'bar')


def test_replace_std_streams():
    """Tests that we correctly replace standard system streams and then restore them to their old values"""
    out = six.StringIO()
    err = six.StringIO()

    with ReplaceStdStreams(out, err):
        try:
            print('hello world')
            raise ValueError('test exception')
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()

    nose.tools.eq_(out.getvalue(), 'hello world\n')
    nose.tools.ok_('test exception' in err.getvalue())

    # outside of the 'with' block the out an err buffers should no longer be getting changed
    try:
        print('outside scope')
        raise ValueError('outside scope')
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()

    nose.tools.ok_('outside scope' not in out)
    nose.tools.ok_('outside scope' not in err)
