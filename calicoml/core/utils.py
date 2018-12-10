# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function


import collections

import sys
from datetime import datetime

import os
import re

import logging
import nose

import numpy as np
import pandas as pd
import prettytable


def assert_equal_with_nan(actual, expected):
    """
    :param actual: The observed value
    :param expected: The expected value
    :return: Whether the observed and expected values are equal
    """
    if np.isnan(actual) and np.isnan(expected):
        return True
    return nose.tools.assert_equal(actual, expected)


def is_numeric(x):
    """Tests to see if a character is numeric"""
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def binarize_seq(xs, pos_value):
    """Binarizes a sequence into 1s (positive) and 0s (negative) given a positive value. Everything not equal
        to the positive value will be set to 0"""
    return np.asarray([1 if x == pos_value else 0 for x in xs])


def prevalence(y, pos_value=None):
    """\
    Computes prevalence given truth labels.

    :param y: truth labels (must be 0 or 1)
    :param pos_value: value to treat as a positive. If None, will assume input is a sequence of 0s and 1s.
    :return: prevalence between 0 and 1 (a float)

    """
    if pos_value is not None:
        y = binarize_seq(y, pos_value)
    assert set(y).issubset({0, 1})
    return float(np.sum(y)) / float(len(y))


def format_p_value(p_val, significance=False):
    """Pretty formats a p value"""
    def format_value():
        """Formats the value"""
        if not np.isfinite(p_val) or not 0.0 <= p_val <= 1.0:
            raise ValueError('Bad p value: {}'.format(p_val))
        elif p_val in (0, 1):
            return '{:.1f}'.format(p_val)
        elif p_val < 0.001:
            return '{:.2e}'.format(p_val)
        elif p_val < 0.01:
            return '{:.3f}'.format(p_val)
        else:
            return '{:.2f}'.format(p_val)

    def format_sig():
        """Formats the significance indicator"""
        if not significance:
            return ""
        elif p_val <= 0.01:
            return "*" * min(4, int(np.floor(np.log10(1.0 / p_val))) - 1)
        elif p_val <= 0.05:
            return "~"
        else:
            return ""

    return format_value() + format_sig()


def assert_rows_are_concordant(df, ignore_columns=None):
    """\
    Asserts that rows are concordant (have same values) in all but the specified columns

    :param df: DataFrame to check
    :param ignore_columns: columns to ignore. Default: None
    :return: dictionary of concordant columns and their (one unique) value

    """
    if ignore_columns is None:
        ignore_columns = []

    unique_col_values = {col: df[col].unique() for col in df.columns if col not in ignore_columns}
    bad_col_values = {col: unique_vals for col, unique_vals in unique_col_values.items() if len(unique_vals) > 1}
    if len(bad_col_values) > 0:
        raise ValueError('Expected unique values, but got multiple values for at least one DataFrame column: {}'.format(
            ', '.join(['{}={}'.format(k, v) for k, v in bad_col_values.items()])))

    return {k: v for k, (v,) in unique_col_values.items() if k not in ignore_columns}


def partially_reorder_columns(df, new_column_order):
    """\
    Rearranges columns in a DataFrame so that the given subset always comes first and in the specified order.
    Useful for cases when you have a DataFrame with many columns but only care about the order of a handful (e.g.
    you might want sample IDs to come first).

    :param df: DataFrame whose columns to re-order
    :param new_column_order: new order for columns. Columns present in the DataFrame but not in new_column_order will
    be moved to the end in their original order.
    :return: DataFame with re-arranged columns

    """
    return pd.DataFrame(df, columns=new_column_order + [col for col in df.columns if col not in new_column_order])


def with_numpy_arrays(func):
    """Decorator: automatically converts all lists and tuples to numpy arrays"""
    def convert_one(arg):
        """Converts a single argument if list or tuple, or returns it unchanged"""
        if isinstance(arg, (list, tuple)):
            return np.asarray(arg)
        else:
            return arg

    def inner(*args, **kwargs):
        """Closure over the annotated function"""
        return func(*[convert_one(x) for x in args],
                    **{k: convert_one(v) for k, v in kwargs.items()})

    return inner


def format_dictionary(adict, sep=', ', sort_first=True):
    """\
    Formats a dictionary into a string of key=value pairs.

    :param adict: dictionary to format
    :param sep: separator. Default ','
    :param sort_first: whether to sort the key-value pairs first before formatting
    :return: a string with the formatted dictionary

    """
    preprocess = sorted if sort_first else lambda x: x
    return sep.join('{}={}'.format(k, v) for k, v in preprocess(iter(adict.items())))


def format_scikit_estimator(estim):
    """\
    Formats a scikit estimator into a human-readable string, using the class name and the parameters returned
    by estimator.get_params()
    :param estim: the scikit estimator to format
    :return: a string with the formatted estimator

    """
    return '{}({})'.format(estim.__class__.__name__, format_dictionary(estim.get_params()))


def clone_random_state(random_state):
    """Clones a numpy RandomState object"""
    cloned_state = np.random.RandomState()
    cloned_state.set_state(random_state.get_state())
    return cloned_state


def majority_label(iterable):
    """Returns the most common element in an iterable"""
    most_common_elt, _ = collections.Counter(iterable).most_common(1)[0]
    return most_common_elt


def get_filename_without_extension(path):
    """Gets the file name without extension"""
    name = os.path.basename(path)
    if '.' in name:
        return re.match(r'([^\.]+)\.\w+', name).group(1)
    else:
        return name


class ReplaceStdStreams(object):
    """Replaces sys.stdout and sys.stderr with custom streams. To be used in 'with' blocks"""
    def __init__(self, new_stdout=sys.stdout, new_stderr=sys.stderr):
        self.new_stdout = new_stdout
        self.new_stderr = new_stderr

        self.old_stdout, self.old_stderr = None, None

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        sys.stdout = self.new_stdout
        sys.stderr = self.new_stderr

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class TimeIt(object):
    """Times the wrapped block"""
    def __init__(self, name):
        self.name = name
        self.start, self.elapsed = None, None

    def __enter__(self):
        self.start = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = datetime.now() - self.start

        print('{} took {}'.format(self.name, self.elapsed))


def setup_analytics_logger(name):
    """Creates a standard analytics logger"""
    logger = logging.getLogger(name)
    if hasattr(logger, 'initialized') and logger.initialized:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-30s: %(levelname)-8s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.initialized = True
    return logger


def pretty_print_dataframe(df, float_format='.4', max_rows=None, index=False, **kwargs):
    """Pretty prints a pandas DataFrame"""
    if max_rows is not None and len(df) > max_rows:
        footer = "\n    ({}/{} rows not shown)".format(len(df) - max_rows, len(df))
        df = df.head(max_rows)
    else:
        footer = ""

    columns = list(df.columns) if not index else [df.index.name or 'index'] + list(df.columns)

    table = prettytable.PrettyTable(columns)
    for row_values in df.itertuples():
        table.add_row(row_values[1:] if not index else list(row_values))
    table.float_format = float_format
    str_table = table.get_string() + footer
    print(str_table, **kwargs)
    return str_table
