# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from calicoml.core import utils
from calicoml.core.data.sources import PandasDataSource
from calicoml.core.serialization.serializer import get_class_name
from calicoml.core.utils import binarize_seq

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder


class Problem(object):
    """\
    Describes a machine learning problem, including the data, the target outcome and the features to use
    for classification

    """
    def __init__(self, data, features, outcome_column, positive_outcome, label_list=None):
        if data is None:
            raise ValueError('Data cannot be null')
        elif len(features) == 0:
            raise ValueError('Feature set is empty')
        elif len(set(features)) != len(features):
            raise ValueError('Features not unique')

        self.data = PandasDataSource(data) if isinstance(data, pd.DataFrame) else data
        self.features = list(features)
        self.outcome_column = outcome_column
        self.positive_outcome = positive_outcome
        self.label_list = None
        if self.positive_outcome is None:
            if label_list is not None:
                self.label_list = sorted(label_list)
            else:
                label_encoder = LabelEncoder()
                label_encoder.fit(self.dataframe[self.outcome_column])
                if len(label_encoder.classes_) == 2:
                    print(" Warning: please define positive_outcome argument for\
                    Binary problem to get improved reporting")
                else:
                    self.label_list = label_encoder.classes_
                print("setting label_list: " + str(self.label_list))
        self._X = None

        if outcome_column in features:
            raise ValueError('Outcome variable {} overlaps with the feature set'.format(self.outcome_column))
        elif len(set(self.data.dataframe.columns)) != len(self.data.dataframe.columns):
            raise ValueError('Problem has duplicate columns')

        self._check_frame()

    @property
    def datatypes(self):
        """Returns a dictionary of datatypes for each column in this Problem"""
        return {col: self.dataframe[col].dtype for col in self.dataframe.columns}

    def set_data(self, data):
        """Returns a new Problem with the DataFrame replaced. Note that this operation is *NOT* in place"""
        return Problem(data, self.features, self.outcome_column, self.positive_outcome, self.label_list)

    def clone(self, deep_copy=False):
        """Creates a shallow copy of this Problem instance"""
        data_to_use = pd.DataFrame(self.data.dataframe, copy=True) if deep_copy else self.data
        return Problem(data_to_use, self.features, self.outcome_column, self.positive_outcome, self.label_list)

    def _check_frame(self):
        """\
        Checks the integrity of the input data frame

        :return: None
        """
        df = self.data.dataframe
        for feat in self.features:
            if feat not in df.columns:
                raise ValueError('No such column: {}. Other columns: {}'.format(feat, ', '.join(df.columns)))

        if self.outcome_column not in df.columns:
            raise ValueError('Outcome column not found: {}'.format(self.outcome_column))

    def vectorize(self, keep_discrete_columns=True, expected_numeric=None, expected_discrete=None,
                  vectorizer=None):
        """\
        Converts classification features into a numeric matrix using DictVectorizer followed by an Imputer.
        Provided as a convenience for simple analyses -- generally you will want to tailor feature preprocessing
        to your specific problem.
        Now uses a ProblemVectorizer. Expected discrete/numeric can be left as None to skip preprocessing.

        :param keep_discrete_columns: if True (default), original discrete columns will be kept in addition to their
           encoded indicators. If False, only the indicators will be kept.
        :param expected_numeric: List of dataframe heading names that are expected to be numeric
        :param expected_discrete: List of dataframe heading names that are expected to be strings
        :param vectorizer: the vectorizer to use (optional)
        :return: new Problem with the features vectorized

        """

        vectorizer = vectorizer or ProblemVectorizer(expected_numeric, expected_discrete)
        return vectorizer.fit_apply(self, keep_discrete_columns)

    def serialize(self, serializer):
        """Serializes this Problem instance into Python primitives"""
        return {'__class__': get_class_name(Problem),
                'data': serializer.serialize(self.data),
                'features': serializer.serialize(self.features),
                'outcome_column': serializer.serialize(self.outcome_column),
                'positive_outcome': serializer.serialize(self.positive_outcome)}

    @staticmethod
    def deserialize(serialized_obj, serializer):
        """Deserializes this Problem instance from Python primitives"""
        return Problem(data=serializer.deserialize(serialized_obj['data']),
                       features=serializer.deserialize(serialized_obj['features']),
                       outcome_column=serializer.deserialize(serialized_obj['outcome_column']),
                       positive_outcome=serializer.deserialize(serialized_obj['positive_outcome']))

    def iloc(self, idx):
        """Like DataFrame.iloc, but returns a subsetted Problem instance"""
        return Problem(self.dataframe.iloc[idx], self.features, self.outcome_column, self.positive_outcome,
                       self.label_list)

    @property
    def prevalence(self):
        """Fraction of positive instances"""
        return utils.prevalence(self.y, 1.0)

    @property
    def n_features(self):
        """Returns the total number of features"""
        return len(self.features)

    @property
    def n_samples(self):
        """Returns the total number of samples"""
        return len(self.dataframe)

    @property
    def n_positives(self):
        """Counts the number of positive samples"""
        if self.positive_outcome is None:
            ValueError('Positive outcome is defined only for Binary Classification Problem')
        return np.sum(self.y)

    @property
    def n_negatives(self):
        """Counts the number of negative samples"""
        if self.positive_outcome is None:
            ValueError('Negative outcome is defined only for Binary Classification Problem')
        return self.n_samples - self.n_positives

    @property
    def should_be_binary(self):
        """Problem is defined as binary one vs. all"""
        return self.positive_outcome is not None

    @property
    def dataframe(self):
        """Returns the underlying DataFrame"""
        return self.data.dataframe

    @property
    def features_dataframe(self):
        """Returns a view of the underlying DataFrame with only the feature columns"""
        return pd.DataFrame(self.dataframe, columns=self.features)

    @property
    def metadata_dataframe(self):
        """Returns a view of the underlying DataFrame without the feature columns"""
        return pd.DataFrame(self.dataframe,
                            columns=[feat for feat in self.dataframe.columns if feat not in self.features])

    @property
    def sample_ids(self):
        """Returns ordered sample IDs in this Problem instance"""
        return list(self.dataframe.index)

    @property
    def X(self):
        """Converts the feature DataFrame into a scikit-style matrix of samples (rows) and features (columns)"""
        if self._X is None:
            self._X = np.asarray(self.features_dataframe.values)
        return self._X

    @property
    def y(self):
        """Gets a binary numpy array of labels. 1=positive. 0=negative"""
        if self.should_be_binary:
            return binarize_seq(self.dataframe[self.outcome_column], self.positive_outcome)
        else:
            if self.label_list is None:
                raise ValueError("Multiclass problem without initialized label list")
            return np.searchsorted(self.label_list, self.dataframe[self.outcome_column])

    def __add__(self, other):
        """Concatentates samples from two Problem instances, returning a combined Problem. The problems
        need to have the exact same features and outcomes"""
        if not isinstance(other, Problem):
            raise ValueError('Can only concatenate Problem instances')
        elif tuple(other.features) != tuple(self.features):
            raise ValueError("Feature sets are not the same (or are in different order)")
        elif other.outcome_column != self.outcome_column:
            raise ValueError("Outcome columns differ")
        elif other.positive_outcome != self.positive_outcome:
            raise ValueError("Positive outcome differs")
        elif len(other) == 0:
            return self.clone()
        else:
            return Problem(pd.concat([self.dataframe, other.dataframe]), self.features, self.outcome_column,
                           self.positive_outcome)

    def __iter__(self):
        """Iterates over sample rows"""
        for _, row in self.dataframe.iterrows():
            yield row

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return Problem(self.dataframe[item], self.features, self.outcome_column, self.positive_outcome, self.label_list)

    def __str__(self):
        return '[Problem with {} samples, {} features. Outcome: {}=={}. Prevalence: {:.2%}]'.format(
            self.n_samples, self.n_features, self.outcome_column, self.positive_outcome, self.prevalence)


class ProblemVectorizer(object):
    """\
    Class implementing improved vectorization. Cleans out non-numbers from
    expected numeric data; converts non-strings to strings in expected nominal
    data.
    """
    def __init__(self, expected_numeric=None, expected_discrete=None, permissive=False):
        """\
        Given a list of values (presumed to be numeric/float/int)
        clean out any nonnumerics, changing them to nans
        Return the result.
        Expected numeric/discrete can be None, causing their preprocessing steps
        to be skipped.

        :param expected_numeric: List of dataframe heading names that are expected to be numeric.
        :param expected_discrete: List of headings that should be strings.
        :param permissive: Boolean. If False, vectorize throws an error if
                           an expected numeric/discrete column is missing. Defaults to False
        """
        self.expected_numeric = expected_numeric if expected_numeric is not None else []
        self.expected_discrete = expected_discrete if expected_discrete is not None else []
        self.permissive = permissive

        self._vectorization_pipeline = None
        self._vectorized_features = None

    def _check_columns(self, columns, df):
        """\
        If self.permissive is true, filters out non-existent columns and returns them. If self.permissive is false,
        asserts that all columns are actually present and throws a ValueError if they are not.

        :param columns: list of columns to check
        :param df: DataFrame to check against
        :return: list of columns

        """
        if self.permissive:
            return [col for col in columns if col in df.columns]
        else:
            missing_columns = [col for col in columns if col not in df.columns]
            if len(missing_columns) != 0:
                raise ValueError('Columns not found: {}'.format(', '.join(sorted(missing_columns))))
            return columns

    def _preprocess_expected_columns(self, df):
        for feature in self._check_columns(self.expected_numeric, df):
            df[feature] = self.preprocess_numeric(df[feature].tolist())
        for feature in self._check_columns(self.expected_discrete, df):
            df[feature] = self.preprocess_discrete(df[feature].tolist(), "discrete")
        return [row.to_dict() for _, row in df.iterrows()]

    def fit(self, problem):
        """\
        Fits the vectorizer on a Problem instance. Calling fit() is a prerequisite for calling apply().
        :param problem: Problem instance to fit on
        :return: self

        """
        features_as_dict = self._preprocess_expected_columns(problem.features_dataframe)
        vectorizer = DictVectorizer(sparse=False)
        self._vectorization_pipeline = Pipeline([('vectorize', vectorizer),
                                                 ('impute', Imputer())]).fit(features_as_dict)
        self._vectorized_features = vectorizer.feature_names_
        return self

    def fit_apply(self, problem, keep_discrete_columns=True):
        """Fits the vectorizer first, then vectorizes the problem"""
        self.fit(problem)
        return self.apply(problem, keep_discrete_columns=keep_discrete_columns)

    def apply(self, problem, keep_discrete_columns=True):
        """\
        Runs actual vectorization on a dataframe. Vectorizes the original dataframe in-place.

        :param problem: Problem instance to vectorize
        :param keep_discrete_columns: if True (default), original discrete columns will be kept in addition to their
           encoded indicators. If False, only the indicators will be kept.
        :return: Vectorized Problem.

        """
        features_as_dict = self._preprocess_expected_columns(problem.features_dataframe)

        X = self._vectorization_pipeline.transform(features_as_dict)
        vectorized_feat_df = pd.DataFrame(columns=list(self._vectorized_features), data=X,
                                          index=problem.dataframe.index)

        # Fix for NaNs causing column overwriting
        # get set of one-hot-encoded columns in the original problem dataframe
        encoded_columns = {column.split("=")[0] for column in vectorized_feat_df.columns if "=" in column}
        for column in vectorized_feat_df.columns:  # Scan all the vectorized features
            # if column should be discrete then preserve the original values
            if column in problem.dataframe and column in encoded_columns:
                del vectorized_feat_df[column]

        if keep_discrete_columns:
            meta_cols = [col for col in problem.dataframe.columns if col not in vectorized_feat_df.columns]
            meta_df = pd.DataFrame(problem.dataframe, columns=meta_cols)
        else:
            meta_df = problem.metadata_dataframe

        vectorized_df = pd.concat([meta_df, vectorized_feat_df], axis=1)

        return Problem(vectorized_df, list(vectorized_feat_df.columns),
                       problem.outcome_column, problem.positive_outcome, problem.label_list)

    def nan_helper(self, x):
        """if x is a float, return x, else return NaN"""
        try:
            return float(x)
        except ValueError:
            return float('nan')

    def preprocess_numeric(self, values):
        """\
        Given a list of values (presumed to be numeric/float/int)
        clean out any nonnumerics, changing them to nans
        Return the result

        :param values: List of input values
        :return: List of input values with non-numbers converted to NaN
        """
        return [self.nan_helper(x) for x in values]

    def preprocess_discrete(self, values, prefix):
        """\
        Given a list of values (presumed to be strings)
        clean out any nonstrings, changing them to <prefix>=str(<value>)
        Return the result

        :param values: List of input values
        :param prefix: Prefix (strings) to fix nonstrings with
        :return: List of input values with non-strings converted to prefix/strings
        """
        return [prefix + "=" + str(v) if not isinstance(v, str) else v for v in values]

    def vectorize_features(self):
        """ returns vectorize feature names"""
        return self._vectorized_features

    def serialize(self, serializer):
        """Serializes the ProblemVectorizer to Python primitives"""
        return {'__class__': get_class_name(ProblemVectorizer),
                'expected_numeric': serializer.serialize(self.expected_numeric),
                'expected_discrete': serializer.serialize(self.expected_discrete),
                'permissive': self.permissive,
                'vectorization_pipeline': serializer.serialize(self._vectorization_pipeline),
                'vectorized_features': serializer.serialize(self._vectorized_features)}

    @staticmethod
    def deserialize(serialized_obj, serializer):
        """Deserializes a ProblemVectorizer from Python primitives"""
        vec = ProblemVectorizer(expected_numeric=serializer.deserialize(serialized_obj['expected_numeric']),
                                expected_discrete=serializer.deserialize(serialized_obj['expected_discrete']),
                                permissive=serialized_obj['permissive'])

        # pylint: disable=protected-access
        vec._vectorization_pipeline = serializer.deserialize(serialized_obj['vectorization_pipeline'])

        # pylint: disable=protected-access
        vec._vectorized_features = serializer.deserialize(serialized_obj['vectorized_features'])
        return vec
