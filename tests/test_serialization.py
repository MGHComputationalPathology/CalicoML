# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

import json
from calicoml.core.problem import Problem, ProblemVectorizer
from calicoml.core.serialization.serializer import Serializer

import nose
import os

import numpy as np

from calicoml.core.algo.learners import SelectAndClassify
from calicoml.core.algo.learners import Pipeline as ApproachPipeline
from calicoml.core.serialization.model import ClassificationModel
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
# pylint: disable=no-name-in-module

from tests.test_learners import mock_problem
from tests.test_reporting import with_temporary_directory


def serializer_roundtrip(serializer, obj):
    """Serializes an object to a file, then deserializes it and returns the result"""
    @with_temporary_directory
    def helper(tmp_dir, serializer, obj):
        """Helper function: takes care of creating and deleting the temp directory for the output"""
        path = os.path.join(tmp_dir, 'out.txt')
        with open(path, 'w') as f:
            try:
                json.dump(serializer.serialize(obj), f)
            except ValueError as e:
                print("test_serialization.serializer_roundtrip - invalid serialization:")
                print(str(serializer.serialize(obj)))
                raise e

        with open(path, 'r') as f:
            return serializer.deserialize(json.load(f))

    return helper(serializer, obj)  # pylint: disable=no-value-for-parameter


def test_primitives():
    """Validates that we can serialize/deserialize primitive types"""
    def checkme(val):
        """Utility"""
        serializer = Serializer()
        nose.tools.eq_(val, serializer.serialize(val))
        nose.tools.eq_(val, serializer_roundtrip(serializer, val))

    yield checkme, None
    yield checkme, 7
    yield checkme, -7
    yield checkme, 3.1415
    yield checkme, 2.71
    yield checkme, 'abc'
    yield checkme, "{'test': 2, 'another_test': '\n'}"
    yield checkme, []
    yield checkme, list(range(100))
    yield checkme, list(range(100)) + ['a', None, 3.1415, 2.71, -7, 7]
    yield checkme, {}
    yield checkme, {'a': 2}
    yield checkme, {'a': list(range(100)), 'b': "{'test': 2, 'another_test': '\n'}"}
    yield checkme, (1)
    yield checkme, True
    yield checkme, False


def test_numpy_arrays():
    """Validates that we can serialize/deserialize numpy arrays"""
    # pylint wrongly thinks np.linspace(..) returns a tuple, and then complains that
    # it doesn't have a .reshape() method. Hence the disable below.
    #
    # pylint: disable=no-member

    def checkme(val):
        """Utility"""
        serializer = Serializer()
        nose.tools.assert_not_equal(type(serializer.serialize(val)), type(val))  # should do non-trivial encoding
        np.testing.assert_array_equal(serializer.deserialize(serializer.serialize(val)), val)

    yield checkme, np.arange(100)
    yield checkme, np.linspace(0.0, 1.0, 200)
    yield checkme, np.linspace(0.0, 1.0, 200).reshape((10, 20))
    yield checkme, np.linspace(0.0, 1.0, 200).reshape((20, 10))
    yield checkme, np.linspace(0.0, 1.0, 200).reshape((100, 2))
    yield checkme, np.linspace(0.0, 1.0, 200).reshape((1, 200))
    yield checkme, np.linspace(0.0, 1.0, 200).reshape((4, 5, 10))


def test_scikit_classifier_serialization():
    """Verifies that we can serialize/deserialize a simple scikit classifier"""
    def checkme(make_classifier):
        """Utility"""
        prob = mock_problem()
        logr = make_classifier().fit(prob.X, prob.y)

        decoded_logr = serializer_roundtrip(Serializer(), logr)
        np.testing.assert_array_equal(logr.predict_proba(prob.X),
                                      decoded_logr.predict_proba(prob.X))

    yield checkme, LogisticRegression
    yield checkme, GaussianNB
    yield checkme, RandomForestClassifier


def test_scikit_pipeline_serialization():
    """Validates that we can serialize/deserialize complex scikit pipelines"""
    def checkme(n_features, k, make_classifier):
        """Utility"""
        prob = mock_problem(n_features=n_features)

        pipe = Pipeline([('select', SelectKBest(k=k)), ('classify', make_classifier())]).fit(prob.X, prob.y)
        decoded_piple = serializer_roundtrip(Serializer(), pipe)
        np.testing.assert_array_equal(pipe.predict_proba(prob.X),
                                      decoded_piple.predict_proba(prob.X))

    for make_cls in [LogisticRegression, GaussianNB, RandomForestClassifier]:
        yield checkme, 10, 2, make_cls
        yield checkme, 10, 'all', make_cls
        yield checkme, 100, 27, make_cls


def test_learning_approach_serialization():
    """Tests end-to-end serialization of a LearningApproach"""
    @with_temporary_directory
    def checkme(working_dir, n_samples, n_features, k, make_classifier, test_vectorize):
        """Utility"""
        assert n_samples % 4 == 0
        model_path = os.path.join(working_dir, 'model.txt')
        prob = mock_problem(n_samples=n_samples, n_features=n_features)
        if test_vectorize:
            df = prob.dataframe
            df['discrete_1'] = ['foo', 'bar'] * int(n_samples / 2)
            df['discrete_2'] = ['foo', 'bar', 'baz', float('nan')] * int(n_samples / 4)
            df['continuous_with_missing'] = [0, 1, 2, float('nan')] * int(n_samples / 4)
            prob = Problem(df, prob.features + ['discrete_1', 'discrete_2', 'continuous_with_missing'],
                           prob.outcome_column, prob.positive_outcome)
            preprocess = ProblemVectorizer()
        else:
            preprocess = None

        approach = SelectAndClassify(SelectKBest(k=k), make_classifier(), preprocess=preprocess).fit(prob)
        model = ClassificationModel(approach, prob)

        model.write(model_path)
        reconstituted_model = ClassificationModel.read(model_path)

        model.validate()
        reconstituted_model.validate()

        np.testing.assert_array_equal(model.approach.apply(prob),
                                      reconstituted_model.approach.apply(prob))

        if preprocess is not None:
            approach_pipeline = ApproachPipeline([('preprocess', preprocess)])
            approach_with_pipeline = SelectAndClassify(SelectKBest(k=k), make_classifier(),
                                                       preprocess=approach_pipeline).fit(prob)
            # test approach serialization with Pipeline from learners.py
            model_with_pipeline = ClassificationModel(approach_with_pipeline, prob)
            model_path2 = os.path.join(working_dir, 'model2.txt')
            model_with_pipeline.write(model_path2)
            reconstituted_model2 = ClassificationModel.read(model_path2)
            reconstituted_model2.validate()
            np.testing.assert_array_almost_equal(model.approach.apply(prob),
                                                 reconstituted_model2.approach.apply(prob), 14)

    for n_samples in [100, 200]:
        for n_features in [10, 100]:
            for k in [2, 10, 'all']:
                for make_cls in [LogisticRegression, GaussianNB]:
                    yield checkme, n_samples, n_features, k, make_cls, False
                    yield checkme, n_samples, n_features, k, make_cls, True


def test_svc():
    """Tests SVC serialization"""
    def checkme(params, probability):
        """
        Test helper function.

        :param params: SVC constructor params as a dictioanry
        :param probability: whether to test the output of predict_proba
        :return: None

        """
        prob = mock_problem(n_samples=100, n_features=3)
        svc = SVC(**params).fit(prob.X, prob.y)
        reconstituted_svc = serializer_roundtrip(Serializer(), svc)
        np.testing.assert_array_equal(reconstituted_svc.decision_function(prob.X), svc.decision_function(prob.X))

        if probability:
            np.testing.assert_array_equal(reconstituted_svc.predict_proba(prob.X), svc.predict_proba(prob.X))

    for params in [{'kernel': 'linear'},
                   {'kernel': 'poly', 'degree': 2},
                   {'kernel': 'poly', 'degree': 3},
                   {'kernel': 'poly', 'degree': 4},
                   {'kernel': 'poly', 'degree': 4, 'coef0': 0.0},
                   {'kernel': 'poly', 'degree': 4, 'coef0': 0.5},
                   {'kernel': 'rbf'},
                   {'kernel': 'rbf', 'gamma': 0.1},
                   {'kernel': 'rbf', 'gamma': 1.0},
                   {'kernel': 'sigmoid', 'coef0': 0.0},
                   {'kernel': 'sigmoid', 'coef0': 1.0}]:
        for probability in [True, False]:
            for cvalue in [0.1, 1.0]:
                params['C'] = cvalue
                params['probability'] = probability
                yield checkme, params, probability


def test_pca():
    """Tests PCA serialization"""
    def checkme(params):
        """
        Test helper function.

        :param params: PCA constructor params as a dictionary
        :return: None

        """
        prob = mock_problem(n_samples=100, n_features=4)
        pca = PCA(**params).fit(prob.X)
        reconstituted_pca = serializer_roundtrip(Serializer(), pca)
        np.testing.assert_allclose(reconstituted_pca.transform(prob.X), pca.transform(prob.X))

    for params in [{'n_components': 2},
                   {'n_components': 3}]:
        yield checkme, params


def test_scaler():
    """Tests StandardScaler serialization"""
    def checkme(params):
        """
        Test helper function.

        :param params: StandardScaler constructor params as a dictionary
        :return: None

        """
        prob = mock_problem(n_samples=100, n_features=4)
        scaler = StandardScaler(**params).fit(prob.X)
        reconstituted_scaler = serializer_roundtrip(Serializer(), scaler)
        np.testing.assert_allclose(reconstituted_scaler.transform(prob.X), scaler.transform(prob.X))

    params = {}
    for param_with_mean in [True, False]:
        params['with_mean'] = param_with_mean
        for param_with_std in [True, False]:
            params['with_std'] = param_with_std
            yield checkme, params


def test_rfc():
    """ Tests random forest classifier"""
    def checkme(params):
        """
        Test helper function.

        :param params: PCA constructor params as a dictioanry
        :return: None

        """
        prob = mock_problem(n_samples=100, n_features=4)
        rfc = RandomForestClassifier(**params)
        rfc.fit(prob.X, prob.y)
        reconstituted_rfc = serializer_roundtrip(Serializer(), rfc)

        np.testing.assert_array_equal(reconstituted_rfc.predict(prob.X), rfc.predict(prob.X))
        np.testing.assert_array_equal(reconstituted_rfc.predict_proba(prob.X), rfc.predict_proba(prob.X))

    params = {}
    for trees in [10, 20, 30]:
        for features in [2, 3]:
            params['n_estimators'] = trees
            params['max_features'] = features
            yield checkme, params


def test_dtc():
    """
    Test decision tree classifier
    """
    def checkme(params):
        """
        Test helper function.

        :param params: DecisionTreeClassifier constructor params as a dictioanry
        :return: None

        """
        prob = mock_problem(n_samples=100, n_features=18)
        tree = DecisionTreeClassifier(**params).fit(prob.X, prob.y)
        reconstituted_dtc = serializer_roundtrip(Serializer(debug_deserialize=False), tree)
        np.testing.assert_array_equal(reconstituted_dtc.predict_proba(prob.X), tree.predict_proba(prob.X))

    params = {}
    for features in [3, 6, 9, 12]:
        params['max_features'] = features
        yield checkme, params


@with_temporary_directory
def test_model_validation(working_dir):
    """Validates that we fail if a model has been corrupted or otherwise produces bad output"""
    model_path = os.path.join(working_dir, 'model.txt')
    prob = mock_problem()
    approach = SelectAndClassify(SelectKBest(k=7), LogisticRegression()).fit(prob)
    model = ClassificationModel(approach, prob)
    model.write(model_path)

    # Change an expected score for a sample -- this should cause model loading to fail because actual
    # classifier output will no longer match the expected output
    with open(model_path, 'r') as f:
        model_string = '\n'.join(f.readlines())
        nose.tools.ok_(str(model.expected_scores[17]) in model_string)
        bad_model_string = model_string.replace(str(model.expected_scores[17]),
                                                str(model.expected_scores[17] + 0.5))

    with open(model_path, 'w') as f:
        f.write(bad_model_string)

    nose.tools.assert_raises(ValueError, lambda: ClassificationModel.read(model_path))
