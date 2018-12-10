# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

import json

import six

from calicoml.core.algo.learners import SelectAndClassify
from calicoml.core.problem import Problem, ProblemVectorizer
from calicoml.core.serialization.model import ClassificationModel
from calicoml.core.tools import predict_web_service
import nose
from calicoml.core.tools.predict_web_service import format_datatypes

import numpy as np
import pandas as pd
from six import wraps, BytesIO
from sklearn.linear_model import LogisticRegression


def mock_problem():
    """ creates mock problem """
    X = np.random.normal(size=(100, 2))
    y = np.asarray([1] * 50 + [0] * 50)
    df = pd.DataFrame({'featA': X[:, 0],
                       'featB': X[:, 1],
                       'featC': ['foo', 'bar'] * 50,
                       'y': y})
    prob = Problem(df, ['featA', 'featB', 'featC'], 'y', 1)
    return prob


def mock_model():
    """Creates a simple mock model for testing"""
    prob = mock_problem()
    logit = SelectAndClassify(selector=None, classifier=LogisticRegression(),
                              preprocess=ProblemVectorizer(), name="test model").fit(prob)

    return ClassificationModel(logit, prob)


def with_mock_service(func):
    """Sets up a mock model and service and calls the function"""
    @wraps(func)
    def inner(*args, **kwargs):
        """Wrapper around the annotated function"""
        model = mock_model()
        predict_web_service.model = model
        app = predict_web_service.app.test_client()
        return func(app, model, *args, **kwargs)

    return inner


def test_datatype_mapping():
    """Validates that we correctly map internal python/numpy types to either 'numeric' or 'text'"""
    def checkme(in_type, out_type):
        """Utility: checks a single type"""
        nose.tools.eq_(format_datatypes({'test': in_type}), {'test': out_type})

    yield checkme, int, 'numeric'
    yield checkme, float, 'numeric'
    yield checkme, np.int, 'numeric'
    yield checkme, np.int_, 'numeric'
    yield checkme, np.int32, 'numeric'
    yield checkme, np.int64, 'numeric'
    yield checkme, np.float, 'numeric'
    yield checkme, np.float_, 'numeric'
    yield checkme, np.float64, 'numeric'
    yield checkme, np.float128, 'numeric'

    yield checkme, type(b'foo'), 'binary' if six.PY3 else 'text'
    yield checkme, type('foo'), 'text'


@with_mock_service
def test_model_info(app, _):
    """Tests that we return correct model metadata"""
    response = json.loads(app.get('/info').get_data(as_text=True))
    nose.tools.eq_(response['name'], 'test model')
    nose.tools.eq_(response['outcome'], 'y')
    nose.tools.eq_(response['datatypes'], {'featA': 'numeric', 'featB': 'numeric', 'featC': 'text'})
    nose.tools.eq_(response['positive_outcome'], 1)
    nose.tools.eq_(response['training_set']['prevalence'], 0.5)
    nose.tools.eq_(response['training_set']['n_features'], 3)
    nose.tools.eq_(response['training_set']['n_samples'], 100)
    nose.tools.assert_list_equal(response['features'], ['featA', 'featB', 'featC'])


def ws_predict(app, df, features):
    """Utility: calls /predict with samples from the given DataFrame, and returns the results as a DataFrame"""
    samples = [{feat: row[feat] for feat in features}
               for _, row in df.iterrows()]
    response = app.post('/predict', data=json.dumps(samples), headers={'content-type': 'application/json'})
    return pd.DataFrame(json.loads(response.get_data(as_text=True)).get("scores"))


@with_mock_service
def test_train_set_predict(app, model):
    """Validates prediction on training set samples"""
    predictions = ws_predict(app, model.training_problem.dataframe, ['featA', 'featB', 'featC'])
    np.testing.assert_allclose(predictions['score'].values, model.expected_scores)


@with_mock_service
def test_novel_predict(app, model):
    """Validates prediction on novel samples"""
    df = pd.DataFrame({'featA': np.random.normal(100),
                       'featB': np.random.normal(100),
                       'featC': ['foo', 'bar', 'bar', 'foo'] * 25})
    predictions = ws_predict(app, df, ['featA', 'featB', 'featC'])
    np.testing.assert_allclose(predictions['score'].values, model.predict(df)['score'].values)


@with_mock_service
def test_upload(app, model):
    """Validates that we can upload files"""
    fake_file = six.StringIO()
    df = model.training_problem.dataframe
    df.to_csv(fake_file, sep='\t')
    fake_file.seek(0)

    response = app.post('/upload', buffered=True,
                        content_type='multipart/form-data',
                        data={'file': (BytesIO(fake_file.getvalue().encode('utf-8')), 'samples.txt')})
    response_df = pd.DataFrame(json.loads(response.get_data(as_text=True)))
    np.testing.assert_allclose(response_df['featA'].values, df['featA'].values)
    np.testing.assert_allclose(response_df['featB'].values, df['featB'].values)
    nose.tools.assert_list_equal(list(response_df['featC']), list(df['featC'].values))


@with_mock_service
def test_get_training_set(app, model):
    """Tests that the service returns valid training data"""
    response = json.loads(app.get('/training_data').get_data(as_text=True))
    training_df = pd.DataFrame(response)

    np.testing.assert_allclose(training_df['featA'].values, model.training_problem.dataframe['featA'].values)
    np.testing.assert_allclose(training_df['featB'].values, model.training_problem.dataframe['featB'].values)
    nose.tools.assert_list_equal(list(training_df['featC']), list(model.training_problem.dataframe['featC'].values))
