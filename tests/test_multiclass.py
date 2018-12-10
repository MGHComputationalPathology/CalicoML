# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from calicoml.core.algo.cross_validation import CVSplitGenerator
from calicoml.core.algo.learners import SelectAndClassify
from calicoml.core.metrics import f_pearson, accuracy_from_confusion_matrix
from calicoml.core.pipeline.ml_runner import CrossValidatedAnalysis, SerialRunner, LearningParameters
from calicoml.core.problem import Problem
from calicoml.core.reporting import ClassificationReport, ReportRenderer
from calicoml.core.serialization.model import ClassificationModel

from tests.test_reporting import with_temporary_directory, mock_coords_data


import os
import pandas as pd
import nose
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def iris_to_df(iris):
    """ Make dataframe for multiclass classification from iris data"""
    X, y = iris.data, iris.target
    iris_column_data = X.T.tolist()
    iris_column_names = ["col" + str(idx) for idx in range(X.shape[1])]
    data = {}
    for ind, iris_data_column in enumerate(iris_column_data):
        data[iris_column_names[ind]] = iris_data_column
    data['Target'] = y.tolist()

    return pd.DataFrame(data)


def compute_average_accuracy(results):
    """averages accuracy computed in results for cross-validate test runs"""
    average_accuracy = 0.0
    count_accurace_sample = 0
    test_key = 'test'
    for cv_result in results:
        cv_test_metric = cv_result[test_key]['metrics']
        average_accuracy += cv_test_metric['accuracy']
        count_accurace_sample += 1
    if count_accurace_sample == 0:
        return 0.0
    average_accuracy /= count_accurace_sample
    return average_accuracy


@with_temporary_directory
def test_multiclass(working_dir):
    """ Tests machine learning classification workfloor with multiclass for iris dataset
        see http://scikit-learn.org/stable/modules/multiclass.html """

    out_dir = os.path.join(working_dir, 'learn_output')
    model_path = os.path.join(out_dir, 'model.txt')

    iris = datasets.load_iris()

    df = iris_to_df(iris)

    features = [feat for feat in df.columns if feat not in ['Target']]

    prob = Problem(df, features, "Target", positive_outcome=None)
    rnd = np.random.RandomState(2016)
    approach = SelectAndClassify(SelectKBest(score_func=f_pearson, k=3), RandomForestClassifier(random_state=rnd))

    learn_params = LearningParameters(metrics={'auc': roc_auc_score, 'accuracy': accuracy_from_confusion_matrix},
                                      treat_as_binary=False)
    cvg = CVSplitGenerator(prob, n_folds=10, n_repartitions=10, random_state=rnd)

    cv = CrossValidatedAnalysis(prob, approach, cv_generator=cvg,
                                runner=SerialRunner(),
                                params=learn_params)

    results = cv.run()
    renderer = ReportRenderer(out_dir)
    ClassificationReport(renderer, False, prob.label_list).generate(results)
    nose.tools.ok_(os.path.exists(os.path.join(out_dir, 'sample_confusion_matrix.txt')))
    average_accuracy = compute_average_accuracy(results)
    nose.tools.assert_almost_equal(0.95, average_accuracy, delta=0.01)

    classifier = SelectAndClassify(SelectKBest(score_func=f_pearson, k=3), RandomForestClassifier(random_state=2016),
                                   name='test multiclass model').fit(prob)
    model = ClassificationModel(classifier, prob)
    model.write(model_path)

    read_model = ClassificationModel.read(model_path)

    auc_average = read_model.training_auc
    nose.tools.assert_almost_equal(1.0, auc_average, delta=1e-6)


def test_multiclass_auc():
    """ Tests auc value for multiclass problem"""
    data = []
    class_values = ['A', 'B', 'C', 'D']
    for index_class in range(4):
        data, _ = mock_coords_data(data, index_class, class_values[index_class], None, True)

    df = pd.DataFrame(columns=['coord0', 'coord1', 'class'], data=data)
    prob = Problem(df, ['coord0', 'coord1'], 'class', None)
    classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(),
                                   name='test multiclass model').fit(prob)
    model = ClassificationModel(classifier, prob)
    auc_average = model.training_auc
    nose.tools.assert_almost_equal(0.853333333, auc_average, delta=1e-6)

    prob_binary = Problem(df, ['coord0', 'coord1'], 'class', 'A')
    classifier_binary = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(),
                                          name='binary model').fit(prob_binary)
    model_binary = ClassificationModel(classifier_binary, prob_binary)
    auc_binary = model_binary.training_auc
    nose.tools.assert_almost_equal(auc_binary, auc_average, delta=1e-6)


def test_multiclass_label_subset():
    """ Tests y_score for multiclass problem with training set
    having subset of possible classes """
    data = []
    data2 = []
    class_values = ['A', 'B', 'C', 'D']
    for index_class in range(4):
        data, data2 = mock_coords_data(data, index_class, class_values[index_class], data2, True)

    df = pd.DataFrame(columns=['coord0', 'coord1', 'class'], data=data)
    prob = Problem(df, ['coord0', 'coord1'], 'class', None)
    df2 = pd.DataFrame(columns=['coord0', 'coord1', 'class'], data=data2)
    prob2 = Problem(df2, ['coord0', 'coord1'], 'class', None, prob.label_list)

    classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(),
                                   name='test multiclass model').fit(prob2)

    y_pred = classifier.predict(prob2)
    y_score = classifier.prediction_probabilities(prob2)
    # check that "C" class has probabilities 0
    for i_row in range(y_pred.shape[0]):
        nose.tools.assert_almost_equal(0.0, y_score[i_row, 2], delta=1e-6)
