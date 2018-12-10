# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from calicoml.core.algo.learners import SelectAndClassify
from calicoml.core.serialization.model import ClassificationModel

import nose
from calicoml.core.tools import learn, predict

import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from tests.test_learners import mock_problem
from tests.test_reporting import with_temporary_directory


def mock_input(working_dir):
    """\
    Mocks an input problem in the given directory, returning a pair of (1) path to the problem file and
    (2) the Problem instance

    """
    prob_path = os.path.join(working_dir, 'problem.txt')
    prob = mock_problem(n_samples=100, n_features=10)
    prob.dataframe.to_csv(prob_path, sep='\t', index=True, index_label='sample_id')
    return prob_path, prob


@with_temporary_directory
def test_learn_tool(working_dir):
    """Tests that the learn.py command line tool works as expected"""
    prob_path, prob = mock_input(working_dir)
    out_dir = os.path.join(working_dir, 'learn_output')
    os.mkdir(out_dir)

    opts, results = learn.main([prob_path, 'y', '1', out_dir, '--cv_repartitions', '2', '--cv_k', '2',
                                '--index_col', 'sample_id'])

    nose.tools.assert_greater(len(results), 0)  # make sure we actually ran something
    nose.tools.eq_(len(results), len(opts.classifiers) * len(opts.n_features))

    for approach in opts.make_learning_approaches():
        nose.tools.ok_(approach.name in [result.name for result in list(results.keys())])
        nose.tools.ok_(os.path.exists(os.path.join(out_dir, approach.name)))
        for result_file in ['roc.pdf', 'mean_metrics.txt', 'cv_metrics.txt', 'selected_features.txt',
                            'mean_sample_scores.txt', 'score_plots.pdf']:
            nose.tools.ok_(os.path.exists(os.path.join(out_dir, approach.name, result_file)))

        scores_df = pd.read_csv(os.path.join(out_dir, approach.name, 'mean_sample_scores.txt'), sep='\t')
        nose.tools.eq_(len(scores_df), len(prob))
        nose.tools.assert_list_equal(sorted(scores_df['sample']), sorted(prob.sample_ids))

    # Make sure we generated the comparative report
    for result_file in ['mean_metrics.txt', 'mean_scores.txt', 'score_plots.pdf']:
        nose.tools.ok_(os.path.exists(os.path.join(out_dir, result_file)))


@with_temporary_directory
def test_predict_tool(working_dir):
    """Tests that the predict.py command line tool works as expected"""
    out_dir = os.path.join(working_dir, 'learn_output')
    model_path = os.path.join(out_dir, 'model.txt')
    predictions_path = os.path.join(out_dir, 'predictions.txt')

    # Mock up some input data
    prob_path, prob = mock_input(working_dir)
    os.mkdir(out_dir)

    # Train a model and save it to a file
    classifier = SelectAndClassify(SelectKBest(k=5), GaussianNB(), name='test model').fit(prob)
    model = ClassificationModel(classifier, prob)
    model.write(model_path)

    # Run the predict tool with the model using the training data loaded from a file, and validate that
    # the returned predictions match
    predict.main([model_path, prob_path, predictions_path, '--index_col', 'sample_id'])

    expected_predictions = pd.DataFrame({'sample_id': prob.sample_ids, 'score': classifier.apply(prob)})
    actual_predictions = pd.read_csv(predictions_path, sep='\t')

    np.testing.assert_allclose(actual_predictions['score'].values, expected_predictions['score'].values)
