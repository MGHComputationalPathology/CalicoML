# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

import json

from calicoml.core.metrics import ROC
from calicoml.core.problem import Problem
from calicoml.core.serialization.serializer import Serializer, get_class_name
from calicoml.core.metrics import compute_averaged_metrics

import numpy as np
import pandas as pd

from calicoml.core.utils import pretty_print_dataframe


def roc_auc_function(y_one_vs_all, scores_y):
    """ Function to access AUC computations of ROC instance """
    return ROC.from_scores(y_one_vs_all, scores_y).auc


class ClassificationModel(object):
    """\
    A fully packaged up, fitted classification model: can be written and read from disk and provides functions
    for classifying new samples. Also does correctness checks on loading.

    """
    def __init__(self, approach, training_problem, expected_scores=None, model_id=None):
        """\

        :param approach: a fitted LearningApproach
        :param training_problem: the problem on which the LearningApproach was trained
        :param expected_scores: expected predictions for all samples in the training problem (optional)
        :param id: model id

        """
        self.id = model_id
        self.approach = approach
        self.training_problem = training_problem
        self.expected_scores = expected_scores

        if self.expected_scores is not None and len(self.expected_scores) != len(self.training_problem):
            raise ValueError('Length of expected scores ({}) does not match the length of the training '
                             'problem ({})'.format(len(self.expected_scores), len(self.training_problem)))
        elif expected_scores is None:
            self.expected_scores = self.approach.apply(self.training_problem)

    @property
    def features(self):
        """Returns list of features used for prediction"""
        return self.training_problem.features

    @property
    def outcome(self):
        """Returns the name of the outcome"""
        return self.training_problem.outcome_column

    @property
    def positive_outcome(self):
        """Returns outcome value considered positive (i.e. mapped to 1)"""
        return self.training_problem.positive_outcome

    @property
    def training_auc(self):
        """Computes AUC on the training set"""
        if not self.training_problem.should_be_binary:
            return self.averaged_training_auc()

        return ROC.from_scores(self.training_problem.y, self.expected_scores).auc

    def averaged_training_auc(self):
        """ Compute averaged auc for multiclass classification """
        scores = self.approach.prediction_probabilities(self.training_problem)
        return compute_averaged_metrics(self.training_problem.y, scores, roc_auc_function)

    def serialize(self, serializer=None):
        """\
        Serializes this ClassificationModel into Python primitives

        :param serializer: serializer to use (optional)
        :return: a dictionary of Python primitives

        """
        serializer = serializer or Serializer()
        return {'__class__': get_class_name(ClassificationModel),
                'approach': serializer.serialize(self.approach),
                'problem': serializer.serialize(self.training_problem),
                'expected_scores': serializer.serialize(self.expected_scores),
                "id": self.id}

    def validate(self, fail_if_different=True):
        """\
        Validates that actual predictions from this ClassificationModel match the expected scores

        :param fail_if_different: whether to fail with a ValueError if any of the scores don't match. Default: True
        :return: a DataFrame with validation results

        """
        expected_score = self.expected_scores
        actual_score = self.approach.apply(self.training_problem)
        if len(expected_score.shape) == 2:
            data_to_check = {'sample': self.training_problem.sample_ids, 'truth': self.training_problem.y}
            columns_validation = ['sample', 'truth']
            separate_scores = True
        else:
            data_to_check = {'sample': self.training_problem.sample_ids,
                             'truth': self.training_problem.y,
                             'expected_score': expected_score,
                             'actual_score': actual_score}
            columns_validation = ['sample', 'truth', 'expected_score', 'actual_score']
            separate_scores = False
        validation_df = pd.DataFrame(data=data_to_check, columns=columns_validation)
        if not separate_scores:
            validation_df['is_correct'] = np.isclose(validation_df['expected_score'].values,
                                                     validation_df['actual_score'].values)
        else:
            validation_df['is_correct'] = [np.allclose(expected_score[index], actual_score[index]) for
                                           index in range(expected_score.shape[0])]
        incorrect_sdf = validation_df[~validation_df['is_correct']]
        if len(incorrect_sdf) > 0 and fail_if_different:
            pretty_print_dataframe(incorrect_sdf)
            raise ValueError('Model validation failed: scores differ for {}/{} samples'.format(
                len(incorrect_sdf), len(validation_df)))
        return validation_df if not separate_scores else validation_df, expected_score, actual_score

    def _check_prediction_input(self, df):
        """Validates that a DataFrame has all the required columns for prediction, and returns a Problem instance
        that the underlying learning approach can be invoked on"""
        missing_features = sorted(set(self.training_problem.features) - set(df.columns))
        if len(missing_features) > 0:
            raise ValueError("Input is missing features (count={}): {}".format(len(missing_features),
                                                                               ', '.join(missing_features)))

        # TODO FIXME: LearningApproaches require a Problem instance when calling apply(). This is not ideal
        # because Problems assume an outcome column, which might not be known when applying to new data.
        # Here we just mock a null outcome column, but we should consider changing the interface so that
        # apply() accepts a data frame directly.
        classification_columns = self.training_problem.features + [self.training_problem.outcome_column]
        classification_df = pd.DataFrame(df, columns=classification_columns)
        return Problem(classification_df, self.training_problem.features, self.training_problem.outcome_column,
                       self.training_problem.positive_outcome, self.training_problem.label_list)

    def predict(self, df, join=False):
        """\
        Applies this model to data in a pandas DataFrame, returning a new dataframe with predictions.

        :param df: input DataFrame. Has to contain all features required for classification.
        :param join: if True, will append predictions to the input DataFrame. Otherwise, will return a new DataFrame
         with only the sample IDs and predictions. Default: False
        :return: DataFrame with predicted scores

        """
        problem = self._check_prediction_input(df)
        results_df = pd.DataFrame(index=pd.Index(problem.sample_ids)) if not join else pd.DataFrame(df, copy=True)
        results_df['score'] = self.approach.apply(problem)
        return results_df

    @staticmethod
    def deserialize(serialized_obj, serializer=None):
        """\
        Deserializes a ClassificationModel from Python primitives. Also validates that model output
        matches the expected scores.

        :param serialized_obj: serialized ClassificationModel
        :param serializer: serializer to use (optional)
        :return: a ClassificationModel

        """
        serializer = serializer or Serializer()
        model = ClassificationModel(serializer.deserialize(serialized_obj['approach']),
                                    serializer.deserialize(serialized_obj['problem']),
                                    expected_scores=serializer.deserialize(serialized_obj['expected_scores']),
                                    model_id=serializer.deserialize(serialized_obj.get('id', None)))
        model.validate()
        return model

    def write(self, path, serializer=None, validate=True):
        """\
        Saves this ClassificationModel to a file

        :param path: where to save the model
        :param serializer: serializer to use (optional)
        :param validate: if True, will attempt to load the model and validate its output. Default: True
        :return: None

        """
        with open(path, 'w') as f:
            json.dump(self.serialize(serializer=serializer), f)

        if validate:
            reconstituted_model = ClassificationModel.read(path, serializer=serializer)
            reconstituted_model.validate(fail_if_different=True)

    @staticmethod
    def read(path, serializer=None):
        """\
        Loads a ClassificationModel from a file.

        :param path: where to load the model from
        :param serializer: serializer to use (optional)
        :return: a ClassificationModel

        """
        with open(path, 'r') as f:
            return ClassificationModel.deserialize(json.load(f), serializer=serializer)
