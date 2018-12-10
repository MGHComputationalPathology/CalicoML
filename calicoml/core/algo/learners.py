# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

import operator

import sklearn

from calicoml.core.algo.cross_validation import RepeatedStratifiedKFold
from calicoml.core.serialization.serializer import get_class_name, get_class_by_name
from calicoml.core.utils import format_scikit_estimator

import numpy as np
from sklearn import clone
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.scorer import roc_auc_scorer

# The builtin in Python 2 is not a builtin in Python 3
# pylint: disable=redefined-builtin
from functools import reduce


class LearningApproach(object):
    """\
    Encapsulates a general learning approach that can be fitted and applied to Problems.

    """
    def __init__(self, name=None):
        """\

        :param name: name of this learning approach (optional)

        """
        self.name = name

    def fit(self, problem):
        """\
        Trains the approach. Equivalent to scikit's fit

        :param problem: the Problem instance to train on
        :return: self

        """
        raise NotImplementedError()

    def apply(self, problem, binary_positive_score=True):
        """\
        Applies an already fitted the approach to a data set, returning approach-specific results.

        :param problem: problem to apply the approach to
        :param binary_positive_score: boolean kwarg for binary score
        :return: a dictionary of results

        """
        raise NotImplementedError()

    def fit_apply(self, problem):
        """\
        Shorthand for fit() followed by apply() on the same Problem instance

        :param problem: problem to fit and apply to
        :return: a dictionary of results

        """
        return self.fit(problem).apply(problem)

    def get_parameters(self):
        """Gets a dictionary of parameters describing this LearningApproach"""
        raise NotImplementedError()

    def serialize(self, serializer):
        """Serializes this LearningApproach to Python primitives using the given serializer object"""
        serialized_learner = {'__class__': get_class_name(self.__class__)}
        serialized_learner.update({k: serializer.serialize(v) for k, v in list(self.get_parameters().items())})
        return serialized_learner

    @staticmethod
    def deserialize(serialized_obj, serializer):
        """Deserializes this LearningApproach from Python primitives using the given serializer object"""
        clazz = get_class_by_name(serialized_obj['__class__'])
        return clazz(**{serializer.deserialize(k): serializer.deserialize(v)
                        for k, v in serialized_obj.items() if k != '__class__'})


def safe_clone(estim):
    """Clones an estimator, or returns None if the estimator is None"""
    return clone(estim) if estim is not None else None


def filter_none_steps(steps):
    """Filters out pipeline steps whose estimators are None"""
    return [(step_name, transform) for step_name, transform in steps if transform is not None]


class SelectAndClassify(LearningApproach):
    """Describes a MachineLearning approach"""
    # pylint: disable=too-many-arguments
    def __init__(self, selector, classifier, selector_grid=None, classifier_grid=None, feature_engineering=None,
                 grid_search_cv_folds=4, grid_search_cv_repartitions=10, grid_search_scorer=roc_auc_scorer,
                 randomized_grid_size_cutoff=40, name=None, model=None, selected_features=None,
                 feature_p_values=None, preprocess=None, random_state=None, verbose=False):
        """\

        :param selector: feature selector (e.g. SelectKBest)
        :param classifier: classifier (e.g. LogisticRegression)
        :param selector_grid: dictionary of hyperparameter choices for the feature selector
        :param classifier_grid: dictionary of hyperparameter choices for the classifier
        :param feature_engineering: any scikit transform
        :param grid_search_cv_folds: number of CV folds to run for the grid search
        :param grid_search_cv_repartitions: CV repartitions to run for the grid search
        :param grid_search_scorer: scorer for the grid search (higher is better). Must have the signature:
        estimator, X, y -> score.
        :param randomized_grid_size_cutoff: maximum size of the grid for exhaustive search. At and above this cutoff,
         we'll use a randomized search with randomized_grid_size_cutoff iterations. Use None to always run
          exhaustive search. Default: 40
        :param name: name of this learning approach (optional)
        :param preprocess: preprocessor to apply to the Problem before training the model. Default: None
        :param random_state: random state to use
        :param verbose: if True, will generate extra debugging output when running

        """
        super(SelectAndClassify, self).__init__(name=name)
        self.selector = selector
        self.classifier = classifier

        self.selector_grid = selector_grid if selector_grid is not None else {}
        self.classifier_grid = classifier_grid if classifier_grid is not None else {}
        self.feature_engineering = feature_engineering
        self.grid_search_cv_folds = grid_search_cv_folds
        self.grid_search_cv_repartitions = grid_search_cv_repartitions
        self.grid_search_scorer = grid_search_scorer
        self.randomized_grid_size_cutoff = randomized_grid_size_cutoff
        self.preprocess = preprocess
        self.random_state = random_state
        self.verbose = verbose

        self.model = model
        self.selected_features = selected_features
        self.feature_p_values = feature_p_values

    def fit(self, problem):
        if self.preprocess is not None:
            self.preprocess = self.preprocess.fit(problem)
            problem = self._preprocess(problem)

        slct = safe_clone(self.selector)
        model = sklearn.pipeline.Pipeline(
            filter_none_steps([('feature_engineering', safe_clone(self.feature_engineering)),
                               ('select', slct),
                               ('classify', safe_clone(self.classifier))]))
        model = model.set_params(**self._grid_search(problem, model))

        self.model = model.fit(problem.X, problem.y)

        if slct is not None:
            if self.verbose:
                constant_feats = [feat for feat in problem.features if len(set(problem.dataframe[feat].values)) == 1]
                if len(constant_feats) > 0:
                    print('WARNING: Constant features: {}'.format(', '.join(constant_feats)))

            if len(slct.get_support()) != len(problem.features):
                # Feature engineering changed the feature set so we can't tell which features were selected
                support = np.asarray([True] * len(problem.features))
            else:
                support = slct.get_support()

            self.selected_features = list(np.asarray(problem.features)[support])
            self.feature_p_values = dict(list(zip(problem.features, slct.pvalues_)))
        else:
            self.selected_features = list(problem.features)
            self.feature_p_values = [float('nan')] * problem.n_features
        return self

    def _get_grid(self):
        """\
        Creates a grid spec to use with scikit's grid search objects.

        :return: tuple of grid spec, grid size

        """
        grid = {} if self.selector is None else {'select__{}'.format(k): v for k, v in self.selector_grid.items()}
        grid.update({'classify__{}'.format(k): v for k, v in self.classifier_grid.items()})

        param_count = sum([len(v) for v in list(grid.values())])
        if param_count == 0:
            grid_size = 0
        else:
            grid_size = reduce(operator.mul, [len(v) for v in list(grid.values())], 1)
        return grid, grid_size

    def _grid_search(self, problem, model):
        """\
        Runs grid search to find the best hyperparameters for the model.

        :param problem: a Problem instance
        :param model: model for which to find the hyperparameters
        :return: a dictionary with the optimal values of hyperparameters found through CV

        """
        grid, grid_size = self._get_grid()
        if grid_size == 0:
            return {}
        grid_search_cv = RepeatedStratifiedKFold(problem.y, n_folds=self.grid_search_cv_folds,
                                                 n_repartitions=self.grid_search_cv_repartitions,
                                                 random_state=self.random_state)

        if self.randomized_grid_size_cutoff is None or grid_size < self.randomized_grid_size_cutoff:
            search = GridSearchCV(model, grid, scoring=self.grid_search_scorer, cv=grid_search_cv)
        else:
            search = RandomizedSearchCV(model, grid, scoring=self.grid_search_scorer, cv=grid_search_cv,
                                        n_iter=self.randomized_grid_size_cutoff,
                                        random_state=self.random_state)

        return search.fit(problem.X, problem.y).best_params_

    def _preprocess(self, problem):
        """Applies the preprocessing transform (if any) and returns the transformed Problem instance"""
        return self.preprocess.apply(problem) if self.preprocess is not None else problem

    def apply(self, problem, binary_positive_score=True):
        """ Returns probability or score """
        if self.model is None:
            raise ValueError("Model is None. Did you forget to call fit?")

        problem = self._preprocess(problem)

        if hasattr(self.model, 'predict_proba'):
            y_score = self.model.predict_proba(problem.X)
            if binary_positive_score and problem.should_be_binary:
                return y_score[:, 1]
            else:
                return y_score
        else:
            y_score = self.model.decision_function(problem.X)
            if len(y_score.shape) != 1:
                raise ValueError("Model does not have predict_proba method to score."
                                 "Please implement directly or using outcome of decision function.")
            return y_score

    def predict(self, problem):
        """ Returns predicted class """
        if self.model is None:
            raise ValueError("Model is None. Did you forget to call fit?")

        problem = self._preprocess(problem)

        return self.model.predict(problem.X)

    def matrix_at_predict_input(self, problem):
        """ Returns prediction problem and X right before calling actual predict method
        Used to explain decision rationale """
        if self.model is None:
            raise ValueError("Model is None. Did you forget to call fit?")
        if not isinstance(self.model, sklearn.pipeline.Pipeline):
            raise ValueError("Model is Pipeline. Did you forget to call fit?")
        input_problem = self._preprocess(problem)
        matrix_xt = input_problem.X
        for _, transform in self.model.steps[:-1]:
            matrix_xt = transform.transform(matrix_xt)
        return matrix_xt

    @staticmethod
    def map_label_to_class_index(label_classes, classifier_classes):
        """ Returns map from class names in model to labels in the problem"""
        # original input label corresponding to each position in self.model.classes_
        classes_in_model = [label_classes[idx] for idx in classifier_classes]

        # original input label -> index within self.model.classes_
        return dict(list(zip(classes_in_model, list(range(len(classifier_classes))))))

    def prediction_probabilities(self, problem):
        """ Returns score matrix of scores for each class for each row of prob.X
        Should be used instead of apply """
        if self.model is None:
            raise ValueError("Model is None. Did you forget to call fit?")

        problem = self._preprocess(problem)
        if problem.label_list is None or not hasattr(self.model, 'classes_')\
                or len(problem.label_list) == len(self.model.classes_):
            return self.model.predict_proba(problem.X)

        label_to_class = SelectAndClassify.map_label_to_class_index(problem.label_list, self.model.classes_)

        y_pred = self.model.predict_proba(problem.X)
        result = np.zeros((y_pred.shape[0], len(problem.label_list)))

        for index_y in range(len(problem.label_list)):
            label = problem.label_list[index_y]
            if label in label_to_class:
                index_pred = label_to_class[label]
                result[:, index_y] = y_pred[:, index_pred]

        return result

    def get_parameters(self):
        return {'type': self.__class__.__name__,
                'model': self.model,
                'selected_features': self.selected_features,
                'feature_p_values': self.feature_p_values}

    def serialize(self, serializer):
        return {'__class__': get_class_name(SelectAndClassify),
                'name': self.name,
                'model': serializer.serialize(self.model),
                'preprocess': serializer.serialize(self.preprocess)}

    @staticmethod
    def deserialize(serialized_obj, serializer):
        return SelectAndClassify(None, None, name=serialized_obj['name'],
                                 model=serializer.deserialize(serialized_obj['model']),
                                 preprocess=serializer.deserialize(serialized_obj['preprocess']))

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return ' -> '.join([format_scikit_estimator(estim)
                                for estim in [self.feature_engineering, self.selector, self.classifier]
                                if estim is not None])


class Pipeline(object):
    """\
    Like scikit's Pipeline, but for Problems. Describes a sequence of transformations where the output
    of one transformation serves as input of the next one in the sequence.

    """
    def __init__(self, steps):
        """

        :param steps: List of tuples describing the steps. Each tuple should consist of the step name
         and the step transform.

        """
        self.steps = steps

    def fit(self, problem):
        """Fits the pipeline on a Problem instance"""
        current_problem = problem
        for _, step in self.steps:
            step = step.fit(current_problem)
            current_problem = step.apply(current_problem)
        return self

    def apply(self, problem):
        """Applies the pipeline to a Problem instance"""
        current_problem = problem
        for _, step in self.steps:
            current_problem = step.apply(current_problem)
        return current_problem

    def serialize(self, serializer):
        """Serializes this Pipeline instance into Python primitives"""

        return {'__class__': get_class_name(Pipeline),
                'steps': serializer.serialize(self.steps)}

    @staticmethod
    def deserialize(serialized_object, serializer):
        """Rebuild this Pipeline instance from JSON"""
        return Pipeline(serializer.deserialize(serialized_object['steps']))
