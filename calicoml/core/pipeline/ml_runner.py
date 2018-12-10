# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from collections import defaultdict, OrderedDict
from multiprocessing import Pool

import six

from calicoml.core.algo.cross_validation import CVSplitGenerator, LearningCurveCVGenerator
from calicoml.core.metrics import compute_averaged_metrics

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


class LearningParameters(object):
    """Describes the low-level details of how to run a machine learning problem"""
    def __init__(self, metrics=None, random_state=None, treat_as_binary=True):
        """\

        :param metrics: dictionary of evaluation metrics
        :param random_state: random state to use. If None, will use the default pre-seeded RandomState. Default: None

        """
        self.random_state = random_state or np.random.RandomState(0xC0FFEE)
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = {'auc': roc_auc_score}
        self.treat_as_binary = treat_as_binary


def _check_train_and_test(train, test):
    """Validates the integrity of train and test sets"""
    common_samples = set(train.dataframe.index) & set(test.dataframe.index)
    if common_samples:
        raise ValueError('Training and test sets overlap. Common samples (n={}): {}'.format(
            len(common_samples), ', '.join(sorted([str(sample) for sample in common_samples]))))


def train_and_evaluate_model(approach, train, test, params):
    """\
    Trains the given ML approach on the training set and applies it to the test set.

    :param approach: LearningApproach to use
    :param train: training data (a Problem instance)
    :param test: test data (a Problem instance)
    :param params: LearningParameters to use
    :return: dictionary of results

    """
    _check_train_and_test(train, test)

    # Train the model
    approach = approach.fit(train)

    # Record results
    results = {}
    for prob_name, prob in [('train', train), ('test', test)]:
        # Temporary until all usages in metrics and reporting are cleaned
        y_score = approach.apply(prob) if params.treat_as_binary else approach.prediction_probabilities(prob)
        results[prob_name] = defaultdict(dict)
        y_pred = None if params.treat_as_binary else approach.predict(prob)
        if not params.treat_as_binary:
            results[prob_name]['scores_matrix'] = y_score

        # TODO: this is not general enough and only works for classifiers
        if not params.treat_as_binary:
            results[prob_name]['confusion_matrix'] = confusion_matrix(prob.y, y_pred)

        for metric_name, compute_metric in params.metrics.items():
            if params.treat_as_binary:
                results[prob_name]['metrics'][metric_name] = compute_metric(prob.y, y_score)
            else:
                if six.PY2:
                    from inspect import formatargspec, getargspec
                    metric_fcn_args = formatargspec(*getargspec(compute_metric))  # pylint: disable=deprecated-method
                    requires_confusion_matrix = 'confusion_matrix' in metric_fcn_args
                else:
                    from inspect import signature  # pylint: disable=no-name-in-module
                    metric_fcn_args = str(signature(compute_metric))
                    requires_confusion_matrix = 'confusion_matrix' in metric_fcn_args

                if requires_confusion_matrix:
                    confusion_matrix_prob = results[prob_name]['confusion_matrix']
                    results[prob_name]['metrics'][metric_name] = compute_metric(prob.y, y_score,
                                                                                confusion_matrix_prob)
                else:
                    results[prob_name]['metrics'][metric_name] = compute_averaged_metrics(
                        prob.y, y_score, compute_metric)

        results[prob_name]['n_samples'] = len(prob)
        results[prob_name]['n_features'] = prob.X.shape[1]
        results[prob_name]['sample'] = list(prob.dataframe.index)
        results[prob_name]['scores'] = y_score
        results[prob_name]['truth'] = prob.y
        results[prob_name]['outcome'] = list(prob.dataframe[prob.outcome_column])
        results[prob_name]['positive_outcome'] = prob.positive_outcome
        results[prob_name] = dict(results[prob_name])  # back to a non-default dict
    results['approach'] = approach.get_parameters()
    results['best_choice'] = get_classifier_name(approach.best_choice) if hasattr(approach, "best_choice") else "None"
    return results


def get_classifier_name(clf, name="None"):
    """ Get classifier name recursivley (for nested classifiers) """
    name = clf.__class__.__name__ if name == "None" else name
    if hasattr(clf, "classifier"):
        clf_name = clf.classifier.__class__.__name__
        name = "_".join([name, clf_name])
        return get_classifier_name(clf.classifier, name=name)
    return name


class Analysis(object):
    """\
    The base class for analyses. Doesn't currently do much.

    """
    def __init__(self, name):
        self.name = name

    def run(self):
        """Runs the analysis"""
        print('\n\n')
        print('\n'.join('{:20s}: {}'.format(str(k), v) for k, v in self.get_info().items()))
        print('\n\n')

    def get_info(self):
        """Returns a dictionary of human-readable parameters of this analysis"""
        params = OrderedDict()
        params['Analysis Name'] = self.get_parameters()['name']
        params['Analysis Type'] = self.get_parameters()['type']
        return params

    def get_parameters(self):
        """\
        Gets parameters describing this analysis.

        :return: a dictionary of parameters

        """
        return {'type': self.__class__.__name__,
                'name': self.name}


class CrossValidatedAnalysis(Analysis):
    """\
    Cross-validates a learning approach on a problem.

    """
    def __init__(self, problem, approach, params=None, name=None, cv_generator=None, runner=None):
        """\

        :param problem: a Problem instance on which the approach is to be evaluated
        :param approach: a LearningApproach instance
        :param params: a LearningParameters instance
        :param name: name for this analysis
        :param cv_generator: cross validation generator constructor: problem, random_state -> CVSplitGenerator.
        :param runner: Runner object to use. If left blank, a parallel runner object is created and used.
        If None, will use the default. Default: None.

        """
        super(CrossValidatedAnalysis, self).__init__(name or "Cross Validated Analysis")
        self.problem = problem
        self.approach = approach
        self.params = params or LearningParameters()
        self.runner = runner if runner is not None else ParallelRunner()

        if cv_generator is not None:
            self.cv_generator = cv_generator
        else:
            self.cv_generator = CVSplitGenerator(problem, random_state=self.params.random_state)

    def get_info(self):
        params = super(CrossValidatedAnalysis, self).get_info()
        params['Data'] = '{} samples, {} features'.format(*self.problem.X.shape)
        params['Outcome'] = '{} == {}'.format(self.problem.outcome_column, self.problem.positive_outcome)
        params['Approach'] = str(self.approach)
        return params

    def run(self):
        """\
        Kicks off the run and blocks until all CV splits have finished.

        :return: a list of results from individual CV splits

        """
        def try_get(getter_callable, default='NA', fmt=None):
            """\
            Utility: given a callable, returns a default value if the call raises a KeyError. Note that this is more
            general than using dict.get(), given that it returns the default for the entire expression that tries
            to access the missing key.

            """
            try:
                return fmt.format(getter_callable()) if fmt is not None else getter_callable()
            except KeyError:
                return default

        super(CrossValidatedAnalysis, self).run()
        results = []

        # enqueue tasks
        scheduled_tasks = [self.runner.run_async(CVTask(self.approach, train, test, self.params))
                           for train, test in self.cv_generator]
        self.runner.close()

        # parse results
        for cv_idx, task in enumerate(scheduled_tasks):
            split_results = task.get()  # get result from parallel task, block until done
            res = ["[CV {:4d}/{:4d}]", "---", "Train samples: {:5d}", "|", "Test samples: {:5d}", "|",
                   "Features: {:5d}", "|", "Selected features: {}", "|", "Test AUC: {}", "|", "Best Choice: {}"]
            res = "\t".join(res).format(
                cv_idx + 1, self.cv_generator.n_total_splits,
                split_results['train']['n_samples'],
                split_results['test']['n_samples'],
                self.problem.n_features,
                # pylint: disable=cell-var-from-loop
                try_get(lambda: len(split_results['approach']['selected_features']), fmt='{:3d}'),
                # pylint: disable=cell-var-from-loop
                try_get(lambda: split_results['test']['metrics']['auc'], fmt='{:.3f}'),
                split_results.get("best_choice", "None"))
            print(res)
            split_results['cv_index'] = cv_idx
            results.append(split_results)
        return results


class LearningCurveAnalysis(Analysis):
    """\
    Implements learning curve analysis on a problem.
    The training set is subsampled and the resulting analysis evaluted
    to predict how beneficial additional training data would be.
    """

    def __init__(self, problem, approach, params=None, name=None, cv_generator=None, fractions=None, runner=None):
        """\

        :param problem: a Problem instance on which the approach is to be evaluated
        :param approach: a LearningApproach instance
        :param params: a LearningParameters instance
        :param name: name for this analysis
        :param fractions: any sequentially iterable collection (such as a list or array) of floats/doubles in [0,1] to
         use as subsampling rates for training sets.
        :param cv_generator: Base CV generator to use. For each fraction, a LearningCurveCVGenerator is created around
         this.
        :param runner: Runner object to use. If left blank, a parallel runner object is created and used.
        If None, will use the default. Default: None.

        """
        super(LearningCurveAnalysis, self).__init__(name or "Learning Curve Analysis")
        self.problem = problem
        self.approach = approach
        self.params = params or LearningParameters()
        self.fractions = fractions if fractions is not None else np.linspace(0.1, 1.0, 10)
        self.runner = runner if runner is not None else ParallelRunner()
        if cv_generator is not None:
            self.cv_generator = cv_generator
        else:
            self.cv_generator = CVSplitGenerator(problem, random_state=self.params.random_state)

    def get_info(self):
        params = super(LearningCurveAnalysis, self).get_info()
        params['Data'] = '{} samples, {} features'.format(*self.problem.X.shape)
        params['Outcome'] = '{} == {}'.format(self.problem.outcome_column, self.problem.positive_outcome)
        params['Approach'] = str(self.approach)
        params['Fractions'] = str(self.fractions)
        return params

    def run(self):
        """\
        Runs the analysis; blocks until finished.
        :return: a list of results - a result per fraction
        """
        super(LearningCurveAnalysis, self).run()

        # Maciej's suggestion - no need to enqueue tasks or extract results,
        # make CrossValidatedAnalysis do that for LearningCurveAnalysis
        results = {}
        for fraction in self.fractions:
            cv_gen = LearningCurveCVGenerator(fraction, self.cv_generator.clone(),
                                              random_state=self.params.random_state)
            fraction_cv = CrossValidatedAnalysis(self.problem, self.approach, params=self.params, name=self.name,
                                                 cv_generator=cv_gen, runner=self.runner)
            results[fraction] = fraction_cv.run()
        return results


class Task(object):
    """\
    Container for an task.
    Runner objects run tasks (normally or in parallel).
    Intuitively, think of a Task as a box that Analysis creates and sends to a factory (Runner object)
    The base class is a dummy, meant to be extended
    """

    def run(self):
        """Runs the task"""
        raise NotImplementedError()


class CVTask(Task):
    """\
    Container for an CrossValidationAnalysis task.
    """

    def __init__(self, approach, train, test, params):
        """\
        Builds a cross validation analysis task.

        :param approach: Approach used in cross validation analysis, almost always self.approach in
        CrossValidationAnalysis object

        :param train: train (training set) in cv_splits
        :param test: test (testing set) in cv_splits
        :param params: parameters to cross validation
        """
        self.approach = approach
        self.train = train
        self.test = test
        self.params = params

    def run(self):
        """Runs the task, returning the cross-validation results for the given training/test split"""
        return train_and_evaluate_model(self.approach, self.train, self.test, self.params)


class SerialRunner(object):
    """A simple Runner object that runs tasks as they are sent. Mainly used for testing and comparing results
    with a parallel runner."""

    def run_async(self, task):
        """Dummy function at the moment; just return a ResultContainer containing task.run"""
        return ResultContainer(task.run())

    def close(self):
        """Does nothing - only defined to provide a unified Runner interface"""
        pass


class ResultContainer(object):
    """Mocks up returned object from ParallelRunner run_async with get function"""

    def __init__(self, result):
        self.result = result

    def get(self):
        """Returns the contained result"""
        return self.result


def _multiprocessing_run_wrapper(task):
    """\
    Returns task.run().

    FIXME: This function is necessary because pickle is broken and can't handle instance methods

    :param task: Task instance to run
    :return: result of calling task.run()

    """
    return task.run()


class ParallelRunner(object):
    """Runs tasks in parallel"""

    def __init__(self, poolsize=None):
        """\
        creates ParallelRunner with a given pool of processes

        :param poolsize: Size of process pool as integer. Default is None (use the number of CPUs).
        """
        self.pool = Pool(processes=poolsize)

    def run_async(self, task):
        """\
        runs task and returns a handle to it
        """
        return self.pool.apply_async(_multiprocessing_run_wrapper, [task])

    def close(self):
        """Closes the ParallelRunner's pool"""
        self.pool.close()
