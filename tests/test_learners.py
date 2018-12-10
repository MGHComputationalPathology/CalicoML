# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from collections import Counter
from calicoml.core.algo.cross_validation import CVSplitGenerator
from calicoml.core.algo.learners import SelectAndClassify, LearningApproach, Pipeline
from calicoml.core.pipeline.ml_runner import train_and_evaluate_model, LearningParameters, CrossValidatedAnalysis, \
    SerialRunner, ParallelRunner, LearningCurveAnalysis
from calicoml.core.problem import Problem, ProblemVectorizer
import nose
from nose.tools import nottest

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class LoggingApproach(LearningApproach):
    """A pass-through approach that logs its inputs"""
    def __init__(self, base_approach):
        super(LoggingApproach, self).__init__()
        self.base_approach = base_approach
        self.fit_problems = []  # problems we called fit() with
        self.apply_problems = []  # problems we called apply() with

    def fit(self, problem):
        """Fits the approach. Pass-through to the underlying approach."""
        self.fit_problems.append(problem)
        self.base_approach.fit(problem)
        return self

    def apply(self, problem, binary_positive_score=True):
        """Pass-through to the underlying approach"""
        self.apply_problems.append(problem)
        return self.base_approach.apply(problem, binary_positive_score)

    def get_parameters(self):
        params = self.base_approach.get_parameters()
        params['model'] = self
        return params


def mock_problem(n_samples=1000, n_features=100, theta=0.5):
    """\
    Creates a mock problem with class-differentiated features.

    :param n_samples: number of samples
    :param n_features: number of features
    :param theta: measure of class separation in sigma units
    :return: a Problem instance

    """
    if n_samples % 2 != 0:
        raise ValueError('Number of samples have to be a multiple of 2')

    rand = np.random.RandomState(0x12345)
    X = rand.normal(size=(n_samples, n_features))
    y = np.zeros(X.shape[0])
    y[:int(X.shape[0] / 2)] = 1
    X[y == 1] += theta
    df = pd.DataFrame(columns=['feat{}'.format(idx) for idx in range(X.shape[1])],
                      data=X)
    df['y'] = y
    df['train_or_test'] = ['train', 'test'] * int(n_samples / 2)
    return Problem(df, [col for col in df if 'feat' in col], 'y', 1)


def test_train_and_evaluate_model():
    """Tests that train_and_evaluate_model works as expected"""
    def checkme(n_samples, n_features, n_features_to_select, theta):
        """Utility"""
        prob = mock_problem(n_samples, n_features, theta)
        train = prob[prob.dataframe['train_or_test'] == 'train']
        test = prob[prob.dataframe['train_or_test'] == 'test']

        approach = LoggingApproach(SelectAndClassify(SelectKBest(k=n_features_to_select), LogisticRegression()))
        results = train_and_evaluate_model(approach, train, test, LearningParameters())
        model = results['approach']['model']

        # Simple sanity checks
        nose.tools.assert_is_not_none(model)
        nose.tools.eq_(results['train']['n_samples'], len(train))
        nose.tools.eq_(results['test']['n_samples'], len(test))
        nose.tools.eq_(len(results['approach']['selected_features']), n_features_to_select)
        nose.tools.assert_greater(results['train']['metrics']['auc'], 0.75)
        nose.tools.assert_greater(results['test']['metrics']['auc'], 0.75)
        nose.tools.assert_greater(results['train']['metrics']['auc'], results['test']['metrics']['auc'])

        for name, sub_prob in [('train', train), ('test', test)]:
            nose.tools.assert_list_equal(results[name]['outcome'], list(sub_prob.dataframe[sub_prob.outcome_column]))
            nose.tools.eq_(results[name]['positive_outcome'], 1)

        # Make sure we trained on the right samples
        nose.tools.eq_(len(model.fit_problems), 1)  # fitting the same model multiple times might cause problems
        np.testing.assert_array_equal(train.X, model.fit_problems[-1].X)
        np.testing.assert_array_equal(train.y, model.fit_problems[-1].y)

        # Make sure we applied the model to both training and test
        for X in [train.X, test.X]:
            nose.tools.ok_(any([np.array_equal(X, applied_prob.X) for applied_prob in model.apply_problems]))

    for n_samples in [100, 1000, 10000]:
        for n_features in [10, 100]:
            for n_features_to_select in [5, 10]:
                for theta in [0.5, 0.75]:
                    yield checkme, n_samples, n_features, n_features_to_select, theta


def test_null_feature_selector():
    """Validates that SelectAndClassify works with a null feature selector"""
    def make_fixed_rs():
        """Utility: makes a fixed random state for use in this test"""
        return np.random.RandomState(0xC0FFEE)

    prob = mock_problem()

    # selector=None and SelectKBest(k='all') should produce identical predictions
    no_select_approach = SelectAndClassify(None, LogisticRegression(random_state=make_fixed_rs()),
                                           classifier_grid={'C': [0.5, 1.0]},
                                           random_state=make_fixed_rs()).fit(prob)
    select_all_approach = SelectAndClassify(SelectKBest(k='all'),
                                            LogisticRegression(random_state=make_fixed_rs()),
                                            classifier_grid={'C': [0.5, 1.0]},
                                            random_state=make_fixed_rs()).fit(prob)

    # There should be no selection step in the underlying model
    nose.tools.eq_(len(no_select_approach.model.steps), len(select_all_approach.model.steps) - 1)

    # We should still be logging the right features
    nose.tools.assert_list_equal(no_select_approach.selected_features, prob.features)

    # Scores should be identical as k='all'
    np.testing.assert_allclose(no_select_approach.apply(prob), select_all_approach.apply(prob))


def test_overlapping_train_and_test():
    """Validates that we fail if samples overlap between training and test"""
    prob = mock_problem()
    train = prob[prob.dataframe['train_or_test'] == 'train']
    test = prob[prob.dataframe['train_or_test'] == 'test']

    approach = SelectAndClassify(SelectKBest(k='all'), LogisticRegression())
    params = LearningParameters()
    train_and_evaluate_model(approach, train, test, params)  # no overlap -- should just work
    nose.tools.assert_raises(ValueError, lambda: train_and_evaluate_model(approach, train, train, params))  # oops
    nose.tools.assert_raises(ValueError, lambda: train_and_evaluate_model(approach, test, test, params))  # oops


def cv_analysis_run_one(task_runner=None):
    """\
    Does one CV analysis run, then validates and returns the results
    :param task_runner: Task runner to use for the CV analysis mockup
    :return: CV analysis results
    """
    prob = mock_problem(n_samples=1000, n_features=100)
    approach = SelectAndClassify(SelectKBest(k=17), LogisticRegression(random_state=np.random.RandomState(0xC0FFEE)))
    cv_generator = CVSplitGenerator(prob, n_folds=10, n_repartitions=2, random_state=np.random.RandomState(0xC0FFEE))
    analysis = CrossValidatedAnalysis(prob, approach, cv_generator=cv_generator, runner=task_runner)

    results = analysis.run()
    nose.tools.eq_(len(results), cv_generator.n_total_splits)  # One per CV split
    for field_name in ['metrics', 'n_samples']:
        nose.tools.ok_(all([field_name in r['train'] for r in results]))
        nose.tools.ok_(all([field_name in r['test'] for r in results]))
    return results


def assert_cv_equal(results_1, results_2):
    """\
    Are the CV analysis results equal?
    :param results_1: One CV analysis result set
    :param results_2: The other CV analysis result set
    :return: None. Testing is done with nose.tools.assert
    """

    for field_name in ['metrics', 'n_samples']:
        nose.tools.eq_([r['train'][field_name] for r in results_1],
                       [r['train'][field_name] for r in results_2])
        nose.tools.eq_([r['test'][field_name] for r in results_1],
                       [r['test'][field_name] for r in results_2])

    nose.tools.eq_([tuple(r['approach']['selected_features']) for r in results_1],
                   [tuple(r['approach']['selected_features']) for r in results_2])


def test_cv_analysis():
    """Tests an end-to-end cross validation run"""

    results_1 = cv_analysis_run_one(task_runner=SerialRunner())
    results_2 = cv_analysis_run_one(task_runner=SerialRunner())

    # Results should be reproducible: same input and run parameters should give the exact same results
    assert_cv_equal(results_1, results_2)


def test_mproc():
    """Tests serial versus parallel runner"""

    results_1 = cv_analysis_run_one(task_runner=ParallelRunner())
    results_2 = cv_analysis_run_one(task_runner=SerialRunner())

    # Results should be reproducible: same input and run parameters should give the exact same results
    assert_cv_equal(results_1, results_2)


def learning_curves_run_one(fractions=None):
    """Runs a single LearningCurves analysis and validates the output"""
    prob = mock_problem(n_samples=1000, n_features=100)
    approach = SelectAndClassify(SelectKBest(k=17), LogisticRegression(random_state=np.random.RandomState(0xC0FFEE)))
    cv_generator = CVSplitGenerator(prob, n_folds=10, n_repartitions=2, random_state=np.random.RandomState(0xC0FFEE))
    analysis = LearningCurveAnalysis(prob, approach, cv_generator=cv_generator, fractions=fractions,
                                     runner=SerialRunner())

    results = analysis.run()
    nose.tools.eq_(len(results), len(fractions))  # One per fraction
    for fraction in sorted(results):
        nose.tools.eq_(len(results[fraction]), cv_generator.n_total_splits)
        for field_name in ['metrics', 'n_samples']:
            nose.tools.ok_(all([field_name in r['train'] for r in results[fraction]]))
            nose.tools.ok_(all([field_name in r['test'] for r in results[fraction]]))

        seen_test_samples = Counter()
        for split_results in results[fraction]:
            # No train/test overlap
            nose.tools.eq_(set(split_results['train']['sample']) & set(split_results['test']['sample']), set())

            # Size of the test set should be 10% of problem size
            nose.tools.eq_(len(split_results['test']['sample']), 0.1 * prob.n_samples)

            # Size of the training set should be 90% of problem size * fraction
            nose.tools.assert_almost_equal(len(split_results['train']['sample']), 0.9 * fraction * prob.n_samples,
                                           delta=1)

            # Record test samples
            seen_test_samples.update(split_results['test']['sample'])

        # Must have seen all test samples, all of them the same number of times
        nose.tools.eq_(set(seen_test_samples.keys()), set(prob.sample_ids))
        nose.tools.eq_(set(seen_test_samples.values()), set([cv_generator.n_repartitions]))

        # Test sets should be identical across fractions. Otherwise difference between fractions will be a product
        # of both the training set size and different CV splits, but we only really care about the former.
        test_sets_by_fraction = {fraction: tuple([tuple(sorted(set(split_results['test']['sample'])))
                                                  for split_results in results[fraction]])
                                 for fraction in sorted(results.keys())}

        nose.tools.eq_(len(set(test_sets_by_fraction.values())), 1)
    return results


def test_learning_curves_analysis():
    """Tests an end-to-end learning curve cross validation run"""
    results_1 = learning_curves_run_one(np.linspace(0.1, 1.0, 5))
    results_2 = learning_curves_run_one(np.linspace(0.1, 1.0, 5))

    # Results should be reproducible: same input and run parameters should give the exact same results
    for fraction in sorted(results_1):
        assert_cv_equal(results_1[fraction], results_2[fraction])


def test_grid():
    """Verifies that we generate correct grids given selector and classifier parameters"""
    def checkme(selector_grid, classifier_grid, expected_grid, expected_size):
        """Utility"""
        learner = SelectAndClassify(SelectKBest(), LogisticRegression(), selector_grid=selector_grid,
                                    classifier_grid=classifier_grid)
        actual_grid, actual_grid_size = learner._get_grid()  # pylint: disable=protected-access
        nose.tools.eq_(actual_grid, expected_grid)
        nose.tools.eq_(actual_grid_size, expected_size)

    yield checkme, {}, {}, {}, 0
    yield checkme, {'k': [5]}, {}, {'select__k': [5]}, 1
    yield checkme, {'k': [5, 10]}, {}, {'select__k': [5, 10]}, 2
    yield checkme, {}, {'C': [1.0]}, {'classify__C': [1.0]}, 1
    yield checkme, {}, {'C': [0.5, 1.0]}, {'classify__C': [0.5, 1.0]}, 2
    yield checkme, {'k': [5, 10]}, {'C': [0.5, 1.0]}, {'select__k': [5, 10], 'classify__C': [0.5, 1.0]}, 4
    yield checkme, {'k': [5, 10]}, {'C': [0.5, 1.0], 'penalty': ['l1', 'l2']}, {'select__k': [5, 10],
                                                                                'classify__C': [0.5, 1.0],
                                                                                'classify__penalty': ['l1', 'l2']}, 8


@nottest
def make_test_grid_scorer(optimal_params):
    """\
    Makes a scorer which assigns higher scores to parameters which closely match the specified optimal set. A perfect
    match gets a score of 0, with imperfect matches getting increasingly negative scores.
    :param optimal_params: set of optimal parameters
    :return: score (a float)

    """
    def compute_error(actual_val, expected_val):
        """Computes the error between the expected and actual parameters"""
        if isinstance(actual_val, (float, int)) and isinstance(expected_val, (float, int)):
            return abs(actual_val - expected_val)
        else:
            # This is somewhat arbitrary but it has the correct direction/gradient, which is what matters here
            return 1.0 if actual_val != expected_val else -1.0

    # pylint: disable=unused-argument
    def scorer(estim, *args, **kwargs):
        """The actual scorer. Closes over the optimal_params argument"""
        estim_params = estim.get_params()
        return -sum(compute_error(estim_params[k], optimal_params[k]) for k in list(optimal_params.keys()))
    return scorer


def test_exhaustive_grid_search():
    """Verifies that exhaustive grid search works as expected"""
    def checkme(selector_grid, classifier_grid, optimal_params):
        """Utility: runs grid search and verifies that we selected the right parameters"""
        prob = mock_problem()
        learner = SelectAndClassify(SelectKBest(), LogisticRegression(), selector_grid=selector_grid,
                                    classifier_grid=classifier_grid,
                                    grid_search_scorer=make_test_grid_scorer(optimal_params),
                                    grid_search_cv_folds=2, grid_search_cv_repartitions=1,
                                    randomized_grid_size_cutoff=None)
        model_params = learner.fit(prob).model.get_params()
        params_to_check = sorted(optimal_params.keys())
        nose.tools.assert_list_equal([(k, model_params[k]) for k in params_to_check],
                                     [(k, optimal_params[k]) for k in params_to_check])

    yield checkme, {}, {}, {}

    yield checkme, {'k': [10]}, {}, {'select__k': 10}
    yield checkme, {'k': [10, 20, 30]}, {}, {'select__k': 10}
    yield checkme, {'k': [10, 20, 30]}, {}, {'select__k': 20}
    yield checkme, {'k': [10, 20, 30]}, {}, {'select__k': 30}

    yield checkme, {}, {'C': [1.0]}, {'classify__C': 1.0}
    yield checkme, {}, {'C': [1.0, 2.0, 3.0]}, {'classify__C': 1.0}
    yield checkme, {}, {'C': [1.0, 2.0, 3.0]}, {'classify__C': 2.0}
    yield checkme, {}, {'C': [1.0, 2.0, 3.0]}, {'classify__C': 3.0}

    big_slct_grid = {'k': [10, 20, 30]}
    big_cls_grid = {'C': [0.5, 1.0, 2.0], 'penalty': ['l1', 'l2'], 'tol': [0.0001, 0.001]}

    yield checkme, big_slct_grid, big_cls_grid, {'select__k': 10,
                                                 'select__score_func': f_classif,
                                                 'classify__C': 1.0,
                                                 'classify__penalty': 'l1',
                                                 'classify__tol': 0.0001}
    yield checkme, big_slct_grid, big_cls_grid, {'select__k': 20,
                                                 'select__score_func': f_classif,
                                                 'classify__C': 2.0,
                                                 'classify__penalty': 'l2',
                                                 'classify__tol': 0.001}
    yield checkme, big_slct_grid, big_cls_grid, {'select__k': 10,
                                                 'select__score_func': f_classif,
                                                 'classify__C': 0.5,
                                                 'classify__penalty': 'l2',
                                                 'classify__tol': 0.0001}


def test_randomized_grid_search():
    """Verifies that randomized grid search works as expected"""
    error_log = []

    def checkme(optimal_params):
        """Utility: runs grid search and verifies that we selected (approximately) the right parameters"""
        np.random.seed(0xC0FFEE)
        prob = mock_problem(n_samples=100)
        learner = SelectAndClassify(SelectKBest(), LogisticRegression(), selector_grid={'k': [10, 20]},
                                    classifier_grid={'C': np.linspace(0.5, 1.0, 1000)},
                                    grid_search_scorer=make_test_grid_scorer(optimal_params),
                                    grid_search_cv_folds=2, grid_search_cv_repartitions=1,
                                    randomized_grid_size_cutoff=100)
        model_params = learner.fit(prob).model.get_params()
        for param_name in sorted(optimal_params.keys()):
            # Might not be exactly optimal, but should be close
            tolerance = 0.05 * abs(optimal_params[param_name])
            nose.tools.assert_almost_equal(model_params[param_name], optimal_params[param_name],
                                           delta=tolerance)

        error_log.append(make_test_grid_scorer(optimal_params)(learner.model, prob.X, prob.y))

    error_log = []
    for optimal_c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for optimal_k in [10, 20]:
            yield checkme, {'classify__C': optimal_c, 'select__k': optimal_k}

    # Given that the search is non-exhaustive, we should have gotten at least a few unique solutions (with different
    # scores)
    nose.tools.assert_greater(len(set(error_log)), 1)


def test_feature_engineering():
    """Tests feature engineering"""
    prob = mock_problem()

    classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(), feature_engineering=PCA(n_components=2))
    model = classifier.fit(prob).model
    steps = dict(model.steps)

    nose.tools.ok_('PCA' in str(classifier))
    nose.tools.ok_('feature_engineering' in steps)
    nose.tools.assert_is_not_none(steps['feature_engineering'].components_)

    # Check that classifier.apply() works
    nose.tools.eq_(len(classifier.apply(prob)), prob.n_samples)

    # Test that SelectAndClassify still works without feature engineering
    classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression())
    model = classifier.fit(prob).model
    steps = dict(model.steps)
    nose.tools.ok_('PCA' not in str(classifier))
    nose.tools.ok_('feature_engineering' not in steps)


def test_preprocessing():
    """Tests feature preprocessing"""
    base_prob = mock_problem()
    base_prob.features.append('discrete_feat')

    # Derive a problem with a single discrete feature perfectly correlated with the label
    df = pd.DataFrame(base_prob.dataframe, copy=True)
    df['discrete_feat'] = 'negative'
    df['discrete_feat'].values[base_prob.y == 1] = 'positive'

    # Verify that a manual upfront vectorize is equivalent to passing a vectorizer as the preprocess step
    # to SelectAndClassify
    prob = base_prob.set_data(df)
    vectorized_prob = ProblemVectorizer().fit_apply(prob)

    baseline_classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(), preprocess=None)
    preprocess_classifier = SelectAndClassify(SelectKBest(k='all'), LogisticRegression(),
                                              preprocess=ProblemVectorizer())

    # First make sure that the baseline classifier cannot be fit on the unvectorized data
    nose.tools.assert_raises(ValueError, lambda: baseline_classifier.fit_apply(prob))

    baseline_scores = baseline_classifier.fit_apply(vectorized_prob)
    preprocess_scores = preprocess_classifier.fit_apply(prob)

    np.testing.assert_allclose(baseline_scores, preprocess_scores)


class CountingTransform(object):
    """Trivial counting transform: adds a new feature equal to the (unique) value of the last feature + 1"""
    def __init__(self):
        self.count = None

    def fit(self, problem):
        """Remembers the unique value of the last feature present in the Problem"""
        last_feature_values = set(problem.X[:, -1])
        if len(last_feature_values) != 1:
            raise ValueError('Expected exactly one unique feature value, but got: {}'.format(last_feature_values))
        self.count, = last_feature_values
        return self

    @property
    def feature_name(self):
        """Name of the added feature"""
        return 'feat{}'.format(self.count + 1)

    def apply(self, problem):
        """Adds a new feature equal to self.count + 1 to the Problem"""
        df = pd.DataFrame(problem.dataframe)
        df[self.feature_name] = self.count + 1
        return Problem(df, problem.features + [self.feature_name], problem.outcome_column, problem.positive_outcome)


def test_pipeline():
    """Validates that pipelines work as expected"""
    prob = Problem(pd.DataFrame({'feat0': [0] * 100, 'y': [0, 1] * 50}), ['feat0'], 'y', 1)
    pipe = Pipeline([('step{}'.format(idx), CountingTransform()) for idx in range(50)])
    pipe.fit(prob)

    transformed_prob = pipe.apply(prob)
    nose.tools.eq_(transformed_prob.X.shape[0], 100)  # same number of samples
    nose.tools.eq_(transformed_prob.X.shape[1], 51)  # started with 1 feature, and added one extra for each transform

    for idx in range(transformed_prob.X.shape[1]):
        np.testing.assert_array_equal(transformed_prob.X[:, idx], [idx] * prob.X.shape[0])


def test_map_label_to_class_index():
    """ test utility SelectAndClassify.map_label_to_class_index """
    nose.tools.eq_(SelectAndClassify.map_label_to_class_index(['A', 'B', 'C', 'D'], [0, 1, 3]),
                   {'A': 0, 'B': 1, 'D': 2})
    nose.tools.eq_(SelectAndClassify.map_label_to_class_index(['A', 'B', 'C', 'D'], [0, 3]),
                   {'A': 0, 'D': 1})
    nose.tools.eq_(SelectAndClassify.map_label_to_class_index(['A', 'B', 'C', 'D'], [2]),
                   {'C': 0})
    nose.tools.eq_(SelectAndClassify.map_label_to_class_index(['A', 'B', 'C', 'D'], [0, 1, 2]),
                   {'A': 0, 'B': 1, 'C': 2})
