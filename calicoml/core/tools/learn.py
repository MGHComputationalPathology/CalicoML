# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from calicoml.core.algo.cross_validation import CVSplitGenerator
from calicoml.core.data.sources import PandasDataSource
from calicoml.core.pipeline.ml_runner import CrossValidatedAnalysis
from calicoml.core.problem import Problem, ProblemVectorizer
from calicoml.core.reporting import ClassificationReport, ReportRenderer, ComparativeClassificationReport
from calicoml.core.serialization.model import ClassificationModel

import os
import pandas as pd

from argparse import ArgumentParser
from calicoml.core.algo.learners import SelectAndClassify
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class LearnerOptions(object):
    """Stores configuration parameters for the Learner tool"""
    # pylint: disable=too-many-arguments
    def __init__(self, input_file, target_label, positive_value, features, n_features, classifiers,
                 output_dir, id_col=None, cv_k=10, cv_repartitions=10, separator='\t',
                 save_models=False):
        """\

        :param input_file: path to the input file
        :param target_label: column containing the target outcome label
        :param positive_value: which of the outcome values to treat as a positive
        :param features: names of columns to use as features
        :param n_features: feature selectors to run
        :param classifiers: classifiers to run
        :param output_dir: directory where to save the output
        :param id_col: column to use as sample ID (default: first column in the file)
        :param cv_k: number of CV folds to run for each repartition
        :param cv_repartitions: number of CV repartitions to run
        :param separator: Field separator. Default: tab
        :param save_models: Whether to output models. Default: False

        """
        self.input_file = input_file
        self.target_label, self.positive_value = target_label, positive_value
        self.features = features
        self.n_features = n_features
        self.classifiers = classifiers
        self.id_col = id_col
        self.cv_k = cv_k
        self.cv_repartitions = cv_repartitions
        self.separator = separator
        self.output_dir = output_dir
        self.save_models = save_models

    def make_learning_approaches(self):
        """Creates LearningApproaches from the learner options"""
        for k in self.n_features:
            for cls in self.classifiers:
                yield SelectAndClassify(SelectKBest(k=k), cls,
                                        name='SelectKBest(k={}) -> {}'.format(k, cls.__class__.__name__),
                                        preprocess=ProblemVectorizer())

    def make_problem(self):
        """Creates a Problem instance using the current options"""
        df = pd.read_csv(self.input_file, sep=self.separator, index_col=0 if self.id_col is None else self.id_col)

        # pylint wrongly thinks that df is a tuple (it's a DataFrame), hence the disable below
        # pylint: disable=no-member
        all_features = [col for col in df.columns if col != self.target_label]
        return Problem(PandasDataSource(df, path=self.input_file),
                       features=self.features if self.features is not None else all_features,
                       outcome_column=self.target_label, positive_outcome=self.positive_value)


def run_learner(options):
    """Runs the learner tool with the given options"""
    prob = options.make_problem()
    results = {}
    for approach in options.make_learning_approaches():
        cv_gen = CVSplitGenerator(prob, n_folds=options.cv_k, n_repartitions=options.cv_repartitions)
        results[approach] = CrossValidatedAnalysis(prob, approach, cv_generator=cv_gen).run()
        report = ClassificationReport(renderer=ReportRenderer(os.path.join(options.output_dir, str(approach))))
        report.generate(results[approach])

        if options.save_models:
            # Retrain the approach on the full dataset
            trained_approach = approach.fit(prob)
            model = ClassificationModel(trained_approach, prob)
            model.write(os.path.join(options.output_dir, str(approach), 'model.txt'))

    comparative_report = ComparativeClassificationReport(renderer=ReportRenderer(options.output_dir))
    comparative_report.generate(results)
    return results


def parse_n_features(n_features):
    """Utility: parses the number of features from a list of strings. Expects integers or the special value 'all'"""
    return [k if k == 'all' else int(k) for k in n_features]


def main(args=None):
    """The main method"""
    parser = ArgumentParser()
    parser.add_argument('input_file', help="Tab-separated input files with one row per sample")
    parser.add_argument('target_label', help="Name of the column with the target label")
    parser.add_argument('positive_value', help="Value of the target label to treat as positive")
    parser.add_argument('output_dir', help="Directory where to save the output")
    parser.add_argument('--index_col', default=None, help="Column to use as the sample ID. Default: first in file")
    parser.add_argument('--features', nargs='+', default=None, help="Features to use for classification. Default: "
                                                                    "everything other than the target")
    parser.add_argument('--n_features', nargs='+', default=[5, 10], help="Number of features to select")
    parser.add_argument('--cv_k', type=int, default=10, help="Number of CV folds to use")
    parser.add_argument('--cv_repartitions', type=int, default=10, help="Number of times to repeat k-fold CV")
    parser.add_argument('--separator', default='\t', help="Input file separator. Default: tab")
    parser.add_argument('--save_models', action='store_true', help="If set, will save models in the output directory")
    parsed_args = parser.parse_args(args)

    try:
        # Special case for outcomes which are integers, e.g. 0 vs 1
        parsed_args.positive_value = int(parsed_args.positive_value)
    except ValueError:
        pass

    options = LearnerOptions(parsed_args.input_file, parsed_args.target_label, parsed_args.positive_value,
                             parsed_args.features, parse_n_features(parsed_args.n_features),
                             [GaussianNB(), LogisticRegression()],
                             output_dir=parsed_args.output_dir, id_col=parsed_args.index_col,
                             cv_k=parsed_args.cv_k, cv_repartitions=parsed_args.cv_repartitions,
                             separator=parsed_args.separator,
                             save_models=parsed_args.save_models)
    return options, run_learner(options)


if __name__ == "__main__":
    main()
