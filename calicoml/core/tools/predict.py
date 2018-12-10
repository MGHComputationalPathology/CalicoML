# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""


from __future__ import print_function

import os
import pandas as pd

from argparse import ArgumentParser
from calicoml.core.serialization.model import ClassificationModel
from calicoml.core.utils import binarize_seq

from sklearn.metrics import roc_auc_score


def get_formatted_auc(y_true, y_pred):
    """Calculates and formats the AUC, or returns a string indicating that the problem is degenerate"""
    if len(set(y_true)) != 2:
        return float('NA. Problem not binary. Unique labels: {}'.format(', '.join(sorted(set(y_true)))))
    else:
        return '{:.3f}'.format(roc_auc_score(y_true, y_pred))


def main(args=None):
    """The main method"""
    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('input_data')
    parser.add_argument('output')
    parser.add_argument('--index_col', default=None, help="Column to use as the sample ID. Default: first in file")
    parser.add_argument('--separator', default='\t', help="Input file separator. Default: tab")
    parser.add_argument('--join', action='store_true', help="If true, will join predictions with the input data. "
                                                            "Default: False.")
    parsed_args = parser.parse_args(args)

    print("Loading {}...".format(os.path.basename(parsed_args.model)))
    model = ClassificationModel.read(parsed_args.model)
    print("  => Model loaded OK: {}".format(str(model.approach)))

    print("\nLoading data...")
    df = pd.read_csv(parsed_args.input_data, sep=parsed_args.separator, index_col=parsed_args.index_col or 0)
    print("  => {} rows, {} columns".format(len(df), len(df.columns)))  # pylint: disable=no-member

    print("\nRunning the model...")
    results_df = model.predict(df, join=parsed_args.join)

    if model.training_problem.outcome_column in df.columns:  # pylint: disable=no-member
        y = binarize_seq(df[model.training_problem.outcome_column], pos_value=model.training_problem.positive_outcome)
        print('  => AUC: {}'.format(get_formatted_auc(y, results_df['score'])))
    else:
        print('  => OK, but no AUC because outcomes are absent')

    print("\nSaving...")
    results_df.to_csv(parsed_args.output, index_label=parsed_args.index_col or "sample_id", sep=parsed_args.separator)
    print("  => {}".format(parsed_args.output))

    print("\nDone!")


if __name__ == "__main__":
    main()
