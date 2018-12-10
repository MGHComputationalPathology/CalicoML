# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import unicode_literals
from __future__ import print_function
from collections import defaultdict, Counter
from calicoml.core.metrics import ROC, accuracy_from_confusion_matrix
from calicoml.core.utils import assert_rows_are_concordant, partially_reorder_columns

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from scipy.stats import ttest_ind


class ReportRenderer(object):
    """\
    Renders reports. Currently simply writes them to files in the specified directly, but in the
    future can be extended to support databases, a web UI etc.

    """
    def __init__(self, directory, create_directory=True, tsv=True, xls=False,
                 width=14, height=8):
        """\

        :param directory: where to store the reports

        """
        self.tables, self.plots = {}, {}

        if isinstance(directory, (list, tuple)):
            directory = os.path.join(*directory)

        self.directory = directory
        self.tsv = tsv
        self.xls = xls
        self.width, self.height = width, height

        if create_directory and not os.path.exists(directory):
            os.mkdir(directory)

    def add_table(self, name, df, index=False, index_col=None):
        """\
        Adds a table to the report.

        :param name: name of the table
        :param df: a pandas DataFrame with table contents
        :param index: whether to save the table's index. Default: False
        :param index_col: column name to use for the index
        :return: None

        """
        self.tables[name] = df

        if self.tsv:
            df.to_csv(os.path.join(self.directory, '{}.txt'.format(name)), sep=str('\t'), encoding='utf-8',
                      index=index, index_label=index_col)
        if self.xls:
            df.to_excel(os.path.join(self.directory, '{}.xlsx'.format(name)), index=index, index_label=index_col)

    def add_plot(self, name, plots):
        """\
        Adds a plot to the report.

        :param name: name of the plot
        :param plots: one or more seaborn instances
        :return: None

        """
        initial_backend = matplotlib.pyplot.get_backend()
        if initial_backend != 'pdf':
            matplotlib.pyplot.switch_backend('pdf')
        from matplotlib.backends.backend_pdf import PdfPages

        self.plots[name] = plots

        if not hasattr(plots, '__iter__'):
            plots = [plots]

        plots_array = []
        with PdfPages(os.path.join(self.directory, '{}.pdf'.format(name))) as pdf_pages:
            for a_plt in plots:
                try:
                    plots_array.append(a_plt)
                    pdf_pages.savefig(a_plt)
                except AttributeError as err:
                    print(err)
        matplotlib.pyplot.switch_backend(initial_backend)
        return plots_array


class Report(object):
    """\
    Base class for reports.

    """
    def __init__(self, renderer):
        """\

        :param renderer: the renderer to use

        """
        self.renderer = renderer

    def generate(self, results):
        """\
        Generates the report from analysis results.

        :param results: analysis results
        :return:

        """
        raise NotImplementedError()

    def plot_roc(self, df, outcome_col='outcome', feature_col='score'):
        """ make ROC plot from dataframe with binary outcome data """
        unique_values = np.unique(df[outcome_col])
        if len(unique_values) > 2:
            return None
        roc = ROC.from_scores(df[outcome_col], df[feature_col])
        auc_ci = ROC.from_scores(df[outcome_col], df[feature_col]).auc_ci
        sns.set('talk', 'whitegrid', 'dark', font_scale=1.0, font='Arial',
                rc={"lines.linewidth": 1, 'grid.linestyle': '--'})
        fpr = roc.dataframe['fpr']
        tpr = roc.dataframe['tpr']
        roc_auc = auc(fpr, tpr)
        line_width = 1

        # PLOT ROC
        plt.figure()
        sns.set(font_scale=1)
        sns.set_style("whitegrid")
        roc_plot = sns.lineplot(fpr, tpr, color='darkorange', lw=line_width, legend=False)
        roc_plot.plot([0, 1], [0, 1], color='navy', lw=line_width, ls="--")
        roc_plot.set_xlim([0.0, 1.0])
        roc_plot.set_ylim([0.0, 1.05])
        roc_plot.set_xlabel('False Positive Rate')
        roc_plot.set_ylabel('True Positive Rate')
        roc_plot.set_title('Receiver Operating Characteristic\nAUC={}'.format(auc_ci))
        roc_plot.legend(('ROC curve (AUC = %0.2f)' % roc_auc,), loc="lower right")
        roc_plot_fig = roc_plot.get_figure()
        return roc_plot_fig

    def plot_scores(self, df, score_auc=None, outcome_col='outcome', feature_col='score'):
        """ plots boxplot and distribution plot of outcome/feature data (outcome must be binary) """
        unique_values = np.unique(df[outcome_col])
        unique_features = [str(feature) for feature in unique_values]
        if len(unique_values) == 2:
            # PLOT BOXPLOT
            plt.figure()
            sns.set_style("whitegrid")
            score_plot = sns.boxplot(x=df['truth'], y=df[feature_col], showfliers=False, color='white')
            sns.swarmplot(x=df['truth'], y=df[feature_col], ax=score_plot)
            score_plot.set_ylim(0.0, 1.05)
            score_plot.set(xticklabels=unique_features)
            score_plot.set_ylabel(str(feature_col))
            score_plot.set_xlabel('')
            score_plot.set_title('CV Scores. AUC={}'.format(score_auc))
            _, p_value = ttest_ind(df['truth'] == 1, df['truth'] == 0, equal_var=False)
            score_plot.text(.94, .95, 'p={}'.format(round(p_value, 3)), ha='center', va='center',
                            transform=score_plot.transAxes, fontsize=8)
            score_plot_fig = score_plot.get_figure()

            # PLOT DISTRIBUTION of outcomes
            plt.figure()
            outcome_0 = df[df['truth'] == 0]
            outcome_1 = df[df['truth'] == 1]
            dist_plot = sns.distplot(outcome_0[feature_col], bins=30, color='blue')
            sns.distplot(outcome_1[feature_col], bins=30, ax=dist_plot, color='orange')
            dist_plot_fig = dist_plot.get_figure()
            return score_plot_fig, dist_plot_fig
        else:
            plt.figure()
            sns.set_style("whitegrid")
            score_plot = sns.boxplot(x=df[outcome_col], y=df[feature_col], showfliers=False, color='white')
            sns.swarmplot(x=df[outcome_col], y=df[feature_col], ax=score_plot)
            score_plot.set_ylim(0.0, 1.05)
            score_plot.set_ylabel(str(feature_col))
            score_plot.set(xticklabels=unique_features)
            score_plot.set_title('CV Scores. AUC={}'.format(score_auc))
            score_plot_fig = score_plot.get_figure()
            return score_plot_fig


class ClassificationReport(Report):
    """\
    Generates reports for cross-validated classifiers

    """
    def __init__(self, renderer, output_train_scores=False, label_list=None):
        """

        :param renderer: renderer to use
        :param output_train_scores: whether to output CV sample scores for the training samples. Default: False

        """
        super(ClassificationReport, self).__init__(renderer)
        self.output_train_scores = output_train_scores
        self.label_list = label_list

    def summarize_performance(self, cv_results):
        """\
        Summarizes classification metrics.

        :param cv_results: list of results from each cross validation split
        :return: DataFrame with performance numbers, and also a dataframe with row averages

        """
        perf_df = self.generate_performance_metrics_dataframe(cv_results)

        # Compute averages across CV splits
        metric_cols = [col for col in perf_df.columns if col.startswith('train_') or col.startswith('test_')]
        average_row = assert_rows_are_concordant(perf_df, ignore_columns=['cv_index', "best_choice"] + metric_cols)
        average_row['best_choice'] = perf_df.groupby("best_choice").count().sort_values("cv_index").index[-1] \
            if "best_choice" in perf_df.columns else "None"
        average_row.update({metric: perf_df[metric].mean() for metric in metric_cols})

        return perf_df, pd.DataFrame([average_row])

    def generate_performance_metrics_dataframe(self, cv_results):
        """\
        Returns a pandas dataframe containing performance metrics.
        Functionality refactored outside of summarize_performance so ComparativeLearningApproachReport
        can use it.

        :param cv_results: list of results from each cross validation split
        :return: DataFrame with performance numbers

        """
        perf_df = pd.DataFrame()
        perf_df['cv_index'] = [r['cv_index'] for r in cv_results]
        perf_df['approach_type'] = [r['approach']['type'] for r in cv_results]
        perf_df['best_choice'] = [r.get('best_choice', 'None') for r in cv_results]
        perf_df['n_features'] = [r['train']['n_features'] for r in cv_results]
        for metric in list(cv_results[0]['test']['metrics'].keys()):
            perf_df['train_{}'.format(metric)] = [r['train']['metrics'][metric] for r in cv_results]
            perf_df['test_{}'.format(metric)] = [r['test']['metrics'][metric] for r in cv_results]
        return perf_df

    def summarize_features(self, cv_results):
        """\
        Summarizes info about which features were selected in cross validation.

        :param cv_results: list of results from each cross validation split
        :return: DataFrame with feature statistics

        """
        def median_or_nan(lst):
            """Returns the median if list is non-empty, or nan otherwise"""
            return np.median(lst) if len(lst) > 0 else float('nan')

        feature_counts = Counter()
        feature_p_vals = defaultdict(list)
        for r in cv_results:
            feature_counts.update(r['approach'].get('selected_features', []))
            for feat, p_val in r['approach'].get('feature_p_values', {}).items():
                feature_p_vals[feat].append(p_val)

        df = pd.DataFrame(list(feature_counts.items()), columns=['feature', 'times_selected'])
        df['median_p_value'] = [median_or_nan(feature_p_vals.get(feat)) for feat in df['feature']]
        df['frequency'] = df['times_selected'] / len(cv_results)
        df['n_cv_splits'] = len(cv_results)
        return df.sort_values('median_p_value', ascending=True)

    def summarize_scores(self, cv_results):
        """\
        Summarizes sample scores.

        :param cv_results: list of results from each cross validation split
        :return: DataFrame with sample scores

        """
        scores_column_order = ['subset', 'sample', 'outcome', 'positive_outcome', 'truth', 'score']

        def cv_result_to_frame(cv_result):
            """Converts results from a single CV split into a DataFrame"""
            frames = []

            for subset in ['train', 'test']:
                y_score = cv_result[subset]['scores']
                y_truth = cv_result[subset]['truth']
                if len(y_score) > 0 and isinstance(y_score[0], np.ndarray):
                    if y_score[0].shape[0] != 2:
                        y_score = [y_score[index, y_truth[index]] for index in range(len(y_score))]
                    else:
                        # binary case with score of 2 computed for each sample with 1 being positive_outcome column
                        y_score = y_score[:, 1]
                sdf = pd.DataFrame({'sample': cv_result[subset]['sample'],
                                    'score': y_score,
                                    'truth': y_truth,
                                    'outcome': cv_result[subset]['outcome']})
                sdf['cv_index'] = cv_result['cv_index']
                sdf['positive_outcome'] = cv_result[subset]['positive_outcome']
                sdf['subset'] = subset
                frames.append(partially_reorder_columns(sdf, ['cv_index'] + scores_column_order))
            return pd.concat(frames, ignore_index=True)

        cv_scores_df = pd.concat([cv_result_to_frame(r) for r in cv_results], ignore_index=True)
        if not self.output_train_scores:
            cv_scores_df = cv_scores_df[cv_scores_df['subset'] != 'train']

        # Compute average scores across CV splits
        averages = []
        for _, sample_sdf in cv_scores_df.groupby(by=['sample', 'subset']):
            average_row = assert_rows_are_concordant(sample_sdf, ignore_columns=['cv_index', 'score'])
            average_row['score'] = sample_sdf['score'].mean()
            averages.append(average_row)

        return cv_scores_df, partially_reorder_columns(pd.DataFrame(averages), scores_column_order)

    def compute_with_averaging_for_multiclass(self, y_truth, score_truth, fcn):
        """ Computes value for binary cases or averaging for multiclass using upper bound for score in
             case of wrong prediction
        """
        result = 0.0
        unique_values = np.unique(y_truth)
        if len(unique_values) > 2:
            for class_value in unique_values:
                indicator_truth = [1 if y == class_value else 0 for y in y_truth]
                indicator_score_estimate = [score_truth[ind] if y_truth[ind] == class_value
                                            else 1.0 - score_truth[ind] for ind in range(len(y_truth))]
                result += fcn(indicator_truth, indicator_score_estimate)
            result /= len(unique_values)
        else:
            result = fcn(y_truth, score_truth)
        return result

    def get_score_plots(self, mean_scores_df):
        """\
        Generates score plots.

        :param mean_scores_df: DataFrame with mean sample scores
        :return: list of plots

        """

        if len(mean_scores_df['truth'].unique()) == 2:
            roc_auc = ROC.from_scores(mean_scores_df['truth'], mean_scores_df['score']).auc_ci
        else:
            roc_auc = self.compute_with_averaging_for_multiclass(mean_scores_df['truth'], mean_scores_df['score'],
                                                                 roc_auc_score)
        return self.plot_scores(mean_scores_df, roc_auc)

    def generate(self, results):
        """\
        Generates the classification report.
        :param results: list of results from each cross validation split
        :return: None

        """
        cv_perf, mean_perf = self.summarize_performance(results)
        self.renderer.add_table('cv_metrics', cv_perf)
        self.renderer.add_table('mean_metrics', mean_perf)
        test_key = 'test'
        if len(results) > 0 and test_key in results[0] and 'confusion_matrix' in results[0][test_key]:
            best_accuracy = -1.0
            best_accuracy_index = -1
            for index, cv_result in enumerate(results):
                accuracy_at_index = accuracy_from_confusion_matrix(cv_result[test_key]['truth'],
                                                                   cv_result[test_key]['scores'],
                                                                   cv_result[test_key]['confusion_matrix'])
                if accuracy_at_index > best_accuracy:
                    best_accuracy = accuracy_at_index
                    best_accuracy_index = index
            if best_accuracy_index >= -1 and 'confusion_matrix' in results[best_accuracy_index][test_key]:
                cv_confusion_matrix = pd.DataFrame(results[best_accuracy_index][test_key]['confusion_matrix'])\
                    if self.label_list is None else\
                    pd.DataFrame(data=results[best_accuracy_index][test_key]['confusion_matrix'],
                                 columns=self.label_list)
                self.renderer.add_table('sample_confusion_matrix', cv_confusion_matrix)
                print(" best accuracy " + str(best_accuracy))

        cv_scores, mean_scores = self.summarize_scores(results)
        self.renderer.add_table('cv_sample_scores', cv_scores)
        self.renderer.add_table('mean_sample_scores', mean_scores)
        self.renderer.add_plot('score_plots', self.get_score_plots(mean_scores))

        self.renderer.add_table('selected_features', self.summarize_features(results))
        unique_values = np.unique(mean_scores['truth'])
        if len(unique_values) == 2:
            self.renderer.add_plot('roc', self.plot_roc(mean_scores, outcome_col='truth'))


class ComparativeClassificationReport(Report):
    """Report comparing learning approaches"""

    def __init__(self, renderer):
        """\

        :param renderer: the renderer to use

        """
        # pylint: disable=useless-super-delegation
        super(ComparativeClassificationReport, self).__init__(renderer)

    def get_concatenated_metrics(self, results):
        """\
        Generates the base concatenated report data from analysis results.

        :param results: Dictionary mapping LearningApproaches to results
        :return:

        """
        reporter = ClassificationReport(self.renderer)
        reports = []
        for approach in results:
            _, perf_df = reporter.summarize_performance(results[approach])
            perf_df["approach"] = str(approach)
            reports.append(perf_df)
        return pd.concat(reports)

    def get_concatenated_scores(self, results):
        """\
        Generates the concatenated scores from analysis results.

        :param results: Dictionary mapping LearningApproaches to results
        :return:

        """
        report = ClassificationReport(self.renderer)
        scores = []
        for approach in results:
            _, average_scores = report.summarize_scores(results[approach])
            average_scores["approach"] = str(approach)
            scores.append(average_scores)
        return pd.concat(scores)

    def get_score_plots(self, results):
        """\
        Generates the report from analysis results.

        :param results: Dictionary mapping LearningApproaches to results
        :return: list of plots ([boxplot, distribution])

        """
        score_df = self.get_concatenated_scores(results)
        return self.plot_scores(score_df)

    def generate(self, results):
        """\
        Generates the comparative report.
        :param cv_results: Dictionary mapping LearningApproaches to results (same input as get_score_plots)
        :return: None

        """
        self.renderer.add_table('mean_scores', self.get_concatenated_scores(results))
        self.renderer.add_table('mean_metrics', self.get_concatenated_metrics(results))
        self.renderer.add_plot('score_plots', self.get_score_plots(results))
