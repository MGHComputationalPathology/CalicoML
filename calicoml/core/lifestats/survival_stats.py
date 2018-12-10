# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

import matplotlib
try:
    matplotlib.use('Agg')
except (ImportError, RuntimeError):
    import traceback
    traceback.print_exc()
    print("Could not use matplotlib Agg backend. Plotting will not work")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from calicoml.core.metrics import ConfidenceInterval
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter


def nan_to_json(x):
    """ converts nan float to json string """
    if isinstance(x, float):
        return 'nan' if np.isnan(x) else x
    else:
        return x


def ci_to_json(ci_object):
    """ result as json """
    return {"estimate": ci_object.estimate, "low": ci_object.low,
            "high": ci_object.high, "pval": nan_to_json(ci_object.pval)}


class Survival(object):
    """Kaplan-Meier curves and other survival analysis utilities for overall cohort"""
    def __init__(self, df, time_col, event_col, label='overall_estimate'):
        """ construct Survival object """
        self.df = df

        self.time_col = time_col
        self.event_col = event_col

        self.kmf_fit = None
        self.naf_fit = None
        self.label = label

    @property
    def time(self):
        """ time from time column """
        return np.asarray(self.df[self.time_col])

    @property
    def event(self):
        """ event from event column """
        return np.asarray(self.df[self.event_col])

    def _fit_kaplan_meier(self):
        """ private method to fit Kaplan-Meier curve """
        if self.kmf_fit is not None:  # already fitted
            return

        # Overall
        kmf_fit = KaplanMeierFitter()
        kmf_fit.fit(self.time, event_observed=self.event, label=self.label)

        naf_case = NelsonAalenFitter()
        naf_case.fit(self.time, event_observed=self.event, label=self.label)

        self.kmf_fit = kmf_fit
        self.naf_fit = naf_case

    def plot_kaplan_meier(self, path=None, ax_plot=None):
        """ plots Kaplan-Meier curve """
        self._fit_kaplan_meier()
        kmf_plot = self.kmf_fit.plot(show_censors=True) if ax_plot is None else self.kmf_fit.plot(ax=ax_plot,
                                                                                                  show_censors=True)
        if path is None:
            return kmf_plot
        plt.savefig(path)
        return None

    def plot_nelson_aalen(self, path=None, ax_plot=None):
        """ plots Nelson-AAlen curve """
        self._fit_kaplan_meier()
        naf_plot = self.naf_fit.plot() if ax_plot is None else self.naf_fit.plot(ax=ax_plot)
        if path is None:
            return naf_plot
        plt.savefig(path)
        return None

    @property
    def hazards(self):
        """ hazard values"""
        def confidence(naf):
            """ wrapper around CI constructor"""
            return ConfidenceInterval(naf.cumulative_hazard_.values[:, 0][-1],
                                      naf.confidence_interval_['{}_lower_0.95'.format(self.label)].values[-1],
                                      naf.confidence_interval_['{}_upper_0.95'.format(self.label)].values[-1],
                                      float('nan'))
        self._fit_kaplan_meier()
        return confidence(self.naf_fit)

    def get_incidence(self, fraction=True):
        """ incidence ratio """
        def get_incidence(df):
            """ ratio of incidences"""
            num = float(np.sum(df[self.event_col] == 1))
            denominator = len(df) if fraction else 1
            return num / denominator
        return get_incidence(self.df)

    @property
    def hazard_points(self):
        """ outputs hazard points """
        def get_points(naf):
            """ wrapper to output each hazard point"""
            return [{"time": idx, "hazard": row[self.label]} for idx, row in naf.cumulative_hazard_.iterrows()]

        return get_points(self.naf_fit)

    def compare(self, other):
        """ makes comparative survival object """
        return ComparativeSurvival(self, other)

    def __getitem__(self, item):
        return Survival(self.df[item], self.time_col, self.event_col)

    def stratify(self, group_col, labels=None, group_col_value=1):
        """ makes comparative survival object based on binary column"""
        cases = self.df[self.df[group_col] == group_col_value]
        controls = self.df[self.df[group_col] != group_col_value]
        controls_sv = Survival(controls, self.time_col, self.event_col,
                               label=labels[0] if labels is not None else 'control')
        cases_sv = Survival(cases, self.time_col, self.event_col,
                            label=labels[1] if labels is not None else 'cases')
        return ComparativeSurvival(controls_sv, cases_sv)

    def to_json(self):
        """ output json with data """
        return {'cumulative_hazards': ci_to_json(self.hazards),
                'incidences': {'estimate': self.get_incidence(True)}}


class ComparativeSurvival(object):
    """coparative survival analysis utilities"""
    def __init__(self, survival0, survival1):
        """ construct ComparativeSurvival object """
        self.time_col1 = survival0.time_col
        self.time_col2 = survival1.time_col
        self.event_col1 = survival0.event_col
        self.event_col2 = survival1.event_col
        self.group_labels = [survival0.label, survival1.label]
        self.survival0 = survival0
        self.survival1 = survival1

        self._cf = None
        self._hr = None

    def plot_kaplan_meier(self, path, ax_plot=None):
        """ plots Kaplan-Meier curve """
        ax_plot1 = self.survival0.plot_kaplan_meier(ax_plot=ax_plot)
        self.survival1.plot_kaplan_meier(ax_plot=ax_plot1)
        plt.savefig(path)

    def plot_nelson_aalen(self, path):
        """ plots Nelson-AAlen curve """
        ax_plot1 = self.survival0.plot_nelson_aalen()
        self.survival1.plot_nelson_aalen(ax_plot=ax_plot1)
        plt.savefig(path)

    def _fit_cox(self):
        """ private method to fit Cox model """
        if self._cf is not None:
            return

        cox_df1 = pd.DataFrame(self.survival0.df, columns=[self.time_col1, self.event_col1])
        cox_df1[self.survival1.label] = 0
        cox_df2 = pd.DataFrame(self.survival1.df, columns=[self.time_col2, self.event_col2])
        if self.time_col1 != self.time_col2:
            cox_df2 = cox_df2.rename(columns={self.time_col2: self.time_col1})
        if self.event_col1 != self.event_col2:
            cox_df2 = cox_df2.rename(columns={self.event_col2: self.event_col1})
        cox_df2[self.survival1.label] = 1
        cox_df = cox_df1.append(cox_df2, ignore_index=True)

        cox_fitted = CoxPHFitter(normalize=False)
        cox_fitted.fit(cox_df, self.time_col1, event_col=self.event_col1, include_likelihood=False)

        self._cf = cox_fitted

    @property
    def cox_model_fitted(self):
        """ Cox model """
        if self._cf is None:
            self._fit_cox()
        return self._cf

    def hazard_ratio(self):
        """ computes hazard ratio """
        if self._hr is None:
            hazard, = np.exp(self.cox_model_fitted.hazards_[self.survival1.label])
            low, high = np.exp(self.cox_model_fitted.confidence_intervals_[self.survival1.label])
            pval, = self.cox_model_fitted.summary['p']
            self._hr = ConfidenceInterval(hazard, low, high, pval)

        return self._hr

    @property
    def hazards(self):
        """ hazard values"""
        return [self.survival0.hazards, self.survival1.hazards]

    def get_incidences(self, fraction=True):
        """ incidens ratios for each group"""
        denominator1 = self.survival0.df.shape[0]
        denominator2 = self.survival1.df.shape[0]
        denominator = denominator1 + denominator2 if fraction else 1
        return [float(self.survival0.get_incidence()) * denominator1 / denominator,
                float(self.survival1.get_incidence()) * denominator2 / denominator]

    @property
    def incidence_contingency(self):
        """ computes incidence contigency """
        n_case = self.survival1.df.shape[0]
        n_control = self.survival0.df.shape[0]

        n_event_case, n_event_control = self.get_incidences(False)
        return [[n_event_control, n_control - n_event_control],
                [n_event_case, n_case - n_event_case]]

    @property
    def hazard_points(self):
        """ outputs hazard points """
        return dict(list(zip(self.group_labels, [self.survival0.hazard_points, self.survival1.hazard_points])))

    def to_json(self):
        """ output json with data """
        return {'hazard_ratio': ci_to_json(self.hazard_ratio()),
                'cumulative_hazards': dict(list(zip(self.group_labels,
                                                    [ci_to_json(x) for x in self.hazards]))),
                'incidences': dict(list(zip(self.group_labels,
                                            [{'estimate': x} for x in self.get_incidences(True)])))}
