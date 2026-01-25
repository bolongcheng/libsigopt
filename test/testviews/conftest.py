# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np
import pytest

from libsigopt.aux.adapter_info_containers import DomainInfo, GPModelInfo, MetricsInfo
from libsigopt.aux.constant import TASK_SELECTION_STRATEGY_A_PRIORI
from libsigopt.compute.misc.constant import NonzeroMeanType

from testaux.utils import (
    DEFAULT_NOISE_PER_POINT,
    TEST_FAILURE_PROB,
    form_points_being_sampled,
    form_points_sampled,
    form_points_to_evaluate,
    form_random_hyperparameter_dict,
    form_random_unconstrained_categorical_domain,
)


def form_domain_info(domain):
    return DomainInfo(
        constraint_list=domain.constraint_list,
        domain_components=domain.domain_components,
        force_hitandrun_sampling=domain.force_hitandrun_sampling,
        priors=domain.priors,
    )


def form_model_info(
    domain,
    task_options,
    num_metrics,
    nonzero_mean_type,
    use_tikhonov,
):
    return GPModelInfo(
        hyperparameters=form_random_hyperparameter_dict(
            domain,
            add_task_length=task_options.size,
            num_metrics=num_metrics,
            use_tikhonov=use_tikhonov,
        ),
        max_simultaneous_af_points=5432,
        nonzero_mean_info=form_nonzero_mean_data(domain.dim, nonzero_mean_type),
        task_selection_strategy=TASK_SELECTION_STRATEGY_A_PRIORI if task_options.size else None,
    )


def form_nonzero_mean_data(dim, mean_type):
    if mean_type == NonzeroMeanType.CUSTOM:
        raise ValueError("This will need some work to make work with tasks")
    return {"mean_type": mean_type, "poly_indices": None}


def form_metrics_info(
    num_optimized_metrics,
    num_constraint_metrics,
    num_stored_metrics,
    metric_objectives,
    optimized_metric_thresholds=None,
    constraint_metric_thresholds=None,
):
    num_metrics = num_optimized_metrics + num_constraint_metrics + num_stored_metrics
    shuffled_index = np.random.permutation(np.arange(num_metrics, dtype=int))
    optimized_metrics_index = shuffled_index[:num_optimized_metrics]
    constraint_metrics_index = shuffled_index[num_optimized_metrics : num_optimized_metrics + num_constraint_metrics]

    user_specified_thresholds = np.full(num_metrics, np.nan)
    assert optimized_metric_thresholds is None or num_optimized_metrics == len(optimized_metric_thresholds)
    assert constraint_metric_thresholds is None or num_constraint_metrics == len(constraint_metric_thresholds)
    if optimized_metric_thresholds is not None:
        user_specified_thresholds[optimized_metrics_index] = optimized_metric_thresholds
    if constraint_metric_thresholds is not None:
        user_specified_thresholds[constraint_metrics_index] = constraint_metric_thresholds

    requires_pareto_frontier_optimization = False
    if num_optimized_metrics == 2:
        requires_pareto_frontier_optimization = True

    if not metric_objectives:
        metric_objectives = ["maximize" for _ in range(num_metrics)]
    return MetricsInfo(
        requires_pareto_frontier_optimization=requires_pareto_frontier_optimization,
        observation_budget=np.random.randint(50, 200),
        user_specified_thresholds=user_specified_thresholds,
        objectives=metric_objectives,
        optimized_metrics_index=optimized_metrics_index,
        constraint_metrics_index=constraint_metrics_index,
    )


class ZigoptSimulator(object):
    def __init__(
        self,
        dim,
        num_sampled,
        num_optimized_metrics=1,
        num_constraint_metrics=0,
        num_stored_metrics=0,
        num_to_sample=0,
        num_being_sampled=0,
        noise_per_point=DEFAULT_NOISE_PER_POINT,
        nonzero_mean_type=NonzeroMeanType.CONSTANT,
        use_tikhonov=False,
        num_tasks=0,
        failure_prob=TEST_FAILURE_PROB,
        metric_objectives=None,
        optimized_metric_thresholds=None,
        constraint_metric_thresholds=None,
    ):
        self.dim = dim
        self.num_sampled = num_sampled
        self.num_to_sample = num_to_sample
        self.num_being_sampled = num_being_sampled
        self.nonzero_mean_type = nonzero_mean_type
        self.noise_per_point = noise_per_point
        self.num_metrics = num_optimized_metrics + num_constraint_metrics + num_stored_metrics
        self.use_tikhonov = use_tikhonov
        self.failure_prob = failure_prob
        self.metric_objectives = metric_objectives
        self.num_optimized_metrics = num_optimized_metrics
        self.num_constraint_metrics = num_constraint_metrics
        self.num_stored_metrics = num_stored_metrics
        self.optimized_metric_thresholds = optimized_metric_thresholds
        self.constraint_metric_thresholds = constraint_metric_thresholds
        self.num_tasks = num_tasks

    def form_gp_ei_categorical_inputs(self, parallelism_method):
        task_options = np.sort(np.random.random(self.num_tasks) if self.num_tasks else [])
        domain = form_random_unconstrained_categorical_domain(self.dim)
        points_sampled = form_points_sampled(
            domain,
            self.num_sampled,
            self.noise_per_point,
            self.num_metrics,
            task_options,
            failure_prob=self.failure_prob,
        )
        points_to_evaluate = form_points_to_evaluate(domain, self.num_to_sample, task_options=task_options)
        points_being_sampled = form_points_being_sampled(
            domain,
            self.num_being_sampled,
            task_options=task_options,
        )
        model_info = form_model_info(
            domain,
            task_options,
            self.num_metrics,
            self.nonzero_mean_type,
            self.use_tikhonov,
        )

        view_input = {
            "domain_info": form_domain_info(domain),
            "model_info": model_info,
            "parallelism": parallelism_method,
            "points_being_sampled": points_being_sampled,
            "points_sampled": points_sampled,
            "points_to_evaluate": points_to_evaluate,
            "tag": {"experiment_id": -1},
            "metrics_info": form_metrics_info(
                self.num_optimized_metrics,
                self.num_constraint_metrics,
                self.num_stored_metrics,
                self.metric_objectives,
                self.optimized_metric_thresholds,
                self.constraint_metric_thresholds,
            ),
            "task_options": task_options,
        }

        return view_input

    def form_gp_next_points_view_input_from_domain(self, domain, parallelism_method):
        task_options = np.sort(np.random.random(self.num_tasks) if self.num_tasks else [])
        points_sampled = form_points_sampled(
            domain,
            self.num_sampled,
            self.noise_per_point,
            self.num_metrics,
            task_options,
            failure_prob=self.failure_prob,
        )
        points_being_sampled = form_points_being_sampled(
            domain,
            self.num_being_sampled,
            task_options=task_options,
        )
        model_info = form_model_info(
            domain,
            task_options,
            self.num_metrics,
            self.nonzero_mean_type,
            self.use_tikhonov,
        )

        view_input = {
            "domain_info": form_domain_info(domain),
            "model_info": model_info,
            "num_to_sample": self.num_to_sample,
            "parallelism": parallelism_method,
            "points_sampled": points_sampled,
            "points_being_sampled": points_being_sampled,
            "tag": {"experiment_id": -1},
            "metrics_info": form_metrics_info(
                self.num_optimized_metrics,
                self.num_constraint_metrics,
                self.num_stored_metrics,
                self.metric_objectives,
                self.optimized_metric_thresholds,
                self.constraint_metric_thresholds,
            ),
            "task_options": task_options,
        }
        return view_input

    def form_gp_next_points_categorical_inputs(self, parallelism_method):
        domain = form_random_unconstrained_categorical_domain(self.dim)
        view_input = self.form_gp_next_points_view_input_from_domain(domain, parallelism_method)
        return view_input, domain

    def form_search_next_points_view_input_from_domain(self, domain, parallelism_method):
        view_input = self.form_gp_next_points_view_input_from_domain(domain, parallelism_method)
        return view_input, domain

    def form_search_next_points_categorical_inputs(self, parallelism_method):
        domain = form_random_unconstrained_categorical_domain(self.dim)
        view_input = self.form_gp_next_points_view_input_from_domain(domain, parallelism_method)
        return view_input, domain

    def form_spe_next_points_view_input_from_domain(self, domain):
        task_options = np.sort(np.random.random(self.num_tasks) if self.num_tasks else [])
        points_sampled = form_points_sampled(
            domain,
            self.num_sampled,
            self.noise_per_point,
            self.num_metrics,
            task_options,
            failure_prob=self.failure_prob,
        )
        points_being_sampled = form_points_being_sampled(
            domain,
            self.num_being_sampled,
            task_options=task_options,
        )

        view_input = {
            "domain_info": form_domain_info(domain),
            "max_simultaneous_af_points": 1000,
            "num_to_sample": self.num_to_sample,
            "points_sampled": points_sampled,
            "points_being_sampled": points_being_sampled,
            "tag": {"experiment_id": -1},
            "metrics_info": form_metrics_info(
                self.num_optimized_metrics,
                self.num_constraint_metrics,
                self.num_stored_metrics,
                self.metric_objectives,
                self.optimized_metric_thresholds,
                self.constraint_metric_thresholds,
            ),
            "task_options": task_options,
        }
        view_input["metrics_info"].observation_budget = np.random.randint(50, 200)

        return view_input

    def form_spe_next_points_inputs(self):
        domain = form_random_unconstrained_categorical_domain(self.dim)
        view_input = self.form_spe_next_points_view_input_from_domain(domain)
        return view_input, domain

    def form_spe_search_next_points_inputs(self):
        domain = form_random_unconstrained_categorical_domain(self.dim)
        view_input = self.form_spe_next_points_view_input_from_domain(domain)
        return view_input, domain

    def form_gp_hyper_opt_categorical_inputs(self):
        task_options = np.sort(np.random.random(self.num_tasks) if self.num_tasks else [])
        domain = form_random_unconstrained_categorical_domain(self.dim)
        points_sampled = form_points_sampled(
            domain,
            self.num_sampled,
            self.noise_per_point,
            self.num_metrics,
            task_options,
            failure_prob=self.failure_prob,
        )
        model_info = form_model_info(
            domain,
            task_options,
            self.num_metrics,
            self.nonzero_mean_type,
            self.use_tikhonov,
        )

        view_input = {
            "domain_info": form_domain_info(domain),
            "model_info": model_info,
            "points_sampled": points_sampled,
            "tag": {"experiment_id": -1},
            "metrics_info": form_metrics_info(
                self.num_optimized_metrics,
                self.num_constraint_metrics,
                self.num_stored_metrics,
                self.metric_objectives,
                self.optimized_metric_thresholds,
                self.constraint_metric_thresholds,
            ),
            "task_options": task_options,
        }

        return view_input, domain

    def form_random_search_view_input_from_domain(self, domain):
        task_options = np.sort(np.random.random(self.num_tasks) if self.num_tasks else [])
        view_input = {
            "domain_info": form_domain_info(domain),
            "num_to_sample": self.num_to_sample,
            "task_options": task_options,
            "tag": {"experiment_id": -1},
        }
        return view_input

    def form_random_search_view_input(self):
        domain = form_random_unconstrained_categorical_domain(self.dim)
        view_input = self.form_random_search_view_input_from_domain(domain)
        return view_input, domain


@pytest.fixture
def zigopt_simulator_factory():
    def _make_simulator(
        dim,
        num_sampled,
        num_optimized_metrics=1,
        num_constraint_metrics=0,
        num_stored_metrics=0,
        num_to_sample=0,
        num_being_sampled=0,
        noise_per_point=DEFAULT_NOISE_PER_POINT,
        nonzero_mean_type=NonzeroMeanType.CONSTANT,
        use_tikhonov=False,
        num_tasks=0,
        failure_prob=TEST_FAILURE_PROB,
        metric_objectives=None,
        optimized_metric_thresholds=None,
        constraint_metric_thresholds=None,
    ):
        return ZigoptSimulator(
            dim=dim,
            num_sampled=num_sampled,
            num_optimized_metrics=num_optimized_metrics,
            num_constraint_metrics=num_constraint_metrics,
            num_stored_metrics=num_stored_metrics,
            num_to_sample=num_to_sample,
            num_being_sampled=num_being_sampled,
            noise_per_point=noise_per_point,
            nonzero_mean_type=nonzero_mean_type,
            use_tikhonov=use_tikhonov,
            num_tasks=num_tasks,
            failure_prob=failure_prob,
            metric_objectives=metric_objectives,
            optimized_metric_thresholds=optimized_metric_thresholds,
            constraint_metric_thresholds=constraint_metric_thresholds,
        )

    return _make_simulator
