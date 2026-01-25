# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np

from libsigopt.aux.adapter_info_containers import PointsContainer
from libsigopt.aux.constant import CATEGORICAL_EXPERIMENT_PARAMETER_NAME
from libsigopt.compute.domain import CategoricalDomain, ContinuousDomain, DomainComponent, DomainConstraint


def form_random_unconstrained_categorical_domain(dim, categoricals_allowed=True, quantized_allowed=True):
    domain_components: list[DomainComponent] = []
    for _ in range(dim):
        if np.random.random() < 0.1 and quantized_allowed:
            domain_components.append(
                {
                    "var_type": "quantized",
                    "elements": list(sorted(np.random.choice(50, 4, replace=False) / 10 - 2.2)),
                }
            )
        elif np.random.random() < 0.25 and categoricals_allowed:
            domain_components.append(
                {
                    "var_type": "categorical",
                    "elements": list(range(np.random.randint(2, 5))),
                }
            )
        elif np.random.random() < 0.5:
            bounds = (np.random.randint(-10, 0), np.random.randint(0, 10))
            domain_components.append({"var_type": "int", "elements": bounds})
        else:
            random_number = np.random.random()
            if random_number < 0.333:
                random_values = np.random.uniform(-2, 3, size=(2,))
            elif random_number < 0.666:
                random_values = np.random.gamma(0.3, 1.0, size=(2,))
            else:
                random_values = np.random.uniform(-34567, 12345, size=(2,))
            sorted_values = sorted(random_values)
            domain_components.append(
                {
                    "var_type": "double",
                    "elements": (sorted_values[0], sorted_values[1]),
                }
            )
    return CategoricalDomain(domain_components)


def form_random_constrained_categorical_domain(n_double_param=5, n_int_param=5, n_cat_param=1, n_quantized_param=1):
    assert n_double_param >= 5
    assert n_int_param >= 5
    assert n_cat_param >= 1
    assert n_quantized_param >= 1
    dim = n_double_param + n_int_param + n_cat_param + n_quantized_param
    idx_shuffled = np.arange(dim)
    np.random.shuffle(idx_shuffled)
    idx_double = idx_shuffled[0:n_double_param]
    idx_int = idx_shuffled[n_double_param : n_double_param + n_int_param]
    idx_cat = idx_shuffled[n_double_param + n_int_param : n_double_param + n_int_param + n_cat_param]
    idx_quantized = idx_shuffled[n_double_param + n_int_param + n_cat_param :]

    # Form domain components
    domain_components_maybe_none: list[DomainComponent | None] = [None] * dim
    for i in idx_double:
        bounds = (0, np.random.randint(1, 5))
        domain_components_maybe_none[i] = {
            "var_type": "double",
            "elements": bounds,
        }
    for i in idx_int:
        bounds = (5, np.random.randint(10, 20))
        domain_components_maybe_none[i] = {
            "var_type": "int",
            "elements": bounds,
        }
    for i in idx_cat:
        domain_components_maybe_none[i] = {
            "var_type": "categorical",
            "elements": list(range(np.random.randint(2, 5))),
        }
    for i in idx_quantized:
        domain_components_maybe_none[i] = {
            "var_type": "quantized",
            "elements": list(sorted(np.random.choice(50, 4, replace=False) / 10 - 2.2)),
        }
    domain_components = [c for c in domain_components_maybe_none if c is not None]
    assert len(domain_components) == dim

    # Form constraints
    constraint_list: list[DomainConstraint] = []
    constraint_weights_int = [0] * dim
    idx_constraint = np.random.choice(idx_int, 2, replace=False)
    for i in idx_constraint:
        constraint_weights_int[i] = -1
    constraint_list.append(
        {"weights": constraint_weights_int, "rhs": -18, "var_type": "int"},
    )

    constraint_weights_int = [0] * dim
    idx_constraint = np.random.choice(idx_int, 2, replace=False)
    for i in idx_constraint:
        constraint_weights_int[i] = 1
    constraint_list.append(
        {"weights": constraint_weights_int, "rhs": 12, "var_type": "int"},
    )

    constraint_weights_double = [0.0] * dim
    idx_constraint = np.random.choice(idx_double, 2, replace=False)
    for i in idx_constraint:
        constraint_weights_double[i] = -1
    constraint_list.append(
        {"weights": constraint_weights_double, "rhs": -1.5, "var_type": "double"},
    )

    constraint_weights_double = [0.0] * dim
    idx_constraint = np.random.choice(idx_double, 2, replace=False)
    for i in idx_constraint:
        constraint_weights_double[i] = 1
    constraint_list.append(
        {"weights": constraint_weights_double, "rhs": 0.5, "var_type": "double"},
    )

    return CategoricalDomain(domain_components, constraint_list)


DEFAULT_NOISE_PER_POINT = 1e-10
TEST_FAILURE_PROB = 0.1


# TODO(RTL-96): Clean this up to have a minimum number of points in the domain
def form_random_hyperparameter_dict(domain, use_tikhonov=False, add_task_length=False, num_metrics=1):
    list_of_hyperparameter_dict = []
    for _ in range(num_metrics):
        alpha = np.random.gamma(1, 0.1)
        tikhonov = np.random.gamma(1, 0.1) if use_tikhonov else None
        task_length = 0.19 if add_task_length else None
        length_scales = []
        for dc in domain:
            if dc["var_type"] == CATEGORICAL_EXPERIMENT_PARAMETER_NAME:
                length_scales.append(np.random.uniform(0.5, 2.0, len(dc["elements"])).tolist())
            else:
                length_scales.append([np.random.gamma(1, 0.1) * (dc["elements"][1] - dc["elements"][0])])
        list_of_hyperparameter_dict.append(
            {
                "alpha": alpha,
                "length_scales": length_scales,
                "tikhonov": tikhonov,
                "task_length": task_length,
            }
        )
    return list_of_hyperparameter_dict


# NOTE: Some potential issues with snap_cats as this is currently constructed
def form_points_sampled(
    domain,
    num_sampled,
    noise_per_point,
    num_metrics,
    task_options,
    snap_cats=False,
    failure_prob=TEST_FAILURE_PROB,
):
    points = domain.generate_quasi_random_points_in_domain(num_sampled)
    if isinstance(domain, ContinuousDomain) and snap_cats:
        for k, this_closed_interval in enumerate(domain.domain_bounds):
            if np.all(this_closed_interval == np.array([0, 1])):
                points[:, k] = np.round(points[:, k])
    values = np.random.uniform(-0.1, 0.1, (num_sampled, num_metrics))
    failures = np.random.random(num_sampled) < failure_prob

    return PointsContainer(
        points=points,
        values=values,
        value_vars=np.full_like(values, noise_per_point),
        failures=failures,
        task_costs=np.random.choice(task_options, size=failures.shape) if task_options.size else None,
    )


def form_points_being_sampled(domain, num_points_being_sampled, task_options=None):
    return PointsContainer(
        points=domain.generate_quasi_random_points_in_domain(num_points_being_sampled),
        task_costs=np.random.choice(task_options, size=num_points_being_sampled) if task_options.size else None,
    )


form_points_to_evaluate = form_points_being_sampled
