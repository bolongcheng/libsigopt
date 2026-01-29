# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PointsContainer:
    points: np.ndarray
    values: np.ndarray | None = None
    value_vars: np.ndarray | None = None
    failures: np.ndarray | None = None
    task_costs: np.ndarray | None = None


@dataclass(slots=True)
class DomainInfo:
    constraint_list: list[Any]
    domain_components: list[Any] | None = None
    force_hitandrun_sampling: bool = False
    priors: list[Any] | None = None

    @property
    def dim(self) -> int:
        if self.domain_components is None:
            return 0
        return len(self.domain_components)


@dataclass(slots=True)
class MetricsInfo:
    requires_pareto_frontier_optimization: bool
    observation_budget: int
    user_specified_thresholds: list[Any]
    objectives: list[Any]
    optimized_metrics_index: list[int]
    constraint_metrics_index: list[int]

    @property
    def has_optimization_metrics(self) -> bool:
        return len(self.optimized_metrics_index) > 0

    @property
    def has_constraint_metrics(self) -> bool:
        return len(self.constraint_metrics_index) > 0

    @property
    def has_optimized_metric_thresholds(self) -> bool:
        if len(self.optimized_metrics_index) == 0:
            return False
        return any(self.user_specified_thresholds[i] is not None for i in self.optimized_metrics_index)


@dataclass(slots=True)
class GPModelInfo:
    hyperparameters: list[dict[str, Any]]
    max_simultaneous_af_points: int
    nonzero_mean_info: dict[str, Any]
    task_selection_strategy: str | None = None
