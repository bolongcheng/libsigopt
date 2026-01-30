# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PointsContainer:
    points: NDArray[np.number]
    values: NDArray[np.number] | None = None
    value_vars: NDArray[np.number] | None = None
    failures: NDArray[np.bool_] | None = None
    task_costs: NDArray[np.number] | None = None


@dataclass(slots=True)
class DomainInfo:
    constraint_list: list[dict[str, Any]]
    domain_components: list[dict[str, Any]] | None = None
    force_hitandrun_sampling: bool = False
    priors: list[dict[str, Any]] | None = None

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
    objectives: list[str]
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
