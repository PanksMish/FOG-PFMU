"""
Reward Shaping
==============
Implements the reward functions from Section III.B and IV.B.

Eq. 5:  R_t^i = -(α·Σ_l q_{t,l} + β·D_t + η·L_t)
Eq. 19: R_t^i += -ξ·Σ_{j∈N(i)} |q_{t,i} - q_{t,j}|
"""

import numpy as np


def base_reward(
    queue_lengths: np.ndarray,
    accumulated_delay: float,
    switching_penalty: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    eta: float = 0.2,
) -> float:
    """
    Base reward (Eq. 5).

    Parameters
    ----------
    queue_lengths      : per-lane queue lengths
    accumulated_delay  : D_t, total vehicle-delay so far
    switching_penalty  : L_t, 1.0 if phase changed this step else 0.0
    alpha, beta, eta   : weighting coefficients

    Returns
    -------
    float : reward (negative = cost)
    """
    return -(
        alpha * float(queue_lengths.sum())
        + beta * accumulated_delay
        + eta * switching_penalty
    )


def coordination_reward(
    base_r: float,
    local_queue_total: float,
    neighbour_queue_totals: np.ndarray,
    xi: float = 0.3,
) -> float:
    """
    Coordination-aware reward (Eq. 19).

    Adds a spatial imbalance penalty to the base reward.

    R_t^i = base_r - ξ · Σ_{j∈N(i)} |q_{t,i} - q_{t,j}|

    Parameters
    ----------
    base_r                  : reward from base_reward()
    local_queue_total       : Σ_l q_{t,l}^i
    neighbour_queue_totals  : array of total queues at each neighbour j
    xi                      : coordination weight ξ

    Returns
    -------
    float : shaped reward
    """
    if len(neighbour_queue_totals) == 0:
        return base_r
    imbalance = float(np.sum(np.abs(local_queue_total - neighbour_queue_totals)))
    return base_r - xi * imbalance


def multi_objective_reward(
    queue_lengths: np.ndarray,
    throughput: float,
    delay: float,
    w_queue: float = 0.5,
    w_throughput: float = 0.3,
    w_delay: float = 0.2,
) -> float:
    """
    Multi-objective reward for A-MORL baseline.
    Balances queue minimisation with throughput maximisation.
    """
    normalised_queue = queue_lengths.sum() / (queue_lengths.size * 100 + 1e-6)
    normalised_throughput = throughput / 1800.0  # normalise by saturation flow
    normalised_delay = delay / 60.0              # normalise by 60s
    return (
        -w_queue * normalised_queue
        + w_throughput * normalised_throughput
        - w_delay * normalised_delay
    )
