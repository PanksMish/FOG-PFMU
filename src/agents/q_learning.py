"""
Q-Learning Utilities
====================
Standalone Q-learning helpers shared across agent variants (F-RL, A-MORL, PFMU).
"""

import numpy as np
from typing import Dict, Tuple


def q_update(
    q_table: Dict[tuple, np.ndarray],
    state_key: tuple,
    action: int,
    reward: float,
    next_state_key: tuple,
    action_dim: int,
    lr: float,
    gamma: float,
    done: bool = False,
) -> float:
    """
    Standard Q-learning update (Eq. 20).

    Q(s,a) ← Q(s,a) + λ·[R + γ·max_a'Q(s',a') - Q(s,a)]

    Returns
    -------
    float : TD error magnitude
    """
    if state_key not in q_table:
        q_table[state_key] = np.zeros(action_dim)
    if next_state_key not in q_table:
        q_table[next_state_key] = np.zeros(action_dim)

    q_current = q_table[state_key][action]
    q_next_max = 0.0 if done else np.max(q_table[next_state_key])
    target = reward + gamma * q_next_max
    td_error = target - q_current
    q_table[state_key][action] += lr * td_error
    return abs(td_error)


def epsilon_greedy(
    q_table: Dict[tuple, np.ndarray],
    state_key: tuple,
    action_dim: int,
    epsilon: float,
) -> int:
    """ε-greedy action selection."""
    if np.random.random() < epsilon or state_key not in q_table:
        return np.random.randint(action_dim)
    return int(np.argmax(q_table[state_key]))


def compute_jain_fairness(values: np.ndarray) -> float:
    """
    Jain's fairness index: J = (Σx_i)² / (n · Σx_i²)
    Range [1/n, 1]: 1 = perfectly fair.
    """
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 1.0
    return float((values.sum() ** 2) / (n * (values ** 2).sum()))
