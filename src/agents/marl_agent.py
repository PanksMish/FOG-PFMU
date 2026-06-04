"""
Coordination-Aware MARL Agent
==============================
Implements the multi-agent reinforcement learning agent from Section IV.B.

Key equations:
  - Eq. 19: Shaped reward with neighbour penalty
  - Eq. 20: Q-learning update rule
  - Eq. 8:  State augmentation with neighbour queues
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MARLAgent:
    """
    Tabular Q-learning agent with coordination-aware reward shaping.

    The reward shaping (Eq. 19) adds a penalty for imbalanced queues
    between neighbouring intersections, reducing congestion spillover.

    Q-learning update (Eq. 20):
      Q(s,a) ← Q(s,a) + λ[R + γ·max_a Q(s',a) - Q(s,a)]
    """

    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        config: dict,
    ):
        self.id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim

        marl_cfg = config["marl"]
        self.lr = marl_cfg["learning_rate"]           # λ (lambda) in Eq. 20
        self.gamma = marl_cfg["discount_factor"]      # γ
        self.epsilon = marl_cfg["epsilon_start"]
        self.epsilon_end = marl_cfg["epsilon_end"]
        self.epsilon_decay = marl_cfg["epsilon_decay"]
        self.xi = marl_cfg["coordination_weight"]     # ξ (xi) in Eq. 19

        # Discretise state space for tabular Q-learning
        # Bins per dimension: 10 levels is sufficient for 6-dim state
        self._num_bins = 10
        self._bins = [
            np.linspace(0, 1, self._num_bins + 1)[1:-1]
            for _ in range(state_dim)
        ]

        # Q-table: state_dim → _num_bins^state_dim (approximate, use hash)
        # For scalability we use a dictionary Q-table
        self.Q: dict = {}

        # Metrics
        self.total_updates = 0
        self.episode_rewards: list = []

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection (Algorithm 1, line 4).

        Parameters
        ----------
        state : np.ndarray  shape (state_dim,)

        Returns
        -------
        int : action index (signal phase)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        s_key = self._state_key(state)
        q_vals = self._get_q_values(s_key)
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    # Reward Shaping (Eq. 19)
    # ------------------------------------------------------------------

    def compute_shaped_reward(
        self,
        base_reward: float,
        local_queues: np.ndarray,
        neighbour_queue_summaries: np.ndarray,
    ) -> float:
        """
        Add coordination penalty to base reward (Eq. 19):

          R_t^i = base_R - ξ · Σ_{j∈N(i)} |q_{t,i} - q_{t,j}|

        The penalty discourages large queue imbalances between neighbours,
        reducing congestion spillover effects.

        Parameters
        ----------
        base_reward               : float from Intersection.compute_reward()
        local_queues              : np.ndarray shape (num_lanes,)
        neighbour_queue_summaries : np.ndarray shape (num_neighbours,)
                                    Each value is the total queue at neighbour j
        Returns
        -------
        float : shaped reward
        """
        if len(neighbour_queue_summaries) == 0:
            return base_reward

        local_total = float(local_queues.sum())
        imbalance = np.sum(np.abs(local_total - neighbour_queue_summaries))
        shaped = base_reward - self.xi * imbalance
        return float(shaped)

    # ------------------------------------------------------------------
    # Q-Learning Update (Eq. 20)
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Q-learning update (Eq. 20):
          Q(s,a) ← Q(s,a) + λ[R + γ·max_a Q(s',a) - Q(s,a)]

        Parameters
        ----------
        state      : current state
        action     : action taken
        reward     : shaped reward received
        next_state : resulting state
        done       : whether episode ended
        """
        s_key = self._state_key(state)
        s_next_key = self._state_key(next_state)

        q_current = self._get_q_values(s_key)
        q_next = self._get_q_values(s_next_key)

        target = reward + (0.0 if done else self.gamma * np.max(q_next))
        td_error = target - q_current[action]
        q_current[action] += self.lr * td_error

        # Write back
        if s_key not in self.Q:
            self.Q[s_key] = np.zeros(self.action_dim)
        self.Q[s_key][action] = q_current[action]

        self.total_updates += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_key(self, state: np.ndarray) -> tuple:
        """Discretise continuous state into a hashable tuple."""
        discretised = tuple(
            int(np.digitize(state[d], self._bins[d]))
            for d in range(self.state_dim)
        )
        return discretised

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Return Q-values for state_key, initialising to zeros if unseen."""
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.action_dim)
        return self.Q[state_key].copy()

    def reset_episode(self):
        """Called at the start of each episode (epsilon is not reset)."""
        pass

    def get_policy_entropy(self) -> float:
        """Measure policy determinism (lower = more deterministic)."""
        if not self.Q:
            return float(np.log(self.action_dim))
        entropies = []
        for q_vals in self.Q.values():
            probs = np.exp(q_vals - q_vals.max())
            probs /= probs.sum()
            ent = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(ent)
        return float(np.mean(entropies))
