"""
Intersection MDP
================
Implements the per-intersection Markov Decision Process from Section III.B.

Equations:
  - Eq. 3:  M = (S, A, P, R, γ)   — MDP tuple
  - Eq. 4:  s_t^i = {q_{t,l}, φ_t, δ_t}  — local state
  - Eq. 5:  R_t^i = -(α·Σ_l q_{t,l} + β·D_t + η·L_t)  — base reward
"""

import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class Intersection:
    """
    Single signalised intersection modelled as a local MDP.

    State  (Eq. 4): [queue lengths per lane | active phase | elapsed duration]
    Action         : Choose next signal phase (0 to num_phases-1)
    Reward (Eq. 5) : Negative weighted sum of queues, delay, switching penalty
    """

    def __init__(self, intersection_id: int, config: dict):
        self.id = intersection_id
        self.config = config
        self.num_lanes = config["traffic"]["lanes_per_intersection"]
        self.num_phases = config["traffic"]["num_phases"]
        self.phase_min = config["traffic"]["phase_duration_min"]
        self.phase_max = config["traffic"]["phase_duration_max"]

        # Reward weights (Eq. 5)
        marl_cfg = config["marl"]
        self.alpha = marl_cfg["alpha"]         # queue weight
        self.beta = marl_cfg["beta"]           # delay weight
        self.eta = marl_cfg["eta"]             # switching penalty

        # State variables
        self.queue_lengths: np.ndarray = np.zeros(self.num_lanes)
        self.active_phase: int = 0
        self.phase_elapsed: int = 0            # δ_t: steps in current phase
        self.accumulated_delay: float = 0.0   # D_t
        self.prev_phase: int = -1              # for switching penalty

        # History for metrics
        self.phase_switches: int = 0
        self.total_throughput: float = 0.0

    # ------------------------------------------------------------------
    # State (Eq. 4)
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """
        Return local state s_t^i = [q_{t,l}, φ_t, δ_t].

        Returns
        -------
        np.ndarray  shape: (num_lanes + 2,)
            [queue_lane_0, ..., queue_lane_k, active_phase_norm, elapsed_norm]
        """
        state = np.concatenate([
            self.queue_lengths / (self.config["traffic"]["saturation_flow"] / 3600),
            [self.active_phase / (self.num_phases - 1)],
            [self.phase_elapsed / self.phase_max],
        ])
        return state.astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self.num_lanes + 2

    @property
    def action_dim(self) -> int:
        return self.num_phases

    # ------------------------------------------------------------------
    # Reward (Eq. 5)
    # ------------------------------------------------------------------

    def compute_reward(self, departures: np.ndarray) -> float:
        """
        Base local reward (Eq. 5):
          R_t^i = -(α·Σ_l q_{t,l} + β·D_t + η·L_t)

        Parameters
        ----------
        departures : np.ndarray  shape (num_lanes,)

        Returns
        -------
        float : reward value (negative = cost)
        """
        queue_cost = self.alpha * self.queue_lengths.sum()
        self.accumulated_delay += self.queue_lengths.sum()  # proxy: each queued veh = 1s delay
        delay_cost = self.beta * self.accumulated_delay
        switch_cost = self.eta * (1.0 if self.active_phase != self.prev_phase else 0.0)
        return -(queue_cost + delay_cost + switch_cost)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def update(
        self,
        action: int,
        queue_lengths: np.ndarray,
        departures: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Apply action, update state, return reward.

        Parameters
        ----------
        action       : chosen signal phase
        queue_lengths: updated queues from TrafficNetwork
        departures   : vehicles that departed this step

        Returns
        -------
        (reward, info_dict)
        """
        self.prev_phase = self.active_phase

        # Phase switching logic
        if action != self.active_phase:
            self.active_phase = action
            self.phase_elapsed = 0
            self.accumulated_delay = 0.0
            self.phase_switches += 1
        else:
            self.phase_elapsed = min(self.phase_elapsed + 1, self.phase_max)

        self.queue_lengths = queue_lengths.copy()
        self.total_throughput += departures.sum()
        reward = self.compute_reward(departures)

        info = {
            "queue_total": float(self.queue_lengths.sum()),
            "throughput": float(departures.sum()),
            "phase_switches": self.phase_switches,
            "reward": reward,
        }
        return reward, info

    def reset(self):
        self.queue_lengths[:] = 0
        self.active_phase = 0
        self.phase_elapsed = 0
        self.accumulated_delay = 0.0
        self.prev_phase = -1
        self.phase_switches = 0
        self.total_throughput = 0.0
