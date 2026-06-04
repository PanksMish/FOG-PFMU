"""
Shared Congestion Index
=======================
Implements φ_t (Eq. 22) — the coupling variable between traffic and parking.

Eq. 22: φ_t = κ1·q̄_t + κ2·w̄_t

This index enables coordinated decision-making across the traffic and
parking subsystems within a unified optimisation framework (Section IV.D).
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CongestionIndex:
    """
    Computes and maintains the shared congestion index φ_t.

    φ_t serves as the coupling variable between traffic signal control
    and parking allocation (Eq. 22):

        φ_t = κ1·q̄_t + κ2·w̄_t

    where:
        q̄_t = average queue length (normalised)
        w̄_t = average parking waiting time (normalised)
        κ1, κ2 = weighting coefficients

    The index is computed per fog node and shared periodically
    with neighbouring nodes (Section IV.D, step 4).
    """

    def __init__(self, config: dict, history_len: int = 60):
        cfg = config["congestion_index"]
        self.kappa1 = cfg["kappa1"]    # queue weight
        self.kappa2 = cfg["kappa2"]    # parking wait-time weight

        # Normalisation constants
        self._max_queue = config["traffic"]["saturation_flow"] / 3600 * 10  # veh/step
        self._max_wait = 300.0  # seconds

        # Smoothed history
        self._queue_history = deque(maxlen=history_len)
        self._wait_history = deque(maxlen=history_len)
        self._phi_history = deque(maxlen=history_len)

        self._current_phi: float = 0.0

    def compute(self, avg_queue: float, avg_wait_time: float) -> float:
        """
        Compute φ_t (Eq. 22).

        Parameters
        ----------
        avg_queue     : q̄_t — average queue length across lanes
        avg_wait_time : w̄_t — average parking waiting time (seconds)

        Returns
        -------
        float : φ_t ∈ [0, 1]
        """
        q_norm = min(avg_queue / max(self._max_queue, 1e-6), 1.0)
        w_norm = min(avg_wait_time / max(self._max_wait, 1e-6), 1.0)

        phi = self.kappa1 * q_norm + self.kappa2 * w_norm
        phi = float(np.clip(phi, 0.0, 1.0))

        self._queue_history.append(q_norm)
        self._wait_history.append(w_norm)
        self._phi_history.append(phi)
        self._current_phi = phi
        return phi

    @property
    def current(self) -> float:
        """Most recently computed φ_t."""
        return self._current_phi

    @property
    def smoothed(self) -> float:
        """Exponentially-smoothed φ_t over history window."""
        if not self._phi_history:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(self._phi_history)))
        weights /= weights.sum()
        return float(np.dot(list(self._phi_history), weights))

    def reset(self):
        self._queue_history.clear()
        self._wait_history.clear()
        self._phi_history.clear()
        self._current_phi = 0.0
