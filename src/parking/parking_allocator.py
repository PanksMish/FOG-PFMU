"""
EASA+ Parking Allocator
========================
Implements the congestion-coupled parking allocation from Section IV.C.

Equations:
  - Eq. 9:   o_p(t) ∈ {0,1}  — binary slot occupancy
  - Eq. 10:  x_{vp} ∈ {0,1}  — allocation decision variable
  - Eq. 11:  min Σ_{v,p} c_{vp}·x_{vp}  — assignment objective
  - Eq. 12:  Σ_p x_{vp} = 1  ∀v  — each vehicle gets one slot
  - Eq. 13:  Σ_v x_{vp} ≤ 1  ∀p  — each slot at most one vehicle
  - Eq. 14:  c_{vp} = λ1·d_{vp} + λ2·T_{vp} + λ3·θ_p  — base cost
  - Eq. 21:  c*_{vp} = c_{vp} + λ4·φ_t  — congestion-aware cost (PFMU)

Algorithm 2 in the paper: PFMU Parking Allocation (EASA+)
"""

import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ParkingSlot:
    """Represents a single parking slot."""

    def __init__(self, slot_id: int, distance_from_node: float):
        self.id = slot_id
        self.distance = distance_from_node   # proxy for d_{vp}
        self.occupied: bool = False           # o_p(t) ∈ {0,1}
        self.occupant_vehicle: Optional[int] = None
        self.occupancy_duration: int = 0     # steps occupied

    def reserve(self, vehicle_id: int):
        self.occupied = True
        self.occupant_vehicle = vehicle_id
        self.occupancy_duration = 0

    def release(self):
        self.occupied = False
        self.occupant_vehicle = None
        self.occupancy_duration = 0

    @property
    def occupancy_rate(self) -> float:
        """θ_p: local congestion / occupancy indicator."""
        return 1.0 if self.occupied else 0.0


class ParkingAllocator:
    """
    EASA+ (Enhanced Adaptive Slot Allocation with congestion coupling).

    Implements Algorithm 2 with the congestion-aware cost function (Eq. 21).

    Parameters
    ----------
    node_id : int
        Fog node identifier
    config  : dict
        Full simulation configuration
    """

    def __init__(self, node_id: int, config: dict):
        self.node_id = node_id
        cfg = config["parking"]

        # Slot capacity drawn from uniform [min, max]
        rng = np.random.default_rng(seed=node_id)
        capacity = int(rng.integers(
            cfg["slot_capacity_min"],
            cfg["slot_capacity_max"] + 1
        ))
        self.capacity = capacity

        # Create slots with random distances [50, 500] metres
        distances = rng.uniform(50, 500, size=capacity)
        self.slots: List[ParkingSlot] = [
            ParkingSlot(slot_id=j, distance_from_node=float(distances[j]))
            for j in range(capacity)
        ]

        # Cost weights (Eq. 14 / 21)
        self.lambda1 = cfg["lambda1"]   # distance
        self.lambda2 = cfg["lambda2"]   # wait time
        self.lambda3 = cfg["lambda3"]   # occupancy
        self.lambda4 = cfg["lambda4"]   # congestion index (PFMU only)

        # Parking duration model (truncated normal in minutes)
        self.dur_min = cfg["parking_duration_min"]
        self.dur_max = cfg["parking_duration_max"]
        self._dur_mean = (self.dur_min + self.dur_max) / 2
        self._dur_std = (self.dur_max - self.dur_min) / 4

        # Stats
        self.total_requests: int = 0
        self.successful_allocations: int = 0
        self.search_time_history: List[float] = []

        logger.debug(f"ParkingAllocator node {node_id}: {capacity} slots")

    # ------------------------------------------------------------------
    # Algorithm 2 — EASA+
    # ------------------------------------------------------------------

    def allocate(self, vehicle_id: int, congestion_index: float = 0.0) -> Optional[int]:
        """
        Allocate a parking slot for vehicle_id (Algorithm 2).

        1. Compute cost c*_{vp} for each available slot (Eq. 21)
        2. Select slot with minimum cost
        3. Reserve slot or return None if lot full

        Parameters
        ----------
        vehicle_id        : unique vehicle identifier
        congestion_index  : φ_t shared congestion index from fog node

        Returns
        -------
        int : slot ID if allocated, None if lot is full
        """
        import time
        t0 = time.perf_counter()
        self.total_requests += 1

        available = [s for s in self.slots if not s.occupied]

        if not available:
            return None  # Forward to neighbour (Algorithm 2, line 8)

        # Compute cost for each available slot (Eq. 21)
        costs = np.array([
            self._compute_cost(slot, congestion_index)
            for slot in available
        ])

        # Select minimum-cost slot (Eq. 11)
        best_idx = int(np.argmin(costs))
        best_slot = available[best_idx]
        best_slot.reserve(vehicle_id)

        search_time_ms = (time.perf_counter() - t0) * 1000
        self.search_time_history.append(search_time_ms)
        self.successful_allocations += 1

        logger.debug(
            f"Node {self.node_id}: allocated slot {best_slot.id} "
            f"to vehicle {vehicle_id} (cost={costs[best_idx]:.3f})"
        )
        return best_slot.id

    def _compute_cost(self, slot: ParkingSlot, phi: float) -> float:
        """
        Compute allocation cost c*_{vp} (Eq. 21):
          c*_{vp} = λ1·d_{vp} + λ2·T_{vp} + λ3·θ_p + λ4·φ_t

        Parameters
        ----------
        slot : ParkingSlot
        phi  : shared congestion index φ_t

        Returns
        -------
        float : allocation cost
        """
        d_vp = slot.distance / 500.0                    # normalised distance
        T_vp = self._expected_wait_time(slot)           # expected wait
        theta_p = slot.occupancy_rate                   # local occupancy
        return (
            self.lambda1 * d_vp
            + self.lambda2 * T_vp
            + self.lambda3 * theta_p
            + self.lambda4 * phi
        )

    def _expected_wait_time(self, slot: ParkingSlot) -> float:
        """Estimate waiting time based on occupancy density (proxy)."""
        occupied_count = sum(1 for s in self.slots if s.occupied)
        occupancy_rate = occupied_count / max(self.capacity, 1)
        return occupancy_rate  # normalised [0,1]

    # ------------------------------------------------------------------
    # Release Slots (periodic)
    # ------------------------------------------------------------------

    def tick(self, time_resolution_s: int = 1):
        """
        Called each time step to release expired parking reservations.
        Parking duration follows truncated normal distribution.
        """
        steps_per_minute = 60 / time_resolution_s
        for slot in self.slots:
            if slot.occupied:
                slot.occupancy_duration += 1
                # Sample departure probability based on truncated normal duration
                mean_steps = self._dur_mean * steps_per_minute
                std_steps = self._dur_std * steps_per_minute
                # Probability of departure at this step (simplified hazard rate)
                if slot.occupancy_duration > mean_steps:
                    if np.random.random() < 0.05:  # 5% chance per step past mean
                        slot.release()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def utilisation_rate(self) -> float:
        """Fraction of slots currently occupied."""
        occupied = sum(1 for s in self.slots if s.occupied)
        return occupied / max(self.capacity, 1)

    @property
    def success_ratio(self) -> float:
        """Fraction of requests successfully allocated."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_allocations / self.total_requests

    @property
    def avg_search_time_ms(self) -> float:
        """Average parking search time in milliseconds."""
        if not self.search_time_history:
            return 0.0
        return float(np.mean(self.search_time_history))

    def reset(self):
        for slot in self.slots:
            slot.release()
        self.total_requests = 0
        self.successful_allocations = 0
        self.search_time_history.clear()
