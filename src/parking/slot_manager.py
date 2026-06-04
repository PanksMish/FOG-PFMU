"""
Slot Manager
============
Global view of all 8 parking facilities for logging and cloud-side analytics.
Synchronises slot occupancy data to the cloud layer (archival only).
"""

import numpy as np
from typing import List, Dict
from src.parking.parking_allocator import ParkingAllocator
import logging

logger = logging.getLogger(__name__)


class SlotManager:
    """
    Coordinates all parking facilities across the simulation.

    Provides:
    - Global occupancy statistics
    - Occupancy synchronisation for cloud logging
    - Aggregate parking performance metrics
    """

    def __init__(self, allocators: List[ParkingAllocator]):
        self.allocators = allocators

    @property
    def global_utilisation(self) -> float:
        """Global average slot utilisation across all facilities."""
        rates = [a.utilisation_rate for a in self.allocators]
        return float(np.mean(rates))

    @property
    def global_success_ratio(self) -> float:
        rates = [a.success_ratio for a in self.allocators]
        valid = [r for r in rates if r > 0]
        return float(np.mean(valid)) if valid else 0.0

    @property
    def global_avg_search_time_s(self) -> float:
        """Global average parking search time in seconds."""
        times = []
        for a in self.allocators:
            if a.search_time_history:
                times.extend(a.search_time_history)
        return float(np.mean(times)) / 1000.0 if times else 0.0  # ms → s

    def get_occupancy_snapshot(self) -> Dict:
        """Return occupancy data for cloud-side logging (Table II)."""
        return {
            "global_utilisation": self.global_utilisation,
            "global_success_ratio": self.global_success_ratio,
            "avg_search_time_s": self.global_avg_search_time_s,
            "per_facility": [
                {
                    "node_id": a.node_id,
                    "capacity": a.capacity,
                    "utilisation": a.utilisation_rate,
                    "success_ratio": a.success_ratio,
                }
                for a in self.allocators
            ],
        }

    def tick(self, time_resolution_s: int = 1):
        """Advance all parking allocators by one step."""
        for a in self.allocators:
            a.tick(time_resolution_s)
