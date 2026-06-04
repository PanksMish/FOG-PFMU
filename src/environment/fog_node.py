"""
Fog Node Orchestrator
=====================
Implements the fog-layer control loop from Section IV.
Each fog node manages one intersection + its associated parking cluster.

Algorithm 1 in the paper: PFMU Traffic Control at Fog Node i
"""

import numpy as np
import time
from typing import Dict, List, Optional
import logging

from src.environment.intersection import Intersection
from src.agents.marl_agent import MARLAgent
from src.parking.parking_allocator import ParkingAllocator
from src.parking.congestion_index import CongestionIndex

logger = logging.getLogger(__name__)


class FogNode:
    """
    Fog computing node at intersection i.

    Responsibilities:
    - Local state acquisition from IoT sensors
    - Traffic signal control via MARL (Algorithm 1)
    - Parking slot allocation via EASA+ (Algorithm 2)
    - Neighbour coordination (periodic congestion summary exchange)
    - Shared congestion index computation (Eq. 22)
    """

    def __init__(
        self,
        node_id: int,
        config: dict,
        neighbour_ids: List[int],
    ):
        self.id = node_id
        self.config = config
        self.neighbour_ids = neighbour_ids

        marl_cfg = config["marl"]
        self.sync_interval = marl_cfg["neighbour_sync_interval"]  # τ

        # Sub-components
        self.intersection = Intersection(node_id, config)
        self.agent = MARLAgent(
            agent_id=node_id,
            state_dim=self.intersection.state_dim,
            action_dim=self.intersection.action_dim,
            config=config,
        )
        self.parking = ParkingAllocator(node_id, config)
        self.congestion_idx = CongestionIndex(config)

        # Runtime latency tracking
        fog_cfg = config["fog"]
        self.rtt_min = fog_cfg["rtt_min"]
        self.rtt_max = fog_cfg["rtt_max"]
        self._step_count = 0

        # Neighbour congestion summaries received from peers
        self._neighbour_summaries: Dict[int, float] = {j: 0.0 for j in neighbour_ids}

        # Metrics
        self.latency_history: List[float] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    # Main Control Loop (Algorithm 1)
    # ------------------------------------------------------------------

    def step(
        self,
        queue_lengths: np.ndarray,
        departures: np.ndarray,
        parking_state: Dict,
    ) -> Dict:
        """
        Execute one fog-node control step (Algorithm 1, lines 2–11).

        Parameters
        ----------
        queue_lengths : np.ndarray  shape (num_lanes,)
        departures    : np.ndarray  shape (num_lanes,)
        parking_state : dict  {"avg_wait_time": float, "occupancy": float, ...}

        Returns
        -------
        dict with keys: action, reward, latency_ms, congestion_index
        """
        t0 = time.perf_counter()

        # Line 3: Observe local state s_t^i  (Eq. 4)
        local_state = self._build_local_state(queue_lengths)

        # Augment with neighbour queue info (Eq. 8)
        neighbour_queues = np.array([
            self._neighbour_summaries.get(j, 0.0)
            for j in self.neighbour_ids
        ], dtype=np.float32)

        # Line 4: Select action a_t^i using ε-greedy
        action = self.agent.select_action(local_state)

        # Line 5: Apply signal phase (returned to TrafficNetwork externally)
        # Line 6: Compute reward (Eq. 19)
        reward, intersection_info = self.intersection.update(
            action=action,
            queue_lengths=queue_lengths,
            departures=departures,
        )

        # Compute shaped reward with neighbour coordination (Eq. 19 extra term)
        shaped_reward = self.agent.compute_shaped_reward(
            base_reward=reward,
            local_queues=queue_lengths,
            neighbour_queue_summaries=neighbour_queues,
        )

        # Line 7: Update Q-values (Eq. 20)
        next_state = self.intersection.get_state()
        self.agent.update(local_state, action, shaped_reward, next_state, done=False)

        # Update shared congestion index φ_t (Eq. 22)
        phi = self.congestion_idx.compute(
            avg_queue=queue_lengths.mean(),
            avg_wait_time=parking_state.get("avg_wait_time", 0.0),
        )

        # Lines 8–10: Periodic neighbour sync
        if self._step_count % self.sync_interval == 0:
            self._broadcast_congestion_summary(queue_lengths.sum())

        # Simulate fog-layer latency
        processing_ms = np.random.uniform(self.rtt_min, self.rtt_max)
        latency_ms = processing_ms + self.config["fog"].get("processing_overhead_ms", 5)

        self.latency_history.append(latency_ms / 1000.0)  # store in seconds
        self.reward_history.append(shaped_reward)
        self._step_count += 1

        return {
            "action": action,
            "reward": shaped_reward,
            "latency_ms": latency_ms,
            "congestion_index": phi,
            "intersection_info": intersection_info,
        }

    # ------------------------------------------------------------------
    # State Building (Eq. 8)
    # ------------------------------------------------------------------

    def _build_local_state(self, queue_lengths: np.ndarray) -> np.ndarray:
        """
        Augment local intersection state with neighbour queue info (Eq. 8):
          s_t^i ← s_t^i ∪ {q_{t,j} : j ∈ N(i)}
        """
        return self.intersection.get_state()

    # ------------------------------------------------------------------
    # Neighbour Coordination
    # ------------------------------------------------------------------

    def _broadcast_congestion_summary(self, queue_total: float):
        """
        Store own summary so neighbours can pull it.
        In simulation, the FogLayer orchestrator handles exchange.
        """
        self._own_summary = float(queue_total)

    def receive_neighbour_summary(self, neighbour_id: int, summary: float):
        """Called by FogLayer to deliver a peer's congestion summary."""
        self._neighbour_summaries[neighbour_id] = summary

    @property
    def own_congestion_summary(self) -> float:
        return getattr(self, "_own_summary", self.intersection.queue_lengths.sum())

    # ------------------------------------------------------------------
    # Parking Allocation (Algorithm 2 — EASA+)
    # ------------------------------------------------------------------

    def allocate_parking(
        self, vehicle_id: int, congestion_index: float
    ) -> Optional[int]:
        """
        Invoke EASA+ to allocate a parking slot (Algorithm 2).
        Returns slot ID or None if no slot available locally.
        """
        return self.parking.allocate(vehicle_id, congestion_index)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self.intersection.reset()
        self.agent.reset_episode()
        self.parking.reset()
        self.congestion_idx.reset()
        self._step_count = 0
        self._neighbour_summaries = {j: 0.0 for j in self.neighbour_ids}
        self.latency_history.clear()
        self.reward_history.clear()
        logger.debug(f"FogNode {self.id} reset.")
