"""
Grid Topology & Fog Layer Orchestrator
=======================================
Sets up the 5×5 intersection grid and orchestrates all fog nodes.
Handles neighbour synchronisation (Algorithm 1, lines 8–10).
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from src.environment.traffic_network import TrafficNetwork
from src.environment.fog_node import FogNode

logger = logging.getLogger(__name__)


class FogLayer:
    """
    Top-level orchestrator for the fog-layer deployment.

    Manages:
    - 25 fog nodes (5×5 grid)
    - 8 parking facilities
    - Inter-node congestion summary exchange
    - Unified step loop for the full simulation
    """

    def __init__(self, config: dict, method: str = "PFMU"):
        self.config = config
        self.method = method
        self.num_nodes = config["simulation"]["num_intersections"]
        self.rows = config["simulation"]["grid_rows"]
        self.cols = config["simulation"]["grid_cols"]

        # Traffic simulation layer
        self.traffic_network = TrafficNetwork(config)

        # One fog node per intersection
        self.fog_nodes: List[FogNode] = []
        for i in range(self.num_nodes):
            neighbours = self.traffic_network.get_neighbours(i)
            node = FogNode(node_id=i, config=config, neighbour_ids=neighbours)
            self.fog_nodes.append(node)

        # Global metrics
        self._step = 0
        logger.info(
            f"FogLayer initialised: method={method}, "
            f"{self.num_nodes} nodes, {self.rows}×{self.cols} grid"
        )

    # ------------------------------------------------------------------
    # Simulation Step
    # ------------------------------------------------------------------

    def step(self, parking_requests: List[Dict] = None) -> Dict:
        """
        Execute one global time step across all fog nodes.

        1. Each fog node selects an action (signal phase)
        2. Traffic network advances with those phases
        3. Each fog node updates its Q-values
        4. Neighbour summaries are exchanged
        5. Parking allocation is handled per request

        Returns
        -------
        dict: aggregate metrics for this step
        """
        parking_requests = parking_requests or []

        # --- Phase 1: Action selection (before network step) ---
        phases = np.array([
            self.fog_nodes[i].agent.select_action(
                self.fog_nodes[i].intersection.get_state()
            )
            for i in range(self.num_nodes)
        ])

        # --- Phase 2: Traffic network step ---
        net_result = self.traffic_network.step(phases)
        queues = net_result["queues"]        # (num_nodes, num_lanes)
        departures = net_result["departure_rates"]

        # --- Phase 3: Fog node updates ---
        node_results = []
        congestion_indices = []
        total_latency = 0.0

        for i, node in enumerate(self.fog_nodes):
            parking_state = {"avg_wait_time": 0.0, "occupancy": 0.0}
            result = node.step(
                queue_lengths=queues[i],
                departures=departures[i],
                parking_state=parking_state,
            )
            node_results.append(result)
            congestion_indices.append(result["congestion_index"])
            total_latency += result["latency_ms"]

        # --- Phase 4: Neighbour summary exchange (Algorithm 1, lines 8–10) ---
        self._exchange_neighbour_summaries()

        # --- Phase 5: Parking allocation ---
        parking_results = self._process_parking_requests(
            parking_requests, congestion_indices
        )

        # --- Aggregate metrics ---
        avg_latency_s = (total_latency / self.num_nodes) / 1000.0
        avg_throughput = net_result["throughput"] / self.num_nodes
        avg_queue = queues.sum(axis=1).mean()
        avg_phi = float(np.mean(congestion_indices))

        self._step += 1

        return {
            "step": self._step,
            "avg_latency_s": avg_latency_s,
            "avg_throughput_vph": avg_throughput,
            "avg_queue_length": avg_queue,
            "avg_congestion_index": avg_phi,
            "parking_results": parking_results,
            "node_results": node_results,
        }

    # ------------------------------------------------------------------
    # Neighbour Exchange
    # ------------------------------------------------------------------

    def _exchange_neighbour_summaries(self):
        """
        Distribute each node's congestion summary to its neighbours.
        Implements the periodic communication in Algorithm 1, lines 8–10.
        """
        summaries = {
            i: self.fog_nodes[i].own_congestion_summary
            for i in range(self.num_nodes)
        }
        for i, node in enumerate(self.fog_nodes):
            for j in node.neighbour_ids:
                node.receive_neighbour_summary(j, summaries[j])

    # ------------------------------------------------------------------
    # Parking
    # ------------------------------------------------------------------

    def _process_parking_requests(
        self, requests: List[Dict], congestion_indices: List[float]
    ) -> List[Dict]:
        """
        For each parking request, find the nearest fog node and invoke EASA+.
        Forwarding to neighbour nodes if local lot is full (Algorithm 2, lines 7–9).
        """
        results = []
        for req in requests:
            vehicle_id = req["vehicle_id"]
            preferred_node = req.get("nearest_node", 0) % self.num_nodes
            phi = congestion_indices[preferred_node]

            slot = self.fog_nodes[preferred_node].allocate_parking(vehicle_id, phi)
            if slot is None:
                # Forward to neighbours (Algorithm 2, line 8)
                for neighbour_id in self.fog_nodes[preferred_node].neighbour_ids:
                    phi_n = congestion_indices[neighbour_id]
                    slot = self.fog_nodes[neighbour_id].allocate_parking(vehicle_id, phi_n)
                    if slot is not None:
                        break

            results.append({
                "vehicle_id": vehicle_id,
                "slot": slot,
                "success": slot is not None,
            })
        return results

    # ------------------------------------------------------------------
    # Episode Management
    # ------------------------------------------------------------------

    def reset(self, seed: int = None):
        """Reset all nodes and traffic network for a new episode."""
        self.traffic_network.reset(seed=seed)
        for node in self.fog_nodes:
            node.reset()
        self._step = 0
        logger.debug(f"FogLayer reset (seed={seed})")

    def set_epsilon(self, epsilon: float):
        """Broadcast epsilon to all agents (for training schedule)."""
        for node in self.fog_nodes:
            node.agent.epsilon = epsilon

    def get_all_latencies(self) -> np.ndarray:
        """Return all recorded latency values across all nodes."""
        all_lat = []
        for node in self.fog_nodes:
            all_lat.extend(node.latency_history)
        return np.array(all_lat)
