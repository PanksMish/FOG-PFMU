"""
Traffic Network Model
=====================
Implements the directed graph G = (I, E) and queue dynamics from Section III.A.

Equations:
  - Eq. 1:  G = (I, E)  — traffic network as directed graph
  - Eq. 2:  q_{t+1,i} = q_{t,i} + a_{t,i} - d_{t,i}  — queue dynamics
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrafficNetwork:
    """
    Models the urban traffic network as a directed graph.

    Attributes
    ----------
    graph : nx.DiGraph
        Directed graph where nodes are intersections and edges are road links.
    queues : np.ndarray
        Shape (num_intersections, num_lanes). Queue lengths q_{t,i}.
    arrival_rates : np.ndarray
        Current arrival rates a_{t,i} per lane.
    departure_rates : np.ndarray
        Current departure rates d_{t,i} per lane.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rows = config["simulation"]["grid_rows"]
        self.cols = config["simulation"]["grid_cols"]
        self.num_intersections = config["simulation"]["num_intersections"]
        self.num_lanes = config["traffic"]["lanes_per_intersection"]
        self.saturation_flow = config["traffic"]["saturation_flow"]  # veh/h
        self.time_resolution = config["simulation"]["time_resolution"]

        # Build topology: I (intersections) and E (edges)
        self.graph: nx.DiGraph = self._build_grid_topology()
        self.intersection_ids: List[int] = list(self.graph.nodes)
        self.neighbours: Dict[int, List[int]] = {
            i: list(self.graph.successors(i)) + list(self.graph.predecessors(i))
            for i in self.intersection_ids
        }

        # State tensors: shape (num_intersections, num_lanes)
        self.queues = np.zeros((self.num_intersections, self.num_lanes))
        self.arrival_rates = np.zeros((self.num_intersections, self.num_lanes))
        self.departure_rates = np.zeros((self.num_intersections, self.num_lanes))

        # Demand intensity function (non-homogeneous Poisson)
        self._demand_min = config["traffic"]["demand_min"] / 3600  # veh/s
        self._demand_max = config["traffic"]["demand_max"] / 3600
        self._current_step = 0

        logger.info(
            f"TrafficNetwork initialised: {self.rows}x{self.cols} grid, "
            f"{self.num_intersections} intersections, {len(self.graph.edges)} edges"
        )

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------

    def _build_grid_topology(self) -> nx.DiGraph:
        """
        Build a 5×5 directed grid graph (bidirectional roads).
        Node IDs are row-major: node i is at (i//cols, i%cols).
        """
        G = nx.DiGraph()
        for r in range(self.rows):
            for c in range(self.cols):
                node = r * self.cols + c
                G.add_node(node, row=r, col=c)
                # Horizontal edges
                if c + 1 < self.cols:
                    right = r * self.cols + (c + 1)
                    G.add_edge(node, right)
                    G.add_edge(right, node)
                # Vertical edges
                if r + 1 < self.rows:
                    down = (r + 1) * self.cols + c
                    G.add_edge(node, down)
                    G.add_edge(down, node)
        return G

    # ------------------------------------------------------------------
    # Demand Modelling
    # ------------------------------------------------------------------

    def _poisson_intensity(self, t: int) -> float:
        """
        Non-homogeneous Poisson intensity λ(t) capturing peak-hour fluctuations.
        Peaks at morning (t=7h) and evening (t=17h) rush hours.
        """
        hour = (t * self.time_resolution) / 3600 % 24
        # Two Gaussian peaks for AM/PM rush
        am_peak = np.exp(-0.5 * ((hour - 8.0) / 1.5) ** 2)
        pm_peak = np.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2)
        base = 0.3
        scale = am_peak * 0.7 + pm_peak * 0.6 + base
        return self._demand_min + scale * (self._demand_max - self._demand_min)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, signal_phases: np.ndarray) -> Dict:
        """
        Advance the traffic network by one time step.

        Implements Eq. 2: q_{t+1,i} = q_{t,i} + a_{t,i} - d_{t,i}

        Parameters
        ----------
        signal_phases : np.ndarray
            Shape (num_intersections,). Active phase index for each intersection.

        Returns
        -------
        dict with keys: queues, arrival_rates, departure_rates, throughput
        """
        t = self._current_step
        lam = self._poisson_intensity(t)

        # Arrivals: Poisson draws per lane
        arrivals = np.random.poisson(
            lam * self.time_resolution,
            size=(self.num_intersections, self.num_lanes)
        ).astype(float)

        # Departures: determined by active phase and saturation flow
        departures = self._compute_departures(signal_phases)

        # Downstream spillover: departed vehicles from i arrive at j ∈ N(i)
        spillover = self._compute_spillover(departures)

        # Queue update (Eq. 2)
        self.queues = np.clip(
            self.queues + arrivals + spillover - departures,
            0, None
        )
        self.arrival_rates = arrivals
        self.departure_rates = departures
        self._current_step += 1

        throughput = departures.sum() * 3600 / self.time_resolution  # veh/h
        return {
            "queues": self.queues.copy(),
            "arrival_rates": arrivals,
            "departure_rates": departures,
            "throughput": throughput,
        }

    def _compute_departures(self, phases: np.ndarray) -> np.ndarray:
        """
        Green phases allow departures up to saturation flow rate.
        Each phase serves two opposing lane pairs.
        """
        sat = (self.saturation_flow / 3600) * self.time_resolution  # veh/step
        departures = np.zeros_like(self.queues)
        for i in range(self.num_intersections):
            phase = int(phases[i])
            # Two lanes get green in each phase
            green_lanes = [(phase * 2) % self.num_lanes,
                           (phase * 2 + 1) % self.num_lanes]
            for lane in green_lanes:
                departures[i, lane] = min(self.queues[i, lane], sat)
        return departures

    def _compute_spillover(self, departures: np.ndarray) -> np.ndarray:
        """
        Vehicles leaving intersection i enter queue at downstream j ∈ N(i).
        Distributes departures uniformly across successors.
        """
        spillover = np.zeros_like(departures)
        for i in self.intersection_ids:
            successors = list(self.graph.successors(i))
            if not successors:
                continue
            for lane in range(self.num_lanes):
                per_successor = departures[i, lane] / len(successors)
                for j in successors:
                    spillover[j, lane % self.num_lanes] += per_successor
        return spillover

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_queue_lengths(self) -> np.ndarray:
        """Return current queue lengths. Shape: (num_intersections, num_lanes)."""
        return self.queues.copy()

    def get_aggregate_queue(self, intersection_id: int) -> float:
        """Return total queue length at intersection i across all lanes."""
        return float(self.queues[intersection_id].sum())

    def get_neighbours(self, intersection_id: int) -> List[int]:
        """Return neighbour intersection IDs N(i)."""
        return self.neighbours[intersection_id]

    def reset(self, seed: Optional[int] = None):
        """Reset network state."""
        if seed is not None:
            np.random.seed(seed)
        self.queues[:] = 0
        self.arrival_rates[:] = 0
        self.departure_rates[:] = 0
        self._current_step = 0
        logger.debug("TrafficNetwork reset.")
