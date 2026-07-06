"""
b2_csia/trust_propagation.py
============================
Trust Propagation Engine.

Implements an Evidence-Constrained Trust Propagation Engine. Performs uncertainty-aware
belief diffusion across the Observability Graph, utilizing Dempster-Shafer 
mass function representations, reliability discounting, Sybil penalties, 
and supports pluggable propagation strategies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple, Optional, Protocol
from b2_csia.uncertainty import MassFunction, Provenance
from b2_csia.observability_graph import ObservabilityGraph


class PropagationStrategy(Protocol):
    """Protocol defining the interface for pluggable propagation strategies."""

    name: str

    def propagate(
        self,
        graph: ObservabilityGraph,
        initial_beliefs: Dict[int, MassFunction],
        config: Dict[str, Any],
    ) -> Tuple[Dict[int, MassFunction], Dict[str, Any]]:
        """Propagate beliefs across the observability graph.

        Parameters
        ----------
        graph : ObservabilityGraph
            The active observability graph representing V2X node relationships.
        initial_beliefs : Dict[int, MassFunction]
            The baseline local trust beliefs keyed by station_id.
        config : Dict[str, Any]
            The configurations under `trust_propagation` section.

        Returns
        -------
        propagated_beliefs : Dict[int, MassFunction]
            The final propagated belief states.
        metadata : Dict[str, Any]
            Explainability attributes and convergence details.
        """
        ...


class PropagationRegistry:
    """Registry managing pluggable trust propagation strategies."""

    def __init__(self) -> None:
        self._strategies: Dict[str, PropagationStrategy] = {}

    def register(self, strategy: PropagationStrategy) -> None:
        """Register a propagation strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[PropagationStrategy]:
        """Fetch a registered strategy by name."""
        return self._strategies.get(name)


# ===========================================================================
# Default Belief Diffusion Strategy
# ===========================================================================

class BeliefDiffusionStrategy:
    """Propagates Dempster-Shafer mass functions via linear belief diffusion."""

    name: str = "belief_diffusion"

    def propagate(
        self,
        graph: ObservabilityGraph,
        initial_beliefs: Dict[int, MassFunction],
        config: Dict[str, Any],
    ) -> Tuple[Dict[int, MassFunction], Dict[str, Any]]:
        # Read parameters from config
        damping = float(config.get("damping_factor", 0.30))
        max_iterations = int(config.get("max_iterations", 20))
        tolerance = float(config.get("convergence_tolerance", 0.0001))
        sybil_penalty = float(config.get("sybil_penalty", 0.40))
        min_edge_conf = float(config.get("minimum_edge_confidence", 0.50))
        min_ev_quality = float(config.get("minimum_evidence_quality", 0.50))

        # Check nodes
        nodes = list(graph.nodes)
        if not nodes:
            return {}, {"iterations": 0, "status": "empty_graph"}

        # Ensure all graph nodes have an entry in initial beliefs
        current_beliefs = {}
        for nid in nodes:
            if nid in initial_beliefs:
                current_beliefs[nid] = initial_beliefs[nid]
            else:
                # Default benign baseline if missing
                current_beliefs[nid] = MassFunction(1.0, 0.0, 0.0)

        initial_states = {nid: current_beliefs[nid] for nid in nodes}

        # Iteration variables
        iterations = 0
        converged = False

        while iterations < max_iterations and not converged:
            next_beliefs = {}
            max_change = 0.0

            for i in nodes:
                # Compute weighted neighboring contributions
                neighbors = []
                total_alpha = 0.0
                neighbor_weighted_masses = []

                for j in nodes:
                    if i == j:
                        continue
                    key = (i, j) if i < j else (j, i)
                    edge = graph.edges.get(key)
                    if not edge:
                        continue

                    # Apply filter constraints
                    if edge.confidence < min_edge_conf:
                        continue
                    q_j = graph.node_qualities.get(j)
                    q_j_score = q_j.score if q_j else 1.0
                    if q_j_score < min_ev_quality:
                        continue

                    # 1. Base edge weight
                    w_ji = edge.weight

                    # 2. Sybil penalty: if similarity is extremely high, penalize propagation
                    # to prevent malicious nodes from mutually reinforcing.
                    is_clone = edge.contributing_factors.get("kinematic", 1.0) > 0.8 and \
                               edge.contributing_factors.get("lane", 1.0) > 0.8
                    if is_clone:
                        w_ji *= (1.0 - sybil_penalty)

                    # 3. Evidence constraint discount factor
                    # alpha = observability_weight * edge_confidence * evidence_quality * neighbor_confidence
                    m_j = current_beliefs[j]
                    alpha_ji = w_ji * edge.confidence * q_j_score * (1.0 - m_j.uncertainty)

                    if alpha_ji > 0.001:
                        # Discount neighbor's mass function before mixing
                        discounted_mj = m_j.discount(alpha_ji)
                        neighbor_weighted_masses.append((alpha_ji, discounted_mj))
                        neighbors.append(j)
                        total_alpha += alpha_ji

                # Fuse neighborhood mass functions
                if total_alpha > 0.0:
                    # Linear mixture of neighbor mass functions
                    mix_A = sum(alpha * m.m_A for alpha, m in neighbor_weighted_masses) / total_alpha
                    mix_not_A = sum(alpha * m.m_not_A for alpha, m in neighbor_weighted_masses) / total_alpha
                    mix_Theta = sum(alpha * m.m_Theta for alpha, m in neighbor_weighted_masses) / total_alpha

                    # Merge provenances
                    merged_prov = initial_states[i].provenance
                    for _, m in neighbor_weighted_masses:
                        merged_prov = merged_prov.merge(m.provenance)

                    m_neighbor_blend = MassFunction(mix_A, mix_not_A, mix_Theta, merged_prov)
                else:
                    # No neighbors in range -> no change
                    m_neighbor_blend = initial_states[i]

                # Update node state using convex blend with damping factor
                # m_i(t+1) = (1 - gamma) * m_i(0) + gamma * m_neighbor_blend
                m_init = initial_states[i]
                new_A = (1.0 - damping) * m_init.m_A + damping * m_neighbor_blend.m_A
                new_not_A = (1.0 - damping) * m_init.m_not_A + damping * m_neighbor_blend.m_not_A
                new_Theta = (1.0 - damping) * m_init.m_Theta + damping * m_neighbor_blend.m_Theta

                next_m = MassFunction(new_A, new_not_A, new_Theta, m_init.provenance.merge(m_neighbor_blend.provenance))
                next_beliefs[i] = next_m

                # Calculate L1-norm change for convergence check
                m_curr = current_beliefs[i]
                diff = abs(next_m.m_A - m_curr.m_A) + abs(next_m.m_not_A - m_curr.m_not_A)
                max_change = max(max_change, diff)

            current_beliefs = next_beliefs
            iterations += 1
            if max_change < tolerance:
                converged = True

        explanation_metadata = {
            "iterations_to_converge": iterations,
            "converged": converged,
            "damping_factor": damping,
            "strategy": self.name,
        }

        return current_beliefs, explanation_metadata


# ===========================================================================
# Personalized PageRank Strategy
# ===========================================================================

class PersonalizedPageRankStrategy:
    """Propagates trust using Personalized PageRank over the constraint matrix."""

    name: str = "personalized_pagerank"

    def propagate(
        self,
        graph: ObservabilityGraph,
        initial_beliefs: Dict[int, MassFunction],
        config: Dict[str, Any],
    ) -> Tuple[Dict[int, MassFunction], Dict[str, Any]]:
        # Uses power iteration method mapped onto DS MassFunctions
        damping = float(config.get("damping_factor", 0.30))
        max_iterations = int(config.get("max_iterations", 20))
        tolerance = float(config.get("convergence_tolerance", 0.0001))

        nodes = list(graph.nodes)
        if not nodes:
            return {}, {"iterations": 0, "status": "empty"}

        # PageRank transition matrix construction
        transition_matrix: Dict[int, Dict[int, float]] = {u: {} for u in nodes}
        for u in nodes:
            total_w = 0.0
            temp_weights = {}
            for v in nodes:
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                edge = graph.edges.get(key)
                if edge:
                    # Base weight constrained by confidence
                    w = edge.weight * edge.confidence
                    temp_weights[v] = w
                    total_w += w

            # Normalize transitions
            if total_w > 0.0:
                for v, w in temp_weights.items():
                    transition_matrix[u][v] = w / total_w

        # Power iterations
        current_beliefs = {nid: initial_beliefs.get(nid, MassFunction(1.0, 0.0, 0.0)) for nid in nodes}
        iterations = 0
        converged = False

        while iterations < max_iterations and not converged:
            next_beliefs = {}
            max_change = 0.0

            for i in nodes:
                # PageRank equation: m_i = (1 - d) * m_init + d * sum( transition_ji * m_j )
                m_init = initial_beliefs.get(i, MassFunction(1.0, 0.0, 0.0))

                neighbor_sum_A = 0.0
                neighbor_sum_not_A = 0.0
                neighbor_sum_Theta = 0.0
                prov = m_init.provenance

                for j in nodes:
                    if j in transition_matrix and i in transition_matrix[j]:
                        transition_prob = transition_matrix[j][i]
                        m_j = current_beliefs[j]
                        neighbor_sum_A += transition_prob * m_j.m_A
                        neighbor_sum_not_A += transition_prob * m_j.m_not_A
                        neighbor_sum_Theta += transition_prob * m_j.m_Theta
                        prov = prov.merge(m_j.provenance)

                new_A = (1.0 - damping) * m_init.m_A + damping * neighbor_sum_A
                new_not_A = (1.0 - damping) * m_init.m_not_A + damping * neighbor_sum_not_A
                new_Theta = (1.0 - damping) * m_init.m_Theta + damping * neighbor_sum_Theta

                next_m = MassFunction(new_A, new_not_A, new_Theta, prov)
                next_beliefs[i] = next_m

                m_curr = current_beliefs[i]
                diff = abs(next_m.m_A - m_curr.m_A) + abs(next_m.m_not_A - m_curr.m_not_A)
                max_change = max(max_change, diff)

            current_beliefs = next_beliefs
            iterations += 1
            if max_change < tolerance:
                converged = True

        return current_beliefs, {
            "iterations_to_converge": iterations,
            "converged": converged,
            "strategy": self.name,
        }


# ===========================================================================
# Propagation Engine
# ===========================================================================

class TrustPropagationEngine:
    """Evidence-Constrained Trust Propagation Engine.

    Aggregates propagation strategies and executes the selected algorithm.
    """

    def __init__(self) -> None:
        self.registry = PropagationRegistry()
        self.registry.register(BeliefDiffusionStrategy())
        self.registry.register(PersonalizedPageRankStrategy())

    def propagate(
        self,
        graph: ObservabilityGraph,
        initial_beliefs: Dict[int, MassFunction],
        config: Dict[str, Any],
    ) -> Tuple[Dict[int, MassFunction], Dict[str, Any]]:
        """Executes trust propagation over the graph.

        Parameters
        ----------
        graph : ObservabilityGraph
            The observability graph.
        initial_beliefs : Dict[int, MassFunction]
            Baseline beliefs.
        config : Dict[str, Any]
            Trust propagation configurations.
        """
        # Read strategy from config (default to belief_diffusion)
        strategy_name = str(config.get("strategy", "belief_diffusion")).strip().lower()
        strategy = self.registry.get(strategy_name)
        if not strategy:
            strategy = self.registry.get("belief_diffusion")
            if not strategy:
                raise RuntimeError("Default BeliefDiffusionStrategy not found in registry")

        return strategy.propagate(graph, initial_beliefs, config)
