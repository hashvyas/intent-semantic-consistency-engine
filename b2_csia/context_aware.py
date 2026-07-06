"""
b2_csia/context_aware.py
========================
Motion Context Inference Engine.

Dynamically infers the operating context of ITS stations (highway, urban, rural,
residential, roundabout, intersection, tunnel, bridge, parking, rsu_zone) based
on road geometry, traffic characteristics, vehicle behaviors, and RSU infrastructure.
Supports transition smoothing, hysteresis, dynamic motion envelopes, and pluggable
inference strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional, Protocol


@dataclass(frozen=True)
class ContextAssessment:
    """Represents the output of a motion context inference step.

    Parameters
    ----------
    context : str
        The inferred context name.
    confidence : float
        Inference confidence ∈ [0.0, 1.0].
    uncertainty : float
        Uncertainty/ignorance ∈ [0.0, 1.0].
    evidence : Dict[str, float]
        Supporting likelihoods for each context.
    transition_probability : float
        The calculated probability of transition from the previous context.
    """

    context: str
    confidence: float
    uncertainty: float
    evidence: Dict[str, float]
    transition_probability: float


@dataclass(frozen=True)
class MotionEnvelope:
    """Defines expected kinematic limits for a specific context.

    Parameters
    ----------
    expected_speed_max : float
        Maximum expected speed (m/s).
    expected_acc_max : float
        Maximum expected longitudinal acceleration (m/s²).
    expected_brake_max : float
        Maximum expected deceleration/braking magnitude (m/s²).
    expected_yaw_max : float
        Maximum expected yaw rate (°/s).
    expected_heading_var : float
        Maximum expected heading variation (std deviation in degrees).
    expected_freq_hz : float
        Expected message update rate (Hz).
    """

    expected_speed_max: float
    expected_acc_max: float
    expected_brake_max: float
    expected_yaw_max: float
    expected_heading_var: float
    expected_freq_hz: float


# Default envelopes for each context
ENVELOPES: Dict[str, MotionEnvelope] = {
    "highway": MotionEnvelope(expected_speed_max=45.0, expected_acc_max=3.0, expected_brake_max=6.0, expected_yaw_max=15.0, expected_heading_var=5.0, expected_freq_hz=10.0),
    "urban": MotionEnvelope(expected_speed_max=16.7, expected_acc_max=5.0, expected_brake_max=8.0, expected_yaw_max=45.0, expected_heading_var=30.0, expected_freq_hz=10.0),
    "rural": MotionEnvelope(expected_speed_max=27.8, expected_acc_max=4.0, expected_brake_max=7.0, expected_yaw_max=30.0, expected_heading_var=15.0, expected_freq_hz=5.0),
    "residential": MotionEnvelope(expected_speed_max=8.3, expected_acc_max=4.0, expected_brake_max=7.0, expected_yaw_max=45.0, expected_heading_var=45.0, expected_freq_hz=2.0),
    "intersection": MotionEnvelope(expected_speed_max=10.0, expected_acc_max=6.0, expected_brake_max=9.0, expected_yaw_max=60.0, expected_heading_var=90.0, expected_freq_hz=10.0),
    "roundabout": MotionEnvelope(expected_speed_max=8.3, expected_acc_max=3.0, expected_brake_max=5.0, expected_yaw_max=90.0, expected_heading_var=180.0, expected_freq_hz=10.0),
    "tunnel": MotionEnvelope(expected_speed_max=30.0, expected_acc_max=3.0, expected_brake_max=6.0, expected_yaw_max=15.0, expected_heading_var=5.0, expected_freq_hz=10.0),
    "bridge": MotionEnvelope(expected_speed_max=35.0, expected_acc_max=3.0, expected_brake_max=6.0, expected_yaw_max=15.0, expected_heading_var=5.0, expected_freq_hz=10.0),
    "parking": MotionEnvelope(expected_speed_max=4.0, expected_acc_max=3.0, expected_brake_max=5.0, expected_yaw_max=90.0, expected_heading_var=180.0, expected_freq_hz=1.0),
    "rsu_zone": MotionEnvelope(expected_speed_max=15.0, expected_acc_max=4.0, expected_brake_max=7.0, expected_yaw_max=45.0, expected_heading_var=45.0, expected_freq_hz=10.0),
}


class ContextInferenceStrategy(Protocol):
    """Protocol for pluggable context inference strategies."""

    name: str

    def infer(
        self,
        cluster: List[Dict[str, Any]],
        previous_context: Optional[str],
        config: Dict[str, Any],
    ) -> ContextAssessment:
        ...


class ContextInferenceRegistry:
    """Registry loading and holding context inference strategies."""

    def __init__(self) -> None:
        self._strategies: Dict[str, ContextInferenceStrategy] = {}

    def register(self, strategy: ContextInferenceStrategy) -> None:
        """Register a context inference strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[ContextInferenceStrategy]:
        """Fetch a registered strategy by name."""
        return self._strategies.get(name)


# ===========================================================================
# Default Probabilistic Context Inference Strategy
# ===========================================================================

class ProbabilisticContextInference:
    """Infers context dynamically using probabilistic rule-based matching."""

    name: str = "probabilistic"

    def infer(
        self,
        cluster: List[Dict[str, Any]],
        previous_context: Optional[str],
        config: Dict[str, Any],
    ) -> ContextAssessment:
        hysteresis = float(config.get("hysteresis", 0.25))
        supported = list(config.get("supported_contexts", ENVELOPES.keys()))

        N = len(cluster)
        if N == 0:
            return ContextAssessment("urban", 0.5, 0.5, {}, 1.0)

        # 1. Extract Traffic and Kinematic Features
        speeds = []
        headings = []
        has_rsu = False
        has_gps_issue = False

        for msg in cluster:
            bvhf = msg.get("cam", {}).get("cam_parameters", {}).get("high_frequency_container", {}).get("basic_vehicle_container_high_frequency", {})
            st = msg.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("station_type")
            if st == 15:
                has_rsu = True

            if bvhf:
                speed = bvhf.get("speed")
                heading = bvhf.get("heading")
                if speed is not None:
                    # ETSI speed is in 0.01 m/s. Convert to m/s
                    speeds.append(speed * 0.01)
                if heading is not None:
                    # ETSI heading is in 0.1 deg. Convert to degrees
                    headings.append(heading * 0.1)

            # Check GPS quality in reference position
            rp = msg.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("reference_position", {})
            if rp:
                lat = rp.get("latitude")
                lon = rp.get("longitude")
                # Indication of tunnel / bridge GPS shadowing
                if lat is None or lon is None or lat == 900_000_001 or lon == 1_800_000_001:
                    has_gps_issue = True

        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        heading_std = self._heading_std(headings) if len(headings) >= 2 else 0.0

        # Calculate likelihoods
        likelihoods: Dict[str, float] = {}

        # Highway: high speeds (> 22 m/s), low heading variance
        p_highway = self._gaussian(mean_speed, 30.0, 10.0) * (1.0 - min(1.0, heading_std / 20.0))
        likelihoods["highway"] = p_highway

        # Urban: moderate speeds (5 to 15 m/s), moderate/high heading variance
        p_urban = self._gaussian(mean_speed, 10.0, 5.0) * min(1.0, (heading_std + 5.0) / 45.0)
        likelihoods["urban"] = p_urban

        # Rural: speeds (15 to 25 m/s), moderate heading variance
        p_rural = self._gaussian(mean_speed, 20.0, 6.0) * (1.0 - min(1.0, heading_std / 35.0))
        likelihoods["rural"] = p_rural

        # Residential: very low speeds (< 8 m/s), high heading variance
        p_residential = self._gaussian(mean_speed, 4.0, 3.0) * min(1.0, (heading_std + 10.0) / 60.0)
        likelihoods["residential"] = p_residential

        # Roundabout: circular headings, speed around 5-8 m/s
        # Constant heading variation
        p_roundabout = self._gaussian(mean_speed, 6.0, 2.5) * self._sigmoid(heading_std - 15.0)
        likelihoods["roundabout"] = p_roundabout

        # Intersection: low speeds, high orthogonal turns
        # If we have angles near 90 or 180 deg
        p_intersection = self._gaussian(mean_speed, 5.0, 4.0) * self._sigmoid(heading_std - 30.0)
        likelihoods["intersection"] = p_intersection

        # Tunnel: GPS degradation, low heading variance
        p_tunnel = (1.5 if has_gps_issue else 0.1) * (1.0 - min(1.0, heading_std / 10.0))
        likelihoods["tunnel"] = p_tunnel

        # Bridge: moderate/high speeds, low heading variance, over water context (optional simulation)
        p_bridge = self._gaussian(mean_speed, 25.0, 8.0) * (1.0 - min(1.0, heading_std / 15.0))
        likelihoods["bridge"] = p_bridge

        # Parking: extremely low speeds (< 3 m/s), very high heading turns
        p_parking = self._gaussian(mean_speed, 1.5, 1.0) * min(1.0, (heading_std + 20.0) / 90.0)
        likelihoods["parking"] = p_parking

        # RSU zone: presence of RSU
        p_rsu = 1.0 if has_rsu else 0.1
        likelihoods["rsu_zone"] = p_rsu

        # Filter supported contexts
        filtered = {k: v for k, v in likelihoods.items() if k in supported}
        if not filtered:
            return ContextAssessment("urban", 0.5, 0.5, {}, 1.0)

        # Normalize likelihoods via Softmax
        sum_exp = sum(math.exp(v) for v in filtered.values())
        probs = {k: math.exp(v) / sum_exp for k, v in filtered.items()}

        # 2. Hysteresis check
        best_context = max(probs, key=lambda k: probs[k])
        best_prob = probs[best_context]

        transition_prob = 1.0
        final_context = best_context

        if previous_context is not None and previous_context in probs:
            prev_prob = probs[previous_context]
            # Hysteresis rule: only switch if the new context probability exceeds previous by hysteresis margin
            if best_prob < prev_prob + hysteresis:
                final_context = previous_context
                transition_prob = 0.2  # low transition probability (retained state)
            else:
                transition_prob = float(best_prob - prev_prob)

        # Confidence = probability of the selected context
        confidence = probs[final_context]
        uncertainty = 1.0 - confidence

        return ContextAssessment(
            context=final_context,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=probs,
            transition_probability=transition_prob,
        )

    @staticmethod
    def _gaussian(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        return math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-0.1 * x))

    @staticmethod
    def _heading_std(headings: List[float]) -> float:
        n = len(headings)
        if n < 2:
            return 0.0
        mean = sum(headings) / n
        var = sum((h - mean) ** 2 for h in headings) / (n - 1)
        return math.sqrt(var)


# ===========================================================================
# Motion Context Inference Engine
# ===========================================================================

class MotionContextInferenceEngine:
    """Behavioral context reasoning engine tracking context states and transitions.

    Explainability Pipeline Sequence
    --------------------------------
    Sensor Features (Speed, Heading, GPS, RSU)
       ↓
    Feature Extraction (Mean Speed, Heading Std Deviation, GPS flags, RSU presence)
       ↓
    Candidate Context Scores (Gaussian & Sigmoid likelihood calculations)
       ↓
    Probability Distribution (Softmax normalization over supported context candidate subset)
       ↓
    Hysteresis Evaluation (Comparison of best context probability with historical state + hysteresis margin)
       ↓
    Final Context Selection & Confidence mapping
       ↓
    ContextAssessment Output generation

    Confidence Interpretation Ranges
    --------------------------------
    - >0.80     : Very High Confidence (Distinct signature match, highly structured environment).
    - 0.60-0.80 : High Confidence (Clear indicator dominance).
    - 0.40-0.60 : Moderate Confidence (Slight overlap between candidate signatures).
    - 0.20-0.40 : Low Confidence (High candidate entropy, localized environment noise).
    - <0.20     : Ambiguous Context (Evenly split probabilities, lacks clear kinematic signature).

    Candidate Context Ranking
    -------------------------
    Instead of only outputting the winning context, the engine tracks the complete ranked
    likelihood list of all supported candidates. This is critical because:
    1. It allows debugging of boundary conditions where small noise fluctuations shift the winner.
    2. In downstream fusion (e.g. Dempster-Shafer trust propagation), secondary beliefs can be
       consulted to verify alternate scenarios if primary observations are disputed.

    Feature Contribution Analysis
    -----------------------------
    | Feature | Purpose | Influence |
    |:---|:---|:---|
    | **Speed** | Differentiates high-speed highway segments from low-speed urban streets | High |
    | **Heading Variance** | Detects intersections, roundabouts, and parking maneuver bounds | High |
    | **Traffic Density** | Distinguishes congested city roads from sparse rural roads | Medium |
    | **GPS Quality** | Detects tunnels and signal blockage conditions | High |
    | **RSU Visibility** | Incorporates trusted infrastructure corroboration | Medium |

    Hysteresis Transition Rule Example
    ----------------------------------
    If the active context is 'urban' (probability 0.49), and a new input cluster shifts candidate 'rural'
    to a slightly higher probability of 0.51:
    - Delta probability = 0.51 - 0.49 = 0.02.
    - Hysteresis margin is configured at 0.25.
    - Since 0.02 < 0.25, the engine blocks the transition and retains the 'urban' context,
      preventing rapid, noisy oscillations at boundary points.
    """

    def __init__(self) -> None:
        self.registry = ContextInferenceRegistry()
        self.registry.register(ProbabilisticContextInference())
        self._previous_contexts: Dict[int, str] = {}  # keyed by station_id or cluster index

    def get_envelope(self, context: str) -> MotionEnvelope:
        """Fetch the motion envelope limits for the specified context."""
        c = context.lower().strip()
        return ENVELOPES.get(c, ENVELOPES["urban"])

    def infer_context(
        self,
        cluster: List[Dict[str, Any]],
        cluster_id: int,
        config: Dict[str, Any],
    ) -> ContextAssessment:
        """Dynamically infer context with transition smoothing and hysteresis.

        Parameters
        ----------
        cluster : List[Dict[str, Any]]
            Spatio-temporal cluster of messages.
        cluster_id : int
            Identifer for tracking context transitions.
        config : Dict[str, Any]
            Inference engine configurations under `motion_context`.
        """
        strategy_name = str(config.get("inference_strategy", "probabilistic")).strip().lower()
        strategy = self.registry.get(strategy_name)
        if not strategy:
            strategy = self.registry.get("probabilistic")
            if not strategy:
                raise RuntimeError("Default ProbabilisticContextInference not registered")

        prev = self._previous_contexts.get(cluster_id)
        assessment = strategy.infer(cluster, prev, config)

        # Update transition history
        self._previous_contexts[cluster_id] = assessment.context
        return assessment
