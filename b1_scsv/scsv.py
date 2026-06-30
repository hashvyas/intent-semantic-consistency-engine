"""
b1_scsv/scsv.py
===============
B1 – Sender Certificate Semantic Validation (SCSV)
Part of the ISCE STB V2X Security Pipeline.

Purpose
-------
Validate that the *kind* of message being sent is semantically consistent
with the *type of station* (vehicle, pedestrian, RSU, etc.) that claims to
be sending it.  This is a lightweight, rule-based check that can catch:

  * Impersonation: a passenger car sending RSU-exclusive messages.
  * Misconfiguration or spoofed headers: an "unknown" station type
    broadcasting anything at all.
  * Message-type mismatch: a VRU device originating infrastructure messages.

This module is intentionally dependency-free beyond the Python standard
library and PyYAML.  It performs **no text / regex scanning** of message
content; it operates purely on enumerated station_type and message_type
identifiers as defined in ETSI ITS standards.

Data interface
--------------
Messages arrive pre-decoded as nested Python dicts (typically from JSON).
Callers are expected to extract:

    station_type  ← cam.cam_parameters.basic_container.station_type
    message_type  ← header.message_id  (resolved to string, e.g. "CAM")

and call ``SCSV.check(station_type, message_type)``.

Return value
------------
``SCSV.check()`` returns a float score:

    1.0  → message passes (allow)
    0.0  → message is blocked
    intermediate values are supported for future confidence-weighted rules

References
----------
* ETSI EN 302 637-2  – Cooperative Awareness Message (CAM)
* ETSI EN 302 637-3  – Decentralised Environmental Notification (DENM)
* ETSI TS 103 900    – ITS Station Types
* ETSI TS 102 894-2  – Common Data Dictionary
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel score values – callers may compare against these constants.
# ---------------------------------------------------------------------------
SCORE_ALLOW: float = 1.0
SCORE_BLOCK: float = 0.0

# Wildcard token used in rule definitions
_WILDCARD = "*"

# Path to the shared pipeline config, relative to this file's package root.
_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "isce_config.yaml"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _policy_to_score(policy: str) -> float:
    """Convert the string policy token ('allow' / 'block') to a numeric score.

    Parameters
    ----------
    policy:
        A string, expected to be ``"allow"`` or ``"block"``.

    Returns
    -------
    float
        ``SCORE_ALLOW`` for "allow", ``SCORE_BLOCK`` for anything else.
    """
    return SCORE_ALLOW if str(policy).strip().lower() == "allow" else SCORE_BLOCK


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SCSV:
    """Sender Certificate Semantic Validator.

    Loads a YAML rule table on construction and exposes a single
    ``check(station_type, message_type)`` method that returns a score
    indicating whether the combination is acceptable.

    Parameters
    ----------
    config_path:
        Absolute or relative path to ``isce_config.yaml``.  Defaults to the
        file sitting one level above this package directory.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If the configuration file cannot be parsed.
    """

    def __init__(self, config_path: Optional[str | os.PathLike] = None) -> None:
        config_path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not config_path.exists():
            raise FileNotFoundError(
                f"SCSV: configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        b1_cfg: Dict[str, Any] = raw.get("b1_scsv", {})

        # Default policy when no rule matches
        default_raw: str = b1_cfg.get("default_policy", "allow")
        self._default_score: float = _policy_to_score(default_raw)

        # Known station types (name → integer code)
        self._station_types: Dict[str, int] = raw.get("station_types", {})
        # Known message types (name → integer code)
        self._message_types: Dict[str, int] = raw.get("message_types", {})

        # Rule list: each entry is a dict with keys
        #   station_type, message_type, action, score (optional), note (optional)
        self._rules: List[Dict[str, Any]] = b1_cfg.get("rules", [])

        logger.info(
            "SCSV loaded: %d rules, default_policy=%s",
            len(self._rules),
            "allow" if self._default_score == SCORE_ALLOW else "block",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, station_type: Any, message_type: Any) -> float:
        """Evaluate whether a (station_type, message_type) combination is valid.

        This is the primary entry point for B1.  Call it once per decoded
        message after extracting the station_type from the basic container
        and the message_type string from the header.

        Parameters
        ----------
        station_type:
            The station type string as decoded from the message, e.g.
            ``"passengerCar"``, ``"roadSideUnit"``.  May also be the raw
            integer station_type value; will be resolved to its string name
            if recognised.
        message_type:
            The message type string, e.g. ``"CAM"``, ``"DENM"``.  May also
            be the raw integer message_id; will be resolved if recognised.

        Returns
        -------
        float
            ``1.0`` (SCORE_ALLOW) if the combination is explicitly allowed or
            falls through to an "allow" default.
            ``0.0`` (SCORE_BLOCK) if the combination is explicitly blocked or
            falls through to a "block" default.
            Intermediate values are possible when a rule specifies a custom
            ``score`` field.

        Notes
        -----
        * Unknown or malformed inputs are handled gracefully: the method
          never raises; it falls back to ``default_policy``.
        * No text scanning or regex is used anywhere in this method.
        """
        # -- Normalise inputs to canonical string names --------------------
        st = self._resolve_station_type(station_type)
        mt = self._resolve_message_type(message_type)

        logger.debug("SCSV.check: station_type=%r → %r, message_type=%r → %r", station_type, st, message_type, mt)

        # -- Walk rule list top-to-bottom; first match wins ----------------
        for rule in self._rules:
            rule_st: str = str(rule.get("station_type", _WILDCARD))
            rule_mt: str = str(rule.get("message_type", _WILDCARD))

            st_match = rule_st == _WILDCARD or rule_st == st
            mt_match = rule_mt == _WILDCARD or rule_mt == mt

            if st_match and mt_match:
                # Rule matched – return the rule's score (or derive from action)
                score = self._rule_score(rule)
                logger.debug(
                    "SCSV rule matched: station_type=%r message_type=%r → score=%.2f (note: %s)",
                    rule_st,
                    rule_mt,
                    score,
                    rule.get("note", ""),
                )
                return score

        # No rule matched – apply default policy
        logger.debug(
            "SCSV no rule matched for station_type=%r message_type=%r → default score=%.2f",
            st,
            mt,
            self._default_score,
        )
        return self._default_score

    # ------------------------------------------------------------------
    # Introspection helpers (useful for dashboards / B2 hand-off)
    # ------------------------------------------------------------------

    @property
    def default_score(self) -> float:
        """The numeric score that applies when no rule matches."""
        return self._default_score

    @property
    def known_station_types(self) -> List[str]:
        """Return the list of station type names defined in the config."""
        return list(self._station_types.keys())

    @property
    def known_message_types(self) -> List[str]:
        """Return the list of message type names defined in the config."""
        return list(self._message_types.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_station_type(self, raw: Any) -> str:
        """Normalise a raw station_type value to its canonical string name.

        Resolution order
        ----------------
        1. ``int``        – reverse-lookup by integer in the station_types enum.
        2. digit string   – a string whose stripped form is all digits is
                           converted to ``int`` first, then reverse-looked-up.
                           This handles decoders that serialise enums as
                           ``"5"`` instead of ``5`` or ``"passengerCar"``.
        3. string         – returned as-is (stripped) for direct name matching.
        4. anything else  – converted via ``str()``; if the result is empty or
                           does not match any rule the default policy applies.

        Malformed / unrecognised values are returned as an empty string so
        that no rule can spuriously match them; the default policy will apply.

        Parameters
        ----------
        raw:
            Raw value from the decoded message field.

        Returns
        -------
        str
            Canonical station type name, or ``""`` if unresolvable.
        """
        if raw is None:
            return ""
        # (1) True integer → reverse-lookup
        if isinstance(raw, int):
            reverse = {v: k for k, v in self._station_types.items()}
            return reverse.get(raw, "")
        # (2) Digit-only string → convert to int and reverse-lookup
        try:
            s = str(raw).strip()
        except Exception:  # pragma: no cover – truly bizarre input
            return ""
        if s.isdigit():
            reverse = {v: k for k, v in self._station_types.items()}
            return reverse.get(int(s), "")
        # (3) Treat as a literal name string
        return s

    def _resolve_message_type(self, raw: Any) -> str:
        """Normalise a raw message_type value to its canonical string name.

        Resolution order
        ----------------
        1. ``int``        – reverse-lookup by integer in the message_types enum.
        2. digit string   – a string whose stripped form is all digits is
                           converted to ``int`` first, then reverse-looked-up.
                           This handles decoders that emit ``"1"`` instead of
                           ``1`` or ``"CAM"``.
        3. string         – returned as-is (stripped) for direct name matching.
        4. anything else  – converted via ``str()``; unresolvable values fall
                           through to the default policy.

        Parameters
        ----------
        raw:
            Raw value from the decoded header field.

        Returns
        -------
        str
            Canonical message type name, or ``""`` if unresolvable.
        """
        if raw is None:
            return ""
        # (1) True integer → reverse-lookup
        if isinstance(raw, int):
            reverse = {v: k for k, v in self._message_types.items()}
            return reverse.get(raw, "")
        # (2) Digit-only string → convert to int and reverse-lookup
        try:
            s = str(raw).strip()
        except Exception:  # pragma: no cover
            return ""
        if s.isdigit():
            reverse = {v: k for k, v in self._message_types.items()}
            return reverse.get(int(s), "")
        # (3) Treat as a literal name string
        return s

    @staticmethod
    def _rule_score(rule: Dict[str, Any]) -> float:
        """Extract the numeric score from a rule dict.

        If the rule has an explicit ``score`` key, that value is used
        (clamped to [0, 1]).  Otherwise the ``action`` key is converted
        via ``_policy_to_score``.

        Parameters
        ----------
        rule:
            A rule dict from the YAML configuration.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if "score" in rule:
            try:
                return float(max(0.0, min(1.0, rule["score"])))
            except (TypeError, ValueError):
                pass
        return _policy_to_score(rule.get("action", "block"))
