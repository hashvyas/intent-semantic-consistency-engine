"""
b2_csia/config.py
=================
Startup configuration validation for the B2 CSIA layer (V2).

Validates the ``b2_csia`` section of ``isce_config.yaml`` at
``CSIA.__init__()`` time so that misconfiguration is detected
immediately (fail-fast) rather than surfacing as a runtime error
during cluster analysis.

Usage
-----
Called internally from ``CSIA.__init__()``::

    from b2_csia.config import validate_b2_config
    validate_b2_config(raw_yaml_dict)

Raises
------
ConfigurationError
    If any required key is missing, has the wrong type, has a value
    outside the accepted range, or fusion weights do not sum to ≈ 1.0.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Custom exception (mirrors b1_scsv.config.ConfigurationError)
# ---------------------------------------------------------------------------


class ConfigurationError(ValueError):
    """Raised when the YAML configuration fails validation.

    Inherits from ``ValueError`` for broad-except compatibility.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_required(cfg: Dict[str, Any], key: str, section: str) -> Any:
    """Return ``cfg[key]``, raising ``ConfigurationError`` if absent.

    Parameters
    ----------
    cfg, key, section:
        Section-level config dict, key name, and human-readable section label.
    """
    if key not in cfg:
        raise ConfigurationError(
            f"[{section}] required key '{key}' is missing from isce_config.yaml"
        )
    return cfg[key]


def _require_numeric(
    cfg: Dict[str, Any],
    key: str,
    section: str,
    lo: float = 0.0,
    hi: float = float("inf"),
) -> float:
    """Return ``cfg[key]`` as a float, asserting it is in [*lo*, *hi*].

    Parameters
    ----------
    cfg, key, section:
        Section-level config dict, key name, and section label.
    lo, hi:
        Inclusive bounds (default: non-negative, no upper limit).
    """
    val = _get_required(cfg, key, section)
    if not isinstance(val, (int, float)):
        raise ConfigurationError(
            f"[{section}] '{key}' must be a number, got {type(val).__name__}"
        )
    f = float(val)
    if not (lo <= f <= hi):
        raise ConfigurationError(
            f"[{section}] '{key}' must be in [{lo}, {hi}], got {f}"
        )
    return f


def _optional_numeric(
    cfg: Dict[str, Any],
    key: str,
    section: str,
    lo: float = 0.0,
    hi: float = float("inf"),
) -> None:
    """If *key* is present in *cfg*, assert it is a numeric in [*lo*, *hi*]."""
    if key not in cfg:
        return
    val = cfg[key]
    if not isinstance(val, (int, float)):
        raise ConfigurationError(
            f"[{section}] '{key}' must be a number, got {type(val).__name__}"
        )
    f = float(val)
    if not (lo <= f <= hi):
        raise ConfigurationError(
            f"[{section}] '{key}' must be in [{lo}, {hi}], got {f}"
        )


def _optional_positive_int(cfg: Dict[str, Any], key: str, section: str) -> None:
    """If *key* is present, assert it is a positive integer."""
    if key not in cfg:
        return
    val = cfg[key]
    if not isinstance(val, int) or val <= 0:
        raise ConfigurationError(
            f"[{section}] '{key}' must be a positive integer, got {val!r}"
        )


# ---------------------------------------------------------------------------
# Vehicle profile sub-validator
# ---------------------------------------------------------------------------


def _validate_vehicle_profile(
    profile: Any,
    label: str,
    section: str,
) -> None:
    """Validate a single vehicle profile mapping.

    Parameters
    ----------
    profile:
        The profile sub-dict from YAML.
    label:
        The profile's key in the YAML (e.g. ``"passenger_car"``).
    section:
        Parent section label for error messages.
    """
    if not isinstance(profile, dict):
        raise ConfigurationError(
            f"[{section}] vehicle_profiles.{label} must be a mapping"
        )
    ctx = f"{section}.vehicle_profiles.{label}"

    _optional_numeric(profile, "station_type", ctx, lo=0, hi=255)
    _optional_numeric(profile, "max_acceleration", ctx, lo=0.0)
    _optional_numeric(profile, "max_deceleration", ctx, lo=0.0)
    _optional_numeric(profile, "max_yaw_rate", ctx, lo=0.0)
    _optional_numeric(profile, "expected_update_hz", ctx, lo=0.0)
    _optional_numeric(profile, "heading_tolerance", ctx, lo=0.0, hi=360.0)
    _optional_numeric(profile, "max_speed", ctx, lo=0.0)


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


def validate_b2_config(raw: Dict[str, Any]) -> None:
    """Validate the full YAML root dict for B2-relevant keys.

    Checks the ``b2_csia`` section for structural correctness, numeric
    range validity, and that fusion weights sum to approximately 1.0.

    Parameters
    ----------
    raw:
        The parsed YAML root dict (output of ``yaml.safe_load``).

    Raises
    ------
    ConfigurationError
        On any validation failure.
    """
    section = "b2_csia"

    b2 = raw.get(section)
    if not isinstance(b2, dict):
        raise ConfigurationError(
            f"'{section}' section must be a mapping in isce_config.yaml"
        )

    # ── Required clustering keys ───────────────────────────────────────────
    min_cluster = _require_numeric(b2, "min_cluster_size", section, lo=2, hi=10_000)
    if not isinstance(b2["min_cluster_size"], int):
        raise ConfigurationError(
            f"[{section}] 'min_cluster_size' must be an integer"
        )

    _require_numeric(b2, "spatial_radius_m",  section, lo=0.1)
    _require_numeric(b2, "window_size_ns",     section, lo=1.0)

    # ── Required field-path keys ───────────────────────────────────────────
    for key in ("position_lat_field", "position_lon_field", "timestamp_field"):
        val = _get_required(b2, key, section)
        if not isinstance(val, str) or not val.strip():
            raise ConfigurationError(
                f"[{section}] '{key}' must be a non-empty string"
            )

    # ── kinematic_fields ──────────────────────────────────────────────────
    kf = _get_required(b2, "kinematic_fields", section)
    if not isinstance(kf, list):
        raise ConfigurationError(
            f"[{section}] 'kinematic_fields' must be a list"
        )
    for i, f in enumerate(kf):
        if not isinstance(f, str) or not f.strip():
            raise ConfigurationError(
                f"[{section}] kinematic_fields[{i}] must be a non-empty string"
            )

    # ── semantic_fields ───────────────────────────────────────────────────
    sf = _get_required(b2, "semantic_fields", section)
    if not isinstance(sf, list):
        raise ConfigurationError(
            f"[{section}] 'semantic_fields' must be a list"
        )

    # ── Mahalanobis ───────────────────────────────────────────────────────
    _require_numeric(b2, "mahalanobis_min_samples", section, lo=2, hi=10_000)

    # ── Thresholds ────────────────────────────────────────────────────────
    _require_numeric(b2, "highway_speed_threshold",     section, lo=0.0)
    _require_numeric(b2, "highway_kinematic_threshold", section, lo=0.0)
    _require_numeric(b2, "city_kinematic_threshold",    section, lo=0.0)
    _require_numeric(b2, "kinematic_cap_multiplier",    section, lo=1.0)

    # highway threshold must be ≤ city threshold (city is wider)
    hw = float(b2["highway_kinematic_threshold"])
    cy = float(b2["city_kinematic_threshold"])
    if hw > cy:
        raise ConfigurationError(
            f"[{section}] 'highway_kinematic_threshold' ({hw}) must be ≤ "
            f"'city_kinematic_threshold' ({cy})"
        )

    # ── Entropy bins ──────────────────────────────────────────────────────
    eb = _get_required(b2, "temporal_entropy_bins", section)
    if not isinstance(eb, int) or eb < 2:
        raise ConfigurationError(
            f"[{section}] 'temporal_entropy_bins' must be an integer ≥ 2, got {eb!r}"
        )

    # ── Fusion weights ────────────────────────────────────────────────────
    w_kin = _require_numeric(b2, "weight_kinematic", section, lo=0.0, hi=1.0)
    w_sem = _require_numeric(b2, "weight_semantic",  section, lo=0.0, hi=1.0)
    w_tim = _require_numeric(b2, "weight_timing",    section, lo=0.0, hi=1.0)
    weight_sum = w_kin + w_sem + w_tim
    if abs(weight_sum - 1.0) > 0.01:
        raise ConfigurationError(
            f"[{section}] fusion weights must sum to 1.0 "
            f"(weight_kinematic={w_kin} + weight_semantic={w_sem} + "
            f"weight_timing={w_tim} = {weight_sum:.4f})"
        )

    # ── V2 optional keys ──────────────────────────────────────────────────
    _optional_numeric(b2, "trust_decay_alpha",    section, lo=0.0, hi=1.0)
    _optional_numeric(b2, "trust_recovery_beta",  section, lo=0.0, hi=1.0)
    _optional_positive_int(b2, "trust_history_window", section)

    # ── Vehicle profiles (optional) ────────────────────────────────────────
    vp = b2.get("vehicle_profiles")
    if vp is not None:
        if not isinstance(vp, dict):
            raise ConfigurationError(
                f"[{section}] 'vehicle_profiles' must be a mapping"
            )
        for label, profile in vp.items():
            _validate_vehicle_profile(profile, label, section)


__all__ = ["ConfigurationError", "validate_b2_config"]
