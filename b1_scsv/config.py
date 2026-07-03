"""
b1_scsv/config.py
=================
Startup configuration validation for the B1 SCSV layer (V2).

Validates the ``b1_scsv`` section of ``isce_config.yaml`` at
``SCSV.__init__()`` time so that misconfiguration is detected
immediately (fail-fast) rather than surfacing as a runtime error
during message processing.

Usage
-----
The validator is intentionally **not** a public API.  It is called
internally from ``SCSV.__init__()``::

    from b1_scsv.config import validate_b1_config
    validate_b1_config(raw_yaml_dict)

Raises
------
ConfigurationError
    If any required key is missing, has the wrong type, or has a value
    outside the accepted range.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ConfigurationError(ValueError):
    """Raised when the YAML configuration fails validation.

    Inherits from ``ValueError`` so existing code that catches broad
    exceptions continues to work.  Carry a descriptive human-readable
    message that identifies the offending key and why it was rejected.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_type(cfg: Dict[str, Any], key: str, expected_type: type, section: str) -> Any:
    """Assert *key* exists in *cfg* and has type *expected_type*.

    Parameters
    ----------
    cfg:
        The config dict (section-level, not the full YAML root).
    key:
        Key to look up.
    expected_type:
        Python type the value must be an instance of.
    section:
        Human-readable section name for error messages (e.g. ``"b1_scsv"``).

    Returns
    -------
    Any
        The (validated) value at *cfg[key]*.

    Raises
    ------
    ConfigurationError
        If the key is absent or the value is of the wrong type.
    """
    if key not in cfg:
        raise ConfigurationError(
            f"[{section}] required key '{key}' is missing from isce_config.yaml"
        )
    val = cfg[key]
    if not isinstance(val, expected_type):
        raise ConfigurationError(
            f"[{section}] '{key}' must be {expected_type.__name__}, "
            f"got {type(val).__name__} ({val!r})"
        )
    return val


def _optional_positive(
    cfg: Dict[str, Any],
    key: str,
    section: str,
    allow_zero: bool = False,
) -> None:
    """If *key* is present, assert it is a positive finite number.

    Parameters
    ----------
    cfg, key, section:
        Same semantics as ``_require_type``.
    allow_zero:
        If ``True``, zero is accepted in addition to positive values.
    """
    if key not in cfg:
        return
    val = cfg[key]
    if not isinstance(val, (int, float)):
        raise ConfigurationError(
            f"[{section}] '{key}' must be a number, got {type(val).__name__}"
        )
    if allow_zero:
        if val < 0:
            raise ConfigurationError(
                f"[{section}] '{key}' must be ≥ 0, got {val}"
            )
    else:
        if val <= 0:
            raise ConfigurationError(
                f"[{section}] '{key}' must be > 0, got {val}"
            )


def _optional_range(
    cfg: Dict[str, Any],
    key: str,
    lo: float,
    hi: float,
    section: str,
) -> None:
    """If *key* is present, assert it falls within [*lo*, *hi*].

    Parameters
    ----------
    cfg, key, section:
        Same semantics as ``_require_type``.
    lo, hi:
        Inclusive bounds for the value.
    """
    if key not in cfg:
        return
    val = cfg[key]
    if not isinstance(val, (int, float)):
        raise ConfigurationError(
            f"[{section}] '{key}' must be a number, got {type(val).__name__}"
        )
    if not (lo <= val <= hi):
        raise ConfigurationError(
            f"[{section}] '{key}' must be in [{lo}, {hi}], got {val}"
        )


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


def validate_b1_config(raw: Dict[str, Any]) -> None:
    """Validate the full YAML root dict for B1-relevant keys.

    Checks the ``b1_scsv`` section for structural correctness and value
    ranges, and also validates the shared ``station_types`` and
    ``message_types`` enumerations that SCSV depends on.

    Parameters
    ----------
    raw:
        The parsed YAML root dict (output of ``yaml.safe_load``).

    Raises
    ------
    ConfigurationError
        On any validation failure.
    """
    section = "b1_scsv"

    # ── Shared enumerations ────────────────────────────────────────────────
    st = raw.get("station_types")
    if not isinstance(st, dict) or not st:
        raise ConfigurationError(
            "'station_types' must be a non-empty mapping in isce_config.yaml"
        )
    for name, code in st.items():
        if not isinstance(code, int):
            raise ConfigurationError(
                f"station_types.{name}: code must be an integer, got {type(code).__name__}"
            )

    mt = raw.get("message_types")
    if not isinstance(mt, dict) or not mt:
        raise ConfigurationError(
            "'message_types' must be a non-empty mapping in isce_config.yaml"
        )
    for name, code in mt.items():
        if not isinstance(code, int):
            raise ConfigurationError(
                f"message_types.{name}: code must be an integer, got {type(code).__name__}"
            )

    # ── b1_scsv section ────────────────────────────────────────────────────
    b1 = raw.get(section)
    if not isinstance(b1, dict):
        raise ConfigurationError(
            f"'{section}' section must be a mapping in isce_config.yaml"
        )

    # default_policy
    dp = b1.get("default_policy", "allow")
    if dp not in ("allow", "block"):
        raise ConfigurationError(
            f"[{section}] 'default_policy' must be 'allow' or 'block', got {dp!r}"
        )

    # rules (optional – no rules means default policy always applies)
    rules = b1.get("rules", [])
    if not isinstance(rules, list):
        raise ConfigurationError(
            f"[{section}] 'rules' must be a list, got {type(rules).__name__}"
        )
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ConfigurationError(
                f"[{section}] rules[{idx}] must be a dict, got {type(rule).__name__}"
            )
        action = rule.get("action")
        if action is not None and action not in ("allow", "block"):
            raise ConfigurationError(
                f"[{section}] rules[{idx}].action must be 'allow' or 'block', got {action!r}"
            )
        score = rule.get("score")
        if score is not None:
            if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
                raise ConfigurationError(
                    f"[{section}] rules[{idx}].score must be in [0.0, 1.0], got {score!r}"
                )

    # ── V2 – optional new keys ─────────────────────────────────────────────
    _optional_positive(b1, "replay_cache_ttl_s", section)
    _optional_positive(b1, "timestamp_freshness_ms", section)
    _optional_positive(b1, "cert_rotation_window_s", section)
    _optional_positive(b1, "cert_max_rotations", section)

    plaus = b1.get("plausibility")
    if plaus is not None:
        if not isinstance(plaus, dict):
            raise ConfigurationError(
                f"[{section}] 'plausibility' must be a mapping"
            )
        _optional_positive(plaus, "max_speed_ms", section, allow_zero=True)
        _optional_positive(plaus, "max_acceleration_ms2", section, allow_zero=True)
        _optional_positive(plaus, "max_jerk_ms3", section, allow_zero=True)
        _optional_positive(plaus, "max_heading_change_deg_s", section, allow_zero=True)
        _optional_positive(plaus, "max_yaw_rate_deg_s", section, allow_zero=True)
        _optional_range(plaus, "lat_min", -900_000_000, 0, section)
        _optional_range(plaus, "lat_max", 0, 900_000_000, section)
        _optional_range(plaus, "lon_min", -1_800_000_000, 0, section)
        _optional_range(plaus, "lon_max", 0, 1_800_000_000, section)


__all__ = ["ConfigurationError", "validate_b1_config"]
