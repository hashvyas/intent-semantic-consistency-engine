"""
b1_scsv/test_b1_limitations.py
================================
B1 Layer – Logic Gap / Limitation Demonstration

Purpose
-------
This script is an *evaluation* tool, not a pass/fail test suite.  It
demonstrates three fundamental limitations of the B1 SCSV firewall so that
researchers and operators understand exactly what B1 can and cannot protect
against.

  Gap 1 – Physics Blindness (Spoof Test)
      B1 checks the *badge* (station_type), not the *physics* of the message.
      A packet with an impossible acceleration value (500 m/s^2) but a valid
      passengerCar badge sails straight through.

  Gap 2 – Identity Trust Without Range Validation (Over-Trust Test)
      B1 checks the *station_type code* but ignores the *station_id value*.
      An RSU (type 15) has a wildcard ALLOW rule, so a packet with an absurd
      station_id (e.g. 999999) is still ALLOWED unchallenged.

  Gap 3 – Stateless Replay (Replay Test)
      B1 holds no state between calls.  Ten identical packets (same station_id,
      timestamp, message_id) are all ALLOWED individually because the firewall
      has no memory of prior calls and cannot detect sequence duplication.

Running
-------
Standalone::

    python b1_scsv/test_b1_limitations.py

Via unittest::

    python -m unittest b1_scsv.test_b1_limitations -v
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR     = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b1_scsv.scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "isce_config.yaml"

# ---------------------------------------------------------------------------
# ANSI helpers (colour only on a real TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _ansi_len(code: str) -> int:
    """Return the invisible byte-length of one ANSI escape pair (open + reset)."""
    return (len(code) + len(_RESET)) if _USE_COLOR else 0


# ---------------------------------------------------------------------------
# Field-extraction helpers  (same pattern as the integration tests)
# ---------------------------------------------------------------------------

def _extract_fields(msg: dict) -> tuple:
    """Extract (station_type, message_type) from a decoded CAM message dict."""
    station_type = (
        msg.get("cam", {})
           .get("cam_parameters", {})
           .get("basic_container", {})
           .get("station_type")
    )
    message_type = (
        msg.get("header", {})
           .get("message_id")
    )
    return station_type, message_type


def _run_pipeline(scsv: SCSV, msg: dict) -> dict:
    """Run one message through the SCSV pipeline with full exception safety.

    Returns a dict with:
        score       – float from SCSV.check(), or None on crash
        verdict     – "ALLOWED" | "BLOCKED" | "ERROR"
        crashed     – bool
        exc_name    – exception class name, or "n/a"
        exc_summary – last traceback line, or ""
    """
    try:
        station_type, message_type = _extract_fields(msg)
        score = scsv.check(station_type, message_type)
        verdict = "ALLOWED" if score == SCORE_ALLOW else "BLOCKED"
        return {
            "score":       score,
            "verdict":     verdict,
            "crashed":     False,
            "exc_name":    "n/a",
            "exc_summary": "",
        }
    except Exception as exc:                    # noqa: BLE001
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return {
            "score":       None,
            "verdict":     "ERROR",
            "crashed":     True,
            "exc_name":    type(exc).__name__,
            "exc_summary": tb_lines[-1].strip(),
        }


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------

# Column widths (visible characters)
_W_SCENARIO = 32
_W_INPUT    = 44
_W_VERDICT  =  9


def _pad(text: str, width: int, extra_ansi: int = 0) -> str:
    return text.ljust(width + extra_ansi)


def _divider() -> str:
    return (
        "+"
        + "-" * (_W_SCENARIO + 2)
        + "+"
        + "-" * (_W_INPUT    + 2)
        + "+"
        + "-" * (_W_VERDICT  + 2)
        + "+"
    )


def _print_table_header(title: str) -> None:
    print()
    print(_c(_BOLD, f"  {title}"))
    print()
    div = _divider()
    ab  = _ansi_len(_BOLD)
    print(div)
    h_scen    = _pad(_c(_BOLD, "Scenario"),    _W_SCENARIO, ab)
    h_input   = _pad(_c(_BOLD, "Input Summary"), _W_INPUT,  ab)
    h_verdict = _pad(_c(_BOLD, "B1 Verdict"),  _W_VERDICT,  ab)
    print(f"| {h_scen} | {h_input} | {h_verdict} |")
    print(div)


def _print_table_row(scenario: str, input_summary: str, verdict: str) -> None:
    col_s = _pad(scenario[:_W_SCENARIO], _W_SCENARIO)
    col_i = _pad(input_summary[:_W_INPUT], _W_INPUT)

    if verdict == "ALLOWED":
        col_v = _pad(_c(_GREEN, verdict), _W_VERDICT, _ansi_len(_GREEN))
    elif verdict == "BLOCKED":
        col_v = _pad(_c(_RED, verdict), _W_VERDICT, _ansi_len(_RED))
    else:
        col_v = _pad(_c(_YELLOW, verdict), _W_VERDICT, _ansi_len(_YELLOW))

    print(f"| {col_s} | {col_i} | {col_v} |")


def _print_divider() -> None:
    print(_divider())


def _researcher_note(gap_num: int, note: str) -> None:
    """Print a highlighted researcher's note beneath a gap section."""
    label   = _c(_CYAN, f"  [Researcher Note - Gap {gap_num}]")
    content = _c(_DIM, f"  {note}")
    print()
    print(label)
    print(content)
    print()


# ===========================================================================
# Gap 1 – Physics Blindness (Spoof Test)
# ===========================================================================

# An acceleration of 500 m/s^2 ≈ 51g is physically impossible for any road
# vehicle.  A human body cannot survive more than ~40g for even a fraction of
# a second.  B1 never inspects this field.
_SPOOF_MSG: dict = {
    "header": {
        "station_id": 42001,
        "message_id": 1,                    # CAM
    },
    "cam": {
        "cam_parameters": {
            "basic_container": {
                "station_type": 5,          # passengerCar – a valid, recognised badge
            },
            "high_frequency_container": {
                "basic_vehicle_container_high_frequency": {
                    "speed":        13889,  # ~500 km/h in 0.01 m/s units – also absurd
                    "longitudinal_acceleration": 500.0,   # 500 m/s^2 – physically impossible
                    "heading":      900,
                    "yaw_rate":     0,
                },
            },
        },
    },
}


def run_gap1_physics(scsv: SCSV) -> dict:
    """Gap 1 – B1 cannot detect impossible physical sensor values.

    Expected B1 verdict: ALLOWED (score 1.0).
    The passengerCar + CAM rule fires regardless of the payload values.
    """
    return _run_pipeline(scsv, _SPOOF_MSG)


# ===========================================================================
# Gap 2 – Identity Trust Without Range Validation (Over-Trust Test)
# ===========================================================================

# RSU station_ids are typically allocated in a specific operator range
# (e.g., 0–999 for a given deployment).  B1 has a wildcard ALLOW rule for
# all roadSideUnit messages and never inspects the numeric station_id value.
# A station_id of 999999 is nonsensical for any real deployment but B1
# passes it without question.
_OVERTRUST_MSG: dict = {
    "header": {
        "station_id": 999999,               # Absurd station_id – far outside any real RSU range
        "message_id": 6,                    # SPATEM – infrastructure-only message type
    },
    "cam": {
        "cam_parameters": {
            "basic_container": {
                "station_type": 15,         # roadSideUnit – wildcard ALLOW rule covers all msg types
            },
        },
    },
}


def run_gap2_overtrust(scsv: SCSV) -> dict:
    """Gap 2 – B1 trusts the station_type badge; it never validates station_id ranges.

    Expected B1 verdict: ALLOWED (score 1.0).
    The RSU wildcard rule fires even though station_id=999999 is implausible.
    """
    return _run_pipeline(scsv, _OVERTRUST_MSG)


# ===========================================================================
# Gap 3 – Stateless Replay (Replay Test)
# ===========================================================================

# A real ITS anti-replay mechanism would track (station_id, timestamp, seqNum)
# tuples and reject any duplicate within a time window.  B1 holds no state
# between calls, so the same packet evaluated ten times in a row passes every
# time.
_REPLAY_STATION_ID  = 77001
_REPLAY_TIMESTAMP   = 1751280000   # fixed Unix timestamp – never changes between calls
_REPLAY_MSG_ID      = 1            # CAM
_REPLAY_STATION_TYPE = 5           # passengerCar
_REPLAY_COUNT       = 10

def _make_replay_msg() -> dict:
    """Return one copy of the fixed replay packet."""
    return {
        "header": {
            "station_id": _REPLAY_STATION_ID,
            "message_id": _REPLAY_MSG_ID,
            "generation_delta_time": _REPLAY_TIMESTAMP,  # identical every call
        },
        "cam": {
            "cam_parameters": {
                "basic_container": {
                    "station_type": _REPLAY_STATION_TYPE,
                },
            },
        },
    }


def run_gap3_replay(scsv: SCSV) -> list[dict]:
    """Gap 3 – B1 is stateless; identical packets always pass.

    Runs _REPLAY_COUNT identical messages through the pipeline.
    Expected B1 verdict: ALLOWED for every single one.
    """
    return [_run_pipeline(scsv, _make_replay_msg()) for _ in range(_REPLAY_COUNT)]


# ===========================================================================
# Main demonstration runner
# ===========================================================================

def run_all_gaps() -> dict:
    """Run all three gap demonstrations and return a summary dict.

    Returns
    -------
    dict with keys:
        gap1  – single result dict
        gap2  – single result dict
        gap3  – list of result dicts (one per replay)
    """
    scsv = SCSV(config_path=_CONFIG_PATH)
    return {
        "gap1": run_gap1_physics(scsv),
        "gap2": run_gap2_overtrust(scsv),
        "gap3": run_gap3_replay(scsv),
    }


def print_results(results: dict) -> None:
    """Pretty-print the full limitation report to stdout."""

    sep = "=" * 72

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    print()
    print(_c(_BOLD, sep))
    print(_c(_BOLD, "  ISCE B1 SCSV – Logic Gap / Limitation Demonstration"))
    print(_c(_BOLD, f"  Config: {_CONFIG_PATH}"))
    print(_c(_BOLD, sep))

    # -------------------------------------------------------------------
    # Gap 1 – Physics Blindness
    # -------------------------------------------------------------------
    r1 = results["gap1"]
    _print_table_header("Gap 1 – Physics Blindness (Spoof Test)")
    _print_table_row(
        "Impossible acceleration",
        "type=passengerCar, accel=500 m/s^2, CAM",
        r1["verdict"],
    )
    _print_divider()

    _researcher_note(
        1,
        "B1 is badge-based: it only checks (station_type, message_type).\n"
        "  It has no access to payload sensor fields such as acceleration,\n"
        "  speed, or yaw-rate. A spoofed packet with a valid badge but\n"
        "  physically impossible kinematics passes B1 without any inspection.\n"
        "  Physics-based anomaly detection requires a higher-layer validator\n"
        "  (e.g., a B2 kinematic plausibility checker) operating on the\n"
        "  decoded payload fields.",
    )

    # -------------------------------------------------------------------
    # Gap 2 – Over-Trust (no station_id range validation)
    # -------------------------------------------------------------------
    r2 = results["gap2"]
    _print_table_header("Gap 2 – Identity Over-Trust (station_id Not Validated)")
    _print_table_row(
        "Out-of-range station_id",
        "type=roadSideUnit(15), station_id=999999, SPATEM",
        r2["verdict"],
    )
    _print_divider()

    _researcher_note(
        2,
        "B1's RSU wildcard rule (station_type=roadSideUnit, message_type=*,\n"
        "  action=allow) fires on the badge value alone. It never inspects\n"
        "  the numeric station_id, which operators typically allocate in\n"
        "  a known range. A rogue device claiming the RSU badge with an\n"
        "  implausible station_id (e.g. 999999) receives an unconditional\n"
        "  ALLOW. Preventing this requires a cryptographic PKI layer that\n"
        "  ties the station_id to a verified certificate, or a range-check\n"
        "  rule implemented outside B1.",
    )

    # -------------------------------------------------------------------
    # Gap 3 – Stateless Replay
    # -------------------------------------------------------------------
    r3_list = results["gap3"]
    _print_table_header(
        f"Gap 3 – Stateless Replay ({_REPLAY_COUNT} identical packets)"
    )

    for i, r3 in enumerate(r3_list, start=1):
        _print_table_row(
            f"Replay #{i:>2} of {_REPLAY_COUNT}",
            f"station_id={_REPLAY_STATION_ID}, ts={_REPLAY_TIMESTAMP}, CAM",
            r3["verdict"],
        )

    _print_divider()

    allowed_count = sum(1 for r in r3_list if r["verdict"] == "ALLOWED")
    print()
    print(
        f"  Summary: {allowed_count}/{_REPLAY_COUNT} identical packets "
        + _c(_GREEN, "ALLOWED") + " (0 blocked)"
    )

    _researcher_note(
        3,
        "B1 is a stateless, per-packet validator. Each call to SCSV.check()\n"
        "  is independent: there is no memory of previously seen packets.\n"
        "  An attacker can capture a legitimate passengerCar + CAM packet\n"
        "  and replay it an unlimited number of times; B1 will ALLOW every\n"
        "  replay because it only evaluates the badge in the current packet.\n"
        "  Replay protection requires a stateful anti-replay window (e.g.,\n"
        "  tracking sequence numbers or timestamps per station_id), which\n"
        "  is out of scope for a stateless semantic validator like B1.",
    )

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    print(_c(_BOLD, sep))
    print(_c(_BOLD, "  Summary of B1 Logic Gaps"))
    print(_c(_BOLD, sep))
    print(
        _c(_DIM,
           "  These gaps are not bugs in B1 – they are expected constraints\n"
           "  of a lightweight, stateless, badge-based semantic boundary.\n"
           "  B1's role is to block structurally invalid or semantically\n"
           "  mismatched packets efficiently. Deeper behavioural anomalies\n"
           "  (physics, identity ranges, replay) require additional layers.\n"
        )
    )

    rows = [
        ("Gap 1", "Physics Blindness",             r1["verdict"]),
        ("Gap 2", "Station-ID Over-Trust",          r2["verdict"]),
        ("Gap 3", f"Stateless Replay (all {_REPLAY_COUNT})", f"ALLOWED x{allowed_count}"),
    ]

    ab = _ansi_len(_BOLD)
    _W_GAP   = 7
    _W_NAME  = 30
    _W_VERD  = 16
    gsep = (
        "+"
        + "-" * (_W_GAP  + 2)
        + "+"
        + "-" * (_W_NAME + 2)
        + "+"
        + "-" * (_W_VERD + 2)
        + "+"
    )
    print(gsep)
    print(
        f"| {_pad(_c(_BOLD,'Gap'), _W_GAP, ab)} "
        f"| {_pad(_c(_BOLD,'Name'), _W_NAME, ab)} "
        f"| {_pad(_c(_BOLD,'Observed Verdict'), _W_VERD, ab)} |"
    )
    print(gsep)
    for gap, name, verd in rows:
        v_color = _GREEN if "ALLOWED" in verd else _RED
        print(
            f"| {_pad(gap, _W_GAP)} "
            f"| {_pad(name, _W_NAME)} "
            f"| {_pad(_c(v_color, verd), _W_VERD, _ansi_len(v_color))} |"
        )
    print(gsep)
    print()


# ===========================================================================
# unittest / pytest class
# ===========================================================================

class TestB1Limitations(unittest.TestCase):
    """Unittest wrapper so the limitation demos can be run by pytest.

    These are *documentation* tests: a PASS means B1 behaves exactly as
    expected (i.e., it exhibits the limitation).  If B1 were ever enhanced
    to close one of these gaps, the corresponding test would fail, alerting
    the team to update the documentation.
    """

    _results: dict | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._results = run_all_gaps()
        print_results(cls._results)

    # ------------------------------------------------------------------
    # Gap 1
    # ------------------------------------------------------------------

    def test_gap1_spoof_passes_b1(self) -> None:
        """A physically impossible packet with a valid badge must be ALLOWED by B1.

        This test documents Gap 1 (physics blindness).  If B1 is ever
        extended with a kinematic plausibility check, this test will fail,
        which is the correct signal to update the architecture documentation.
        """
        assert self._results is not None
        r = self._results["gap1"]
        self.assertFalse(
            r["crashed"],
            f"Pipeline must not crash on the spoof packet; got {r['exc_name']}",
        )
        self.assertEqual(
            r["score"],
            SCORE_ALLOW,
            f"Gap 1 expects ALLOWED (1.0) for valid-badge + impossible-physics; "
            f"got score={r['score']} verdict={r['verdict']}",
        )

    # ------------------------------------------------------------------
    # Gap 2
    # ------------------------------------------------------------------

    def test_gap2_overtrust_rsu_allowed(self) -> None:
        """An RSU badge with an absurd station_id must be ALLOWED by B1.

        This test documents Gap 2 (identity over-trust).  B1's wildcard RSU
        rule fires unconditionally on the station_type field; the station_id
        value is ignored.
        """
        assert self._results is not None
        r = self._results["gap2"]
        self.assertFalse(
            r["crashed"],
            f"Pipeline must not crash on the over-trust packet; got {r['exc_name']}",
        )
        self.assertEqual(
            r["score"],
            SCORE_ALLOW,
            f"Gap 2 expects ALLOWED (1.0) for RSU badge regardless of station_id; "
            f"got score={r['score']} verdict={r['verdict']}",
        )

    # ------------------------------------------------------------------
    # Gap 3
    # ------------------------------------------------------------------

    def test_gap3_replay_all_allowed(self) -> None:
        """Every replayed packet must be ALLOWED; B1 has no replay memory.

        This test documents Gap 3 (stateless replay).  All _REPLAY_COUNT
        identical packets must receive SCORE_ALLOW because each call to
        SCSV.check() is independent with no shared state.
        """
        assert self._results is not None
        r3_list = self._results["gap3"]

        crashes = [r for r in r3_list if r["crashed"]]
        self.assertEqual(
            len(crashes), 0,
            f"Pipeline must not crash on replay packets; "
            f"{len(crashes)} crash(es): {[r['exc_name'] for r in crashes]}",
        )

        not_allowed = [r for r in r3_list if r["score"] != SCORE_ALLOW]
        self.assertEqual(
            len(not_allowed), 0,
            f"Gap 3 expects ALL {_REPLAY_COUNT} replay packets to be ALLOWED; "
            f"{len(not_allowed)} were not.",
        )


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    results = run_all_gaps()
    print_results(results)

    # Check that all gaps demonstrated their expected behaviour
    r1_ok = (not results["gap1"]["crashed"]) and (results["gap1"]["score"] == SCORE_ALLOW)
    r2_ok = (not results["gap2"]["crashed"]) and (results["gap2"]["score"] == SCORE_ALLOW)
    r3_ok = all(r["score"] == SCORE_ALLOW for r in results["gap3"])

    all_ok = r1_ok and r2_ok and r3_ok
    if all_ok:
        print(_c(_GREEN, "  [PASS] All three logic gaps demonstrated as expected."))
        sys.exit(0)
    else:
        print(_c(_RED,   "  [FAIL] One or more gaps did not behave as documented."))
        if not r1_ok:
            print(_c(_RED, f"    Gap 1: score={results['gap1']['score']} crashed={results['gap1']['crashed']}"))
        if not r2_ok:
            print(_c(_RED, f"    Gap 2: score={results['gap2']['score']} crashed={results['gap2']['crashed']}"))
        if not r3_ok:
            bad = [r for r in results["gap3"] if r["score"] != SCORE_ALLOW]
            print(_c(_RED, f"    Gap 3: {len(bad)} replay(s) not ALLOWED"))
        sys.exit(1)
