"""
b1_scsv/test_bad_actors.py
==========================
B1 Layer – Negative / Robustness Test (Bad Actors)

Purpose
-------
Feed four deliberately malformed or unregistered CAM packets through the
exact same SCSV pipeline used in the integration tests.  For each packet
we assert two properties:

  1. The pipeline must NOT crash Python (no unhandled KeyError / TypeError).
  2. SCSV.check() must return SCORE_BLOCK (0.0) – every bad actor must be
     blocked by the B1 firewall.

If SCSV raises an exception the test catches it, prints the traceback
summary, and marks the result as FAIL so both correctness AND robustness
of the firewall are evaluated.

Input
-----
Loads  ../b1_bad_actors.json  (relative to this file).
Each entry is a CAM message dict that also carries a ``_comment`` field
describing the type of invalid data it contains.

Output
------
A clean terminal table with columns:

    station_id | invalid data type (_comment) | exception name | B1 action | result

Running
-------
Standalone::

    python b1_scsv/test_bad_actors.py

Via pytest::

    python -m pytest b1_scsv/test_bad_actors.py -v
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap – allow direct execution from the project root or this dir
# ---------------------------------------------------------------------------
_THIS_DIR     = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b1_scsv.scsv import SCSV, SCORE_BLOCK  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_PATH     = _PROJECT_ROOT / "isce_config.yaml"
_BAD_ACTORS_PATH = _PROJECT_ROOT / "b1_bad_actors.json"

# ---------------------------------------------------------------------------
# ANSI helpers (colour only on a real TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


# ---------------------------------------------------------------------------
# Pipeline helpers – identical to the TestSCSVRealCAMSamples pattern
# ---------------------------------------------------------------------------

def _extract_fields(msg: dict) -> tuple:
    """Extract (station_type, message_type) from a decoded CAM message dict.

    Uses .get() at every level so that partial or missing keys return None
    rather than raising KeyError.  This mirrors the field-extraction pattern
    used across the SCSV integration tests and the stress-batch pipeline.
    """
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
    """Run one message through the SCSV evaluation pipeline.

    Wraps the full extraction + check() call in a try/except so that any
    exception (KeyError, TypeError, AttributeError, …) is caught and
    reported rather than crashing the test runner.

    Returns
    -------
    dict with keys:
        score         – float score from SCSV.check(), or None on crash
        action        – "BLOCKED" | "ALLOWED" | "ERROR"
        crashed       – bool
        exc_name      – exception class name, or "—" if no crash
        exc_summary   – one-line traceback summary, or "" if no crash
    """
    try:
        station_type, message_type = _extract_fields(msg)
        score = scsv.check(station_type, message_type)
        action = "BLOCKED" if score == SCORE_BLOCK else "ALLOWED"
        return {
            "score":       score,
            "action":      action,
            "crashed":     False,
            "exc_name":    "n/a",
            "exc_summary": "",
        }
    except Exception as exc:                        # noqa: BLE001
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        summary  = tb_lines[-1].strip()             # last line: "ExcType: msg"
        return {
            "score":       None,
            "action":      "ERROR",
            "crashed":     True,
            "exc_name":    type(exc).__name__,
            "exc_summary": summary,
        }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

# Column widths (content only, not counting borders/spaces)
_W_ID      = 12
_W_COMMENT = 52
_W_EXC     = 16
_W_ACTION  =  9
_W_RESULT  =  6


def _pad(text: str, width: int, ansi_extra: int = 0) -> str:
    """Left-align *text* padded to *width* visible characters.

    ``ansi_extra`` accounts for invisible ANSI escape bytes so that ljust
    still lines up the visible columns correctly.
    """
    return text.ljust(width + ansi_extra)


def _divider() -> str:
    return (
        "+"
        + "-" * (_W_ID      + 2)
        + "+"
        + "-" * (_W_COMMENT + 2)
        + "+"
        + "-" * (_W_EXC     + 2)
        + "+"
        + "-" * (_W_ACTION  + 2)
        + "+"
        + "-" * (_W_RESULT  + 2)
        + "+"
    )


def _print_table(rows: list[dict]) -> None:
    ansi_bold = len(_BOLD) + len(_RESET) if _USE_COLOR else 0
    ansi_color = len(_GREEN) + len(_RESET) if _USE_COLOR else 0  # same for all 3-char codes

    print()
    print(_c(_BOLD, "  B1 SCSV – Bad Actors Negative Test"))
    print(_c(_DIM,  f"  Input : {_BAD_ACTORS_PATH}"))
    print(_c(_DIM,  f"  Config: {_CONFIG_PATH}"))
    print()

    div = _divider()
    print(div)

    # Header row
    h_id      = _pad(_c(_BOLD, "station_id"),    _W_ID,      ansi_bold)
    h_comment = _pad(_c(_BOLD, "invalid data type (_comment)"), _W_COMMENT, ansi_bold)
    h_exc     = _pad(_c(_BOLD, "exception name"), _W_EXC,    ansi_bold)
    h_action  = _pad(_c(_BOLD, "B1 action"),      _W_ACTION, ansi_bold)
    h_result  = _pad(_c(_BOLD, "result"),         _W_RESULT, ansi_bold)
    print(f"| {h_id} | {h_comment} | {h_exc} | {h_action} | {h_result} |")
    print(div)

    for row in rows:
        sid      = _pad(str(row["station_id"]), _W_ID)
        comment  = _pad(str(row["comment"])[:_W_COMMENT], _W_COMMENT)
        exc_raw  = str(row["exc_name"])

        # Colour the exception cell
        if exc_raw == "—":
            exc = _pad(_c(_GREEN, exc_raw),  _W_EXC, ansi_color)
        else:
            exc = _pad(_c(_RED,   exc_raw),  _W_EXC, ansi_color)

        # Colour the action cell
        action_raw = str(row["action"])
        if action_raw == "BLOCKED":
            action = _pad(_c(_GREEN,  action_raw), _W_ACTION, ansi_color)
        elif action_raw == "ALLOWED":
            action = _pad(_c(_RED,    action_raw), _W_ACTION, ansi_color)
        else:
            action = _pad(_c(_YELLOW, action_raw), _W_ACTION, ansi_color)

        # Colour the result cell
        result_raw = str(row["result"])
        if result_raw == "PASS":
            result = _pad(_c(_GREEN, result_raw), _W_RESULT, ansi_color)
        else:
            result = _pad(_c(_RED,   result_raw), _W_RESULT, ansi_color)

        print(f"| {sid} | {comment} | {exc} | {action} | {result} |")

        # Extra detail line if the pipeline crashed
        if row["exc_summary"]:
            detail = f"  !! {row['exc_summary']}"
            print(f"|   {_c(_RED, detail[:_W_ID + _W_COMMENT + _W_EXC + _W_ACTION + _W_RESULT + 12])}")

    print(div)
    print()


# ---------------------------------------------------------------------------
# Core evaluation function (shared by standalone runner and unittest class)
# ---------------------------------------------------------------------------

def evaluate_bad_actors() -> list[dict]:
    """Load b1_bad_actors.json and run each packet through the SCSV pipeline.

    Returns
    -------
    list[dict]
        One result dict per bad actor with keys:
        station_id, comment, score, action, crashed, exc_name,
        exc_summary, result.
    """
    if not _BAD_ACTORS_PATH.exists():
        sys.exit(f"ERROR: bad actors file not found: {_BAD_ACTORS_PATH}")

    with _BAD_ACTORS_PATH.open("r", encoding="utf-8") as fh:
        bad_actors: list[dict] = json.load(fh)

    scsv = SCSV(config_path=_CONFIG_PATH)

    rows: list[dict] = []
    for msg in bad_actors:
        station_id = msg.get("header", {}).get("station_id", "?")
        comment    = msg.get("_comment", "(no comment)")

        pipeline   = _run_pipeline(scsv, msg)

        # PASS = no crash AND score is SCORE_BLOCK
        if pipeline["crashed"]:
            result = "FAIL"
            print(
                f"  [ROBUSTNESS FAILURE] station_id={station_id}\n"
                f"    {pipeline['exc_summary']}",
                file=sys.stderr,
            )
        elif pipeline["score"] == SCORE_BLOCK:
            result = "PASS"
        else:
            result = "FAIL"

        rows.append(
            {
                "station_id":  station_id,
                "comment":     comment,
                **pipeline,
                "result":      result,
            }
        )

    return rows


# ===========================================================================
# unittest / pytest class
# ===========================================================================

class TestBadActors(unittest.TestCase):
    """Negative test suite for the B1 SCSV firewall.

    Tests are split into three focused assertions so that pytest -v shows
    granular failure details.  All three share a single evaluation run
    performed in setUpClass.
    """

    _rows: list[dict] | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._rows = evaluate_bad_actors()
        _print_table(cls._rows)

    # ------------------------------------------------------------------

    def test_pipeline_does_not_crash(self) -> None:
        """The pipeline must not raise for any bad actor.

        A KeyError (missing station_type) or TypeError (wrong data type)
        crashing the firewall is a critical robustness failure, not just a
        correctness failure.
        """
        assert self._rows is not None
        crashes = [r for r in self._rows if r["crashed"]]
        if crashes:
            lines = "\n".join(
                f"  station_id={r['station_id']}  exc={r['exc_name']}: {r['exc_summary']}"
                for r in crashes
            )
            self.fail(
                f"SCSV pipeline crashed on {len(crashes)} bad actor(s) "
                f"(robustness failure):\n{lines}"
            )

    def test_all_bad_actors_blocked(self) -> None:
        """Every bad actor must receive score=0.0 (BLOCKED) from B1.

        An ALLOW decision for any of these malformed/unregistered packets
        means the firewall is NOT fail-closed and could be bypassed.
        """
        assert self._rows is not None
        not_blocked = [
            r for r in self._rows
            if r["score"] is not None and r["score"] != SCORE_BLOCK
        ]
        if not_blocked:
            lines = "\n".join(
                f"  station_id={r['station_id']} ({r['comment'][:50]}): "
                f"action={r['action']}, score={r['score']}"
                for r in not_blocked
            )
            self.fail(
                f"B1 failed to block {len(not_blocked)} bad actor(s):\n{lines}"
            )

    def test_all_results_pass(self) -> None:
        """Summary assertion: every row in the result table must be PASS.

        Combines the no-crash + blocked checks into one top-level assertion
        that mirrors the 'result' column in the terminal table.
        """
        assert self._rows is not None
        failures = [r for r in self._rows if r["result"] != "PASS"]
        if failures:
            lines = "\n".join(
                f"  station_id={r['station_id']} ({r['comment'][:50]}): "
                f"action={r['action']}, exception={r['exc_name']!r}"
                for r in failures
            )
            self.fail(
                f"{len(failures)} bad actor(s) did not receive a PASS result:\n{lines}"
            )


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    rows = evaluate_bad_actors()
    _print_table(rows)

    passed = sum(1 for r in rows if r["result"] == "PASS")
    failed = sum(1 for r in rows if r["result"] == "FAIL")
    total  = len(rows)

    summary_line = f"  Results: {passed}/{total} PASS  |  {failed}/{total} FAIL"
    print(summary_line)
    print()

    if failed:
        print(_c(_RED,   "  [FAIL] BAD-ACTOR TEST FAILED - see rows marked FAIL above."))
        sys.exit(1)
    else:
        print(_c(_GREEN, "  [PASS] All bad actors correctly blocked. B1 firewall is robust."))
        sys.exit(0)
