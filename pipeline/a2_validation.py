"""
pipeline/a2_validation.py
==========================
A2 Validation Runner — validates that the leak-free synthesizer continues to
produce suitable input for B3 and that B3 captures are recorded correctly.

This module is a **pure evaluation layer**.  It does not implement or duplicate
any pipeline logic.  Every scenario is processed by the existing
``ISCEPipeline.run()`` orchestrator exactly as implemented.

What this validates
-------------------
1. Synthesized text contains no B2-derived information (leakage check).
2. The synthesized text passed into B3 is captured and displayed for inspection.
3. B3 output (label, confidence, availability) is recorded per scenario.
4. Predicted labels are compared against expected labels where available.

What this does NOT do
---------------------
* Modify the orchestrator, synthesizer, fusion, B1, B2, or B3.
* Duplicate pipeline logic.
* Retrain any model.
* Modify thresholds or templates.
* Alter ``manual_pipeline_test.py``.

Usage
-----
Single file::

    python pipeline/a2_validation.py scenarios/replay/msg_000.json

Folder (all .json files in directory)::

    python pipeline/a2_validation.py scenarios/replay/

With explicit context::

    python pipeline/a2_validation.py scenarios/replay/ --context highway

With a label-map file (JSON: { "msg_000.json": "MALICIOUS", ... })::

    python pipeline/a2_validation.py scenarios/replay/ --label-map labels.json

With a uniform expected label for all scenarios in a folder::

    python pipeline/a2_validation.py scenarios/replay/ --expected MALICIOUS

Combine with --json-out to write structured results::

    python pipeline/a2_validation.py scenarios/ --recursive --json-out results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of invocation location
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.orchestrator import ISCEPipeline
from pipeline.leakage_validator import SynthesisLeakageValidator

# ---------------------------------------------------------------------------
# Shared pipeline instances (stateless across scenarios)
# ---------------------------------------------------------------------------
_pipeline = ISCEPipeline()
_validator = SynthesisLeakageValidator()


# ---------------------------------------------------------------------------
# Expected-label resolution helpers
# ---------------------------------------------------------------------------

#: Keys checked in the message dict for an embedded expected label (priority order).
_LABEL_KEYS = ("expected_label", "label", "ground_truth")


def _resolve_expected_label(
    msg: Dict[str, Any],
    filename: str,
    label_map: Dict[str, str],
    uniform_label: Optional[str],
) -> Optional[str]:
    """Resolve the expected classification label for a scenario.

    Resolution priority:
    1. ``label_map`` keyed by filename stem (case-insensitive).
    2. ``label_map`` keyed by full filename (case-insensitive).
    3. Embedded label in the message dict under standard keys.
    4. ``is_attacker`` boolean field (``True`` → ``"MALICIOUS"``,
       ``False`` → ``"BENIGN"``).
    5. ``uniform_label`` supplied on the command line.
    6. ``None`` (unknown).

    Parameters
    ----------
    msg:
        The loaded message dictionary.
    filename:
        The basename of the source file (e.g. ``"msg_000.json"``).
    label_map:
        Mapping of filename (or stem) → expected label string.
    uniform_label:
        A single label applied to all scenarios when no other source is found.

    Returns
    -------
    Optional[str]
        The resolved expected label, or ``None`` when unknown.
    """
    stem = pathlib.Path(filename).stem.lower()
    name_lower = filename.lower()

    # 1 & 2. label_map lookup
    for key in (stem, name_lower, filename):
        if key in label_map:
            return label_map[key].upper()

    # 3. Embedded label fields
    for key in _LABEL_KEYS:
        val = msg.get(key)
        if val is not None:
            return str(val).upper()

    # 4. is_attacker boolean
    is_attacker = msg.get("is_attacker")
    if is_attacker is True:
        return "MALICIOUS"
    if is_attacker is False:
        return "BENIGN"

    # 5. Uniform label
    if uniform_label is not None:
        return uniform_label.upper()

    return None


# ---------------------------------------------------------------------------
# Per-scenario result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Structured result for a single scenario evaluation.

    Attributes
    ----------
    scenario_name:
        Filename or display name of the evaluated scenario.
    leakage_pass:
        ``True`` when the synthesized text contains no forbidden B2 vocabulary.
    leakage_violations:
        List of violation description strings when ``leakage_pass`` is ``False``.
    synthesized_text:
        The exact text passed into ``classify_text()`` (B3 input).
    b3_available:
        Whether B3 returned a live classification.
    b3_label:
        B3 predicted label, or ``None`` when unavailable.
    b3_confidence:
        B3 prediction confidence, or ``None`` when unavailable.
    b3_raw:
        Full raw B3 result dict.
    expected_label:
        Expected label, or ``None`` when unknown.
    prediction_result:
        ``"PASS"``, ``"FAIL"``, or ``"UNKNOWN"`` (when expected is absent or
        B3 is unavailable).
    pipeline_decision:
        Final pipeline decision string (``"ACCEPT"`` / ``"CAUTION"`` / ``"REJECT"``).
    total_ms:
        Total pipeline latency in milliseconds.
    error:
        Non-empty when an exception occurred during processing.
    """
    scenario_name:       str
    leakage_pass:        bool
    leakage_violations:  List[str]
    synthesized_text:    str
    b3_available:        bool
    b3_label:            Optional[str]
    b3_confidence:       Optional[float]
    b3_raw:              Dict[str, Any]
    expected_label:      Optional[str]
    prediction_result:   str            # "PASS" | "FAIL" | "UNKNOWN"
    pipeline_decision:   str
    total_ms:            float
    error:               str = ""


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_scenario(
    msg: Dict[str, Any],
    scenario_name: str,
    context: Optional[str],
    expected_label: Optional[str],
) -> ScenarioResult:
    """Run the full pipeline on a single message and collect validation data.

    The orchestrator is called exactly once with ``[msg]`` as the message
    window.  No pipeline logic is re-implemented here.

    Parameters
    ----------
    msg:
        Loaded V2X message dictionary.
    scenario_name:
        Display name for this scenario (typically the filename).
    context:
        Operational context string passed to the orchestrator.
    expected_label:
        Expected B3 label for comparison, or ``None``.

    Returns
    -------
    ScenarioResult
        Fully populated result record.
    """
    try:
        result = _pipeline.run([msg], context)
    except Exception as exc:
        return ScenarioResult(
            scenario_name=scenario_name,
            leakage_pass=False,
            leakage_violations=[],
            synthesized_text="",
            b3_available=False,
            b3_label=None,
            b3_confidence=None,
            b3_raw={},
            expected_label=expected_label,
            prediction_result="UNKNOWN",
            pipeline_decision="ERROR",
            total_ms=0.0,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    # --- Extract synthesized text ---
    synth = result.get("synthesized_message", {})
    synthesized_text: str = synth.get("text", "")

    # --- Leakage validation ---
    leakage_result = _validator.validate(synthesized_text)
    leakage_pass = leakage_result.clean
    leakage_violations = [str(v) for v in leakage_result.violations]

    # --- B3 output ---
    b3_raw: Dict[str, Any] = result.get("b3", {})
    b3_available: bool  = b3_raw.get("available", False)
    b3_label:  Optional[str]   = b3_raw.get("label")
    b3_conf:   Optional[float] = b3_raw.get("confidence")

    # --- Prediction comparison ---
    if expected_label is None:
        prediction_result = "UNKNOWN"
    elif not b3_available or b3_label is None:
        prediction_result = "UNKNOWN"
    elif b3_label.upper() == expected_label.upper():
        prediction_result = "PASS"
    else:
        prediction_result = "FAIL"

    return ScenarioResult(
        scenario_name=scenario_name,
        leakage_pass=leakage_pass,
        leakage_violations=leakage_violations,
        synthesized_text=synthesized_text,
        b3_available=b3_available,
        b3_label=b3_label,
        b3_confidence=b3_conf,
        b3_raw=b3_raw,
        expected_label=expected_label,
        prediction_result=prediction_result,
        pipeline_decision=result.get("decision", "UNKNOWN"),
        total_ms=result.get("latencies", {}).get("total_ms", 0.0),
        error="",
    )


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def _load_scenario(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a JSON scenario file and normalise it to a list of message dicts.

    The file may contain:
    - A single message object  → returns ``[msg]``
    - A JSON array of messages → returns the array as-is

    Parameters
    ----------
    path:
        Absolute or relative path to the JSON file.

    Returns
    -------
    List[dict]
        One or more message dicts.

    Raises
    ------
    ValueError
        If the file does not contain a JSON object or array.
    """
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError(
        f"Unexpected JSON type in {path}: expected object or array, "
        f"got {type(raw).__name__}"
    )


def _collect_json_files(
    path: pathlib.Path,
    recursive: bool,
) -> List[pathlib.Path]:
    """Collect all .json files under ``path``.

    Parameters
    ----------
    path:
        A file or directory path.
    recursive:
        If ``True``, descend into subdirectories.

    Returns
    -------
    List[pathlib.Path]
        Sorted list of .json file paths.
    """
    if path.is_file():
        return [path]
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path.glob(pattern))


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

_SEP = "-" * 60


def _print_scenario_report(sr: ScenarioResult) -> None:
    """Print a formatted per-scenario validation report to stdout.

    Parameters
    ----------
    sr:
        The scenario result to display.
    """
    print(_SEP)
    print()
    print(f"Scenario:          {sr.scenario_name}")
    print()

    if sr.error:
        print(f"ERROR:             {sr.error}")
        print()
        print(_SEP)
        return

    # --- Leakage ---
    leak_status = "PASS" if sr.leakage_pass else "FAIL"
    print(f"Leakage Check:     {leak_status}")
    if not sr.leakage_pass:
        for v in sr.leakage_violations:
            print(f"  !! {v}")
    print()

    # --- Synthesized text ---
    print("Synthesized Text (B3 Input):")
    print(f"  {sr.synthesized_text}")
    print()

    # --- Expected / predicted ---
    print(f"Expected Label:    {sr.expected_label if sr.expected_label else 'N/A'}")

    if sr.b3_available:
        print(f"Predicted Label:   {sr.b3_label}")
        conf_str = f"{sr.b3_confidence:.4f}" if sr.b3_confidence is not None else "N/A"
        print(f"Confidence:        {conf_str}")
    else:
        print(f"Predicted Label:   unavailable  ({sr.b3_raw.get('status', 'no status')})")
        print(f"Confidence:        N/A")

    print()
    print(f"Prediction Result: {sr.prediction_result}")
    print(f"Pipeline Decision: {sr.pipeline_decision}")
    print(f"Latency:           {sr.total_ms:.3f} ms")
    print()
    print(_SEP)


_SUMMARY_SEP = "=" * 60


def _print_folder_summary(results: List[ScenarioResult]) -> None:
    """Print an aggregate summary after processing a folder of scenarios.

    Parameters
    ----------
    results:
        All scenario results collected during a folder run.
    """
    total = len(results)
    if total == 0:
        print("No scenarios processed.")
        return

    errors          = sum(1 for r in results if r.error)
    leakage_passes  = sum(1 for r in results if r.leakage_pass)
    leakage_fails   = sum(1 for r in results if not r.leakage_pass)
    b3_available    = sum(1 for r in results if r.b3_available)
    b3_unavailable  = sum(1 for r in results if not r.b3_available)
    pred_pass       = sum(1 for r in results if r.prediction_result == "PASS")
    pred_fail       = sum(1 for r in results if r.prediction_result == "FAIL")
    pred_unknown    = sum(1 for r in results if r.prediction_result == "UNKNOWN")

    # Accuracy only over scenarios where both expected label and B3 were available
    comparable = [
        r for r in results
        if r.expected_label and r.b3_available and r.b3_label is not None
    ]
    accuracy_str = (
        f"{pred_pass}/{len(comparable)} "
        f"({100.0 * pred_pass / len(comparable):.1f}%)"
        if comparable else "N/A (no B3 predictions with known expected labels)"
    )

    # Average confidence over available B3 results
    confs = [r.b3_confidence for r in results if r.b3_confidence is not None]
    avg_conf_str = f"{sum(confs)/len(confs):.4f}" if confs else "N/A"

    # Average latency
    latencies = [r.total_ms for r in results if not r.error]
    avg_lat_str = f"{sum(latencies)/len(latencies):.3f} ms" if latencies else "N/A"

    # Prediction label distribution
    label_counts: Dict[str, int] = {}
    for r in results:
        if r.b3_label is not None:
            label_counts[r.b3_label] = label_counts.get(r.b3_label, 0) + 1
        elif r.b3_available:
            label_counts["(no label)"] = label_counts.get("(no label)", 0) + 1

    print()
    print(_SUMMARY_SEP)
    print("A2 VALIDATION SUMMARY")
    print(_SUMMARY_SEP)
    print(f"Total scenarios processed:   {total}")
    print(f"Processing errors:           {errors}")
    print()
    print("--- Leakage Validation ---")
    print(f"  PASS:                      {leakage_passes}")
    print(f"  FAIL:                      {leakage_fails}")
    print()
    print("--- B3 Classification ---")
    print(f"  Available:                 {b3_available}")
    print(f"  Unavailable (stub):        {b3_unavailable}")
    if label_counts:
        print("  Label distribution:")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"    {lbl:<20} {cnt}")
    print()
    print("--- Prediction Accuracy ---")
    print(f"  Accuracy:                  {accuracy_str}")
    print(f"  PASS:                      {pred_pass}")
    print(f"  FAIL:                      {pred_fail}")
    print(f"  UNKNOWN (no comparison):   {pred_unknown}")
    print()
    print("--- Performance ---")
    print(f"  Average confidence:        {avg_conf_str}")
    print(f"  Average latency:           {avg_lat_str}")
    print(_SUMMARY_SEP)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_validation(
    target: pathlib.Path,
    context: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    uniform_label: Optional[str] = None,
    recursive: bool = False,
    json_out: Optional[pathlib.Path] = None,
    verbose: bool = True,
) -> List[ScenarioResult]:
    """Run A2 validation over a single file or directory of scenario files.

    This is the primary entry point when the module is used as a library.
    Command-line invocation goes through ``main()`` which calls this function.

    Parameters
    ----------
    target:
        Path to a single .json file or a directory of .json files.
    context:
        Operational context string forwarded to the orchestrator.
    label_map:
        Optional mapping of filename (or stem) → expected label string.
    uniform_label:
        Single expected label applied to all scenarios when no other source
        resolves a label.
    recursive:
        When ``target`` is a directory, descend into subdirectories.
    json_out:
        Optional path to write structured JSON results.
    verbose:
        Print per-scenario reports to stdout.

    Returns
    -------
    List[ScenarioResult]
        One result per processed JSON file.
    """
    label_map = label_map or {}
    files = _collect_json_files(target, recursive)

    if not files:
        print(f"No .json files found under: {target}")
        return []

    is_folder = len(files) > 1 or target.is_dir()

    if verbose and is_folder:
        print(f"A2 Validation — {len(files)} scenario(s) from: {target}")
        if context:
            print(f"Context override: {context}")
        print()

    all_results: List[ScenarioResult] = []

    for json_path in files:
        try:
            messages = _load_scenario(json_path)
        except Exception as exc:
            sr = ScenarioResult(
                scenario_name=json_path.name,
                leakage_pass=False,
                leakage_violations=[],
                synthesized_text="",
                b3_available=False,
                b3_label=None,
                b3_confidence=None,
                b3_raw={},
                expected_label=None,
                prediction_result="UNKNOWN",
                pipeline_decision="ERROR",
                total_ms=0.0,
                error=f"Load error: {exc}",
            )
            all_results.append(sr)
            if verbose:
                _print_scenario_report(sr)
            continue

        # Use the last message in a multi-message file as the target for label
        # resolution (consistent with orchestrator: cluster[-1] is the target).
        target_msg = messages[-1]
        expected = _resolve_expected_label(
            target_msg, json_path.name, label_map, uniform_label
        )

        sr = _evaluate_scenario(
            msg=target_msg,
            scenario_name=json_path.name,
            context=context,
            expected_label=expected,
        )
        all_results.append(sr)

        if verbose:
            _print_scenario_report(sr)

    if verbose and is_folder:
        _print_folder_summary(all_results)

    if json_out is not None:
        _write_json_output(all_results, json_out)

    return all_results


def _write_json_output(results: List[ScenarioResult], path: pathlib.Path) -> None:
    """Serialise all results to a JSON file for downstream processing.

    Parameters
    ----------
    results:
        List of scenario results to serialise.
    path:
        Output file path.
    """
    payload = [asdict(r) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="a2_validation",
        description=(
            "A2 Validation Runner — evaluates the leak-free synthesizer and "
            "captures B3 output for every scenario without modifying the pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        metavar="PATH",
        help="Path to a single .json scenario file or a directory of .json files.",
    )
    parser.add_argument(
        "--context",
        metavar="CTX",
        default=None,
        help=(
            "Operational context label forwarded to the orchestrator "
            "(e.g. 'urban', 'highway', 'rural'). Default: auto-detected or 'unknown'."
        ),
    )
    parser.add_argument(
        "--expected",
        metavar="LABEL",
        default=None,
        dest="uniform_label",
        help=(
            "Expected label applied uniformly to all scenarios in this run "
            "(e.g. MALICIOUS, BENIGN). Overridden by embedded labels and --label-map."
        ),
    )
    parser.add_argument(
        "--label-map",
        metavar="FILE",
        default=None,
        help=(
            "Path to a JSON file mapping filename (or stem) → expected label string. "
            'Example: { "msg_000.json": "MALICIOUS", "msg_001": "BENIGN" }'
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recurse into subdirectories when PATH is a directory.",
    )
    parser.add_argument(
        "--json-out",
        metavar="FILE",
        default=None,
        help="Write structured JSON results to FILE.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-scenario reports; only print the folder summary.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for command-line invocation.

    Parameters
    ----------
    argv:
        Argument list. ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if any leakage violation was detected.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    target = pathlib.Path(args.target)
    if not target.exists():
        print(f"Error: path does not exist: {target}", file=sys.stderr)
        return 1

    # Load optional label map
    label_map: Dict[str, str] = {}
    if args.label_map:
        label_map_path = pathlib.Path(args.label_map)
        try:
            with label_map_path.open(encoding="utf-8") as fh:
                label_map = json.load(fh)
        except Exception as exc:
            print(f"Error loading --label-map {label_map_path}: {exc}", file=sys.stderr)
            return 1

    json_out = pathlib.Path(args.json_out) if args.json_out else None
    verbose  = not args.quiet

    results = run_validation(
        target=target,
        context=args.context,
        label_map=label_map,
        uniform_label=args.uniform_label,
        recursive=args.recursive,
        json_out=json_out,
        verbose=verbose,
    )

    # Print summary in quiet mode (folder summary is always useful)
    if args.quiet and len(results) > 1:
        _print_folder_summary(results)

    # Exit 1 if any leakage was detected (useful for CI)
    if any(not r.leakage_pass for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
