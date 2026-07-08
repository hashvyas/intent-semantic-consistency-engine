"""
pipeline/leakage_validator.py
==============================
Reusable automated validation utility that verifies synthesized scene
descriptions contain no information leaked from B2 (CSIA) reasoning.

Usage
-----
::

    from pipeline.leakage_validator import SynthesisLeakageValidator

    validator = SynthesisLeakageValidator()
    result = validator.validate(synthesized_text)

    if not result.clean:
        for v in result.violations:
            print(v)

The validator is designed to be called:

* During unit test runs (``test_synthesizer_leakage.py``).
* As part of future regression test suites without modification.
* From any pipeline component that needs to assert the absence of leakage.

Design
------
Detection is performed by matching the synthesized text against a canonical
set of compiled regular-expression patterns that capture the vocabulary
associated with B2 reasoning outputs.  Each pattern is assigned a category
label and a human-readable description for diagnostic output.

The validator is intentionally conservative: it is better to flag a false
positive (text contains a suspicious word in a benign context) than to miss
genuine leakage.  Test authors may consult the ``FORBIDDEN_PATTERNS`` list
to understand what is flagged and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, NamedTuple, Pattern, Tuple


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

class _PatternEntry(NamedTuple):
    """A single forbidden-term pattern with metadata."""
    category: str
    description: str
    pattern: "re.Pattern[str]"


def _p(category: str, description: str, regex: str) -> _PatternEntry:
    """Compile a case-insensitive forbidden pattern entry.

    Parameters
    ----------
    category:
        Short category label (e.g. ``"trust"``).
    description:
        Human-readable description of what this pattern catches.
    regex:
        Regular-expression string.

    Returns
    -------
    _PatternEntry
    """
    return _PatternEntry(
        category=category,
        description=description,
        pattern=re.compile(regex, re.IGNORECASE),
    )


#: Canonical catalogue of forbidden B2 vocabulary patterns.
#: Every pattern is compiled once at import time for efficiency.
FORBIDDEN_PATTERNS: List[_PatternEntry] = [
    # --- Trust ---
    _p("trust",       "Bare trust keyword",
       r"\btrust\b"),
    _p("trust",       "TrustScore label (formatted or unformatted)",
       r"\btrust_?score\b"),
    _p("trust",       "Trust level label",
       r"\btrust_?level\b"),
    _p("trust",       "Trust metric label",
       r"\btrust_?metric\b"),
    _p("trust",       "Trust value label",
       r"\btrust_?value\b"),

    # --- Belief / Disbelief ---
    _p("belief",      "Bare belief keyword",
       r"\bbelief\b"),
    _p("disbelief",   "Bare disbelief keyword",
       r"\bdisbelief\b"),

    # --- Uncertainty ---
    _p("uncertainty", "Bare uncertainty keyword",
       r"\buncertainty\b"),

    # --- Confidence (B2-derived) ---
    # NOTE: The word 'confidence' can legitimately appear in raw peer/RSU reports
    # that quote sensor confidence fields from the V2X message payload.
    # However, to maintain a strict leakage-free guarantee, the synthesizer
    # must not emit the word 'confidence' in any form.  If a raw payload field
    # uses 'confidence', the serializer must emit the field name verbatim
    # (e.g. "sensor_confidence=HIGH") without the standalone word.
    _p("confidence",  "Standalone confidence keyword",
       r"\bconfidence\b"),

    # --- Entropy ---
    _p("entropy",     "Bare entropy keyword",
       r"\bentropy\b"),
    _p("entropy",     "TemporalEntropy label",
       r"\btemporal_?entropy\b"),

    # --- Replay ---
    _p("replay",      "Replay keyword",
       r"\breplay\b"),
    _p("replay",      "ReplayProbability label",
       r"\breplay_?prob(ability)?\b"),

    # --- Cluster score ---
    _p("cluster",     "Cluster score label",
       r"\bcluster_?score\b"),
    _p("cluster",     "Kinematic similarity label",
       r"\bkinematic_?similarity\b"),

    # --- Identity consistency ---
    _p("identity",    "Identity consistency label",
       r"\bidentity_?consistency\b"),

    # --- Suspicion / Risk ---
    _p("suspicion",   "Suspicion keyword",
       r"\bsuspicion\b"),
    _p("suspicion",   "Suspicious keyword",
       r"\bsuspicious\b"),
    _p("risk",        "Risk level label",
       r"\brisk_?level\b"),
    _p("risk",        "Risk score label",
       r"\brisk_?score\b"),

    # --- Explicit B2 field labels as they appeared in the old template ---
    _p("trust",       "Old template TrustScore= field",
       r"\bTrustScore\s*="),
    _p("belief",      "Old template Belief= field",
       r"\bBelief\s*="),
    _p("disbelief",   "Old template Disbelief= field",
       r"\bDisbelief\s*="),
    _p("uncertainty", "Old template Uncertainty= field",
       r"\bUncertainty\s*="),
    _p("entropy",     "Old template TemporalEntropy= field",
       r"\bTemporalEntropy\s*="),
    _p("replay",      "Old template ReplayProbability= field",
       r"\bReplayProbability\s*="),
    _p("identity",    "Old template IdentityConsistency= field",
       r"\bIdentityConsistency\s*="),
    _p("cluster",     "Old template KinematicSimilarity= field",
       r"\bKinematicSimilarity\s*="),

    # --- Trust Metadata section header ---
    _p("trust",       "Old template Trust Metadata section",
       r"\bTrust\s+Metadata\b"),

    # --- Behavioral Evidence section header ---
    _p("trust",       "Old template Behavioral Evidence section",
       r"\bBehavioral\s+Evidence\b"),

    # --- Obstacle Alert inferred from trust threshold ---
    _p("trust",       "Old template Obstacle Alert (trust-inferred)",
       r"\bObstacle\s+Alert\s*:"),
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeakageViolation:
    """A single detected instance of forbidden B2 vocabulary in synthesized text.

    Attributes
    ----------
    category:
        High-level violation category (e.g. ``"trust"``, ``"belief"``).
    description:
        Human-readable description of the pattern that matched.
    matched_text:
        The exact substring matched by the pattern.
    character_offset:
        Character offset of the match within the input text.
    """
    category: str
    description: str
    matched_text: str
    character_offset: int

    def __str__(self) -> str:
        return (
            f"[{self.category.upper()}] {self.description}: "
            f"matched {self.matched_text!r} at offset {self.character_offset}"
        )


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of a single ``SynthesisLeakageValidator.validate()`` call.

    Attributes
    ----------
    clean:
        ``True`` if no forbidden patterns were found; ``False`` otherwise.
    violations:
        Ordered list of all detected violations.  Empty when ``clean=True``.
    """
    clean: bool
    violations: List[LeakageViolation] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact human-readable summary of this result.

        Returns
        -------
        str
            ``"CLEAN"`` or a multi-line violation report.
        """
        if self.clean:
            return "CLEAN: no B2 leakage detected."
        lines = [f"LEAKAGE DETECTED — {len(self.violations)} violation(s):"]
        for v in self.violations:
            lines.append(f"  {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SynthesisLeakageValidator:
    """Validates that a synthesized scene description contains no B2 leakage.

    The validator checks the input text against the full ``FORBIDDEN_PATTERNS``
    catalogue and reports every match.

    This class is stateless and thread-safe after construction.  A single
    instance may be reused across many calls without side-effects.

    Parameters
    ----------
    extra_patterns:
        Optional additional ``_PatternEntry`` instances to append to the
        built-in catalogue.  Useful for project-specific extensions without
        modifying this module.

    Examples
    --------
    ::

        validator = SynthesisLeakageValidator()
        result = validator.validate("Station 1001 reports speed=1500.")
        assert result.clean

        result = validator.validate("TrustScore=0.92, Belief=0.81")
        assert not result.clean
        assert result.violations[0].category == "trust"
    """

    def __init__(
        self,
        extra_patterns: Optional[List[_PatternEntry]] = None,
    ) -> None:
        self._patterns: List[_PatternEntry] = list(FORBIDDEN_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def validate(self, text: str) -> ValidationResult:
        """Scan ``text`` for any forbidden B2 vocabulary patterns.

        Every non-overlapping match of every pattern is recorded.  The scan
        is exhaustive: all patterns are checked regardless of whether earlier
        patterns already matched.

        Parameters
        ----------
        text:
            The synthesized scene description text to validate.

        Returns
        -------
        ValidationResult
            ``ValidationResult.clean`` is ``True`` when no violations were found.
        """
        violations: List[LeakageViolation] = []
        for entry in self._patterns:
            for match in entry.pattern.finditer(text):
                violations.append(
                    LeakageViolation(
                        category=entry.category,
                        description=entry.description,
                        matched_text=match.group(0),
                        character_offset=match.start(),
                    )
                )
        # Sort by character offset for deterministic, readable output
        violations.sort(key=lambda v: v.character_offset)
        return ValidationResult(clean=len(violations) == 0, violations=violations)

    def assert_clean(self, text: str) -> None:
        """Assert that ``text`` contains no B2 leakage.

        Convenience wrapper for use in test assertions.  Raises
        ``AssertionError`` with a descriptive message if violations are found.

        Parameters
        ----------
        text:
            The synthesized scene description text to validate.

        Raises
        ------
        AssertionError
            If any forbidden pattern is detected.
        """
        result = self.validate(text)
        if not result.clean:
            raise AssertionError(result.summary())


# ---------------------------------------------------------------------------
# Optional typing import (avoid circular at module level)
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402  (placed after class definitions intentionally)
