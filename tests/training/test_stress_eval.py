"""Tests for matched-baseline stress recovery measurement."""

from __future__ import annotations

from benchmarks.stress_eval import _recovery_time


def test_recovery_is_measured_against_baseline_at_the_same_time() -> None:
    baseline = [1.0] * 12
    stressed = [1.0] * 5 + [0.0, 0.0] + [1.0] * 5

    recovery = _recovery_time(
        stressed,
        baseline,
        event_step=5,
        window=2,
        tolerance=0.1,
        consecutive_windows=2,
    )

    assert recovery == 5
