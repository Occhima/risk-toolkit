from __future__ import annotations

import pytest
from schenberg.core.diagnostics import Diagnostic, DiagnosticReport


def test_empty_report_is_ok() -> None:
    report = DiagnosticReport()
    assert report.ok
    assert not report.has_errors
    assert len(report) == 0
    assert not report


def test_add_is_immutable_and_accumulates() -> None:
    base = DiagnosticReport()
    one = base.add("warning", "W1", "careful")
    two = one.add("error", "E1", "broken", location="swap_fold")

    assert len(base) == 0  # original untouched
    assert len(one) == len(("W1",))
    assert len(two) == len(("W1", "E1"))
    assert one.has_warnings and not one.has_errors
    assert two.has_errors


def test_extend_and_merge() -> None:
    report = DiagnosticReport().extend(
        Diagnostic("info", "I1", "hello"),
        Diagnostic("error", "E1", "bad"),
    )
    merged = report.merge(DiagnosticReport().add("warning", "W1", "hmm"))

    assert len(report) == len(("I1", "E1"))
    assert len(merged) == len(("I1", "E1", "W1"))
    assert merged.has_errors


def test_raise_if_errors() -> None:
    DiagnosticReport().add("warning", "W", "ok").raise_if_errors()  # no raise

    with pytest.raises(ValueError, match="diagnostic error"):
        DiagnosticReport().add("error", "E", "boom", location="here").raise_if_errors()


def test_to_frame_renders_columns() -> None:
    report = DiagnosticReport().add("error", "E1", "broken", location="loc")
    frame = report.to_frame()

    assert frame.columns == ["level", "code", "message", "location"]
    assert frame.select("code").item() == "E1"
    assert frame.select("location").item() == "loc"


def test_explain_lists_diagnostics() -> None:
    report = DiagnosticReport().add("error", "E1", "broken")
    assert "1 diagnostic" in report.explain()
    assert "E1" in report.explain()
    assert "ok" in DiagnosticReport().explain()
