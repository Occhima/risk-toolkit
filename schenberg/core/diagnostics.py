"""DiagnosticReport: an accumulated, inspectable bundle of validation issues.

Most of the engine validates eagerly and raises on the first problem — fine for a
single hard contract. Some checks want to *accumulate*: a structure's fold against
its component schema, a router's branch coverage, a shock's target sources. A
:class:`DiagnosticReport` is an immutable list of :class:`Diagnostic` records that
grows by :meth:`add` / :meth:`extend`, answers :attr:`has_errors`, can
:meth:`raise_if_errors`, and renders to a :class:`polars.DataFrame` for display.

It is deliberately small: a shared vocabulary for "here is everything that's
wrong", not a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

Level = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One issue found by an inspection: a level, a stable code, a message, and an
    optional location (a term name, branch label, source name, ...)."""

    level: Level
    code: str
    message: str
    location: str | None = None

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        return f"{self.level.upper()} {self.code}{where}: {self.message}"


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """An immutable accumulation of :class:`Diagnostic`\\ s."""

    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(d.level == "error" for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        return any(d.level == "warning" for d in self.diagnostics)

    @property
    def ok(self) -> bool:
        return not self.has_errors

    def add(
        self, level: Level, code: str, message: str, location: str | None = None
    ) -> DiagnosticReport:
        """Return a new report with one more diagnostic appended."""
        return DiagnosticReport(self.diagnostics + (Diagnostic(level, code, message, location),))

    def extend(self, *diagnostics: Diagnostic) -> DiagnosticReport:
        """Return a new report with several diagnostics appended."""
        return DiagnosticReport(self.diagnostics + tuple(diagnostics))

    def merge(self, other: DiagnosticReport) -> DiagnosticReport:
        """Concatenate two reports."""
        return DiagnosticReport(self.diagnostics + other.diagnostics)

    def raise_if_errors(self) -> None:
        """Raise a single ``ValueError`` summarizing all error-level diagnostics."""
        errors = [d for d in self.diagnostics if d.level == "error"]
        if errors:
            joined = "\n".join(f"  - {d}" for d in errors)
            raise ValueError(f"{len(errors)} diagnostic error(s):\n{joined}")

    def to_frame(self) -> pl.DataFrame:
        """Render the diagnostics as a Polars frame (a *diagnostic* — eager)."""
        return pl.DataFrame(
            {
                "level": [d.level for d in self.diagnostics],
                "code": [d.code for d in self.diagnostics],
                "message": [d.message for d in self.diagnostics],
                "location": [d.location for d in self.diagnostics],
            },
            schema={"level": pl.Utf8, "code": pl.Utf8, "message": pl.Utf8, "location": pl.Utf8},
        )

    def explain(self) -> str:
        if not self.diagnostics:
            return "DiagnosticReport: ok (no diagnostics)"
        lines = [f"DiagnosticReport: {len(self.diagnostics)} diagnostic(s)"]
        lines += [f"  - {d}" for d in self.diagnostics]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __bool__(self) -> bool:
        return bool(self.diagnostics)
