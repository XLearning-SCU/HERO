"""Per-session stage timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from hero.rollout.models import SessionTiming


# Finer marks (``build``, ``eval``, ``teardown``) still write into the mark
# dictionary for debug logs but are rolled into ``postrun_ms`` in the public
# schema.
_POSTRUN_MARKS: tuple[str, ...] = ("postrun", "build", "eval", "teardown")


@dataclass(slots=True)
class StageTimer:
    """Record monotonic timestamps for session stages."""

    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, event: str) -> None:
        """Mark a stage start or finish."""
        self._marks[f"{stage}_{event}"] = time.monotonic()

    def to_session_timing(self) -> SessionTiming:
        """Return durations for the init/run/post-run lifecycle."""
        return SessionTiming(
            register_to_init_queue_ms=self._span_ms("dispatch_started", "init_started"),
            init_ms=self._duration_ms("init"),
            run_ms=self._duration_ms("run"),
            postrun_ms=self._postrun_span_ms(),
        )

    def _duration_ms(self, stage: str) -> float:
        started = self._marks.get(f"{stage}_started")
        finished = self._marks.get(f"{stage}_finished") or started
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _span_ms(self, start_mark: str, end_mark: str) -> float:
        started = self._marks.get(start_mark)
        finished = self._marks.get(end_mark)
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _postrun_span_ms(self) -> float:
        """Earliest postrun-family started to latest postrun-family finished."""
        starts = [
            self._marks[f"{stage}_started"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_started" in self._marks
        ]
        finishes = [
            self._marks[f"{stage}_finished"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_finished" in self._marks
        ]
        if not starts or not finishes:
            return 0.0
        return max(0.0, (max(finishes) - min(starts)) * 1000.0)
