"""Stage-isolated session dispatcher for gateway execution.

Drives four stages — INIT, READY, RUNNING, POSTRUN — each with an isolated
worker pool. Queued vs. executing within a stage is the `inflight` bool on
`ManagedSession`, so the stage enum stays small. Eval-runtime prewarm is an
ad-hoc background task owned by the session handler, not a dedicated stage.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from hero.agent.models import AgentRunResult
from hero.rollout.models import SessionDispatchRequest, SessionResult
from hero.rollout.timer import StageTimer
from hero.runtime.base import BaseRuntime
from hero.runtime.models import ExecInput

logger = logging.getLogger(__name__)

StageCallback = Callable[["ManagedSession"], Awaitable[None]]
StageTransitionCallback = Callable[["ManagedSession"], None]
_STOP = object()


class SessionStage(str, Enum):
    INIT = "INIT"
    READY = "READY"
    RUNNING = "RUNNING"
    POSTRUN = "POSTRUN"


@dataclass(slots=True)
class DispatcherSnapshot:
    init_queue_depth: int = 0
    init_inflight: int = 0
    ready_depth: int = 0
    run_inflight: int = 0
    postrun_queue_depth: int = 0
    postrun_inflight: int = 0

    @property
    def active_count(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


@dataclass(slots=True)
class ManagedSession:
    """Per-session state flowing through the gateway dispatcher."""

    request: SessionDispatchRequest
    timer: StageTimer
    session_dir: Path
    artifacts_dir: Path
    runtime: BaseRuntime | None = None
    agent_result: AgentRunResult | None = None
    final_result: SessionResult | None = None
    postrun_steps: list[ExecInput] = field(default_factory=list)
    eval_prewarm_task: asyncio.Task | None = None
    cancel_requested: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    execution_deadline: float | None = None
    stage: SessionStage = SessionStage.INIT
    inflight: bool = False

    @property
    def session_id(self) -> str:
        return self.request.session_id


class SessionDispatcher:
    """Drive INIT -> READY -> RUNNING -> POSTRUN with isolated worker pools."""

    def __init__(
        self,
        *,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
    ) -> None:
        if max_init_workers < 1 or max_run_workers < 1 or max_postrun_workers < 1:
            raise ValueError("all stage worker counts must be at least 1")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.on_init: StageCallback | None = None
        self.on_run: StageCallback | None = None
        self.on_postrun: StageCallback | None = None
        self.on_stage_change: StageTransitionCallback | None = None
        self._init_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._postrun_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_slots = asyncio.Semaphore(max_run_workers)
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._workers = [
            *(asyncio.create_task(self._init_worker()) for _ in range(self.max_init_workers)),
            *(asyncio.create_task(self._run_worker()) for _ in range(self.max_run_workers)),
            *(asyncio.create_task(self._postrun_worker()) for _ in range(self.max_postrun_workers)),
        ]
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for managed in sessions:
            managed.cancel_requested = True
            managed.cancel_event.set()
            if managed.runtime is not None:
                await managed.runtime.cancel()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def enqueue(self, managed: ManagedSession) -> None:
        if not self._started:
            raise RuntimeError("dispatcher has not been started")
        async with self._lock:
            if managed.session_id in self._sessions:
                raise ValueError(f"session {managed.session_id} is already enqueued")
            self._sessions[managed.session_id] = managed
        await self._init_queue.put(managed.session_id)

    async def cancel(self, session_id: str) -> bool:
        should_enqueue_postrun = False
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return False
            if managed.cancel_requested:
                return True
            managed.cancel_requested = True
            managed.cancel_event.set()
            # If the session is parked in READY (holding a ready slot), release it
            # and transition to POSTRUN so the postrun worker picks it up.
            if managed.stage == SessionStage.READY and not managed.inflight:
                self._ready_slots.release()
                managed.stage = SessionStage.POSTRUN
                managed.inflight = False
                should_enqueue_postrun = True
            elif managed.stage == SessionStage.INIT and not managed.inflight:
                managed.stage = SessionStage.POSTRUN
                managed.inflight = False
                should_enqueue_postrun = True
        if managed.runtime is not None:
            await managed.runtime.cancel()
        if should_enqueue_postrun:
            self._notify_stage_change(managed)
            await self._postrun_queue.put(session_id)
        return True

    async def active_count(self) -> int:
        return (await self.snapshot()).active_count

    async def snapshot(self) -> DispatcherSnapshot:
        async with self._lock:
            snap = DispatcherSnapshot()
            for managed in self._sessions.values():
                if managed.stage == SessionStage.INIT:
                    if managed.inflight:
                        snap.init_inflight += 1
                    else:
                        snap.init_queue_depth += 1
                elif managed.stage == SessionStage.READY:
                    snap.ready_depth += 1
                elif managed.stage == SessionStage.RUNNING:
                    snap.run_inflight += 1
                elif managed.stage == SessionStage.POSTRUN:
                    if managed.inflight:
                        snap.postrun_inflight += 1
                    else:
                        snap.postrun_queue_depth += 1
            return snap

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def _init_worker(self) -> None:
        while True:
            item = await self._init_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._begin(session_id, SessionStage.INIT)
            if managed is None:
                continue
            if not (managed.cancel_requested or managed.final_result is not None):
                await self._safe_invoke(self.on_init, managed, SessionStage.INIT)
            await self._finish_init(managed)

    async def _run_worker(self) -> None:
        while True:
            item = await self._ready_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._begin(session_id, SessionStage.RUNNING, from_ready=True)
            if managed is None:
                continue
            if not (managed.cancel_requested or managed.final_result is not None):
                await self._safe_invoke(self.on_run, managed, SessionStage.RUNNING)
            await self._transition_to_postrun(managed)

    async def _postrun_worker(self) -> None:
        while True:
            item = await self._postrun_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._begin(session_id, SessionStage.POSTRUN)
            if managed is None:
                continue
            await self._safe_invoke(self.on_postrun, managed, SessionStage.POSTRUN)
            async with self._lock:
                self._sessions.pop(session_id, None)

    async def _safe_invoke(
        self, callback: StageCallback | None, managed: ManagedSession, stage: SessionStage
    ) -> None:
        if callback is None:
            logger.error("Dispatcher stage %s has no callback", stage)
            return
        try:
            await callback(managed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Dispatcher stage %s failed for session %s", stage, managed.session_id
            )

    async def _begin(
        self,
        session_id: str,
        stage: SessionStage,
        *,
        from_ready: bool = False,
    ) -> ManagedSession | None:
        """Mark the session as inflight in *stage*. Returns None if session is gone."""
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return None
            if from_ready:
                # READY slot was granted via semaphore; RUNNING consumes it.
                pass
            managed.stage = stage
            managed.inflight = True
        self._notify_stage_change(managed)
        return managed

    async def _finish_init(self, managed: ManagedSession) -> None:
        """After INIT callback returns, move to READY (waiting a run slot) or POSTRUN."""
        if managed.cancel_requested or managed.final_result is not None:
            await self._move_to_postrun(managed)
            return

        async with self._lock:
            if managed.session_id not in self._sessions:
                return
            managed.stage = SessionStage.READY
            managed.inflight = False
        self._notify_stage_change(managed)

        acquired = await self._acquire_ready_slot(managed)
        if not acquired:
            await self._move_to_postrun(managed, release_ready=False)
            return
        await self._ready_queue.put(managed.session_id)

    async def _transition_to_postrun(self, managed: ManagedSession) -> None:
        # The RUN callback is done (or was skipped). The ready slot is released
        # back to the pool on exit of RUNNING.
        self._ready_slots.release()
        await self._move_to_postrun(managed, release_ready=False)

    async def _move_to_postrun(
        self, managed: ManagedSession, *, release_ready: bool = True
    ) -> None:
        transitioned = False
        async with self._lock:
            if managed.session_id not in self._sessions:
                return
            if managed.stage == SessionStage.POSTRUN:
                return
            if release_ready and managed.stage == SessionStage.READY and not managed.inflight:
                self._ready_slots.release()
            managed.stage = SessionStage.POSTRUN
            managed.inflight = False
            transitioned = True
        if transitioned:
            self._notify_stage_change(managed)
            await self._postrun_queue.put(managed.session_id)

    async def _acquire_ready_slot(self, managed: ManagedSession) -> bool:
        """Race the semaphore against session cancellation. Returns True on acquire."""
        if managed.cancel_event.is_set() or managed.final_result is not None:
            return False
        acquire_task = asyncio.create_task(self._ready_slots.acquire())
        cancel_task = asyncio.create_task(managed.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {acquire_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            cancel_task.cancel()
            if not acquire_task.done():
                acquire_task.cancel()
        # If we managed to acquire the semaphore despite cancellation, release it
        # so it doesn't leak to a later session.
        acquired = acquire_task in done and not acquire_task.cancelled() and acquire_task.exception() is None
        if managed.cancel_event.is_set() or managed.final_result is not None:
            if acquired:
                self._ready_slots.release()
            return False
        return acquired

    def _notify_stage_change(self, managed: ManagedSession) -> None:
        callback = self.on_stage_change
        if callback is None:
            return
        try:
            callback(managed)
        except Exception:
            logger.exception(
                "Dispatcher stage-change callback failed for session %s",
                managed.session_id,
            )
