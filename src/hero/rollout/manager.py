"""Top-level task orchestration for rollout batches."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from hero.platform.events import EventBus
from hero.rollout.balancer import NodeScheduler
from hero.rollout.models import (
    SessionContext,
    SessionResult,
    SessionStatus,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from hero.rollout.pipeline import Pipeline

logger = logging.getLogger(__name__)

_CALLBACK_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int = 0
    errored_sessions: int = 0
    harness: str | None = None
    model: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    results: list[SessionResult] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)
    session_states: dict[str, str] = field(default_factory=dict)


def _harness_from_request(request: TaskRequest) -> str | None:
    if request.agent and request.agent.harness:
        return request.agent.harness
    return None


def _model_from_request(request: TaskRequest) -> str | None:
    if request.agent and request.agent.model_name:
        return request.agent.model_name
    return None


def _mean_reward(results: list[SessionResult]) -> float | None:
    rewards: list[float] = []
    for r in results:
        traces = r.trajectory.traces
        if traces and traces[-1].reward is not None:
            try:
                rewards.append(float(traces[-1].reward))
            except (TypeError, ValueError):
                pass
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


def _mean_traces(results: list[SessionResult]) -> float | None:
    """Average number of traces per session for the task."""
    if not results:
        return None
    counts = [len(r.trajectory.traces) for r in results]
    return sum(counts) / len(counts)


def _mean_completions(results: list[SessionResult]) -> float | None:
    """Average number of raw completions (LLM requests) per session."""
    if not results:
        return None
    counts = [
        int(r.trajectory.metadata.get("record_count") or len(r.trajectory.traces))
        for r in results
    ]
    return sum(counts) / len(counts)


class RolloutManager:
    """Manage the lifecycle of rollout sessions for a single submitted task."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        scheduler: NodeScheduler,
        event_bus: EventBus | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler
        self.event_bus = event_bus or EventBus()
        self._tasks: dict[str, _TaskRecord] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
            except RuntimeError:
                return
        self.event_bus.publish_threadsafe(loop, event_type, payload)

    async def submit_task(self, request: TaskRequest) -> str:
        """Register a task and run it in the background. Returns task_id immediately."""
        self._loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._tasks.get(request.task_id)
            if existing is not None and existing.status == "running":
                raise ValueError(f"task {request.task_id} is already running")
            self._tasks[request.task_id] = _TaskRecord(
                task_id=request.task_id,
                status="running",
                total_sessions=request.num_samples,
                harness=_harness_from_request(request),
                model=_model_from_request(request),
            )
        self._emit(
            "task.created",
            {
                "task_id": request.task_id,
                "status": "running",
                "harness": _harness_from_request(request),
                "model": _model_from_request(request),
                "num_samples": request.num_samples,
            },
        )
        asyncio.create_task(self._run_task_background(request))
        return request.task_id

    async def _run_task_background(self, request: TaskRequest) -> None:
        """Execute a task in the background, updating the record on completion."""
        try:
            result = await self._execute_task(request)
            logger.info("Task %s completed with %d results", request.task_id, len(result.results))
        except Exception:
            logger.exception("Background task %s failed", request.task_id)
            self._emit("task.completed", {"task_id": request.task_id, "status": "failed"})
            return
        self._emit(
            "task.completed",
            {
                "task_id": request.task_id,
                "status": result.status,
                "completed_sessions": len(result.results),
            },
        )
        if request.callback_url:
            await self._post_callback(request.callback_url, result)

    async def _post_callback(self, callback_url: str, result: TaskResult) -> None:
        """Best-effort POST the terminal TaskResult to the trainer's callback URL."""
        try:
            async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    callback_url,
                    json=result.model_dump(mode="json"),
                )
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Callback POST to %s failed for task %s; trainer must fall back to polling",
                callback_url,
                result.task_id,
                exc_info=True,
            )

    def session_state_changed(self, task_id: str, session_id: str, status: str) -> None:
        """Hook invoked from the pipeline whenever a session changes state."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                record.session_states[session_id] = status
                record.updated_at = time.time()
        self._emit(
            "session.state_changed",
            {"task_id": task_id, "session_id": session_id, "status": status},
        )

    async def _execute_task(self, request: TaskRequest) -> TaskResult:
        sessions = [
            SessionContext(
                session_id=f"sk-hero-{uuid.uuid4()}",
                task_id=request.task_id,
                request=request,
                deadline_monotonic=time.monotonic() + request.timeout_seconds,
            )
            for _ in range(request.num_samples)
        ]

        async def _on_result(result: SessionResult) -> None:
            result_path = self.pipeline.result_path_for(result.task_id, result.session_id)
            with self._lock:
                record = self._tasks[request.task_id]
                record.completed_sessions += 1
                if result.status in {SessionStatus.ERROR, SessionStatus.TIMEOUT}:
                    record.errored_sessions += 1
                record.results.append(result)
                record.updated_at = time.time()
                record.session_states[result.session_id] = str(result.status)
                if result_path is not None:
                    record.result_paths.append(result_path)
            self._emit(
                "session.state_changed",
                {
                    "task_id": result.task_id,
                    "session_id": result.session_id,
                    "status": str(result.status),
                },
            )
            self._emit(
                "task.updated",
                {
                    "task_id": request.task_id,
                    "completed_sessions": record.completed_sessions,
                    "total_sessions": record.total_sessions,
                },
            )

        try:
            results = await self.pipeline.run_batch(sessions, on_result=_on_result)
        except Exception:
            with self._lock:
                self._tasks[request.task_id].status = "failed"
                self._tasks[request.task_id].updated_at = time.time()
            raise

        ordered_results = list(results)
        with self._lock:
            record = self._tasks[request.task_id]
            record.status = "completed"
            record.completed_sessions = len(ordered_results)
            record.results = ordered_results
            record.updated_at = time.time()
            result_paths = list(record.result_paths)

        return TaskResult(
            task_id=request.task_id,
            status="completed",
            results=ordered_results,
            result_paths=result_paths,
        )

    def get_task(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            return TaskStatus(
                task_id=record.task_id,
                status=record.status,
                total_sessions=record.total_sessions,
                completed_sessions=record.completed_sessions,
                results=list(record.results),
                result_paths=list(record.result_paths),
            )

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for record in self._tasks.values():
                out.append({
                    "task_id": record.task_id,
                    "status": record.status,
                    "harness": record.harness,
                    "model": record.model,
                    "num_samples": record.total_sessions,
                    "completed_sessions": record.completed_sessions,
                    "errored_sessions": record.errored_sessions,
                    "mean_reward": _mean_reward(record.results),
                    "mean_traces": _mean_traces(record.results),
                    "mean_completions": _mean_completions(record.results),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "source": "live",
                })
            return out

    def list_sessions_for(self, task_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            existing = {r.session_id: r for r in record.results}
            out: list[dict[str, Any]] = []
            for session_id, status in record.session_states.items():
                result = existing.get(session_id)
                if result is not None:
                    traces = result.trajectory.traces
                    reward = traces[-1].reward if traces else None
                    out.append({
                        "session_id": result.session_id,
                        "task_id": result.task_id,
                        "status": str(result.status),
                        "node_id": result.node_id,
                        "reward": reward,
                        "timing": result.timing.model_dump(),
                        "error": result.error,
                    })
                else:
                    out.append({
                        "session_id": session_id,
                        "task_id": task_id,
                        "status": status,
                    })
            return out

    def status(self) -> dict[str, object]:
        with self._lock:
            task_statuses = {
                task_id: record.status
                for task_id, record in self._tasks.items()
            }
        return {
            "tasks": task_statuses,
            "pipeline": self.pipeline.status(),
            "nodes": self.scheduler.stats(),
        }
