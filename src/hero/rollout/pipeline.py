"""Dispatch + collect rollout pipeline for gateway nodes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from hero.platform.events import EventBus
from hero.rollout.balancer import NodeScheduler
from hero.rollout.models import SessionContext, SessionDispatchRequest, SessionResult, SessionStatus
from hero.trajectory.models import Trajectory

logger = logging.getLogger(__name__)

ResultCallback = Callable[[SessionResult], Awaitable[None] | None]


def _trajectory_status(status: str) -> str:
    if status == SessionStatus.TIMEOUT:
        return SessionStatus.TIMEOUT
    if status == SessionStatus.COMPLETED:
        return SessionStatus.COMPLETED
    return SessionStatus.ERROR


class Pipeline:
    """Process rollout sessions by dispatching to gateway nodes and collecting results."""

    def __init__(
        self,
        *,
        callback_url: str,
        save_dir: str | None,
        scheduler: NodeScheduler,
        dispatch_poll_interval_seconds: float = 1.0,
        callback_grace_seconds: float = 180.0,
        event_bus: EventBus | None = None,
    ) -> None:
        self.callback_url = callback_url.rstrip("/")
        self.save_dir = Path(save_dir) if save_dir else None
        self.scheduler = scheduler
        self.dispatch_poll_interval_seconds = dispatch_poll_interval_seconds
        self.callback_grace_seconds = callback_grace_seconds
        self.event_bus = event_bus

        self._client: httpx.AsyncClient | None = None
        self._started = False
        self._lifecycle_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[SessionResult]] = {}
        self._pending_lock = asyncio.Lock()

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self.event_bus is not None:
            await self.event_bus.publish(event_type, payload)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            self._client = httpx.AsyncClient(timeout=30.0)
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            async with self._pending_lock:
                for future in self._pending.values():
                    if not future.done():
                        future.cancel()
                self._pending.clear()
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            self._started = False

    async def run_batch(
        self,
        sessions: list[SessionContext],
        *,
        on_result: ResultCallback | None = None,
    ) -> list[SessionResult]:
        await self.start()
        return await asyncio.gather(
            *(self._dispatch_and_collect(session, on_result) for session in sessions)
        )

    async def accept_callback_result(self, result: SessionResult) -> bool:
        async with self._pending_lock:
            future = self._pending.get(result.session_id)
            if future is None or future.done():
                return False
            future.set_result(result)
            return True

    def status(self) -> dict[str, object]:
        return {
            "pending_sessions": len(self._pending),
        }

    def result_path_for(self, task_id: str, session_id: str) -> str | None:
        path = self._result_path(task_id, session_id)
        return None if path is None else str(path)

    async def _dispatch_and_collect(
        self,
        session: SessionContext,
        callback: ResultCallback | None,
    ) -> SessionResult:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.completion_future = future
        async with self._pending_lock:
            self._pending[session.session_id] = future

        session.timer.mark("dispatch", "started")
        await self._emit(
            "session.state_changed",
            {"task_id": session.task_id, "session_id": session.session_id, "status": "DISPATCHING"},
        )
        try:
            dispatch_request = await self._dispatch_session(session)
            session.timer.mark("dispatch", "finished")
            await self._emit(
                "session.state_changed",
                {
                    "task_id": session.task_id,
                    "session_id": session.session_id,
                    "status": "REGISTERED",
                    "node_id": session.node_id,
                },
            )
            result = await self._wait_for_result(session, dispatch_request, future)
        except TimeoutError as exc:
            logger.warning("Session %s timed out in rollout pipeline", session.session_id)
            result = self._failure_result(session, status=SessionStatus.TIMEOUT, error=str(exc))
        except Exception as exc:
            logger.exception("Dispatch failed for session %s", session.session_id)
            result = self._failure_result(session, error=str(exc))
        finally:
            async with self._pending_lock:
                self._pending.pop(session.session_id, None)

        await asyncio.to_thread(self._persist_result, result)
        session.rollout_result = result
        try:
            if callback is not None:
                maybe_awaitable = callback(result)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            return result
        finally:
            await self._cleanup_session(session)

    async def _dispatch_session(self, session: SessionContext) -> SessionDispatchRequest:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        while True:
            node = self.scheduler.acquire_node()
            if node is None:
                remaining_timeout = self._remaining_timeout_seconds(session)
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                continue

            session.node_id = node.node_id
            session.gateway_url = node.gateway_url
            dispatch_timeout = self._remaining_timeout_seconds(session)
            dispatch_request = SessionDispatchRequest(
                session_id=session.session_id,
                task_id=session.task_id,
                instruction=session.request.instruction,
                remaining_timeout_seconds=session.request.timeout_seconds,
                callback_url=self.callback_url,
                runtime=session.request.runtime,
                agent=session.request.agent,
                builder=session.request.builder,
                evaluator=session.request.evaluator,
                metadata=dict(session.request.metadata),
            )
            try:
                response = await self._client.post(
                    f"{node.gateway_url}/sessions",
                    json=dispatch_request.model_dump(mode="json"),
                    timeout=min(30.0, dispatch_timeout),
                )
                response.raise_for_status()
                return dispatch_request
            except Exception as exc:
                if await self._accepted_duplicate_dispatch(
                    exc, node.gateway_url, session, dispatch_request
                ):
                    return dispatch_request
                self.scheduler.release_reservation(node.node_id)
                self.scheduler.mark_unhealthy(node.node_id)
                try:
                    remaining_timeout = self._remaining_timeout_seconds(session)
                except TimeoutError:
                    raise TimeoutError(
                        "session timeout expired before gateway dispatch completed"
                    ) from exc
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                session.node_id = None
                session.gateway_url = None

    async def _accepted_duplicate_dispatch(
        self,
        exc: Exception,
        gateway_url: str,
        session: SessionContext,
        dispatch_request: SessionDispatchRequest,
    ) -> bool:
        """Treat dispatch errors after gateway accept as successful dispatch.

        A gateway can accept a session and still have the rollout server's POST
        fail before the acknowledgement arrives. A later retry of the same
        single-use session id usually returns 409. If the gateway reports that
        the session belongs to this task, the rollout server should continue to
        wait for its result instead of retrying forever.
        """
        if self._client is None:
            return False

        try:
            response = await self._client.get(
                f"{gateway_url}/sessions/{session.session_id}",
                timeout=5.0,
            )
            response.raise_for_status()
        except Exception:
            log = logger.debug
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409:
                log = logger.warning
            log(
                "Failed to confirm duplicate dispatch for session %s",
                session.session_id,
                exc_info=True,
            )
            return False

        payload = response.json()
        if payload.get("task_id") != dispatch_request.task_id:
            logger.warning(
                "Duplicate session id %s exists on %s for task %s, expected %s",
                session.session_id,
                gateway_url,
                payload.get("task_id"),
                dispatch_request.task_id,
            )
            return False

        logger.info(
            "Confirmed duplicate dispatch for session %s on %s; continuing to wait for result",
            session.session_id,
            gateway_url,
        )
        return True

    async def _wait_for_result(
        self,
        session: SessionContext,
        dispatch_request: SessionDispatchRequest,
        future: asyncio.Future[SessionResult],
    ) -> SessionResult:
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        # Interleave callback-wait and gateway-poll so the poll path is a
        # live safety net (not dead code). Covers two races:
        #   1. Callback HTTP POST dropped/delayed — poll GET finds the result.
        #   2. Gateway flips status→terminal a tick before serializing the
        #      result payload — we re-poll next iteration instead of
        #      synthesizing a failure.
        # REGISTERED is gateway queue time before INIT. Keep measuring it in
        # session timing, but do not spend the execution timeout until INIT starts.
        execution_timeout_started = False
        callback_deadline: float | None = None
        pre_init_poll_interval = self.dispatch_poll_interval_seconds
        result_poll_interval = max(self.dispatch_poll_interval_seconds, 5.0)

        while True:
            if execution_timeout_started:
                assert callback_deadline is not None
                remaining = callback_deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_timeout = min(result_poll_interval, remaining)
            else:
                wait_timeout = pre_init_poll_interval
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                pass
            try:
                status, result = await self._poll_session_state(session, timeout=30.0)
            except Exception as exc:
                logger.debug(
                    "poll_session_result failed for session %s: %s",
                    session.session_id,
                    exc,
                )
                status = None
                result = None
            if result is not None:
                if not future.done():
                    future.set_result(result)
                return result
            if (
                not execution_timeout_started
                and status is not None
                and status != SessionStatus.REGISTERED
            ):
                session.deadline_monotonic = (
                    time.monotonic() + session.request.timeout_seconds
                )
                callback_deadline = self._callback_deadline_monotonic(session)
                execution_timeout_started = True

        raise TimeoutError(
            f"session {dispatch_request.session_id} did not return a terminal result "
            "before the callback deadline"
        )

    async def _poll_session_state(
        self,
        session: SessionContext,
        *,
        timeout: float,
    ) -> tuple[str | None, SessionResult | None]:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        response = await self._client.get(
            f"{session.gateway_url}/sessions/{session.session_id}",
            timeout=min(30.0, timeout),
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        result_payload = payload.get("result")
        status_value = str(status) if status is not None else None
        if isinstance(result_payload, dict):
            return status_value, SessionResult.model_validate(result_payload)
        # Gateway may flip status→terminal before the result payload is
        # serialized into the GET response. Returning None here keeps the
        # outer loop polling until either the payload lands or the callback
        # deadline expires — preventing synthesized empty-trace "failures"
        # that poisoned GRPO batches (see feedback_sglang_tool_parser.md
        # et al).
        return status_value, None

    async def _poll_session_result(
        self,
        session: SessionContext,
        *,
        timeout: float,
    ) -> SessionResult | None:
        _, result = await self._poll_session_state(session, timeout=timeout)
        return result

    def _remaining_timeout_seconds(self, session: SessionContext) -> float:
        remaining = session.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session timeout expired")
        return remaining

    def _callback_deadline_monotonic(self, session: SessionContext) -> float:
        return session.deadline_monotonic + self.callback_grace_seconds

    def _remaining_callback_window_seconds(self, session: SessionContext) -> float:
        remaining = self._callback_deadline_monotonic(session) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session callback deadline expired")
        return remaining

    async def _cleanup_session(self, session: SessionContext) -> None:
        if self._client is None or session.gateway_url is None:
            return

        try:
            response = await self._client.delete(
                f"{session.gateway_url}/sessions/{session.session_id}"
            )
            if response.status_code not in {200, 404}:
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to clean up session %s on gateway %s",
                session.session_id,
                session.gateway_url,
                exc_info=True,
            )

    def _persist_result(self, result: SessionResult) -> None:
        path = self._result_path(result.task_id, result.session_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._storage_payload(result), separators=(",", ":"), default=str)
        )

    def _result_path(self, task_id: str, session_id: str) -> Path | None:
        if self.save_dir is None:
            return None
        return self.save_dir / f"task_{task_id}" / f"ses_{session_id}.json"

    @staticmethod
    def _storage_payload(result: SessionResult) -> dict[str, object]:
        """Return the persisted session artifact shape.

        The on-disk rollout result keeps session-level status/error only.
        Trajectory payloads store the structured trace data without duplicating
        terminal status information.
        """
        payload = result.model_dump(mode="json")
        trajectory = payload.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory.pop("status", None)
            trajectory.pop("error", None)
        return payload

    @staticmethod
    def _failure_result(
        session: SessionContext,
        *,
        status: str = SessionStatus.ERROR,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=session.session_id,
            task_id=session.task_id,
            status=_trajectory_status(status),
            trajectory=Trajectory(
                status=_trajectory_status(status),
                metadata={
                    "builder": session.request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(session.request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=session.timer.to_session_timing(),
            node_id=session.node_id,
            error=error,
            metadata=dict(session.request.metadata),
        )
