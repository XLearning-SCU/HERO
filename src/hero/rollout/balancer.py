"""Gateway-node registration and stage-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import threading

from hero.rollout.models import GatewayNodeInfo, NodeRegistrationRequest, NodeStageMetrics


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class GatewayNode:
    node_id: str
    gateway_url: str
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    healthy: bool
    last_heartbeat: datetime
    heartbeat_interval_seconds: int
    metrics: NodeStageMetrics = field(default_factory=NodeStageMetrics)
    dispatch_reservations: int = 0
    draining: bool = False

    @property
    def total_sessions(self) -> int:
        return self.metrics.total_sessions + self.dispatch_reservations

    @property
    def ready_buffer_target(self) -> int:
        """Ready buffer tracks run capacity one-for-one."""
        return self.max_run_workers

    def to_model(self) -> GatewayNodeInfo:
        return GatewayNodeInfo(
            node_id=self.node_id,
            gateway_url=self.gateway_url,
            max_init_workers=self.max_init_workers,
            max_run_workers=self.max_run_workers,
            max_postrun_workers=self.max_postrun_workers,
            metrics=self.metrics.model_copy(),
            dispatch_reservations=self.dispatch_reservations,
            healthy=self.healthy,
            draining=self.draining,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            last_heartbeat=self.last_heartbeat,
        )


class NodeScheduler:
    """Track registered nodes and assign sessions using stage-aware pressure."""

    def __init__(
        self,
        *,
        bootstrap_nodes: list[dict[str, object]] | None = None,
        stale_factor: float = 2.5,
    ) -> None:
        self._nodes: dict[str, GatewayNode] = {}
        self._lock = threading.RLock()
        self._stale_factor = stale_factor

        for node in bootstrap_nodes or []:
            node_id = str(node["node_id"])
            max_run_workers = max(1, int(node.get("max_run_workers", node.get("capacity", 1))))
            self._nodes[node_id] = GatewayNode(
                node_id=node_id,
                gateway_url=str(node["gateway_url"]).rstrip("/"),
                max_init_workers=max(1, int(node.get("max_init_workers", 4))),
                max_run_workers=max_run_workers,
                max_postrun_workers=max(1, int(node.get("max_postrun_workers", 4))),
                healthy=False,
                last_heartbeat=_utcnow(),
                heartbeat_interval_seconds=max(1, int(node.get("heartbeat_interval_seconds", 30))),
            )

    def register_node(self, request: NodeRegistrationRequest) -> GatewayNodeInfo:
        now = _utcnow()
        with self._lock:
            existing = self._nodes.get(request.node_id)
            draining = existing.draining if existing is not None else False
            metrics = existing.metrics.model_copy() if existing is not None else NodeStageMetrics()
            self._nodes[request.node_id] = GatewayNode(
                node_id=request.node_id,
                gateway_url=request.gateway_url.rstrip("/"),
                max_init_workers=request.max_init_workers,
                max_run_workers=request.max_run_workers,
                max_postrun_workers=request.max_postrun_workers,
                metrics=metrics,
                dispatch_reservations=0,
                healthy=True,
                last_heartbeat=now,
                heartbeat_interval_seconds=request.heartbeat_interval_seconds,
                draining=draining,
            )
            return self._nodes[request.node_id].to_model()

    def heartbeat(self, node_id: str, *, metrics: NodeStageMetrics) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            node.last_heartbeat = _utcnow()
            node.healthy = True
            node.metrics = metrics
            node.dispatch_reservations = 0
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def acquire_node(self) -> GatewayNodeInfo | None:
        with self._lock:
            self._refresh_health_locked()
            candidates = [
                node
                for node in self._nodes.values()
                if node.healthy and not node.draining and self._can_accept_session(node)
            ]
            if not candidates:
                return None

            selected = min(candidates, key=self._node_score)
            selected.dispatch_reservations += 1
            return selected.to_model()

    def refresh_health(self) -> None:
        """Recompute stored health from heartbeat timestamps (call from a tick task)."""
        with self._lock:
            self._refresh_health_locked()

    def release_reservation(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.dispatch_reservations = max(0, node.dispatch_reservations - 1)
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def mark_unhealthy(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.healthy = False
            return node.to_model()

    def drain_node(self, node_id: str) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            node.draining = True
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def get_node(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            return self._fresh_model_locked(node, _utcnow())

    def list_nodes(self) -> list[GatewayNodeInfo]:
        with self._lock:
            now = _utcnow()
            return [
                self._fresh_model_locked(node, now)
                for node in sorted(self._nodes.values(), key=lambda item: item.node_id)
            ]

    def stats(self) -> dict[str, object]:
        with self._lock:
            now = _utcnow()
            return {
                "nodes": [
                    self._fresh_model_locked(node, now).model_dump(mode="json")
                    for node in sorted(self._nodes.values(), key=lambda item: item.node_id)
                ],
            }

    def _fresh_model_locked(self, node: GatewayNode, now: datetime) -> GatewayNodeInfo:
        """Return a snapshot with heartbeat-derived health but no mutation."""
        return replace(node, healthy=self._is_node_healthy(node, now)).to_model()

    def _is_node_healthy(self, node: GatewayNode, now: datetime) -> bool:
        timeout_seconds = max(1.0, node.heartbeat_interval_seconds * self._stale_factor)
        return (now - node.last_heartbeat).total_seconds() <= timeout_seconds

    def _refresh_health_locked(self) -> None:
        now = _utcnow()
        for node in self._nodes.values():
            node.healthy = self._is_node_healthy(node, now)

    @staticmethod
    def _node_score(node: GatewayNode) -> tuple[float, float, float, float, float, str]:
        run_pressure = node.metrics.run_inflight / node.max_run_workers
        postrun_pressure = (
            node.metrics.postrun_inflight + node.metrics.postrun_queue_depth
        ) / node.max_postrun_workers
        init_pressure = (
            node.metrics.init_inflight
            + node.metrics.init_queue_depth
            + node.dispatch_reservations
        ) / node.max_init_workers
        ready_gap = max(0, node.ready_buffer_target - node.metrics.ready_depth)
        admission_limit = (
            node.max_init_workers
            + node.ready_buffer_target
            + node.max_run_workers
            + node.max_postrun_workers
        )
        total_pressure = node.total_sessions / admission_limit
        return (
            run_pressure,
            postrun_pressure,
            init_pressure,
            -float(ready_gap),
            total_pressure,
            node.node_id,
        )

    @staticmethod
    def _can_accept_session(node: GatewayNode) -> bool:
        admission_limit = (
            node.max_init_workers
            + node.ready_buffer_target
            + node.max_run_workers
            + node.max_postrun_workers
        )
        if node.total_sessions >= admission_limit:
            return False
        postrun_backlog = node.metrics.postrun_inflight + node.metrics.postrun_queue_depth
        return postrun_backlog < (node.max_postrun_workers * 2)

    def _require_node_locked(self, node_id: str) -> GatewayNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown gateway node: {node_id}")
        return node

    def _maybe_remove_drained_locked(self, node_id: str, node: GatewayNode) -> None:
        if node.draining and node.total_sessions == 0:
            self._nodes.pop(node_id, None)
