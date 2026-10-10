"""Task list and detail routes (merge live rollout state with FS index)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api")


def _merge_task_dict(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        merged[key] = value
    return merged


async def _fetch_live_task_list(state) -> list[dict[str, Any]]:
    payload = await state.rollout_client.safe_get_json("/tasks", default=None)
    if not payload:
        return []
    if isinstance(payload, dict):
        return payload.get("tasks") or []
    if isinstance(payload, list):
        return payload
    return []


@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: str | None = Query(default=None),
    harness: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    state = request.app.state.platform
    index_tasks = state.fs_index.list_tasks(
        status=status,
        harness=harness,
        since=since,
        limit=limit,
    )
    fs_map = {t.task_id: t.to_dict() for t in index_tasks}
    live_tasks = await _fetch_live_task_list(state)
    for live in live_tasks:
        task_id = live.get("task_id")
        if not task_id:
            continue
        existing = fs_map.get(task_id, {"task_id": task_id})
        fs_map[task_id] = _merge_task_dict(existing, live)
    # Re-apply filters because live overlay may now match.
    tasks = list(fs_map.values())
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    if harness:
        tasks = [t for t in tasks if t.get("harness") == harness]
    tasks.sort(key=lambda t: (t.get("updated_at") or 0), reverse=True)
    return {"tasks": tasks[:limit]}


@router.get("/tasks/{task_id}")
async def get_task(request: Request, task_id: str) -> dict[str, Any]:
    state = request.app.state.platform
    summary = state.fs_index.get_task(task_id)
    payload: dict[str, Any] = summary.to_dict() if summary else {"task_id": task_id}

    # Live overlay from rollout server.
    live = await state.rollout_client.safe_get_json(
        f"/rollout/task/{task_id}", default=None
    )
    if isinstance(live, dict):
        payload = _merge_task_dict(payload, live)
    elif summary is None:
        raise HTTPException(status_code=404, detail="Task not found")

    sessions_payload = await state.rollout_client.safe_get_json(
        f"/tasks/{task_id}/sessions", default=None
    )
    if isinstance(sessions_payload, dict):
        sessions = sessions_payload.get("sessions") or []
    else:
        sessions = []

    fs_sessions = state.fs_index.list_sessions_for(task_id)
    fs_session_map = {s.session_id: s.to_dict() for s in fs_sessions}
    for live_session in sessions:
        sid = live_session.get("session_id")
        if not sid:
            continue
        fs_session_map[sid] = _merge_task_dict(
            fs_session_map.get(sid, {"session_id": sid}),
            live_session,
        )

    payload["sessions"] = list(fs_session_map.values())
    payload.setdefault("save_dir_path", str(state.config.save_dir / f"task_{task_id}"))
    return payload


__all__ = ["router"]
