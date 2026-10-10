"""Concurrency-safe local Docker runtime adapter for HERO.

This module intentionally lives under ``run_rl._shared.runtime`` and is loaded through
``RuntimeSpec.import_path``.  It adds the proven local SWE image lifecycle
controls without changing HERO, Slime, or ProRL's training implementation.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import errno
import fcntl
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import shutil
import tarfile
import time
from typing import Any, AsyncIterator, Awaitable, Callable, TextIO
from urllib.parse import quote, urlparse

from hero.runtime.docker import DockerRuntime

logger = logging.getLogger(__name__)

DockerCommand = Callable[..., Awaitable[tuple[int, str | None, str | None]]]

_TRANSIENT_RE = re.compile(
    r"(?i)(resource temporarily unavailable|fork/exec .*(runc|containerd-shim)|"
    r"failed to start shim|OCI runtime (start|exec) failed.*resource temporarily unavailable|"
    r"cannot connect to the docker daemon|connection refused)"
)
_REMOVAL_IN_PROGRESS_RE = re.compile(
    r"(?i)(removal of container .* already in progress|"
    r"container removal is still in progress)"
)
_CONTAINER_NOT_FOUND_RE = re.compile(r"(?i)no such (?:container|object)")
_CGROUP_COMPATIBILITY_RE = re.compile(
    r"(?i)cgroup[^\n]{0,160}(?:unavailable|not found|does not exist|"
    r"permission denied|operation not permitted|read-only|invalid)|"
    r"(?:permission denied|operation not permitted|read-only)[^\n]{0,160}cgroup"
)

_VFS_IMAGE_VISIBILITY_ATTEMPTS = 5
_VFS_IMAGE_VISIBILITY_RETRY_SECONDS = 0.2


class _InterprocessImageLease:
    """One cross-process image-use lease backed by ``flock``.

    cluster runs HERO rollouts and Slime grading in separate Python
    processes against one VFS Docker daemon.  Their in-process refcounts cannot
    see each other.  A shared lease is therefore held from image inspection
    until the caller has confirmed that its container was deleted; eviction
    takes a non-blocking exclusive lease.  The feature is disabled when
    ``path`` is ``None``, which keeps the established local-overlay2 path
    byte-for-byte equivalent at the Docker command level.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.fd: int | None = None

    async def acquire_shared(self) -> bool:
        return await self._acquire(fcntl.LOCK_SH, blocking=True)

    async def acquire_exclusive(self, *, blocking: bool) -> bool:
        return await self._acquire(fcntl.LOCK_EX, blocking=blocking)

    async def upgrade_exclusive(self) -> None:
        """Drop SH and take EX; callers must re-check image state afterwards."""
        if self.path is None:
            return
        self._relock(fcntl.LOCK_UN)
        acquired = await self._acquire(fcntl.LOCK_EX, blocking=True)
        if not acquired:  # pragma: no cover - blocking flock either succeeds or raises.
            raise RuntimeError(f"could not acquire image lease {self.path}")

    async def downgrade_shared(self) -> None:
        if self.path is not None:
            await self._acquire(fcntl.LOCK_SH, blocking=True)

    async def release(self) -> None:
        fd, self.fd = self.fd, None
        if fd is None:
            return

        # Unlock/close must not queue behind flock waiters in an executor.
        # There is no await between detaching, unlocking and closing the fd.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    async def _acquire(self, operation: int, *, blocking: bool) -> bool:
        if self.path is None:
            return True
        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            while True:
                try:
                    # Waiting in the default executor can exhaust all threads
                    # and prevent holders from releasing their locks. Never
                    # block an OS thread on lock contention; yield to asyncio.
                    fcntl.flock(self.fd, operation | fcntl.LOCK_NB)
                    return True
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    if not blocking:
                        await self.release()
                        return False
                await asyncio.sleep(0.05)
        except BaseException:
            # No background syscall can acquire the lock after cancellation.
            await self.release()
            raise

    def _relock(self, operation: int) -> None:
        if self.fd is None:
            raise RuntimeError("image lease is not open")
        fcntl.flock(self.fd, operation)


@dataclass
class _ProcessUsage:
    """One container PID namespace's usage, collected without subprocesses."""

    rss_bytes: int = 0
    processes: int = 0
    threads: int = 0
    pids: tuple[int, ...] = ()


@dataclass
class _TrackedContainer:
    container_name: str
    session_id: str
    image: str
    init_pid: int
    pid_namespace: int
    memory_limit_bytes: int
    pids_limit: int
    cgroup_limits_verified: bool
    consecutive_over_limit: int = 0
    killed: bool = False


def _pid_namespace_inode(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    try:
        return (proc_root / str(pid) / "ns" / "pid").stat().st_ino
    except OSError:
        return None


def _read_process_status(path: Path) -> tuple[int, int]:
    """Return (RSS bytes, thread count) from one Linux proc status file."""
    rss_bytes = 0
    threads = 1
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0, 0
    return rss_bytes, threads


def _scan_pid_namespaces(
    namespaces: set[int], *, proc_root: Path = Path("/proc")
) -> dict[int, _ProcessUsage]:
    """Aggregate RSS/process/thread usage for PID namespaces in one /proc walk.

    This deliberately uses only filesystem/syscall operations.  It keeps
    working when the host is too short of process slots to fork ``docker top``
    (the failure mode this runtime is intended to contain).
    """
    mutable: dict[int, list[Any]] = {inode: [0, 0, 0, []] for inode in namespaces}
    if not mutable:
        return {}
    try:
        entries = os.scandir(proc_root)
    except OSError:
        return {inode: _ProcessUsage() for inode in namespaces}
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                pid = int(entry.name)
                inode = (Path(entry.path) / "ns" / "pid").stat().st_ino
            except (OSError, ValueError):
                continue
            aggregate = mutable.get(inode)
            if aggregate is None:
                continue
            rss_bytes, threads = _read_process_status(Path(entry.path) / "status")
            if threads <= 0:
                continue
            aggregate[0] += rss_bytes
            aggregate[1] += 1
            aggregate[2] += threads
            aggregate[3].append(pid)
    return {
        inode: _ProcessUsage(
            rss_bytes=int(values[0]),
            processes=int(values[1]),
            threads=int(values[2]),
            pids=tuple(int(pid) for pid in values[3]),
        )
        for inode, values in mutable.items()
    }


def _read_limit(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _container_cgroup_limits(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Read effective kernel memory/pids cgroups for a container init PID."""
    try:
        lines = (
            (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return {}

    controller_paths: dict[str, str] = {}
    unified_path: str | None = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if not controllers:
            unified_path = path
        for controller in controllers.split(","):
            controller_paths[controller] = path

    if unified_path is not None:
        relative = unified_path.lstrip("/")
        base = cgroup_root / relative
        return {
            "version": 2,
            "path": unified_path,
            "isolated": unified_path != "/",
            "memory_limit_bytes": _read_limit(base / "memory.max"),
            "pids_limit": _read_limit(base / "pids.max"),
        }

    memory_path = controller_paths.get("memory")
    pids_path = controller_paths.get("pids")
    memory_base = (
        cgroup_root / "memory" / memory_path.lstrip("/")
        if memory_path is not None
        else None
    )
    pids_base = (
        cgroup_root / "pids" / pids_path.lstrip("/") if pids_path is not None else None
    )
    return {
        "version": 1,
        "memory_path": memory_path,
        "pids_path": pids_path,
        "isolated": bool(
            memory_path not in (None, "/") and pids_path not in (None, "/")
        ),
        "memory_limit_bytes": (
            _read_limit(memory_base / "memory.limit_in_bytes")
            if memory_base is not None
            else None
        ),
        "pids_limit": (
            _read_limit(pids_base / "pids.max") if pids_base is not None else None
        ),
    }


def _self_cgroup_memory_metrics(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[int | None, int | None, int]:
    """Return this job cgroup's (limit, usage, conservative reclaim credit)."""
    try:
        lines = (proc_root / "self" / "cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None, 0
    memory_path: str | None = None
    unified_path: str | None = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if not parts[1]:
            unified_path = parts[2]
        elif "memory" in parts[1].split(","):
            memory_path = parts[2]

    if unified_path is not None:
        base = cgroup_root / unified_path.lstrip("/")
        limit = _read_limit(base / "memory.max")
        usage = _read_limit(base / "memory.current")
        stat_path = base / "memory.stat"
    elif memory_path is not None:
        base = cgroup_root / "memory" / memory_path.lstrip("/")
        limit = _read_limit(base / "memory.limit_in_bytes")
        usage = _read_limit(base / "memory.usage_in_bytes")
        stat_path = base / "memory.stat"
    else:
        return None, None, 0

    file_cache = 0
    active_file = 0
    inactive_file = 0
    shmem = 0
    dirty_file = 0
    writeback_file = 0
    try:
        with stat_path.open(encoding="utf-8") as source:
            for line in source:
                key, _, raw_value = line.partition(" ")
                value = int(raw_value)
                if key in {"file", "cache", "total_cache"}:
                    file_cache = max(file_cache, value)
                elif key in {"active_file", "total_active_file"}:
                    active_file = max(active_file, value)
                elif key in {"inactive_file", "total_inactive_file"}:
                    inactive_file = max(inactive_file, value)
                elif key in {"shmem", "total_shmem"}:
                    shmem = max(shmem, value)
                elif key in {"dirty", "file_dirty", "total_dirty"}:
                    dirty_file = max(dirty_file, value)
                elif key in {"writeback", "file_writeback", "total_writeback"}:
                    writeback_file = max(writeback_file, value)
    except (OSError, ValueError):
        pass
    # Older v1 kernels may expose only active/inactive file counters. Give all
    # clean inactive cache and half of clean active cache reclaim credit: this
    # avoids treating a cache-warm node as full without assuming its entire hot
    # working set can disappear cheaply. Dirty/writeback/tmpfs pages get no
    # credit. Missing active-file counters naturally fall back to inactive-only.
    file_cache = max(file_cache, active_file + inactive_file)
    clean_file = max(0, file_cache - shmem - dirty_file - writeback_file)
    lru_credit = max(
        0,
        inactive_file
        + active_file // 2
        - dirty_file
        - writeback_file,
    )
    reclaim_credit = min(clean_file, lru_credit)
    return limit, usage, reclaim_credit


def _host_available_memory_gb() -> float:
    """Return Linux MemAvailable, which includes reclaimable page cache."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("cannot read host available memory") from exc
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _host_total_memory_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return 0.0
    return 0.0


async def _docker_engine_container_pid(
    container_name: str, *, timeout: float = 5.0
) -> int | None:
    """Read ``State.Pid`` from Docker's HTTP API without spawning a process."""
    docker_host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")

    async def request() -> int | None:
        parsed = urlparse(docker_host)
        if parsed.scheme == "unix":
            socket_path = parsed.path or "/var/run/docker.sock"
            reader, writer = await asyncio.open_unix_connection(socket_path)
            host_header = "localhost"
        elif parsed.scheme in {"tcp", "http", "https"}:
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (2376 if parsed.scheme == "https" else 2375)
            reader, writer = await asyncio.open_connection(
                host, port, ssl=True if parsed.scheme == "https" else None
            )
            host_header = f"{host}:{port}"
        else:
            return None

        try:
            path = f"/containers/{quote(container_name, safe='')}/json"
            writer.write(
                (
                    f"GET {path} HTTP/1.1\r\n"
                    f"Host: {host_header}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
            )
            await writer.drain()
            response = await reader.read()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
        header, separator, body = response.partition(b"\r\n\r\n")
        if not separator:
            return None
        status_line = header.splitlines()[0] if header else b""
        if b" 200 " not in status_line:
            return None
        try:
            payload = json.loads(body)
            pid = int(payload["State"]["Pid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return pid if pid > 0 else None

    try:
        return await asyncio.wait_for(request(), timeout=timeout)
    except (OSError, asyncio.TimeoutError):
        return None


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    value = default if raw in (None, "") else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    value = default if raw in (None, "") else float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class LocalDockerCoordinator:
    """Gateway-process-wide limits and image ownership accounting."""

    def __init__(self) -> None:
        self.load_sem = asyncio.Semaphore(
            _env_int("HERO_LOCAL_DOCKER_LOAD_CONCURRENCY", 6, minimum=1)
        )
        self.run_sem = asyncio.Semaphore(
            _env_int("HERO_LOCAL_DOCKER_RUN_CONCURRENCY", 16, minimum=1)
        )
        self.active_concurrency = _env_int(
            "HERO_LOCAL_DOCKER_ACTIVE_CONCURRENCY", 64, minimum=1
        )
        self.active_sem = asyncio.Semaphore(self.active_concurrency)
        # Preserve the full concurrency limit on the normal path.  Under
        # pressure, delay only new containers and leave running work alone.
        # MemAvailable covers Docker, Ray, model workers, and reclaimable cache
        # in one cheap kernel metric; 0 disables admission control.
        requested_memory_reserve_gb = _env_float(
            "HERO_MIN_AVAILABLE_MEMORY_DURING_RUN_GB", 300.0
        )
        host_total_gb = _host_total_memory_gb()
        self.min_available_memory_gb = (
            min(requested_memory_reserve_gb, 0.40 * host_total_gb)
            if host_total_gb > 0
            else requested_memory_reserve_gb
        )
        self.memory_admission_interval = _env_float(
            "HERO_LOCAL_DOCKER_MEMORY_ADMISSION_INTERVAL_SEC",
            5.0,
            minimum=0.1,
        )
        # An admission reservation is held while a freshly started container
        # reaches its initial working set.  The probe and reservation update
        # share one asyncio lock, so a burst of 64 coroutines cannot all make
        # the decision against the same MemAvailable sample.
        self.admission_reservation_hold_sec = _env_float(
            "HERO_LOCAL_DOCKER_ADMISSION_RESERVATION_HOLD_SEC", 30.0
        )
        self._admission_guard = asyncio.Lock()
        self._reserved_memory_gb = 0.0
        self.image_cache_size = _env_int("HERO_LOCAL_DOCKER_IMAGE_CACHE_SIZE", 16)
        # Shared training/evaluation Docker daemon: leave a substantial
        # recovery reserve for overlay2 metadata and active containers.
        self.min_free_gb = _env_float("HERO_LOCAL_DOCKER_MIN_FREE_GB", 500.0)
        self.storage_wait_timeout = _env_float(
            "HERO_LOCAL_DOCKER_STORAGE_WAIT_TIMEOUT_SEC", 900.0
        )
        self.storage_wait_interval = _env_float(
            "HERO_LOCAL_DOCKER_STORAGE_WAIT_INTERVAL_SEC", 5.0
        )
        self.data_root = Path(
            os.environ.get(
                "HERO_LOCAL_DOCKER_DATA_ROOT",
                str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hero/docker-data"),
            )
        )
        # Existing cluster submissions already export the selected storage
        # driver in their encoded worker command.  Auto-enabling leases for
        # ``vfs`` lets queued jobs use the safe lifecycle without requiring a
        # re-submit; an explicitly empty lease-dir remains an opt-out.  Local
        # IDE jobs use overlay2 and therefore never enter this branch.
        storage_driver = os.environ.get(
            "HERO_LOCAL_DOCKER_STORAGE_DRIVER", ""
        ).strip().lower()
        lease_dir_value = (
            os.environ.get("SWE_DOCKER_IMAGE_LEASE_DIR")
            if storage_driver == "vfs"
            else None
        )
        if lease_dir_value is None and storage_driver == "vfs":
            lease_dir_value = str(self.data_root / ".swe-image-leases")
        self.image_lease_dir = (
            Path(lease_dir_value) if lease_dir_value not in (None, "") else None
        )
        # VFS updates Docker's shared image metadata during load/tag/rmi.  A
        # per-image lease cannot prevent an rmi for image B from racing with a
        # load/tag for image A, so coordinate all mutating image operations
        # against this daemon-wide lock.  Loads/tags take SH and can still run
        # concurrently; rmi takes EX.  overlay2 leaves the path unset and thus
        # keeps its established Docker command path unchanged.
        self.image_mutation_lock_path = (
            self.image_lease_dir / ".daemon-image-mutations.lock"
            if self.image_lease_dir is not None
            else None
        )
        self.remove_images = _env_bool("HERO_LOCAL_DOCKER_RMI", True)
        self._image_locks: dict[str, asyncio.Lock] = {}
        self._image_locks_guard = asyncio.Lock()
        self._image_refcounts: dict[str, int] = {}
        self._image_leases: dict[str, list[_InterprocessImageLease]] = {}
        self._image_lease_keys: dict[str, str] = {}
        self._idle_images: OrderedDict[str, None] = OrderedDict()
        self._idle_guard = asyncio.Lock()

        # One gateway-wide /proc sampler replaces one ``docker top`` process
        # per container.  RUN_DIR is already unique per training invocation.
        telemetry_value = os.environ.get("HERO_LOCAL_DOCKER_TELEMETRY_PATH")
        if telemetry_value is None:
            run_dir = os.environ.get("RUN_DIR")
            telemetry_value = (
                str(Path(run_dir) / "resource_samples.jsonl") if run_dir else ""
            )
        self.telemetry_path = Path(telemetry_value) if telemetry_value else None
        # Keep one buffered descriptor instead of synchronously opening and
        # closing a shared-filesystem file for every container lifecycle event.
        # Critical kill records and batch completion are still flushed eagerly.
        self.telemetry_flush_interval = _env_float(
            "HERO_LOCAL_DOCKER_TELEMETRY_FLUSH_INTERVAL_SEC", 30.0, minimum=0.0
        )
        self._telemetry_sink: TextIO | None = None
        self._telemetry_last_flush = time.monotonic()
        # Accept the previous watchdog variable as a compatibility alias.  The
        # new name reflects that one process-free sampler covers memory and
        # PID pressure together.
        legacy_resource_interval = _env_float(
            "HERO_LOCAL_DOCKER_MEMORY_CHECK_INTERVAL_SEC", 5.0, minimum=0.1
        )
        self.resource_check_interval = _env_float(
            "HERO_LOCAL_DOCKER_RESOURCE_CHECK_INTERVAL_SEC",
            legacy_resource_interval,
            minimum=0.1,
        )
        default_emergency_gb = (
            min(100.0, max(4.0, 0.10 * host_total_gb)) if host_total_gb > 0 else 100.0
        )
        self.emergency_available_memory_gb = _env_float(
            "HERO_LOCAL_DOCKER_EMERGENCY_AVAILABLE_MEMORY_GB",
            default_emergency_gb,
        )
        self._tracked_containers: dict[str, _TrackedContainer] = {}
        self._resource_monitor_task: asyncio.Task[None] | None = None
        self._telemetry_error_logged = False
        self._emergency_pressure_samples = 0
        self._cgroup_budget_guard = asyncio.Lock()
        self._cgroup_budget_ready: bool | None = None

    async def acquire_active(self, reservation_gb: float = 0.0) -> float:
        await self.active_sem.acquire()
        try:
            return await self._wait_for_memory_admission(reservation_gb)
        except BaseException:
            self.active_sem.release()
            raise

    def release_admission_reservation(self, reservation_gb: float) -> None:
        if reservation_gb <= 0:
            return
        # This method is called from the same event loop as admission.  There
        # is no await between the guarded admission check and its increment,
        # so this synchronous decrement cannot interleave with that section.
        self._reserved_memory_gb = max(0.0, self._reserved_memory_gb - reservation_gb)
        self._write_telemetry(
            "admission_reservation_released",
            released_gb=reservation_gb,
            reserved_gb=self._reserved_memory_gb,
        )

    def release_active(self, reservation_gb: float = 0.0) -> None:
        self.release_admission_reservation(reservation_gb)
        self.active_sem.release()

    async def _wait_for_memory_admission(self, reservation_gb: float) -> float:
        """Atomically reserve startup headroom before admitting a container."""
        if self.min_available_memory_gb <= 0:
            return 0.0
        reservation_gb = max(0.0, reservation_gb)

        waiting = False
        while True:
            async with self._admission_guard:
                available_gb = _host_available_memory_gb()
                projected_gb = available_gb - self._reserved_memory_gb - reservation_gb
                admitted = projected_gb >= self.min_available_memory_gb
                if admitted:
                    self._reserved_memory_gb += reservation_gb
            if admitted:
                if waiting:
                    logger.info(
                        "local-Docker memory admission resumed: host available "
                        "%.1f GiB, startup reservations %.1f GiB",
                        available_gb,
                        self._reserved_memory_gb,
                    )
                self._write_telemetry(
                    "admission_reserved",
                    host_available_gb=available_gb,
                    requested_gb=reservation_gb,
                    reserved_gb=self._reserved_memory_gb,
                )
                return reservation_gb
            if not waiting:
                logger.warning(
                    "pausing new local-Docker container admission: host available "
                    "%.1f GiB - reservations %.1f GiB - request %.1f GiB "
                    "< %.1f GiB",
                    available_gb,
                    self._reserved_memory_gb,
                    reservation_gb,
                    self.min_available_memory_gb,
                )
                waiting = True
            await asyncio.sleep(self.memory_admission_interval)

    async def ensure_parent_cgroup_budget(self) -> bool:
        """Initialize the Docker parent memcg after model/Ray have settled.

        The guard prepares writable cgroup files but intentionally does not
        guess how much memory the already-resident trainer needs.  The first
        rollout container computes that budget here. Any error is fail-open:
        callers omit kernel-only Docker flags and retain admission + /proc
        enforcement instead of making the training job unstartable.
        """
        if self._cgroup_budget_ready is not None:
            return self._cgroup_budget_ready

        async with self._cgroup_budget_guard:
            if self._cgroup_budget_ready is not None:
                return self._cgroup_budget_ready
            try:
                memory_limit_path = Path(
                    os.environ["HERO_LOCAL_DOCKER_PARENT_MEMORY_LIMIT_FILE"]
                )
                memory_usage_path = Path(
                    os.environ["HERO_LOCAL_DOCKER_PARENT_MEMORY_USAGE_FILE"]
                )
                pids_limit_path = Path(
                    os.environ["HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT_FILE"]
                )
                pids_current_path = Path(
                    os.environ["HERO_LOCAL_DOCKER_PARENT_PIDS_CURRENT_FILE"]
                )
                cgroup_root = Path(
                    os.environ.get("HERO_LOCAL_DOCKER_CGROUP_ROOT", "/sys/fs/cgroup")
                )
                root_limit, root_usage, reclaim_credit = _self_cgroup_memory_metrics(
                    cgroup_root=cgroup_root
                )
                if root_limit is None or root_limit <= 0:
                    raise RuntimeError("job memory cgroup limit is unavailable")
                if root_usage is None:
                    raise RuntimeError("job memory cgroup usage is unavailable")

                gib = 1024**3
                host_total_bytes = int(_host_total_memory_gb() * gib)
                # cgroup v1 represents "unlimited" with a huge integer rather
                # than the v2 string "max". Bound it by physical RAM before
                # deriving a reserve or it could collapse the parent to 1 GiB.
                if host_total_bytes > 0:
                    root_limit = min(root_limit, host_total_bytes)
                root_limit_gb = root_limit / gib
                # Keep the original 300-GiB safety intent on the 1.5-TiB
                # nodes, but clamp it to 40% on smaller allocations.
                configured_reserve_gb = self.min_available_memory_gb
                reserve_gb = min(
                    max(0.20 * root_limit_gb, configured_reserve_gb),
                    0.40 * root_limit_gb,
                )
                # The helper discounts inactive cache fully and active cache
                # by half. This matches MemAvailable's reclaim-aware intent
                # while protecting hot page cache from aggressive eviction.
                reclaim_credit = min(root_usage, max(0, reclaim_credit))
                root_nonreclaimable = max(0, root_usage - reclaim_credit)
                host_available_bytes = int(_host_available_memory_gb() * gib)
                reserve_bytes = int(reserve_gb * gib)
                # Host MemAvailable captures memory outside this cgroup; the
                # job-working-set term protects against a misleadingly large
                # host value. Choose the more conservative headroom.
                host_budget = host_available_bytes - reserve_bytes
                job_budget = root_limit - root_nonreclaimable - reserve_bytes
                budget = max(0, min(host_budget, job_budget))

                parent_usage = _read_limit(memory_usage_path) or 0
                # Never write a hard limit below live usage. A small margin
                # avoids immediate reclaim thrash while old containers drain.
                budget = max(budget, parent_usage + gib)
                budget = min(budget, root_limit)
                if budget <= 0:
                    raise RuntimeError("computed Docker cgroup budget is zero")

                memory_limit_path.write_text(str(budget), encoding="utf-8")
                observed_budget = _read_limit(memory_limit_path)
                # cgroup v1 may round a byte limit down by part of one page.
                # Accept only that documented granularity, then report/use the
                # effective value rather than pretending the request was exact.
                page_size = max(4096, int(os.sysconf("SC_PAGE_SIZE")))
                if (
                    observed_budget is None
                    or observed_budget <= 0
                    or observed_budget + page_size < budget
                ):
                    raise RuntimeError(
                        "memory budget verification failed: "
                        f"requested={budget}, observed={observed_budget}"
                    )
                budget = min(budget, observed_budget)

                per_container_pids = _env_int("HERO_LOCAL_DOCKER_PIDS_LIMIT", 2048)
                parent_pids_unit = per_container_pids or 2048
                parent_pids = _env_int(
                    "HERO_LOCAL_DOCKER_PARENT_PIDS_LIMIT",
                    min(
                        self.active_concurrency * parent_pids_unit,
                        131072,
                    ),
                    minimum=1,
                )
                pids_current = _read_limit(pids_current_path) or 0
                parent_pids = max(parent_pids, pids_current + 128)
                pids_limit_path.write_text(str(parent_pids), encoding="utf-8")
                observed_pids = _read_limit(pids_limit_path)
                if observed_pids is None or observed_pids < parent_pids:
                    raise RuntimeError(
                        f"pids budget verification failed: {observed_pids}"
                    )

                self._cgroup_budget_ready = True
                self._write_telemetry(
                    "parent_cgroup_budget",
                    enabled=True,
                    memory_budget_bytes=budget,
                    parent_memory_usage_bytes=parent_usage,
                    root_memory_limit_bytes=root_limit,
                    root_memory_usage_bytes=root_usage,
                    root_reclaim_credit_bytes=reclaim_credit,
                    root_nonreclaimable_bytes=root_nonreclaimable,
                    reserve_bytes=reserve_bytes,
                    parent_pids_limit=parent_pids,
                )
                logger.info(
                    "initialized local-Docker parent cgroup: memory %.1f GiB, "
                    "pids %d, reserve %.1f GiB",
                    budget / gib,
                    parent_pids,
                    reserve_gb,
                )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                self._cgroup_budget_ready = False
                self._write_telemetry(
                    "parent_cgroup_budget",
                    enabled=False,
                    error=str(exc),
                )
                logger.error(
                    "cannot initialize local-Docker parent cgroup; continuing "
                    "with atomic admission and /proc monitoring: %s",
                    exc,
                )
            return self._cgroup_budget_ready

    def disable_kernel_cgroups(self, reason: str) -> None:
        """Make subsequent starts use the compatible userspace safeguards."""
        self._cgroup_budget_ready = False
        self._write_telemetry(
            "kernel_cgroup_disabled",
            reason=reason,
        )

    @asynccontextmanager
    async def run_slot(self) -> AsyncIterator[None]:
        async with self.run_sem:
            yield

    def register_container(
        self,
        *,
        container_name: str,
        session_id: str,
        image: str,
        init_pid: int,
        memory_limit_bytes: int,
        pids_limit: int,
        cgroup_limits_verified: bool,
    ) -> bool:
        """Register a live container with the process-free resource sampler."""
        namespace = _pid_namespace_inode(init_pid)
        if namespace is None:
            logger.warning(
                "cannot register %s resource monitor: PID %s disappeared",
                container_name,
                init_pid,
            )
            return False
        self._tracked_containers[container_name] = _TrackedContainer(
            container_name=container_name,
            session_id=session_id,
            image=image,
            init_pid=init_pid,
            pid_namespace=namespace,
            memory_limit_bytes=memory_limit_bytes,
            pids_limit=pids_limit,
            cgroup_limits_verified=cgroup_limits_verified,
        )
        self._write_telemetry(
            "container_registered",
            container=container_name,
            session_id=session_id,
            image=image,
            init_pid=init_pid,
            memory_limit_bytes=memory_limit_bytes,
            pids_limit=pids_limit,
            cgroup_limits_verified=cgroup_limits_verified,
        )
        task = self._resource_monitor_task
        if task is None or task.done():
            self._resource_monitor_task = asyncio.create_task(
                self._resource_monitor_loop(),
                name="hero-local-docker-resource-monitor",
            )
        return True

    def unregister_container(self, container_name: str) -> None:
        record = self._tracked_containers.pop(container_name, None)
        if record is not None:
            self._write_telemetry(
                "container_unregistered",
                container=container_name,
                session_id=record.session_id,
                image=record.image,
            )

    async def _resource_monitor_loop(self) -> None:
        """Sample all managed containers in one /proc walk and no subprocesses."""
        try:
            while self._tracked_containers:
                self._sample_resources()
                await asyncio.sleep(self.resource_check_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Resource monitoring is a fail-open safety layer.  Kernel cgroup
            # limits (when enabled) remain authoritative; admission still uses
            # MemAvailable even if a malformed /proc entry trips this loop.
            logger.exception("local-Docker /proc resource monitor failed")
        finally:
            self._flush_telemetry()
            if self._resource_monitor_task is asyncio.current_task():
                self._resource_monitor_task = None

    def _sample_resources(self) -> dict[str, _ProcessUsage]:
        records = list(self._tracked_containers.values())
        usage_by_namespace = _scan_pid_namespaces(
            {record.pid_namespace for record in records}
        )
        usage_by_name: dict[str, _ProcessUsage] = {}
        victims: list[tuple[_TrackedContainer, _ProcessUsage, str]] = []
        samples: list[dict[str, Any]] = []

        for record in records:
            usage = usage_by_namespace.get(record.pid_namespace, _ProcessUsage())
            usage_by_name[record.container_name] = usage
            # Once the effective kernel memory+pids cgroups have been
            # verified, they are the authoritative per-container counters.
            # Summed process VmRSS double-counts shared pages and must not be a
            # second hard limit. /proc remains useful for telemetry and the
            # host-wide emergency backstop below.
            userspace_limits = not record.cgroup_limits_verified
            memory_over = bool(
                userspace_limits
                and
                record.memory_limit_bytes > 0
                and usage.rss_bytes > record.memory_limit_bytes
            )
            # pids.max accounts tasks (threads), not merely process leaders.
            pids_over = bool(
                userspace_limits
                and record.pids_limit > 0
                and usage.threads >= record.pids_limit
            )
            record.consecutive_over_limit = (
                record.consecutive_over_limit + 1 if memory_over or pids_over else 0
            )
            if record.consecutive_over_limit >= 2 and not record.killed:
                reason = "memory_limit" if memory_over else "pids_limit"
                victims.append((record, usage, reason))
            samples.append(
                {
                    "container": record.container_name,
                    "session_id": record.session_id,
                    "image": record.image,
                    "init_pid": record.init_pid,
                    "rss_bytes": usage.rss_bytes,
                    "processes": usage.processes,
                    "threads": usage.threads,
                    "memory_limit_bytes": record.memory_limit_bytes,
                    "pids_limit": record.pids_limit,
                    "cgroup_limits_verified": record.cgroup_limits_verified,
                }
            )

        try:
            available_gb = _host_available_memory_gb()
        except RuntimeError:
            available_gb = -1.0
        under_emergency_pressure = (
            available_gb >= 0
            and self.emergency_available_memory_gb > 0
            and available_gb < self.emergency_available_memory_gb
        )
        self._emergency_pressure_samples = (
            self._emergency_pressure_samples + 1 if under_emergency_pressure else 0
        )
        if self._emergency_pressure_samples >= 2 and records and not victims:
            live_records = [record for record in records if not record.killed]
            largest = (
                max(
                    live_records,
                    key=lambda item: usage_by_name[item.container_name].rss_bytes,
                )
                if live_records
                else None
            )
            if largest is not None:
                largest_usage = usage_by_name[largest.container_name]
                if largest_usage.rss_bytes > 0:
                    victims.append((largest, largest_usage, "host_memory_emergency"))

        self._write_telemetry(
            "resource_sample",
            host_available_gb=available_gb,
            reserved_gb=self._reserved_memory_gb,
            total_rss_bytes=sum(item.rss_bytes for item in usage_by_name.values()),
            total_processes=sum(item.processes for item in usage_by_name.values()),
            total_threads=sum(item.threads for item in usage_by_name.values()),
            containers=samples,
        )
        for record, usage, reason in victims:
            self._kill_container_processes(record, usage, reason)
        return usage_by_name

    def _kill_container_processes(
        self,
        record: _TrackedContainer,
        usage: _ProcessUsage,
        reason: str,
    ) -> None:
        """SIGKILL one PID namespace without needing Docker/runc to fork."""
        record.killed = True
        logger.error(
            "killing runaway SWE container %s without forking: reason=%s, "
            "RSS=%.1f GiB, processes=%d, threads=%d",
            record.container_name,
            reason,
            usage.rss_bytes / 1024**3,
            usage.processes,
            usage.threads,
        )
        killed = 0
        # Kill the namespace init last so descendants remain discoverable long
        # enough to signal. Never signal the gateway itself even if /proc data
        # is unexpectedly inconsistent.
        current_pid = os.getpid()
        ordered = sorted(usage.pids, key=lambda pid: pid == record.init_pid)
        for pid in ordered:
            if pid in (1, current_pid):
                continue
            # The process can exit after the /proc scan and its numeric PID can
            # be reused before signal delivery.  Re-check the namespace at the
            # last possible moment so a stale sample can never target an
            # unrelated host process.
            if _pid_namespace_inode(pid) != record.pid_namespace:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError):
                continue
        self._write_telemetry(
            "container_killed",
            container=record.container_name,
            session_id=record.session_id,
            image=record.image,
            reason=reason,
            rss_bytes=usage.rss_bytes,
            processes=usage.processes,
            threads=usage.threads,
            signalled_processes=killed,
        )

    def _write_telemetry(self, event: str, **fields: Any) -> None:
        path = self.telemetry_path
        if path is None:
            return
        payload = {
            "timestamp": time.time(),
            "event": event,
            "run_id": os.environ.get("HERO_RUN_ID", "hero"),
            **fields,
        }
        try:
            sink = self._telemetry_sink
            if sink is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                sink = path.open("a", encoding="utf-8", buffering=256 * 1024)
                self._telemetry_sink = sink
            sink.write(json.dumps(payload, sort_keys=True) + "\n")
            now = time.monotonic()
            if (
                event == "container_killed"
                or self.telemetry_flush_interval <= 0
                or now - self._telemetry_last_flush >= self.telemetry_flush_interval
            ):
                self._flush_telemetry(now=now)
        except (OSError, ValueError):
            self._disable_telemetry_sink()
            if not self._telemetry_error_logged:
                self._telemetry_error_logged = True
                logger.warning(
                    "cannot write local-Docker resource telemetry to %s",
                    path,
                    exc_info=True,
                )

    def _flush_telemetry(self, *, now: float | None = None) -> None:
        sink = self._telemetry_sink
        if sink is None:
            return
        try:
            sink.flush()
            self._telemetry_last_flush = time.monotonic() if now is None else now
        except (OSError, ValueError):
            self._disable_telemetry_sink()
            if not self._telemetry_error_logged:
                self._telemetry_error_logged = True
                logger.warning(
                    "cannot flush local-Docker resource telemetry to %s",
                    self.telemetry_path,
                    exc_info=True,
                )

    def _disable_telemetry_sink(self) -> None:
        sink = self._telemetry_sink
        self._telemetry_sink = None
        self.telemetry_path = None
        if sink is not None:
            try:
                sink.close()
            except (OSError, ValueError):
                pass

    def _lease_key(self, image: str, image_tar: Path | None = None) -> str:
        # The image ref is always available, whereas some Slime callers can
        # legitimately use an already-loaded image without an image_tar.  A
        # tar-dependent key would let that caller bypass a tar-backed user's
        # lease for the same image. HERO and Slime therefore both key only on
        # the registry-normalized ref; keep image_tar in the signature because
        # callers still need it for loading.
        _ = image_tar
        return _normalize_image_ref(image)

    def _new_image_lease(
        self, image: str, image_tar: Path | None = None
    ) -> _InterprocessImageLease:
        key = self._lease_key(image, image_tar)
        self._image_lease_keys[image] = key
        path = None
        if self.image_lease_dir is not None:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            path = self.image_lease_dir / f"{digest}.lock"
        return _InterprocessImageLease(path)

    @asynccontextmanager
    async def _daemon_image_mutation(
        self, *, exclusive: bool, blocking: bool = True
    ) -> AsyncIterator[bool]:
        """Coordinate VFS image metadata mutations across worker processes."""
        lease = _InterprocessImageLease(self.image_mutation_lock_path)
        try:
            if exclusive:
                acquired = await lease.acquire_exclusive(blocking=blocking)
            else:
                acquired = await lease.acquire_shared()
            yield acquired
        finally:
            await lease.release()

    async def acquire_image(
        self,
        image: str,
        image_tar: Path | None,
        command: DockerCommand,
    ) -> None:
        await self.wait_for_storage("loading a SWE image", command)
        lock = await self._image_lock(image)
        async with lock:
            async with self._idle_guard:
                self._idle_images.pop(image, None)
            lease = self._new_image_lease(image, image_tar)
            try:
                await lease.acquire_shared()
                if not await self._image_loaded(
                    image, command, require_definitive_missing=self.image_lease_dir is not None
                ):
                    if self.image_lease_dir is not None and self._image_refcounts.get(image, 0) > 0:
                        # We hold the per-image asyncio lock. Existing users
                        # need it to release their SH leases, so waiting for EX
                        # here would block the very releases we depend on.
                        raise RuntimeError(
                            f"Docker image {image!r} is missing while local users still hold leases; "
                            "refusing an exclusive reload to avoid a lock cycle"
                        )
                    if image_tar is None:
                        raise RuntimeError(
                            f"Docker image {image!r} is not loaded and no local image_tar was configured"
                        )
                    # Shared leases protect already-loaded users.  Loading a
                    # missing image needs an exclusive lease so HERO and
                    # Slime cannot both load the same tar.  Re-check after the
                    # non-atomic SH -> EX conversion because another process
                    # may have won the upgrade race.
                    await lease.upgrade_exclusive()
                    if not await self._image_loaded(image, command):
                        if self.image_mutation_lock_path is not None:
                            await asyncio.to_thread(validate_image_tar, image_tar, image)
                        else:
                            validate_image_tar(image_tar, image)
                        if self.image_mutation_lock_path is None:
                            # Preserve the established overlay2 sequencing:
                            # load_sem covers only docker load, while tag is
                            # resolved immediately afterwards.
                            async with self.load_sem:
                                if not await self._image_loaded(image, command):
                                    await self._load_image_tar(image_tar, command)
                            await self._ensure_expected_image_tag(
                                image,
                                image_tar,
                                command,
                                retry_visibility=False,
                            )
                        else:
                            # Take the in-process load slot before the daemon
                            # SH lock.  Queued local loads must not hold SH and
                            # starve an rmi waiting for EX.
                            async with self.load_sem:
                                async with self._daemon_image_mutation(
                                    exclusive=False
                                ):
                                    if not await self._image_loaded(image, command):
                                        await self._load_image_tar(image_tar, command)
                                    await self._ensure_expected_image_tag(
                                        image,
                                        image_tar,
                                        command,
                                        retry_visibility=True,
                                    )
                    await lease.downgrade_shared()
                self._image_refcounts[image] = (
                    self._image_refcounts.get(image, 0) + 1
                )
                self._image_leases.setdefault(image, []).append(lease)
            except BaseException:
                await lease.release()
                raise

    async def _load_image_tar(
        self, image_tar: Path, command: DockerCommand
    ) -> None:
        rc, stdout, stderr = await command(
            "docker",
            "load",
            "-i",
            str(image_tar),
            timeout=1800.0,
            capture=True,
        )
        if rc != 0:
            raise RuntimeError(
                f"docker load failed for {image_tar}: rc={rc}: "
                f"{(stderr or stdout or '')[-1000:]}"
            )

    async def _ensure_expected_image_tag(
        self,
        image: str,
        image_tar: Path,
        command: DockerCommand,
        *,
        retry_visibility: bool,
    ) -> None:
        if await self._image_loaded(image, command):
            return

        # docker load registers the manifest's RAFT RepoTag (e.g.
        # ``docker.m.daocloud.io/swebench/foo:latest``), which newer Docker
        # does NOT alias to the short form.
        for full_tag in _image_tar_tags(image_tar):
            if _normalize_image_ref(full_tag) != _normalize_image_ref(image):
                continue
            if not await self._image_loaded(full_tag, command):
                continue
            rc, _, stderr = await command(
                "docker",
                "tag",
                full_tag,
                image,
                timeout=60.0,
                capture=True,
            )
            if rc == 0:
                visible = (
                    await self._wait_for_vfs_image_visibility(image, command)
                    if retry_visibility
                    else await self._image_loaded(image, command)
                )
                if visible:
                    logger.info(
                        "docker load of %s produced %s; aliased to %s",
                        image_tar,
                        full_tag,
                        image,
                    )
                    return
            else:
                logger.warning(
                    "docker tag %s -> %s failed rc=%s: %s",
                    full_tag,
                    image,
                    rc,
                    (stderr or "")[-400:],
                )

        raise RuntimeError(
            f"docker load of {image_tar} did not produce expected tag {image!r}"
        )

    async def _wait_for_vfs_image_visibility(
        self, image: str, command: DockerCommand
    ) -> bool:
        last_rc = -1
        last_stderr = ""
        for attempt in range(1, _VFS_IMAGE_VISIBILITY_ATTEMPTS + 1):
            last_rc, _, stderr = await command(
                "docker",
                "image",
                "inspect",
                image,
                timeout=30.0,
                capture=True,
            )
            if last_rc == 0:
                return True
            last_stderr = (stderr or "")[-400:]
            if attempt < _VFS_IMAGE_VISIBILITY_ATTEMPTS:
                await asyncio.sleep(_VFS_IMAGE_VISIBILITY_RETRY_SECONDS)
        logger.warning(
            "Docker image %s remained invisible after a successful tag and "
            "%s inspect attempts; last rc=%s: %s",
            image,
            _VFS_IMAGE_VISIBILITY_ATTEMPTS,
            last_rc,
            last_stderr,
        )
        return False

    async def release_image(self, image: str, command: DockerCommand) -> None:
        lock = await self._image_lock(image)
        async with lock:
            count = self._image_refcounts.get(image, 0)
            if count <= 0:
                logger.warning("Docker image refcount underflow for %s", image)
                return
            leases = self._image_leases.get(image, [])
            lease = leases.pop() if leases else None
            if not leases:
                self._image_leases.pop(image, None)
            if lease is not None:
                await lease.release()
            if count > 1:
                self._image_refcounts[image] = count - 1
                return
            self._image_refcounts.pop(image, None)
            if self.remove_images:
                async with self._idle_guard:
                    self._idle_images[image] = None
                    self._idle_images.move_to_end(image)
        await self._evict_idle(command)

    async def wait_for_storage(self, operation: str, command: DockerCommand) -> None:
        if self.min_free_gb <= 0:
            return
        deadline = time.monotonic() + self.storage_wait_timeout
        while self._free_gb() < self.min_free_gb:
            evicted = await self._evict_idle(command, force_all=True)
            free_gb = self._free_gb()
            if free_gb >= self.min_free_gb:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Docker storage stayed below {self.min_free_gb:.1f} GB while "
                    f"{operation}: {free_gb:.1f} GB free after evicting {evicted} images"
                )
            logger.warning(
                "pausing %s: %.1f GB free, %.1f GB required",
                operation,
                free_gb,
                self.min_free_gb,
            )
            await asyncio.sleep(self.storage_wait_interval)

    async def _evict_idle(
        self, command: DockerCommand, *, force_all: bool = False
    ) -> int:
        evicted = 0
        busy_images: set[str] = set()
        while True:
            async with self._idle_guard:
                over_limit = len(self._idle_images) > self.image_cache_size
                disk_pressure = self._free_gb() < self.min_free_gb
                if not self._idle_images or not (
                    force_all or over_limit or disk_pressure
                ):
                    return evicted
                image = next(
                    (
                        candidate
                        for candidate in self._idle_images
                        if candidate not in busy_images
                    ),
                    None,
                )
                if image is None:
                    return evicted
                self._idle_images.pop(image, None)
            lock = await self._image_lock(image)
            async with lock:
                if self._image_refcounts.get(image, 0) > 0:
                    continue
                key = self._image_lease_keys.get(image, self._lease_key(image))
                path = None
                if self.image_lease_dir is not None:
                    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                    path = self.image_lease_dir / f"{digest}.lock"
                eviction_lease = _InterprocessImageLease(path)
                if not await eviction_lease.acquire_exclusive(blocking=False):
                    # A HERO or Slime sandbox still owns this image.  Skip it
                    # and keep scanning: one slow grader must not pin every
                    # unrelated idle image behind it and fill the VFS store.
                    busy_images.add(image)
                    async with self._idle_guard:
                        self._idle_images[image] = None
                    continue
                try:
                    async with self._daemon_image_mutation(
                        exclusive=True, blocking=force_all or disk_pressure
                    ) as mutation_acquired:
                        if not mutation_acquired:
                            # Routine VFS cache trimming must not pin this
                            # image's EX lease behind an unrelated slow load.
                            # Keep it eligible for a later pass and let cached
                            # users proceed. Storage-pressure eviction still
                            # waits; overlay2 has no daemon lease to contend on.
                            busy_images.add(image)
                            async with self._idle_guard:
                                self._idle_images[image] = None
                            continue
                        remove_target = image
                        remove_args = ["docker", "rmi", image]
                        if self.image_lease_dir is not None:
                            # Local tars can carry a registry-prefixed RepoTag and
                            # HERO adds a short alias. Removing only the alias
                            # leaves every VFS layer behind. With an exclusive
                            # cross-process lease it is safe to remove the image ID
                            # (and therefore all aliases) on this isolated daemon.
                            inspect_rc, image_id, _ = await command(
                                "docker",
                                "image",
                                "inspect",
                                "--format",
                                "{{.Id}}",
                                image,
                                timeout=30.0,
                                capture=True,
                            )
                            if inspect_rc == 0 and (image_id or "").strip():
                                remove_target = (image_id or "").strip()
                                remove_args = [
                                    "docker",
                                    "rmi",
                                    "-f",
                                    remove_target,
                                ]
                        rc, _, stderr = await command(
                            *remove_args, timeout=180.0, capture=True
                        )
                        if rc == 0 or not await self._image_loaded(
                            remove_target, command
                        ):
                            evicted += 1
                            continue
                        logger.warning(
                            "failed to evict Docker image %s via %s: %s",
                            image,
                            remove_target,
                            stderr,
                        )
                        busy_images.add(image)
                        async with self._idle_guard:
                            self._idle_images[image] = None
                finally:
                    await eviction_lease.release()

    async def _image_loaded(
        self, image: str, command: DockerCommand, *, require_definitive_missing: bool = False
    ) -> bool:
        rc, stdout, stderr = await command(
            "docker", "image", "inspect", image, timeout=30.0, capture=True
        )
        if rc != 0 and require_definitive_missing:
            diagnostic = (stderr or stdout or "")
            if rc != 1 or not re.search(r"(?i)no such (?:image|object):", diagnostic):
                raise RuntimeError(
                    f"Docker image inspect failed for {image!r}, rc={rc}; "
                    f"not treating this as a missing image: {diagnostic[-500:]}"
                )
        return rc == 0

    async def _image_lock(self, image: str) -> asyncio.Lock:
        async with self._image_locks_guard:
            return self._image_locks.setdefault(image, asyncio.Lock())

    def _free_gb(self) -> float:
        try:
            return shutil.disk_usage(self.data_root).free / 1e9
        except OSError:
            return 0.0


_COORDINATOR: LocalDockerCoordinator | None = None


def get_coordinator() -> LocalDockerCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = LocalDockerCoordinator()
    return _COORDINATOR


class ManagedDockerRuntime(DockerRuntime):
    """HERO DockerRuntime with local tar loading and bounded lifecycle."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._coordinator = get_coordinator()
        self._active_acquired = False
        self._admission_reservation_gb = 0.0
        self._reservation_release_task: asyncio.Task[None] | None = None
        self._image_acquired = False
        self._resource_registered = False
        self._container_init_pid: int | None = None
        # Teardown must outlive a cancelled request.  Keep one shared task so
        # concurrent/repeated stop() calls cannot skip or duplicate cleanup.
        self._stop_task: asyncio.Task[None] | None = None
        # A failed foreground removal must not turn into a permanent orphan.
        # This task retains the runtime (and therefore its active/image
        # ownership) until Docker is healthy enough to confirm deletion.
        self._cleanup_retry_task: asyncio.Task[None] | None = None
        # Docker returns immediately while VFS continues deleting a copied
        # writable tree. Remember that state so the reaper polls for absence
        # instead of submitting another destructive request every cycle.
        self._removal_in_progress_since: float | None = None

    async def exec(self, command: str, **kwargs):  # type: ignore[no-untyped-def]
        limit_gb = _env_float("HERO_LOCAL_DOCKER_ADDRESS_SPACE_LIMIT_GB", 128.0)
        if limit_gb > 0:
            command = f"ulimit -v {int(limit_gb * 1024**2)}; {command}"
        return await super().exec(command, **kwargs)

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("docker runtime was already destroyed")
        vfs = os.environ.get("HERO_LOCAL_DOCKER_STORAGE_DRIVER", "").strip().lower() == "vfs"
        phase_started = time.monotonic()

        def record_phase(phase: str) -> None:
            nonlocal phase_started
            now = time.monotonic()
            if vfs:
                self._coordinator._write_telemetry(
                    "container_start_phase", session_id=self.session_id,
                    container=self._container_name, phase=phase,
                    elapsed_sec=now - phase_started,
                )
            phase_started = now
        image_tar_value = self.spec.kwargs.get("image_tar")
        image_tar = Path(str(image_tar_value)) if image_tar_value else None
        if image_tar is not None and not image_tar.is_absolute():
            # Release JSONL paths are relative to the HERO repository, not cwd.
            image_tar = Path(__file__).resolve().parents[3] / image_tar
        memory_limit_gb = _env_float("HERO_LOCAL_DOCKER_MEMORY_LIMIT_GB", 0.0)
        memory_limit_mb = int(memory_limit_gb * 1024)
        if self.spec.memory_mb is not None:
            memory_limit_mb = (
                min(self.spec.memory_mb, memory_limit_mb)
                if memory_limit_mb
                else self.spec.memory_mb
            )
        # Default to the actual per-container hard cap. This reservation is
        # temporary (see the hold task below), so light images can still ramp
        # to ACTIVE_CONCURRENCY=64 after their cold-start window.
        reservation_gb = _env_float(
            "HERO_LOCAL_DOCKER_ADMISSION_RESERVATION_GB",
            memory_limit_mb / 1024 if memory_limit_mb else 8.0,
        )
        cgroup_mode = (
            os.environ.get("HERO_LOCAL_DOCKER_CGROUP_MODE", "rootless-fallback")
            .strip()
            .lower()
        )
        kernel_cgroups = cgroup_mode == "kernel"
        cgroup_parent = os.environ.get("HERO_LOCAL_DOCKER_CGROUP_PARENT", "").strip()
        pids_limit = (
            _env_int("HERO_LOCAL_DOCKER_PIDS_LIMIT", 2048) if kernel_cgroups else 0
        )
        try:
            self._admission_reservation_gb = await self._coordinator.acquire_active(
                0.0 if vfs else reservation_gb
            )
            self._active_acquired = True
            record_phase("capacity_and_host_floor")
            await self._coordinator.acquire_image(
                self.spec.image, image_tar, self._docker_command
            )
            self._image_acquired = True
            record_phase("image_prepare")
            await self._coordinator.wait_for_storage(
                "starting a SWE container", self._docker_command
            )
            if kernel_cgroups:
                budget_ready = await self._coordinator.ensure_parent_cgroup_budget()
                if not budget_ready:
                    if _env_bool("HERO_LOCAL_DOCKER_REQUIRE_CGROUP_LIMITS", False):
                        raise RuntimeError(
                            "kernel cgroup mode was requested but the parent "
                            "resource budget could not be initialized"
                        )
                    kernel_cgroups = False
                    cgroup_parent = ""
                    pids_limit = 0
            create_args = [
                "docker",
                "create",
                "--name",
                self._container_name,
                "--label",
                "ai.hero.namespace=swegym-local-docker",
                "--label",
                f"ai.hero.run_id={os.environ.get('HERO_RUN_ID', 'hero')}",
                "--label",
                f"ai.hero.session_id={self.session_id}",
            ]
            # Keep numerical libraries inside evaluator containers from sizing
            # a pool against all host CPUs. Eight remains ample for repo tests;
            # 0 explicitly disables these defaults.
            legacy_thread_cap = os.environ.get(
                "HERO_LOCAL_DOCKER_TEST_THREADS", ""
            ).strip()
            thread_cap = _env_int(
                "HERO_LOCAL_DOCKER_NUMERIC_THREADS",
                int(legacy_thread_cap) if legacy_thread_cap else 8,
            )
            if thread_cap > 0:
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    create_args.extend(["-e", f"{name}={thread_cap}"])
            if kernel_cgroups:
                if cgroup_parent:
                    create_args.extend(["--cgroup-parent", cgroup_parent])
                if pids_limit > 0:
                    create_args.extend(["--pids-limit", str(pids_limit)])
            if not self.spec.allow_internet:
                create_args.extend(["--network", "none"])
            elif self.spec.network:
                create_args.extend(["--network", self.spec.network])
            if self.spec.gpus > 0:
                create_args.extend(["--gpus", str(self.spec.gpus)])
            if self.spec.cpus is not None:
                create_args.extend(["--cpus", str(self.spec.cpus)])
            if memory_limit_mb:
                # Always preserve the requested Docker memory setting. Kernel
                # mode enforces it; rootless-fallback mode is contained by the
                # centralized /proc sampler below.
                create_args.extend(["--memory", f"{memory_limit_mb}m"])
            create_args.extend(["-v", f"{self.session_dir}:{self.runtime_session_dir}"])
            for volume in self.spec.kwargs.get("volumes", []):
                create_args.extend(["-v", str(volume)])
            # Most SWE images have no ENTRYPOINT (or an entrypoint wrapper that
            # correctly execs its arguments), so preserve the long-standing
            # ``IMAGE sleep infinity`` command byte-for-byte for them.  A few
            # Multi-SWE archives instead declare ENTRYPOINT ["/bin/bash"].
            # Docker then runs ``/bin/bash sleep infinity``; bash treats
            # ``sleep`` as a script path and the container exits immediately.
            # Supply shell syntax only for that provably incompatible form;
            # retain the image's original ENTRYPOINT itself. This keeps every
            # other image's startup and evaluation semantics unchanged.
            if image_tar is None:
                image_entrypoint = None
            elif vfs:
                image_entrypoint = await asyncio.to_thread(
                    _image_tar_entrypoint, image_tar, self.spec.image
                )
            else:
                image_entrypoint = _image_tar_entrypoint(image_tar, self.spec.image)
            shell_entrypoint_compat = _is_shell_only_entrypoint(image_entrypoint)
            if shell_entrypoint_compat:
                self._coordinator._write_telemetry(
                    "container_entrypoint_compatibility",
                    container=self._container_name,
                    session_id=self.session_id,
                    image=self.spec.image,
                    original_entrypoint=list(image_entrypoint or ()),
                )
                create_args.extend(
                    [self.spec.image, "-c", "exec sleep 2147483647"]
                )
            else:
                create_args.extend([self.spec.image, "sleep", "infinity"])

            def compatibility_args(args: list[str]) -> list[str]:
                result: list[str] = []
                skip_value = False
                for value in args:
                    if skip_value:
                        skip_value = False
                        continue
                    if value in {"--cgroup-parent", "--pids-limit"}:
                        skip_value = True
                        continue
                    result.append(value)
                return result

            record_phase("storage_and_create_args")
            async with self._coordinator.run_slot():
                record_phase("create_slot_wait")
                if vfs:
                    # Preparation is still bounded by active/load slots and
                    # storage pressure checks. Reserve startup memory only
                    # after image preparation and the create-slot wait, while
                    # retaining the atomic host-headroom check before create.
                    self._admission_reservation_gb = await self._coordinator._wait_for_memory_admission(
                        reservation_gb
                    )
                    record_phase("startup_memory_wait")

                async def create_and_start(
                    args: list[str],
                ) -> tuple[str | None, int, str | None]:
                    remove_rc, remove_out, remove_err = await self._docker_command(
                        "docker",
                        "rm",
                        "-f",
                        self._container_name,
                        timeout=self._STOP_TIMEOUT,
                        capture=True,
                    )
                    if vfs:
                        # A CLI timeout does not cancel a Docker daemon
                        # operation. Never issue another same-name create
                        # while deletion is failed, pending, or unobservable.
                        if remove_rc != 0 and not (
                            remove_rc == 1 and _CONTAINER_NOT_FOUND_RE.search(
                                f"{remove_out or ''}\n{remove_err or ''}"
                            )
                        ):
                            return "cleanup", remove_rc, remove_err or remove_out
                        inspect_rc, inspect_out, inspect_err = await self._docker_command(
                            "docker", "container", "inspect", self._container_name,
                            timeout=self._STOP_TIMEOUT, capture=True,
                        )
                        if not (inspect_rc == 1 and _CONTAINER_NOT_FOUND_RE.search(
                            f"{inspect_out or ''}\n{inspect_err or ''}"
                        )):
                            return "cleanup", inspect_rc or 1, (
                                "container removal not confirmed; refusing same-name create: "
                                + (inspect_err or "container still present or inspect failed")
                            )
                    rc, _, stderr = await self._docker_command(
                        *args, timeout=self._START_TIMEOUT, capture=True
                    )
                    if rc != 0:
                        return "create", rc, stderr
                    rc, _, stderr = await self._docker_command(
                        "docker",
                        "start",
                        self._container_name,
                        timeout=self._START_TIMEOUT,
                        capture=True,
                    )
                    if rc != 0:
                        return "start", rc, stderr
                    return None, 0, None

                stage, rc, stderr = await create_and_start(create_args)
                if (
                    stage is not None
                    and kernel_cgroups
                    and not _env_bool("HERO_LOCAL_DOCKER_REQUIRE_CGROUP_LIMITS", False)
                    and (not vfs or (
                        stage in {"create", "start"}
                        and rc == 1
                        and _CGROUP_COMPATIBILITY_RE.search(stderr or "")
                    ))
                ):
                    reason = (
                        f"docker {stage} failed with kernel cgroups: "
                        f"rc={rc}: {stderr}"
                    )
                    logger.error("%s; retrying in compatible fallback mode", reason)
                    self._coordinator.disable_kernel_cgroups(reason)
                    self._coordinator._write_telemetry(
                        "container_start_cgroup_fallback",
                        container=self._container_name,
                        session_id=self.session_id,
                        failed_stage=stage,
                        rc=rc,
                        error=stderr,
                    )
                    kernel_cgroups = False
                    cgroup_parent = ""
                    pids_limit = 0
                    stage, rc, stderr = await create_and_start(
                        compatibility_args(create_args)
                    )
                if stage is not None:
                    raise RuntimeError(f"docker {stage} failed: rc={rc}: {stderr}")
                record_phase("docker_create_start")

            init_pid = await self._inspect_container_pid()
            self._container_init_pid = init_pid
            cgroup_verified = False
            if init_pid is not None:
                cgroup_verified = self._verify_cgroup_limits(
                    init_pid,
                    memory_limit_mb=memory_limit_mb,
                    pids_limit=pids_limit,
                    kernel_cgroups=kernel_cgroups,
                    cgroup_parent=cgroup_parent,
                )
                self._resource_registered = self._coordinator.register_container(
                    container_name=self._container_name,
                    session_id=self.session_id,
                    image=self.spec.image,
                    init_pid=init_pid,
                    memory_limit_bytes=memory_limit_mb * 1024**2,
                    # Also enforce a pids ceiling in /proc fallback mode. This
                    # is a userspace safety net, not a training parameter.
                    pids_limit=(
                        pids_limit
                        if pids_limit > 0
                        else _env_int("HERO_LOCAL_DOCKER_FALLBACK_PIDS_LIMIT", 2048)
                    ),
                    cgroup_limits_verified=cgroup_verified,
                )
            if not self._resource_registered:
                self._coordinator._write_telemetry(
                    "resource_monitor_unavailable",
                    container=self._container_name,
                    session_id=self.session_id,
                    image=self.spec.image,
                    reason=(
                        "container_init_pid_unavailable"
                        if init_pid is None
                        else "container_pid_namespace_unavailable"
                    ),
                )
                if not cgroup_verified:
                    raise RuntimeError(
                        "cannot safely run a container without either verified "
                        "kernel cgroup limits or /proc memory/PID monitoring"
                    )
            self._schedule_admission_reservation_release()
            self._chmod_needed = await self._detect_chmod_needed()
            if self._chmod_needed:
                await self._docker_command(
                    "docker",
                    "exec",
                    "--user",
                    "root",
                    self._container_name,
                    "chmod",
                    "-R",
                    "a+rwX",
                    self.runtime_session_dir,
                    timeout=self._STOP_TIMEOUT,
                )
        except BaseException:
            await self.stop()
            raise

    async def _inspect_container_pid(self) -> int | None:
        # The direct Engine API creates no helper process, so it remains usable
        # when the machine is close to its process/thread ceiling.
        attempts = _env_int(
            "HERO_LOCAL_DOCKER_PID_PROBE_ATTEMPTS", 3, minimum=1
        )
        retry_delay = _env_float(
            "HERO_LOCAL_DOCKER_PID_PROBE_RETRY_SEC", 0.25, minimum=0.0
        )
        for attempt in range(attempts):
            pid = await _docker_engine_container_pid(self._container_name)
            if pid is not None:
                return pid
            if attempt + 1 < attempts and retry_delay > 0:
                await asyncio.sleep(retry_delay)

        # Compatibility fallback for unusual Docker transports. This runs once
        # at startup; periodic monitoring never invokes Docker CLI.
        rc, stdout, stderr = await self._docker_command(
            "docker",
            "inspect",
            "--format",
            "{{.State.Pid}}",
            self._container_name,
            timeout=self._STOP_TIMEOUT,
            capture=True,
        )
        if rc != 0:
            logger.warning(
                "cannot inspect init PID for %s; /proc monitor unavailable: %s",
                self._container_name,
                stderr or stdout,
            )
            return None
        try:
            pid = int((stdout or "").strip())
        except ValueError:
            return None
        return pid if pid > 0 else None

    def _verify_cgroup_limits(
        self,
        init_pid: int,
        *,
        memory_limit_mb: int,
        pids_limit: int,
        kernel_cgroups: bool,
        cgroup_parent: str,
    ) -> bool:
        if not kernel_cgroups:
            self._coordinator._write_telemetry(
                "cgroup_verification",
                container=self._container_name,
                session_id=self.session_id,
                mode="rootless-fallback",
                verified=False,
            )
            return False

        limits = _container_cgroup_limits(
            init_pid,
            cgroup_root=Path(
                os.environ.get("HERO_LOCAL_DOCKER_CGROUP_ROOT", "/sys/fs/cgroup")
            ),
        )
        actual_memory = limits.get("memory_limit_bytes")
        actual_pids = limits.get("pids_limit")
        expected_memory = memory_limit_mb * 1024**2
        memory_ok = expected_memory <= 0 or (
            isinstance(actual_memory, int) and actual_memory <= expected_memory
        )
        pids_ok = pids_limit <= 0 or (
            isinstance(actual_pids, int) and actual_pids <= pids_limit
        )
        parent_ok = bool(limits.get("isolated"))
        if cgroup_parent:
            paths = " ".join(
                str(limits.get(key, "")) for key in ("path", "memory_path", "pids_path")
            )
            parent_ok = parent_ok and cgroup_parent.strip("/") in paths
        verified = memory_ok and pids_ok and parent_ok
        self._coordinator._write_telemetry(
            "cgroup_verification",
            container=self._container_name,
            session_id=self.session_id,
            mode="kernel",
            verified=verified,
            expected_memory_bytes=expected_memory,
            expected_pids=pids_limit,
            expected_parent=cgroup_parent,
            observed=limits,
        )
        if not verified:
            message = (
                f"kernel cgroup limits were not effective for "
                f"{self._container_name}: expected memory={expected_memory}, "
                f"pids={pids_limit}, parent={cgroup_parent!r}; observed={limits}"
            )
            if _env_bool("HERO_LOCAL_DOCKER_REQUIRE_CGROUP_LIMITS", False):
                raise RuntimeError(message)
            logger.error("%s; continuing with /proc safety fallback", message)
            self._coordinator.disable_kernel_cgroups(message)
        return verified

    def _schedule_admission_reservation_release(self) -> None:
        if self._admission_reservation_gb <= 0:
            return
        hold_sec = self._coordinator.admission_reservation_hold_sec
        self._reservation_release_task = asyncio.create_task(
            self._release_admission_reservation_after(hold_sec),
            name=f"hero-admission-reservation-{self.session_id}",
        )

    async def _release_admission_reservation_after(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            self._release_admission_reservation()
        except asyncio.CancelledError:
            raise

    def _release_admission_reservation(self) -> None:
        reservation_gb = self._admission_reservation_gb
        if reservation_gb > 0:
            self._admission_reservation_gb = 0.0
            self._coordinator.release_admission_reservation(reservation_gb)

    async def stop(self) -> None:
        if self._destroyed:
            return
        task = self._stop_task
        if task is None:
            task = asyncio.create_task(
                self._stop_impl(),
                name=f"hero-docker-stop-{self.session_id}",
            )
            self._stop_task = task
            task.add_done_callback(self._stop_task_finished)
        # wait_for() cancellation from the session budget must not cancel the
        # actual docker rm/release sequence.  A later stop() joins this task.
        await asyncio.shield(task)
        if not self._destroyed and self._stop_task is task:
            # Removal exhausted its bounded retries.  Keep the runtime
            # retryable instead of hiding a live orphan behind _destroyed=True.
            self._stop_task = None
            self._ensure_cleanup_retry()

    def _stop_task_finished(self, task: asyncio.Task[None]) -> None:
        """Start the orphan reaper even when the caller cancelling stop left."""
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "Docker stop task failed for %s",
                self._container_name,
                exc_info=True,
            )
        if self._stop_task is task:
            self._stop_task = None
        if not self._destroyed:
            self._ensure_cleanup_retry()

    def _ensure_cleanup_retry(self) -> None:
        task = self._cleanup_retry_task
        if task is None or task.done():
            self._cleanup_retry_task = asyncio.create_task(
                self._retry_cleanup_until_removed(),
                name=f"hero-docker-reaper-{self.session_id}",
            )

    async def _retry_cleanup_until_removed(self) -> None:
        """Keep retrying an orphan without reopening its admission slot."""
        delay = _env_float(
            "HERO_LOCAL_DOCKER_CLEANUP_RETRY_INTERVAL_SEC", 2.0, minimum=0.1
        )
        max_delay = _env_float(
            "HERO_LOCAL_DOCKER_CLEANUP_RETRY_MAX_INTERVAL_SEC",
            30.0,
            minimum=0.1,
        )
        while not self._destroyed:
            await asyncio.sleep(delay)
            task = self._stop_task
            if task is None:
                task = asyncio.create_task(
                    self._stop_impl(),
                    name=f"hero-docker-reap-attempt-{self.session_id}",
                )
                self._stop_task = task
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "background Docker cleanup failed for %s",
                    self._container_name,
                    exc_info=True,
                )
            finally:
                if self._stop_task is task:
                    self._stop_task = None
            delay = min(delay * 2, max_delay)

    async def _stop_impl(self) -> None:
        removed = False
        reservation_task = self._reservation_release_task
        self._reservation_release_task = None
        try:
            if reservation_task is not None:
                reservation_task.cancel()
                try:
                    await reservation_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.warning(
                        "admission reservation cleanup failed for %s",
                        self._container_name,
                        exc_info=True,
                    )
        finally:
            try:
                if (
                    self._chmod_needed is not False
                    and self._removal_in_progress_since is None
                ):
                    try:
                        await self._docker_command(
                            "docker",
                            "exec",
                            "--user",
                            "root",
                            self._container_name,
                            "chmod",
                            "-R",
                            "a+rwX",
                            self.runtime_session_dir,
                            timeout=self._STOP_TIMEOUT,
                        )
                    except Exception:
                        logger.warning(
                            "chmod cleanup failed for %s",
                            self._container_name,
                            exc_info=True,
                        )
                rc, stdout, stderr = await self._remove_container()
                remove_output = f"{stdout or ''}\n{stderr or ''}"
                removed = rc == 0 or bool(
                    _CONTAINER_NOT_FOUND_RE.search(remove_output)
                )
                if not removed:
                    if _REMOVAL_IN_PROGRESS_RE.search(remove_output):
                        logger.info(
                            "Docker container removal still in progress for %s",
                            self._container_name,
                        )
                    else:
                        logger.warning(
                            "docker rm -f failed for %s (rc=%s): %s",
                            self._container_name,
                            rc,
                            stderr or stdout,
                        )
            finally:
                # Fail closed: an unconfirmed container keeps its active slot
                # and image reference.  Releasing them here used to admit new
                # work while the live orphan kept consuming memory, producing
                # an unbounded accumulation after the first fork failure.
                if removed:
                    # Accounting release performs no pressure probe. Keep the
                    # reservation while an unconfirmed orphan is still live,
                    # then release it immediately after deletion succeeds.
                    self._release_admission_reservation()
                    if self._resource_registered:
                        self._resource_registered = False
                        self._coordinator.unregister_container(self._container_name)
                    if (
                        os.environ.get("HERO_LOCAL_DOCKER_STORAGE_DRIVER", "").strip().lower() == "vfs"
                        and self._active_acquired
                    ):
                        # Confirmed deletion ends container capacity ownership.
                        # Image leases/refcounts remain owned until release_image;
                        # slow rmi/daemon-lock waits must not retain this slot.
                        self._active_acquired = False
                        self._coordinator.release_active()
                    try:
                        if self._image_acquired:
                            self._image_acquired = False
                            await self._coordinator.release_image(
                                self.spec.image, self._docker_command
                            )
                    finally:
                        if self._active_acquired:
                            self._active_acquired = False
                            self._coordinator.release_active()
                        self._destroyed = True

    async def _remove_container(self) -> tuple[int, str | None, str | None]:
        """Remove the container, retrying daemon failures during teardown."""
        if self._removal_in_progress_since is not None:
            result = await self._docker_command(
                "docker",
                "container",
                "inspect",
                self._container_name,
                timeout=self._STOP_TIMEOUT,
                capture=True,
            )
            rc, stdout, stderr = result
            output = f"{stdout or ''}\n{stderr or ''}"
            if rc != 0 and _CONTAINER_NOT_FOUND_RE.search(output):
                self._removal_in_progress_since = None
                return 0, stdout, stderr
            if rc != 0:
                return result
            retry_after = _env_float(
                "HERO_LOCAL_DOCKER_REMOVE_IN_PROGRESS_RETRY_SEC",
                120.0,
                minimum=0.0,
            )
            if time.monotonic() - self._removal_in_progress_since < retry_after:
                return 1, stdout, "container removal is still in progress"
            # A daemon restart can lose the pending removal while leaving the
            # container behind. Permit one fresh rm after a long stale window.
            self._removal_in_progress_since = None

        attempts = _env_int("HERO_LOCAL_DOCKER_REMOVE_ATTEMPTS", 3, minimum=1)
        result: tuple[int, str | None, str | None] = (-1, None, None)
        for attempt in range(attempts):
            result = await self._docker_command(
                "docker",
                "rm",
                "-f",
                self._container_name,
                timeout=self._STOP_TIMEOUT,
                capture=True,
            )
            rc, stdout, stderr = result
            output = f"{stdout or ''}\n{stderr or ''}"
            if rc == 0 or _CONTAINER_NOT_FOUND_RE.search(output):
                self._removal_in_progress_since = None
                return result
            if _REMOVAL_IN_PROGRESS_RE.search(output):
                self._removal_in_progress_since = time.monotonic()
                return result
            if attempt + 1 < attempts:
                await asyncio.sleep(min(0.5 * (2**attempt), 2.0))
        return result

    async def _docker_command(
        self,
        *args: str,
        timeout: float | None = None,
        capture: bool = False,
    ) -> tuple[int, str | None, str | None]:
        retries = _env_int("HERO_LOCAL_DOCKER_TRANSIENT_RETRIES", 4)
        result: tuple[int, str | None, str | None] = (-1, None, None)
        for attempt in range(retries + 1):
            try:
                result = await self._run_local_command(
                    *args, timeout=timeout, capture=capture
                )
            except OSError as exc:
                # Treat fork/EAGAIN exactly like Docker's other transient
                # failures so teardown retries instead of abandoning cleanup.
                result = (
                    -1,
                    None,
                    f"local subprocess spawn failed: {exc}",
                )
            rc, stdout, stderr = result
            if rc == 0 or not _TRANSIENT_RE.search(f"{stderr or ''}\n{stdout or ''}"):
                return result
            if attempt == retries:
                break
            await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
        return result


@lru_cache(maxsize=1024)
def _image_tar_entrypoint(
    path: Path, expected_image: str
) -> tuple[str, ...] | None:
    """Read one image's configured ENTRYPOINT without loading or running it.

    Failure to read optional metadata deliberately returns ``None`` so the
    established Docker startup path remains the fallback. Docker image archives
    are immutable for a run, making this safe to cache across the 16 sibling
    trajectories that use the same image.
    """
    try:
        with tarfile.open(path, "r") as archive:
            source = archive.extractfile(archive.getmember("manifest.json"))
            if source is None:
                return None
            manifest = json.load(source)
            if not isinstance(manifest, list):
                return None
            expected_norm = _normalize_image_ref(expected_image)
            item = next(
                (
                    candidate
                    for candidate in manifest
                    if isinstance(candidate, dict)
                    if expected_image in candidate.get("RepoTags", [])
                    or any(
                        _normalize_image_ref(str(tag)) == expected_norm
                        for tag in candidate.get("RepoTags", [])
                    )
                ),
                None,
            )
            if item is None or not isinstance(item.get("Config"), str):
                return None
            config_source = archive.extractfile(archive.getmember(item["Config"]))
            if config_source is None:
                return None
            image_config = json.load(config_source)
    except (
        OSError,
        tarfile.TarError,
        KeyError,
        StopIteration,
        TypeError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(image_config, dict):
        return None
    config = image_config.get("config")
    if not isinstance(config, dict):
        return None
    raw_entrypoint = config.get("Entrypoint")
    if not isinstance(raw_entrypoint, list) or not raw_entrypoint:
        return None
    if not all(isinstance(value, str) for value in raw_entrypoint):
        return None
    return tuple(raw_entrypoint)


def _is_shell_only_entrypoint(entrypoint: tuple[str, ...] | None) -> bool:
    """Whether ``ENTRYPOINT + ['sleep', 'infinity']`` invokes a script by mistake."""
    if entrypoint is None or len(entrypoint) != 1:
        return False
    return Path(entrypoint[0]).name in {"bash", "sh"}


def _image_tar_tags(path: Path) -> list[str]:
    """Return the RepoTags declared inside a Docker image archive manifest."""
    try:
        with tarfile.open(path, "r") as archive:
            source = archive.extractfile(archive.getmember("manifest.json"))
            if source is None:
                return []
            manifest = json.load(source)
    except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError):
        return []
    return [str(tag) for item in manifest for tag in item.get("RepoTags", [])]


def _normalize_image_ref(ref: str) -> str:
    """Strip a registry host component so a ref compares equal to its short form.

    Docker treats ``docker.m.daocloud.io/swebench/foo:latest`` and
    ``swebench/foo:latest`` as the same image, but our tars are saved with the
    full registry-prefixed RepoTag (they were pulled from a mirror). Accepting
    only the bare short name rejected every scikit-learn / sphinx-doc image and
    produced 16 spurious ``infra_failed`` rows in the 0807 v1 eval.
    """
    # A registry component is everything before the first '/', but only when
    # that component contains a '.' or ':' (host:port) or is a known single-word
    # registry. A bare library/officials name like 'nginx' has no slash, and a
    # two-part name like 'swebench/foo' has a slash whose first part is not a
    # host, so neither is stripped here.
    if "/" in ref:
        head, _, rest = ref.partition("/")
        if "." in head or ":" in head or head in {"localhost"}:
            return rest
    return ref


def validate_image_tar(path: Path, expected_image: str) -> None:
    """Reject missing, corrupt, or incorrectly tagged local Docker archives."""
    if not path.is_file():
        raise FileNotFoundError(f"local Docker image tar not found: {path}")
    try:
        with tarfile.open(path, "r") as archive:
            source = archive.extractfile(archive.getmember("manifest.json"))
            if source is None:
                raise RuntimeError("manifest.json is unreadable")
            manifest = json.load(source)
    except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid local Docker image tar {path}: {exc}") from exc
    tags = [str(tag) for item in manifest for tag in item.get("RepoTags", [])]
    expected_norm = _normalize_image_ref(expected_image)
    matched = expected_image in tags or any(
        _normalize_image_ref(t) == expected_norm for t in tags
    )
    if not matched:
        raise RuntimeError(
            f"local Docker image tar {path} contains {tags!r}, expected {expected_image!r}"
        )
