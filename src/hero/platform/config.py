"""Runtime configuration for the platform service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hero.config import TopologyConfig


@dataclass(slots=True)
class PlatformConfig:
    """Resolved configuration for `hero dashboard`."""

    host: str
    port: int
    rollout_url: str
    save_dir: Path
    topology: TopologyConfig
    topology_path: Path

    @classmethod
    def from_topology(
        cls,
        topology_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        rollout_url: str | None = None,
        save_dir: str | Path | None = None,
    ) -> "PlatformConfig":
        topology = TopologyConfig.load(topology_path)
        resolved_rollout = rollout_url or topology.rollout.public_url
        resolved_save_dir = Path(
            save_dir
            if save_dir is not None
            else (topology.rollout.save_dir or "./rollout_results")
        ).resolve()
        return cls(
            host=host,
            port=port,
            rollout_url=resolved_rollout.rstrip("/"),
            save_dir=resolved_save_dir,
            topology=topology,
            topology_path=Path(topology_path).resolve(),
        )
