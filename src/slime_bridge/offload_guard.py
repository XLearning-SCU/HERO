"""Fence HERO requests while an cluster colocated rollout is offloaded."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def create_offload_guard(args):
    # This profile is forced off by the launcher for local/overlay2 jobs.
    if os.environ.get("SLIME_VFS_MULTINODE") != "1" or not args.offload_rollout:
        return None
    if args.rollout_function_path != "slime_bridge.rollout.generate_rollout_hero_async":
        return None

    from hero.config import TopologyConfig

    path = os.environ.get("SLIME_GATEWAY_TOPOLOGY")
    if not path:
        raise RuntimeError("cluster offload guard requires the generated gateway topology")
    topology = TopologyConfig.load(path)
    guard = InferenceOffloadGuard([node.public_url for node in topology.gateway.nodes])
    logger.info("cluster offload guard enabled: gateways=%s", guard.gateway_urls)
    return guard


class InferenceOffloadGuard:
    def __init__(self, gateway_urls):
        self.gateway_urls = tuple(dict.fromkeys(url.rstrip("/") for url in gateway_urls))
        if not self.gateway_urls:
            raise ValueError("Cannot fence inference without gateway URLs")
        self.paused = False

    def pause(self):
        if self.paused:
            return
        # The gateway's inference HTTP ceiling is 900s. Allow its existing
        # requests to finish before freeing their GPU buffers. A failed drain
        # must abort the offload, never proceed with live inference requests.
        with httpx.Client(timeout=httpx.Timeout(930.0, connect=10.0)) as client:
            for url in self.gateway_urls:
                response = client.post(
                    f"{url}/admin/inference/pause", params={"timeout_seconds": 900.0}
                )
                response.raise_for_status()
                status = response.json()
                if status.get("paused") is not True or status.get("inflight") != 0:
                    raise RuntimeError(f"Gateway did not drain inference: {url}: {status}")
        self.paused = True
        logger.info("cluster offload guard: all gateways paused and drained")

    def resume(self):
        if not self.paused:
            return
        # Do NOT call from a finally block: a failed restore must leave traffic
        # fenced. The driver calls this only after weights AND KV are restored.
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            for url in self.gateway_urls:
                response = client.post(f"{url}/admin/inference/resume")
                response.raise_for_status()
                if response.json().get("paused") is not False:
                    raise RuntimeError(f"Gateway did not resume inference: {url}")
        self.paused = False
        logger.info("cluster offload guard: inference resumed after weights and KV restore")
