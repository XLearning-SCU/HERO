"""Opt-in CPU-only breadcrumbs for cluster VFS multi-node training hangs."""

import json
import logging
import os
from pathlib import Path
import socket
import time

logger = logging.getLogger(__name__)


class TrainPhaseTrace:
    def __init__(self, *, rank, rollout_id, step_id):
        self.path = None
        self.context = dict(rank=rank, rollout_id=rollout_id, step_id=step_id)
        directory = os.environ.get("SLIME_TRAIN_PHASE_TRACE_DIR")
        if os.environ.get("SLIME_VFS_MULTINODE") != "1" or not directory:
            return
        try:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / f"rank-{rank}-{socket.gethostname()}-{os.getpid()}.jsonl"
        except OSError:
            logger.warning("Cannot create training phase trace directory", exc_info=True)

    def emit(self, phase, **fields):
        # Never inspect CUDA memory, copy tensors, synchronize, or introduce a
        # collective here. 'forward_return' means host return, not GPU completion.
        if self.path is None:
            return
        try:
            record = dict(self.context, time_ns=time.time_ns(), phase=phase, **fields)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError):
            logger.warning("Disabling failed training phase trace writer", exc_info=True)
            self.path = None
