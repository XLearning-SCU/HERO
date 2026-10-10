"""Slime data-source wrappers used by HERO examples."""

from __future__ import annotations

import math

try:
    from slime.rollout.data_source import RolloutDataSourceWithBuffer
except ImportError as _SLIME_IMPORT_ERROR:
    class RolloutDataSourceWithBuffer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "Slime is required to use CeilEpochRolloutDataSourceWithBuffer."
            ) from _SLIME_IMPORT_ERROR


def ceil_to_batch_size(size: int, batch_size: int) -> int:
    if size <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return math.ceil(size / batch_size) * batch_size


class CeilEpochRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Expose a rounded-up epoch length for fixed-size Slime rollout batches.

    Slime computes `num_rollout_per_epoch = len(data_source) // rollout_batch_size`.
    For datasets whose size is not divisible by the rollout batch size, the
    default floor behavior skips the tail prompts. Returning a rounded-up length
    lets the existing data source wrap only the final few prompts while still
    covering every prompt in the dataset once per epoch.
    """

    def __len__(self) -> int:
        return ceil_to_batch_size(
            super().__len__(),
            int(getattr(self.args, "rollout_batch_size", 1) or 1),
        )
