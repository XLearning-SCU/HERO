# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os

import ray
import torch
from slime.ray.ray_actor import RayActor

# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

RAY_DEFAULT_ENV_VARS = {
    # Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
}


_RAY_INHERITED_ENV_VARS = (
    "SLIME_VFS_MULTINODE",
    "SLIME_TRAIN_PHASE_TRACE_DIR",
    "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "CUDA_HOME",
    "NVTE_CUDA_INCLUDE_DIR", "NVTE_CUDNN_INCLUDE_DIR", "NVTE_CUDNN_LIB_DIR",
    "FLASHINFER_DISABLE_VERSION_CHECK", "FLASHINFER_DISABLE_AOT_META_CHECK",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "NCCL_NVLS_ENABLE",
    "NCCL_DEBUG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_DEVICE_MAX_CONNECTIONS",
    "GATE_THRESHOLD",
    "EFFICIENCY_COEFFICIENT",
    "EFFICIENCY_RATIO",
)


def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    inherited = {k: os.environ[k] for k in _RAY_INHERITED_ENV_VARS if k in os.environ}
    if os.environ.get("SLIME_VFS_MULTINODE") == "1":
        # Preserve the job's chosen interfaces in nested actors as well.
        # Do not change environment propagation for established IDE runs.
        inherited.update({k: os.environ[k] for k in (
            "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"
        ) if os.environ.get(k)})
    return RAY_DEFAULT_ENV_VARS | inherited | (env_vars or {})


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
