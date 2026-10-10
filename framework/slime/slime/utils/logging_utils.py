import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    # Callers use zero-based rollout/update IDs (pre-training eval uses -1).
    # Shift tracking IDs only; leave caller-owned data and checkpoint/scheduler
    # IDs unchanged. Rollout IDs 0..79 display as 1..80. Training IDs still
    # follow the caller formula when update counts vary between rollouts.
    metrics = dict(metrics)
    for key in ("train/step", "rollout/step", "eval/step"):
        if key in metrics:
            metrics[key] += 1

    if args.use_wandb:
        # Explicit step= is ignored in shared mode and can drop out-of-order
        # rollout/train records in other modes. Each row carries its custom
        # axis; W&B's internal history counter is not a training step.
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
