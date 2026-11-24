"""CLI flags and checkpoint evaluation shared by the domain launchers.

Every launcher has two modes:

* **train** (default): online collection + training, with periodic
  validation (``val`` / ``val-alpha`` / ``val-beta``) logged to wandb.
* **eval** (``--eval_last_n_ckpts N``): evaluates the last ``N`` policy
  checkpoints on the domain's held-out test environments under every
  deployment phase the method supports (``test`` / ``test-alpha`` /
  ``test-beta``). Logs to a separate wandb run ``<eval_run_prefix>_<run>``
  and to ``<eval run>/ckpts/eval_results/eval_results.jsonl``, one row per
  (checkpoint epoch, test env, phase).
"""

from __future__ import annotations

import glob
import json
import os
import re
from argparse import ArgumentParser
from typing import Callable, Sequence

from metabfm import config as C


def add_logging_args(parser: ArgumentParser, default_project: str) -> None:
    parser.add_argument("--no_log", action="store_true", help="Disable wandb logging.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", default_project),
        help="wandb project (default: $WANDB_PROJECT, else %(default)s).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="wandb entity (default: $WANDB_ENTITY, else your wandb default).",
    )


def add_eval_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--eval_last_n_ckpts",
        type=int,
        default=0,
        help="Eval mode: skip training and evaluate the last N policy "
        "checkpoints of <run_name> on the held-out test environments.",
    )
    parser.add_argument(
        "--eval_run_prefix",
        type=str,
        default="EVAL",
        help="Eval mode logs to the wandb run and ckpt dir "
        "<eval_run_prefix>_<training run name>.",
    )


def require_fresh_run(run_name: str, dset_root: str, dset_name: str) -> None:
    """Refuse to start a new run on top of an existing one.

    amago reuses existing checkpoint and buffer directories, so a second
    ``train`` with the same run name would train on the first run's replay
    data and write into its checkpoint folder. Resume with ``--ckpt <epoch>``
    instead, or pick a new run name / delete the old run.
    """
    ckpt_dir = os.path.join(C.RL_CKPT_DIR, run_name, "ckpts")
    leftovers = [
        d
        for d in (
            os.path.join(ckpt_dir, "policy_weights"),
            os.path.join(ckpt_dir, "training_states"),
            os.path.join(dset_root, dset_name, "buffer", "fifo"),
            os.path.join(dset_root, dset_name, "buffer", "protected"),
        )
        if os.path.isdir(d) and os.listdir(d)
    ]
    if leftovers:
        raise SystemExit(
            f"Run {run_name!r} already has data:\n  "
            + "\n  ".join(leftovers)
            + "\nResume it with --ckpt <epoch>, or use a new --run_name / PREFIX, "
            "or delete these directories."
        )


def eval_collect_kwargs() -> dict:
    """Experiment kwargs that disable collection and dataset writes."""
    return dict(
        exploration_wrapper_type=None,
        has_dset_edit_rights=False,
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
    )


def _jsonable(v):
    try:
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return float(v)
    except Exception:
        return None


def run_checkpoint_eval(
    experiment,
    source_run_name: str,
    test_envs: Sequence[tuple[str, Callable]],
    timesteps: int,
    last_n: int,
) -> str:
    """Evaluate the last ``last_n`` checkpoints of ``source_run_name``.

    ``test_envs`` is a list of ``(env_tag, make_env)`` pairs. Each pair is
    rolled through every enabled deployment phase via
    ``experiment.evaluate_test``; wandb keys are ``{phase}/{env_tag}``.
    Returns the path of the JSONL results file.
    """
    pw_dir = os.path.join(C.RL_CKPT_DIR, source_run_name, "ckpts", "policy_weights")
    epoch_of = lambda p: int(re.search(r"policy_epoch_(\d+)\.pt$", p).group(1))
    ckpts = sorted(glob.glob(os.path.join(pw_dir, "policy_epoch_*.pt")), key=epoch_of)
    if not ckpts:
        raise FileNotFoundError(f"No policy_epoch_*.pt checkpoints under {pw_dir}")
    ckpts = ckpts[-last_n:]

    results_dir = os.path.join(experiment.ckpt_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "eval_results.jsonl")
    print(
        f"[eval] {source_run_name}: epochs {[epoch_of(p) for p in ckpts]} "
        f"x {[tag for tag, _ in test_envs]} -> {results_path}"
    )
    with open(results_path, "a") as out:
        for path in ckpts:
            epoch = epoch_of(path)
            experiment.load_checkpoint_from_path(path, is_accelerate_state=False)
            experiment.policy.eval()
            experiment.epoch = epoch
            for env_tag, make_env in test_envs:
                print(f"[eval] epoch {epoch}: {env_tag}")
                per_phase = experiment.evaluate_test(
                    make_env,
                    timesteps=timesteps,
                    log_tag=env_tag,
                )
                for phase, logs in per_phase.items():
                    row = {
                        "source_run": source_run_name,
                        "eval_run": experiment.run_name,
                        "epoch": epoch,
                        "env_tag": env_tag,
                        "phase": phase,
                        "metrics": {k: _jsonable(v) for k, v in logs.items()},
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
    print(f"[eval] done: {results_path}")
    return results_path
