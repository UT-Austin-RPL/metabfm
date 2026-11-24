"""ColoredGrid launcher for BilevelAgent (DynamicsLearner + RL).

Trains a BilevelAgent online on ColoredGrid and validates on a held-out task
split via AMAGO's ``make_val_env`` hook every ``--val_interval`` epochs. Eval
mode (``--eval_last_n_ckpts N``) evaluates the last N checkpoints on the same
held-out tasks; see :mod:`metabfm.rl.launch_utils`.

Tasks: ``--n_envs N --n_rewards M`` draws N cell permutations and M reward
grids from :mod:`metabfm.env.reward_pool`; each rollout samples a layout
and a reward grid independently from the N x M cross product.

Gin ladder: the ``--baseline`` preset
(:data:`metabfm.rl.baseline_presets.BASELINE_PRESETS`), then the env
overlay (``--env_config``, default ``env/coloredgrid``), then ``--overlay``
configs, then ``--gin`` bindings. Method configs are env-agnostic; the env
overlay owns all architectural dimensions (phi/psi sizes, transformer depths,
attention windows, critic bin range, ...).

Usage (normally via ``bash scripts/colorgrid.sh <method> <train|eval|dry_run>``):
    python scripts/train_colorgrid.py \\
        --run_name cg_alphabeta_10kx10k --baseline metabfm_alphabeta \\
        --n_envs 10000 --n_rewards 10000 --grid_size 5 --parallel_actors 128
"""

import os
from argparse import ArgumentParser
from datetime import datetime

import gin
import wandb

from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset

from metabfm import config as C
from metabfm.env import ColoredGrid
from metabfm.env.reward_pool import (
    make_cell_perm_split,
    make_reward_split,
)


from metabfm.rl import configs as rl_configs

import metabfm.rl.zero_shot_rl  # noqa: F401
import metabfm.rl.dynamics_learner  # noqa: F401
import metabfm.rl.extrinsic_encoder  # noqa: F401
from metabfm.rl.baseline_presets import BASELINE_PRESETS
from metabfm.rl.launch_utils import (
    add_eval_args,
    add_logging_args,
    eval_collect_kwargs,
    require_fresh_run,
    run_checkpoint_eval,
)
from metabfm.rl.metabfm_experiment import BilevelExperiment


def main():
    parser = ArgumentParser(
        description="Train BilevelAgent on ColoredGrid (regression launcher)."
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--baseline",
        type=str,
        choices=sorted(BASELINE_PRESETS.keys()),
        default="metabfm_alphabeta",
        help="Baseline preset. Resolves to a fixed gin ladder via "
        "BASELINE_PRESETS.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        required=True,
        help="Number of distinct training cell permutations (the 'env' axis).",
    )
    parser.add_argument(
        "--n_rewards",
        type=int,
        required=True,
        help="Number of distinct training reward grids (the 'reward' axis).",
    )
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--parallel_actors", type=int, default=160)
    parser.add_argument("--dloader_workers", type=int, default=4)
    parser.add_argument("--no_eval", action="store_true")
    add_logging_args(parser, default_project="metabfm-colorgrid")
    add_eval_args(parser)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the gin config, envs, and model, write ckpts/config.txt, "
        "then exit before any data collection or training. Implies --no_log.",
    )
    parser.add_argument("--ckpt", type=int, default=None, help="Resume from epoch")
    parser.add_argument(
        "--gin", action="append", default=None, help="Extra gin bindings"
    )
    parser.add_argument(
        "--overlay",
        type=str,
        action="append",
        default=None,
        help="Extra registered gin config(s) layered on top of the preset+env "
        "ladder, e.g. `--overlay dynamics/with_ortho`. Repeatable. Applied AFTER "
        "the env overlay and BEFORE --gin ad-hoc bindings.",
    )
    parser.add_argument("--k", type=int, default=6, help="Episodes per trial (env k)")
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--ckpt_interval", type=int, default=50)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help="Per-element gradient value clipping threshold passed to "
        "BilevelExperiment. Pass 0 (or any non-positive value) to disable "
        "clipping entirely.",
    )
    parser.add_argument(
        "--online_coeff",
        type=float,
        default=None,
        help=(
            "Override TDJepa.online_coeff (weight on the DPG online actor "
            "term, computed from `maximized_critics` / `maximized_phi_predictor` "
            "with gradients). Defaults to the gin binding. At least one of "
            "online_coeff / offline_coeff must be > 0; otherwise the actor "
            "receives no learning signal (asserted in TDJepa.__init__)."
        ),
    )
    parser.add_argument(
        "--offline_coeff",
        type=float,
        default=None,
        help=(
            "Override TDJepa.offline_coeff (weight on the AWAC-style "
            "filtered-BC offline actor term). Defaults to the gin binding."
        ),
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="env/coloredgrid",
        help="Domain overlay gin file, applied last in the ladder so it "
        "overrides component defaults (default: env/coloredgrid).",
    )
    parser.add_argument(
        "--num_collection_batches",
        type=int,
        default=2,
        help="Number of z-proposal rounds per epoch during online ZSRL collection.",
    )
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--steps_anneal", type=int, default=500_000)
    parser.add_argument(
        "--start_collecting_at_epoch",
        type=float,
        default=0.0,
        help=(
            "Epoch at which online collection kicks in.  Default 0 = collect "
            "from the start.  Pass a positive int to delay collection, or "
            "'inf' to disable it entirely and train only on the prefilled "
            "replay buffer (useful for quickly iterating on training-speed "
            "experiments without burning wall-clock on env rollouts)."
        ),
    )
    parser.add_argument(
        "--start_learning_at_epoch",
        type=int,
        default=5,
        help=(
            "Epoch at which gradient updates begin.  Default 5 lets the "
            "online buffer pre-fill before training kicks off.  Larger "
            "values (e.g. 10) give a longer warm-up — useful when the "
            "first few collection rounds need to settle a new method."
        ),
    )
    args = parser.parse_args()
    if args.dry_run:
        args.no_log = True

    grid_size = args.grid_size
    max_ep_steps = 50 if grid_size != 9 else 60
    k = args.k

    # ── Exploration (egreedy for online collection) ──
    exploration_config = {}
    exploration_wrapper_type = cli_utils.switch_exploration(
        exploration_config,
        strategy="egreedy",
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        steps_anneal=args.steps_anneal,
    )

    # ── Gin config ──
    # Ladder: --baseline preset, env overlay, --overlay configs (so their
    # bindings win over env defaults), then --gin ad-hoc bindings.
    baseline_label = args.baseline
    base_configs = BASELINE_PRESETS[baseline_label]

    config_paths = [rl_configs.get(c) for c in base_configs]
    config_paths.append(rl_configs.get(args.env_config))
    for ov in args.overlay or []:
        config_paths.append(rl_configs.get(ov))
    cli_utils.use_config(exploration_config, config_paths, finalize=False)
    # CLI sugar: --online_coeff / --offline_coeff <v> are applied as gin
    # bindings before the raw --gin pass-through so explicit `--gin
    # TDJepa.<key>=...` bindings still win.
    if args.online_coeff is not None:
        gin.parse_config(f"TDJepa.online_coeff = {float(args.online_coeff)}")
    if args.offline_coeff is not None:
        gin.parse_config(f"TDJepa.offline_coeff = {float(args.offline_coeff)}")
    for binding in args.gin or []:
        gin.parse_config(binding)
    gin.finalize()

    gin_ladder_log = list(base_configs) + [args.env_config] + list(args.overlay or [])
    print(f"  Baseline preset: {baseline_label}")
    print("  Gin ladder:")
    for c in gin_ladder_log:
        print(f"    {c}")

    # ── Task split ──
    # ``train`` splits are used for online data collection; ``test`` splits
    # are held out and play the make_val_env role.
    n_envs = args.n_envs
    n_rewards = args.n_rewards
    cell_perm_split = make_cell_perm_split(grid_size, n_train=n_envs, seed=0)
    reward_split = make_reward_split(grid_size, n_train=n_rewards, seed=0)
    train_cell_perms = cell_perm_split.train
    test_cell_perms = cell_perm_split.test
    train_reward_grids = reward_split.train
    test_reward_grids = reward_split.test
    dset_tag = f"envs_{n_envs}_rewards_{n_rewards}"

    # ── Environments ──
    _seed = [0]

    def _make_env(reward_grids, cell_perms, env_name: str):
        def factory():
            seed = _seed[0]
            _seed[0] += 1
            return AMAGOEnv(
                ColoredGrid(
                    n=grid_size,
                    k=k,
                    max_ep_steps=max_ep_steps,
                    randomize_tilt=False,
                    seed=seed,
                    cell_perms=cell_perms,
                    reward_grids=reward_grids,
                ),
                env_name=env_name,
                batched_envs=1,
            )

        return factory

    make_train_env = _make_env(
        reward_grids=train_reward_grids,
        cell_perms=train_cell_perms,
        env_name=f"train_{grid_size}x{grid_size}_{dset_tag}",
    )
    make_val_env = _make_env(
        reward_grids=test_reward_grids,
        cell_perms=test_cell_perms,
        env_name=f"val_{grid_size}x{grid_size}_{dset_tag}",
    )
    # Eval mode tests on the same held-out tasks as in-training validation.
    make_test_env = _make_env(
        reward_grids=test_reward_grids,
        cell_perms=test_cell_perms,
        env_name=f"test_{grid_size}x{grid_size}_{dset_tag}",
    )

    # ── Training ──
    group_name = f"{args.run_name}_{grid_size}x{grid_size}_{dset_tag}"

    for trial in range(args.trial_offset, args.trial_offset + args.trials):
        run_name = f"{group_name}_trial_{trial}"
        source_run_name = run_name
        eval_mode = args.eval_last_n_ckpts > 0
        if eval_mode:
            run_name = f"{args.eval_run_prefix}_{source_run_name}"

        # Per-trial on-disk buffer keyed off the unique run_name, so trials
        # never share a buffer when --trials > 1.
        online_dset_name = f"online_{run_name}"
        if not eval_mode and args.ckpt is None:
            require_fresh_run(run_name, C.BUFFER_DIR, online_dset_name)
        dataset = DiskTrajDataset(
            dset_root=C.BUFFER_DIR,
            dset_name=online_dset_name,
            dset_max_size=1_000_000,
            relabeler=None,
        )

        steps_per_round = max_ep_steps * k + 1
        train_timesteps = args.num_collection_batches * steps_per_round
        eval_timesteps = steps_per_round * 3
        if eval_mode:
            collect_kwargs = eval_collect_kwargs()
        else:
            collect_kwargs = dict(
                exploration_wrapper_type=exploration_wrapper_type,
                has_dset_edit_rights=True,
                start_collecting_at_epoch=args.start_collecting_at_epoch,
                train_timesteps_per_epoch=train_timesteps,
            )

        experiment = BilevelExperiment(
            baseline_preset=baseline_label,
            gin_ladder=gin_ladder_log,
            num_collection_batches=args.num_collection_batches,
            steps_per_round=steps_per_round,
            run_name=run_name,
            ckpt_base_dir=C.RL_CKPT_DIR,
            dataset=dataset,
            val_timesteps_per_epoch=0 if args.no_eval else eval_timesteps,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            parallel_actors=args.parallel_actors,
            env_mode="sync",
            sample_actions_train=True,
            sample_actions_val=True,
            log_to_wandb=not args.no_log,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            stagger_traj_file_lengths=False,
            save_trajs_as="npz",
            padded_sampling="none",
            dloader_workers=args.dloader_workers,
            epochs=args.epochs,
            start_learning_at_epoch=args.start_learning_at_epoch,
            train_batches_per_epoch=0 if eval_mode else 1000,
            always_save_latest=True,
            always_load_latest=False,
            batches_per_update=1,
            lr_warmup_steps=750,
            grad_clip=args.grad_clip,
            l2_coeff=1e-4,
            mixed_precision="no",
            val_interval=args.val_interval,
            ckpt_interval=args.ckpt_interval,
            **collect_kwargs,
        )
        experiment.start()
        if args.dry_run:
            print(f"Dry run: wrote {os.path.join(experiment.ckpt_dir, 'config.txt')}")
            return
        if args.ckpt is not None:
            experiment.load_checkpoint(epoch=args.ckpt)
        if eval_mode:
            run_checkpoint_eval(
                experiment,
                source_run_name,
                [("heldout", make_test_env)],
                timesteps=eval_timesteps,
                last_n=args.eval_last_n_ckpts,
            )
            wandb.finish(quiet=True)
            continue
        experiment.learn()

        done_path = os.path.join(experiment.ckpt_dir, "DONE")
        with open(done_path, "w") as f:
            f.write(f"Completed at {datetime.now().isoformat()}\n")
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
