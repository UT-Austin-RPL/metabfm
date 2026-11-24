"""Wind launcher: BilevelAgent (TDJepa) on the 2-D Wind point-robot task.

Wind (amago's ``WindEnv``, wrapped by :mod:`metabfm.rl.wind`) is a meta-RL
navigation task with a hidden wind vector and a hidden goal on the unit
circle, both fixed for a meta-trial of ``--k_train_episodes`` episodes.

* Train / validation: winds from ``--train_wind_seed``, ``--n_goals`` goals
  from ``--goal_seed`` outside ``--excluded_quadrant``.
* Test-ID: fresh winds (``--test_wind_seed``) and goals (``--test_goal_seed``)
  from the training quadrants.
* Test-OOD: the same, with goals restricted to the held-out quadrant.

Eval mode (``--eval_last_n_ckpts N``) evaluates the last N checkpoints on
Test-ID and Test-OOD; see :mod:`metabfm.rl.launch_utils`. Gin ladder:
``--baseline`` preset, then ``--env_config``, ``--overlay``, ``--gin``.

Usage (normally via ``bash scripts/wind.sh <method> <train|eval|dry_run>``):
    python scripts/train_wind.py --run_name wind_metabfm \\
        --baseline metabfm_alphabeta --epochs 1000 --parallel_actors 80
"""

import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime

import gin
import wandb

from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.envs.builtin.wind import WindEnv
from amago.loading import DiskTrajDataset

from metabfm import config as C
from metabfm.rl.wind import WindDictObs


_WIND_BUFFER_ROOT = C.BUFFER_DIR


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
        description="Train BilevelAgent on the 2-D Wind env (online launcher)."
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
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--parallel_actors", type=int, default=128)
    parser.add_argument("--dloader_workers", type=int, default=4)
    parser.add_argument("--no_eval", action="store_true")
    add_logging_args(parser, default_project="metabfm-wind")
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
    parser.add_argument(
        "--k_train_episodes",
        type=int,
        default=3,
        help="Inner episodes per meta-trial during training. Default 3 "
        "(225 total steps with --max_episode_steps=75).",
    )
    parser.add_argument(
        "--k_eval_episodes",
        type=int,
        default=10,
        help="Inner episodes per meta-trial during validation and checkpoint "
        "eval. Default 10 (751 total steps) probes how well the policy "
        "keeps adapting beyond its training horizon.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=75,
        help="Steps per inner episode (matches the original POMDP-baselines wind setting).",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=80,
        help="Number of unique winds in the discrete task set sampled from "
        "each wind seed. The training set and the held-out test set each "
        "contain n_tasks winds, drawn from disjoint RNG streams.",
    )
    parser.add_argument(
        "--train_wind_seed",
        type=int,
        default=1337,
        help="RNG seed for the train/val wind set. Mirrors the original "
        "POMDP-baselines deterministic seed.",
    )
    parser.add_argument(
        "--test_wind_seed",
        type=int,
        default=9999,
        help="RNG seed for the held-out test wind set used by "
        "checkpoint eval (--eval_last_n_ckpts). Should differ "
        "from --train_wind_seed for a true train/test split.",
    )
    # ── Random goals on the unit circle ──────────────────────────────────
    parser.add_argument(
        "--n_goals",
        type=int,
        default=80,
        help="Number of goals sampled on the unit circle. Each (wind, "
        "goal) pair is a task drawn from the cross product. Pass 0 (or "
        "any non-positive int) to use a single fixed goal at (0, 1).",
    )
    parser.add_argument(
        "--goal_seed",
        type=int,
        default=4242,
        help="Seed for the *training* goal-set RNG (used by both "
        "make_train_env and make_val_env so in-loop validation tracks "
        "training progress on identical goals). Ignored when --n_goals <= 0.",
    )
    parser.add_argument(
        "--test_goal_seed",
        type=int,
        default=8484,
        help="Seed for the held-out *test* goal-set RNG. Distinct from "
        "--goal_seed so checkpoint eval (both ID and OOD) samples fresh "
        "goals never seen during training. Ignored "
        "when --n_goals <= 0.",
    )
    parser.add_argument(
        "--excluded_quadrant",
        type=int,
        default=2,
        choices=(0, 1, 2, 3, 4),
        help="Quadrant of the unit circle held out from training (math "
        "convention, CCW from +x). The training env blocks this quadrant "
        "and the held-out OOD test env restricts to it. Pass 0 to disable "
        "the held-out split (train and OOD test both sample from all four "
        "quadrants — OOD then collapses into ID semantically). Ignored "
        "when --n_goals <= 0.",
    )
    parser.add_argument(
        "--dense_reward",
        action=BooleanOptionalAction,
        default=True,
        help="Use a dense shaping reward exp(-||state - goal||) plus a +1 "
        "success bonus when inside the goal disk (sharp peak ~2.0 in-disk "
        "vs ~1.0 at the boundary), instead of the sparse 0/1 goal-disk "
        "reward. Applied identically to train/val/test envs. Pass "
        "--no-dense_reward to use the sparse signal.",
    )
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
        "--sf_gamma",
        type=float,
        default=None,
        help=(
            "Override TDJepa.sf_gamma (the discount used by phi/psi/"
            "maximized_phi/target_* SF predictors). Defaults to the gin "
            "binding (0.97). The SF Bellman backup uses the deployment "
            "policy \u03c0_{\u03b3_i} but discounts at \u03b3_sf, so picking "
            "\u03b3_sf < \u03b3 = \u03b3_i shortens the SF horizon (lower-"
            "variance bootstrap targets) without changing what policy is "
            "being represented. Set equal to TDJepa.gamma for a "
            "single-gamma SF backup."
        ),
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
        default="env/wind",
        help="Domain overlay gin file, applied last in the ladder so it "
        "overrides component defaults (default: env/wind).",
    )
    parser.add_argument(
        "--num_collection_batches",
        type=int,
        default=1,
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
            "Epoch at which online collection kicks in. Default 0 = collect "
            "from the start. Pass a positive int to delay collection, or "
            "'inf' to disable it entirely and train only on the prefilled "
            "replay buffer."
        ),
    )
    parser.add_argument(
        "--start_learning_at_epoch",
        type=int,
        default=10,
        help="Epoch at which the inner RL learner begins gradient updates.",
    )
    parser.add_argument(
        "--no_allow_val",
        action="store_true",
        help="Skip the zero-alpha eval phase (val / test).",
    )
    parser.add_argument(
        "--no_allow_val_alpha",
        action="store_true",
        help="Skip the alpha-schedule eval phase (val-alpha / test-alpha).",
    )
    parser.add_argument(
        "--no_allow_val_beta",
        action="store_true",
        help="Skip the beta-schedule eval phase (val-beta / test-beta).",
    )
    args = parser.parse_args()
    if args.dry_run:
        args.no_log = True

    exploration_config = {}
    exploration_wrapper_type = cli_utils.switch_exploration(
        exploration_config,
        strategy="egreedy",
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        steps_anneal=args.steps_anneal,
    )

    baseline_label = args.baseline
    base_configs = BASELINE_PRESETS[baseline_label]

    config_paths = [rl_configs.get(c) for c in base_configs]
    config_paths.append(rl_configs.get(args.env_config))
    for ov in args.overlay or []:
        config_paths.append(rl_configs.get(ov))
    cli_utils.use_config(exploration_config, config_paths, finalize=False)
    # CLI sugar: --sf_gamma / --online_coeff / --offline_coeff <v> are
    # applied as gin bindings before the raw --gin pass-through so
    # explicit `--gin TDJepa.<key>=...` bindings still win.
    if args.sf_gamma is not None:
        gin.parse_config(f"TDJepa.sf_gamma = {float(args.sf_gamma)}")
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

    # Matches the original POMDP-baselines / amago WindEnv setting.
    wind_max = 0.08

    # ── Train / test goal splits ──────────────────────────────────────────
    # n_goals <= 0 passes None down so WindEnv uses its single fixed (0, 1)
    # goal. Otherwise training blocks --excluded_quadrant and the OOD test
    # env restricts to it; --excluded_quadrant=0 disables the split.
    if args.n_goals is None or args.n_goals <= 0:
        n_goals = None
        train_goal_quadrants = None
        test_goal_quadrants = None
    else:
        n_goals = args.n_goals
        if args.excluded_quadrant == 0:
            train_goal_quadrants = None
            test_goal_quadrants = None
        else:
            eq = args.excluded_quadrant
            train_goal_quadrants = [q for q in (1, 2, 3, 4) if q != eq]
            test_goal_quadrants = [eq]

    def _make_env_factory(
        env_name: str,
        k: int,
        wind_seed: int,
        goal_seed: int,
        goal_quadrants,
    ):
        def factory():
            base = WindEnv(
                max_episode_steps=args.max_episode_steps,
                n_tasks=args.n_tasks,
                k_episodes=k,
                wind_seed=wind_seed,
                wind_max=wind_max,
                dense_reward=args.dense_reward,
                n_goals=n_goals,
                goal_seed=goal_seed,
                goal_quadrants=goal_quadrants,
            )
            return AMAGOEnv(
                WindDictObs(base),
                env_name=env_name,
                batched_envs=1,
            )

        return factory

    # Train / val share the *training* wind set + *training* goal set so
    # the in-loop val curve tracks training progress (not generalization).
    make_train_env = _make_env_factory(
        env_name="WindEnv-Train",
        k=args.k_train_episodes,
        wind_seed=args.train_wind_seed,
        goal_seed=args.goal_seed,
        goal_quadrants=train_goal_quadrants,
    )
    make_val_env = _make_env_factory(
        env_name="WindEnv-Train",  # same env_name as train: val tracks training
        k=args.k_eval_episodes,
        wind_seed=args.train_wind_seed,
        goal_seed=args.goal_seed,
        goal_quadrants=train_goal_quadrants,
    )
    # Test envs for checkpoint eval: distinct env_names so ID and OOD log as
    # separate wandb series. ID = fresh winds + goals from the training
    # quadrants; OOD = same winds, goals from the held-out quadrant.
    make_test_env_id = _make_env_factory(
        env_name="WindEnv-Test-ID",
        k=args.k_eval_episodes,
        wind_seed=args.test_wind_seed,
        goal_seed=args.test_goal_seed,
        goal_quadrants=train_goal_quadrants,
    )
    make_test_env_ood = _make_env_factory(
        env_name="WindEnv-Test-OOD",
        k=args.k_eval_episodes,
        wind_seed=args.test_wind_seed,
        goal_seed=args.test_goal_seed,
        goal_quadrants=test_goal_quadrants,
    )

    # The "v10" suffix does not depend on --k_eval_episodes, so eval runs with
    # a different k_eval still resolve to the training run's ckpt dir.
    if args.k_train_episodes == 10:
        group_name = f"{args.run_name}_wind_k{args.k_train_episodes}"
    else:
        group_name = f"{args.run_name}_wind_k{args.k_train_episodes}v10"

    for trial in range(args.trial_offset, args.trial_offset + args.trials):
        run_name = f"{group_name}_trial_{trial}"

        # Eval mode runs under a fresh EVAL_* name so the source ckpt dir
        # stays untouched; source_run_name locates its policy_weights/.
        source_run_name = run_name
        if args.eval_last_n_ckpts > 0:
            run_name = f"{args.eval_run_prefix}_{source_run_name}"

        online_dset_name = f"online_{run_name}"
        if args.eval_last_n_ckpts == 0 and args.ckpt is None:
            require_fresh_run(run_name, _WIND_BUFFER_ROOT, online_dset_name)
        dataset = DiskTrajDataset(
            dset_root=_WIND_BUFFER_ROOT,
            dset_name=online_dset_name,
            dset_max_size=100_000,
            relabeler=None,
        )

        train_batches = 0 if args.eval_last_n_ckpts > 0 else 1000

        # Trial geometry: max_episode_steps * k_train_episodes = 75 * 3
        # = 225 env steps, plus the trailing soft-reset observation = 226.
        # Eval uses k_eval_episodes (default 10) -> 75 * 10 + 1 = 751.
        steps_per_round = args.max_episode_steps * args.k_train_episodes + 1
        eval_steps_per_round = args.max_episode_steps * args.k_eval_episodes + 1
        train_timesteps = args.num_collection_batches * steps_per_round
        # Wind step is cheap; one eval trial per actor is already plenty
        # of variance reduction at parallel_actors >= 64.
        eval_rounds_per_actor = 1

        # Wind step is pure numpy: env_mode="sync" wins outright (no IPC,
        # no pickle).
        if args.eval_last_n_ckpts > 0:
            collect_kwargs = dict(env_mode="sync", **eval_collect_kwargs())
        else:
            collect_kwargs = dict(
                env_mode="sync",
                exploration_wrapper_type=exploration_wrapper_type,
                has_dset_edit_rights=True,
                start_collecting_at_epoch=args.start_collecting_at_epoch,
                train_timesteps_per_epoch=train_timesteps,
                num_collection_batches=args.num_collection_batches,
                steps_per_round=steps_per_round,
            )

        experiment = BilevelExperiment(
            baseline_preset=baseline_label,
            gin_ladder=gin_ladder_log,
            run_name=run_name,
            ckpt_base_dir=C.RL_CKPT_DIR,
            dataset=dataset,
            val_timesteps_per_epoch=(
                0 if args.no_eval else eval_steps_per_round * eval_rounds_per_actor
            ),
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            parallel_actors=args.parallel_actors,
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
            train_batches_per_epoch=train_batches,
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
        experiment.allow_val = not args.no_allow_val
        experiment.allow_val_alpha = not args.no_allow_val_alpha
        experiment.allow_val_beta = not args.no_allow_val_beta
        experiment.start()
        if args.dry_run:
            print(f"Dry run: wrote {os.path.join(experiment.ckpt_dir, 'config.txt')}")
            return
        if args.ckpt is not None:
            experiment.load_checkpoint(epoch=args.ckpt)

        if args.eval_last_n_ckpts > 0:
            test_envs = [("ID", make_test_env_id)]
            if n_goals is not None and args.excluded_quadrant != 0:
                test_envs.append(("OOD", make_test_env_ood))
            run_checkpoint_eval(
                experiment,
                source_run_name,
                test_envs,
                timesteps=eval_steps_per_round * eval_rounds_per_actor,
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
