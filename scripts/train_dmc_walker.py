"""DMC Remastered walker meta-RL launcher for BilevelAgent (DynamicsLearner + RL).

Trains on state-based ``walker walk`` over randomized embodiments (torso / foot
geometry, joint ranges, motor damping / armature, ground friction), then
evaluates several reward modes on in-distribution and held-out dynamics-seed
splits. A given dynamics seed is the same body under every reward mode.

Train on ``walk`` rather than ``run`` because the DMC rewards nest (both
multiply a shared stand term by a speed gate, 1 m/s vs 8 m/s): supervising
``walk`` leaves ``run`` as a harder OOD target and ``stand`` as an easier
nested check.

Train mode validates every deployment phase (``val`` / ``val-alpha`` /
``val-beta``) under each core reward mode on the training bodies (wandb keys
``val/walk``, ``val-beta/run``, ...). Eval mode (``--eval_last_n_ckpts N``)
evaluates the last N checkpoints on both body splits under each of
``--eval_rewards`` (``test/test_split/walk``, ...); see
:mod:`metabfm.rl.launch_utils`. Each reward mode needs its own rollouts:
the reward stream is in the actor's context, so behavior is reward-dependent
and rollouts cannot be rescored under another reward.

Usage (normally via ``bash scripts/dmc_walker.sh <method> <train|eval|dry_run>``):
    python scripts/train_dmc_walker.py --run_name dmc_walker_pe_n10 \\
        --baseline pure_extrinsic --n_dynamics_seeds 10 --parallel_actors 24
"""

import itertools
import os
from argparse import ArgumentParser
from datetime import datetime

import gin
import wandb

from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset

from metabfm import config as C
from metabfm.env.dmc_walker import (
    WALKER_CORE_REWARD_MODES,
    WALKER_REWARD_MODES,
    DMCRWalkerMeta,
    make_dynamics_seed_split,
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

#: Reward modes scored in eval mode: the training reward (walk) plus five
#: held-out behaviours.
DEFAULT_EVAL_REWARDS = ["walk", "stand", "run", "walk_backward", "crouch", "crouch_walk"]


class MultiRewardValExperiment(BilevelExperiment):
    """``BilevelExperiment`` whose validation covers several reward modes.

    When ``val_eval_combos`` is set to a list of ``(log_tag, make_env)``
    pairs, ``evaluate_val`` rolls each combo through every deployment phase
    (instead of the single ``make_val_env`` set) and logs under
    ``val/{tag}``, ``val-alpha/{tag}``, ``val-beta/{tag}``.
    """

    val_eval_combos = None  # Optional[list[tuple[str, callable]]]

    def evaluate_val(self) -> None:
        if not self.val_eval_combos:
            return super().evaluate_val()
        timesteps = self.val_timesteps_per_epoch
        if timesteps <= 0:
            return
        for tag, factory in self.val_eval_combos:
            self.evaluate_test(
                factory,
                timesteps=timesteps,
                log_tag=tag,
                phase_prefix="val",
            )


def main():
    parser = ArgumentParser(
        description="Train BilevelAgent on DMC Remastered walker (randomized dynamics)."
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--baseline",
        type=str,
        choices=sorted(BASELINE_PRESETS.keys()),
        default="pure_extrinsic",
        help="One-flag baseline preset selector. Resolves to a fixed gin "
        "ladder via BASELINE_PRESETS. Default pure_extrinsic (alpha=1, "
        "beta=1) is the supervised meta-RL configuration.",
    )

    # ── Task distribution ──
    parser.add_argument(
        "--n_dynamics_seeds",
        type=int,
        default=10,
        help="Number of training dynamics seeds (distinct embodiments). The "
        "sweep axis: {1, 10, 100, 10000}. The held-out test set is fixed "
        "across all values and never overlaps the train pool.",
    )
    parser.add_argument(
        "--n_test_dynamics_seeds",
        type=int,
        default=1000,
        help="Size of the fixed held-out dynamics-seed pool.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for make_dynamics_seed_split. Changing it redraws BOTH "
        "splits, so keep it fixed across the n_dynamics_seeds sweep.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="walk",
        choices=sorted(WALKER_REWARD_MODES.keys()),
        help="Reward function used for training collection. Validation and "
        "checkpoint eval roll every reward in --eval_rewards.",
    )

    # ── Trial geometry ──
    parser.add_argument("--k", type=int, default=2, help="Episodes per train trial.")
    parser.add_argument(
        "--eval_k",
        type=int,
        default=2,
        help="Episodes per eval trial. Kept at the training k: with a "
        "128-step rolling context on a 1001-step trial, longer trials give "
        "the policy no extra memory to exploit.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=250,
        help="Agent steps per inner episode. At frame_skip=2 this is 500 "
        "control steps (half of DMC's usual 1000-step horizon).",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=2,
        help="Control steps per agent step. Reward is summed over the skip.",
    )

    # ── Experiment plumbing ──
    parser.add_argument(
        "--epochs",
        type=int,
        default=1001,
        help="Training length in epochs.",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--parallel_actors", type=int, default=24)
    parser.add_argument(
        "--env_mode",
        type=str,
        default="sync",
        choices=["sync", "async"],
        help="Mujoco steps at ~0.6 ms and each hard reset recompiles the "
        "walker XML (~130 ms, a sizable fraction of a 501-step trial). Sync "
        "is fine at 24 actors; flip to async if wall-clock becomes painful.",
    )
    parser.add_argument(
        "--dset_root",
        type=str,
        default=C.BUFFER_DIR,
        help="Filesystem root for the replay buffer (default: $METABFM_BUFFER_DIR). "
        "Keep it on a local SSD rather than NFS: the buffer is read once per training "
        "batch (32 trajectory files), and NFS reads measure ~67 ms/file "
        "against ~0.04 ms locally, which pins the dataloader workers in "
        "uninterruptible IO wait and idles the GPU. Checkpoints and wandb "
        "logs still go to C.RL_CKPT_DIR -- they are written rarely enough "
        "that NFS latency does not matter.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=8,
        help="Torch dataloader workers per run.",
    )
    parser.add_argument("--no_eval", action="store_true")
    add_logging_args(parser, default_project="metabfm-dmc-walker")
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
        help="Extra registered gin config(s) layered on top of the ladder. "
        "Applied AFTER the env overlay and BEFORE --gin bindings.",
    )
    parser.add_argument(
        "--eval_rounds",
        type=int,
        default=1,
        help="Meta-trials per actor per eval phase. Sample size is "
        "parallel_actors * eval_rounds trials.",
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
    parser.add_argument(
        "--eval_splits",
        type=str,
        action="append",
        default=None,
        choices=["train_split", "test_split"],
        help="Restrict eval mode to these dynamics splits. "
        "Repeatable; defaults to both.",
    )
    parser.add_argument(
        "--eval_rewards",
        type=str,
        action="append",
        default=None,
        choices=sorted(WALKER_REWARD_MODES.keys()),
        help="Reward modes evaluated in eval mode. Repeatable; defaults to "
        f"{DEFAULT_EVAL_REWARDS}.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Write one mp4 per eval trial under <video_dir>/<reward_mode>/. "
        "Only actor 0 records. Observations stay state-based either way.",
    )
    parser.add_argument(
        "--video_trials",
        type=int,
        default=1,
        help="Videos to save per recording env before it stops rendering.",
    )
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--ckpt_interval", type=int, default=50)
    parser.add_argument(
        "--env_config",
        type=str,
        default="env/dmc_walker",
        help="Domain overlay gin file, applied last in the ladder.",
    )
    parser.add_argument(
        "--num_collection_batches",
        type=int,
        default=1,
        help="Number of z-proposal rounds per epoch during online collection.",
    )
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--steps_anneal", type=int, default=500_000)
    parser.add_argument("--start_collecting_at_epoch", type=float, default=0.0)
    parser.add_argument(
        "--start_learning_at_epoch",
        type=int,
        default=10,
        help="Epoch at which the inner RL learner begins gradient updates.",
    )
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument(
        "--online_coeff",
        type=float,
        default=None,
        help="Override TDJepa.online_coeff (DPG online actor term).",
    )
    parser.add_argument(
        "--offline_coeff",
        type=float,
        default=None,
        help="Override TDJepa.offline_coeff (AWAC-style filtered-BC term).",
    )
    args = parser.parse_args()
    if args.dry_run:
        args.no_log = True

    # ── Exploration (egreedy for online collection) ──
    exploration_config = {}
    exploration_wrapper_type = cli_utils.switch_exploration(
        exploration_config,
        strategy="egreedy",
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        steps_anneal=args.steps_anneal,
    )

    # ── Gin ladder ──
    baseline_label = args.baseline
    base_configs = BASELINE_PRESETS[baseline_label]

    config_paths = [rl_configs.get(c) for c in base_configs]
    config_paths.append(rl_configs.get(args.env_config))
    for ov in args.overlay or []:
        config_paths.append(rl_configs.get(ov))
    cli_utils.use_config(exploration_config, config_paths, finalize=False)
    # Severing policy_ctx / z at the actor input is only meaningful for the
    # pure_extrinsic ablation, where it makes the agent a provably SF-free
    # AMAGO actor-critic. Every other preset needs those features: metabfm_*
    # loses its SF branch without them, and the alpha/off + beta/off presets
    # (dyncs_zsrl, vanilla_zsrl) also skip the extrinsic encoder, which would
    # leave the actor with an all-zero input and train silently to garbage.
    # Derived here rather than bound in env/dmc_walker.gin so it cannot fall
    # out of sync with --baseline; an explicit --gin below still wins.
    gin.parse_config(f"TDJepa.zero_policy_ctx = {baseline_label == 'pure_extrinsic'}")
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

    # ── Dynamics-seed split ──
    split = make_dynamics_seed_split(
        n_train=args.n_dynamics_seeds,
        n_test=args.n_test_dynamics_seeds,
        seed=args.split_seed,
    )
    assert set(split.train).isdisjoint(split.test)
    print(
        f"  Dynamics seeds: {len(split.train)} train / {len(split.test)} test "
        f"(disjoint, split_seed={args.split_seed})"
    )
    print(f"  Buffer root: {args.dset_root} ({args.dloader_workers} dloader workers)")
    print(f"  Ckpt root:   {C.RL_CKPT_DIR}")

    # ── Environments ──
    # Each factory owns its own actor counter so the i-th actor of a given
    # factory always gets the same body-sampling seed. Building the eval
    # factories fresh per sweep therefore evaluates the SAME set of bodies at
    # every checkpoint, which keeps checkpoint-to-checkpoint deltas readable.
    def make_env_factory(
        seeds, reward_mode: str, k: int, env_name: str, seed_base: int, video_dir=None
    ):
        counter = itertools.count()

        def factory():
            actor_idx = next(counter)
            # Only actor 0 records: every actor rendering would multiply the
            # render cost by parallel_actors and write redundant videos.
            render_dir = None
            if video_dir is not None and actor_idx == 0:
                render_dir = os.path.join(video_dir, reward_mode)
            return AMAGOEnv(
                DMCRWalkerMeta(
                    dynamics_seeds=seeds,
                    reward_mode=reward_mode,
                    k=k,
                    max_episode_steps=args.max_episode_steps,
                    frame_skip=args.frame_skip,
                    seed=seed_base + actor_idx,
                    render_dir=render_dir,
                    render_trials=args.video_trials,
                ),
                env_name=env_name,
                batched_envs=1,
            )

        return factory

    dset_tag = f"seeds_{args.n_dynamics_seeds}_{args.reward_mode}"

    make_train_env = make_env_factory(
        split.train, args.reward_mode, args.k, f"DMCRWalker-Train-{dset_tag}", 0
    )
    # Needed for Experiment init (builds self.val_envs); evaluate_val is
    # overridden to roll all three reward modes via val_eval_combos instead.
    make_val_env = make_env_factory(
        split.train,
        args.reward_mode,
        args.eval_k,
        f"DMCRWalker-Val-{dset_tag}",
        1_000_000,
    )
    # Train-split bodies, one factory per active reward. Distinct seed_bases
    # so the three reward passes draw independent actor streams on the same
    # body pool. Tags become wandb suffixes: val/run, val-alpha/walk, ...
    val_eval_combos = [
        (
            reward_mode,
            make_env_factory(
                split.train,
                reward_mode,
                args.eval_k,
                f"DMCRWalker-Val-{reward_mode}",
                1_000_000 + 10_000 * i,
            ),
        )
        for i, reward_mode in enumerate(sorted(WALKER_CORE_REWARD_MODES))
    ]

    # (split, reward) combos. Distinct seed_bases so the ID and OOD passes
    # draw independent actor streams, and each reward mode is scored on the
    # same bodies within a split.
    test_combos = [
        ("train_split", split.train, 2_000_000),
        ("test_split", split.test, 3_000_000),
    ]
    if args.eval_splits:
        keep = set(args.eval_splits)
        test_combos = [c for c in test_combos if c[0] in keep]
    test_rewards = args.eval_rewards or DEFAULT_EVAL_REWARDS

    # ── Training ──
    group_name = f"{args.run_name}_dmcwalker_{dset_tag}_k{args.k}"

    for trial in range(args.trial_offset, args.trial_offset + args.trials):
        run_name = f"{group_name}_trial_{trial}"
        source_run_name = run_name
        eval_mode = args.eval_last_n_ckpts > 0
        if eval_mode:
            run_name = f"{args.eval_run_prefix}_{source_run_name}"

        if not eval_mode and args.ckpt is None:
            require_fresh_run(run_name, args.dset_root, f"online_{run_name}")
        dataset = DiskTrajDataset(
            dset_root=args.dset_root,
            dset_name=f"online_{run_name}",
            dset_max_size=50_000,
            relabeler=None,
        )

        # Trial geometry: max_episode_steps * k agent steps plus the initial
        # reset observation. At the defaults, 250 * 2 + 1 = 501, so one
        # collection batch is exactly one meta-trial per actor per epoch.
        steps_per_round = args.max_episode_steps * args.k + 1
        eval_steps_per_round = args.max_episode_steps * args.eval_k + 1
        train_timesteps = args.num_collection_batches * steps_per_round
        eval_rounds_per_actor = args.eval_rounds
        if eval_mode:
            collect_kwargs = eval_collect_kwargs()
        else:
            collect_kwargs = dict(
                exploration_wrapper_type=exploration_wrapper_type,
                has_dset_edit_rights=True,
                start_collecting_at_epoch=args.start_collecting_at_epoch,
                train_timesteps_per_epoch=train_timesteps,
            )

        experiment = MultiRewardValExperiment(
            baseline_preset=baseline_label,
            gin_ladder=gin_ladder_log,
            num_collection_batches=args.num_collection_batches,
            steps_per_round=steps_per_round,
            run_name=run_name,
            ckpt_base_dir=C.RL_CKPT_DIR,
            dataset=dataset,
            val_timesteps_per_epoch=(
                0 if args.no_eval else eval_steps_per_round * eval_rounds_per_actor
            ),
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            parallel_actors=args.parallel_actors,
            env_mode=args.env_mode,
            sample_actions_train=True,
            sample_actions_val=True,
            log_to_wandb=not args.no_log,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            stagger_traj_file_lengths=False,
            save_trajs_as="npz",
            # "left" rather than "none": with max_seq_len=128 on a
            # 501-step trial, "none" restricts window starts to [0, 373] and
            # undersamples the first ~127 steps by up to 128x -- exactly the
            # high-task-uncertainty region this experiment is about.
            padded_sampling="left",
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
        if not args.no_eval:
            experiment.val_eval_combos = val_eval_combos

        experiment.allow_val = not args.no_allow_val
        experiment.allow_val_alpha = not args.no_allow_val_alpha
        experiment.allow_val_beta = not args.no_allow_val_beta
        experiment.start()
        if args.dry_run:
            print(f"Dry run: wrote {os.path.join(experiment.ckpt_dir, 'config.txt')}")
            return
        if args.ckpt is not None:
            experiment.load_checkpoint(epoch=args.ckpt)
        if eval_mode:
            test_envs = [
                (
                    f"{split_tag}/{reward_mode}",
                    make_env_factory(
                        seeds,
                        reward_mode,
                        args.eval_k,
                        f"DMCRWalker-Test-{split_tag}-{reward_mode}",
                        seed_base,
                        video_dir=args.video_dir,
                    ),
                )
                for split_tag, seeds, seed_base in test_combos
                for reward_mode in test_rewards
            ]
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
