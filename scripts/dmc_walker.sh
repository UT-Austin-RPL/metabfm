#!/usr/bin/env bash
# DMC Remastered Walker.
#
#   bash scripts/dmc_walker.sh <method> train|eval|dry_run [extra train_dmc_walker flags]
#
# method:  supervised | bfm_lcl | metabfm | any baseline preset name
# train:   1500 epochs on the walk reward with N_SEEDS training bodies (n1, n10,
#          or n10k). Logs validation (stand / walk / run on training bodies)
#          to wandb.
# eval:    evaluates the last 3 checkpoints (epochs 1400, 1450, 1500) on training
#          and held-out bodies under six rewards (10 episodes per trial,
#          Thompson sigma=0.03; MetaBFM's alpha phase decays alpha 1 -> 0 over
#          10 episodes). Logs to the wandb run EVAL_<run> and writes
#          $METABFM_DIR/rl_ckpts/EVAL_<run>/ckpts/eval_results/eval_results.jsonl.
# dry_run: write ckpts/config.txt for the train config and exit.
#
# Env: PREFIX (default "release"), N_SEEDS (default n10k), METABFM_DIR,
# METABFM_BUFFER_DIR (replay buffers; keep on local disk), WANDB_PROJECT,
# WANDB_ENTITY.

source "$(dirname "$0")/_common.sh"
[[ $# -ge 2 ]] || usage
preset="$(resolve_preset "$1")"
mode="$2"
shift 2
N_SEEDS="${N_SEEDS:-n10k}"
case "$N_SEEDS" in
  n1) n=1 ;; n10) n=10 ;; n10k) n=10000 ;;
  *) echo "N_SEEDS must be n1, n10, or n10k" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
common=(
  --run_name "${PREFIX}_dmcw_${preset}_${N_SEEDS}"
  --baseline "$preset"
  --n_dynamics_seeds "$n"
  --env_mode sync
)
train=(--parallel_actors 32 --epochs 1500 --ckpt_interval 50 --dloader_workers 8)
eval=(
  --eval_last_n_ckpts 3
  --parallel_actors 64
  --eval_k 10
  --dloader_workers 0
  --gin "ThompsonSampling.sigma = 0.03"
)
if [[ "$preset" == metabfm_alphabeta ]]; then
  eval+=(
    --gin "TDJepa.val_alpha_schedule = @alpha_val/LinearDecaySchedule()"
    --gin "alpha_val/LinearDecaySchedule.high = 1.0"
    --gin "alpha_val/LinearDecaySchedule.low = 0.0"
    --gin "alpha_val/LinearDecaySchedule.decay_episodes = 10"
  )
fi

case "$mode" in
  train)   python scripts/train_dmc_walker.py "${common[@]}" "${train[@]}" "$@" ;;
  eval)    python scripts/train_dmc_walker.py "${common[@]}" "${eval[@]}" "$@" ;;
  dry_run) python scripts/train_dmc_walker.py "${common[@]}" "${train[@]}" --dry_run "$@" ;;
  *) usage ;;
esac
