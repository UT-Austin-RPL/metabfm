#!/usr/bin/env bash
# Wind.
#
#   bash scripts/wind.sh <method> train|eval|dry_run [extra train_wind flags]
#
# method:  supervised | bfm_lcl | metabfm | any baseline preset name
# train:   1000 epochs, k=3 episodes per training trial. Logs validation on
#          the training task distribution to wandb.
# eval:    evaluates the last 5 checkpoints of the training run on the
#          held-out Test-ID and Test-OOD tasks (35 episodes per trial,
#          Thompson sigma=6). Logs to the wandb run EVAL_<run> and writes
#          $METABFM_DIR/rl_ckpts/EVAL_<run>/ckpts/eval_results/eval_results.jsonl.
# dry_run: write ckpts/config.txt for the train config and exit.
#
# Env: PREFIX (run-name prefix, default "release"), METABFM_DIR,
# METABFM_BUFFER_DIR, WANDB_PROJECT, WANDB_ENTITY.

source "$(dirname "$0")/_common.sh"
[[ $# -ge 2 ]] || usage
preset="$(resolve_preset "$1")"
mode="$2"
shift 2

common=(
  --run_name "${PREFIX}_${preset}"
  --baseline "$preset"
  --trials 1
  --k_train_episodes 3
  --parallel_actors 80
)
train=(--epochs 1000 --start_learning_at_epoch 5 --k_eval_episodes 10)
eval=(--eval_last_n_ckpts 5 --k_eval_episodes 35 --gin "ThompsonSampling.sigma = 6")

case "$mode" in
  train)   python scripts/train_wind.py "${common[@]}" "${train[@]}" "$@" ;;
  eval)    python scripts/train_wind.py "${common[@]}" "${eval[@]}" "$@" ;;
  dry_run) python scripts/train_wind.py "${common[@]}" "${train[@]}" --dry_run "$@" ;;
  *) usage ;;
esac
