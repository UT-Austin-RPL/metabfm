#!/usr/bin/env bash
# ColorGrid.
#
#   bash scripts/colorgrid.sh <method> train|eval|dry_run [extra train_colorgrid flags]
#
# method:  supervised | bfm_lcl | metabfm | any baseline preset name
# train:   401 epochs on a 5x5 grid with 10k environments (color permutations)
#          and N_REWARDS reward landscapes. Logs validation on held-out tasks
#          to wandb.
# eval:    evaluates the last 5 checkpoints of the training run on the
#          held-out tasks. Logs to the wandb run EVAL_<run> and writes
#          $METABFM_DIR/rl_ckpts/EVAL_<run>/ckpts/eval_results/eval_results.jsonl.
# dry_run: write ckpts/config.txt for the train config and exit.
#
# Env: PREFIX (default "release"), N_REWARDS (default 10000), METABFM_DIR,
# METABFM_BUFFER_DIR, WANDB_PROJECT, WANDB_ENTITY.

source "$(dirname "$0")/_common.sh"
[[ $# -ge 2 ]] || usage
preset="$(resolve_preset "$1")"
mode="$2"
shift 2
N_REWARDS="${N_REWARDS:-10000}"

common=(
  --run_name "${PREFIX}_cg_${preset}_rew${N_REWARDS}"
  --baseline "$preset"
  --n_envs 10000
  --n_rewards "$N_REWARDS"
  --grid_size 5
  --trials 1
  --parallel_actors 128
)
train=(--epochs 401 --start_learning_at_epoch 5)
eval=(--eval_last_n_ckpts 5)

case "$mode" in
  train)   python scripts/train_colorgrid.py "${common[@]}" "${train[@]}" "$@" ;;
  eval)    python scripts/train_colorgrid.py "${common[@]}" "${eval[@]}" "$@" ;;
  dry_run) python scripts/train_colorgrid.py "${common[@]}" "${train[@]}" --dry_run "$@" ;;
  *) usage ;;
esac
