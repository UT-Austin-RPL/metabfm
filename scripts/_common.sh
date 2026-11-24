# Shared helpers for scripts/{wind,colorgrid,dmc_walker}.sh. Source, don't run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREFIX="${PREFIX:-release}"

python -c 'from metabfm import config as C
print(f"[metabfm] checkpoints + eval results: {C.RL_CKPT_DIR}  (METABFM_DIR)")
print(f"[metabfm] replay buffers:             {C.BUFFER_DIR}  (METABFM_BUFFER_DIR)")' >&2

# Paper method names -> baseline presets (metabfm/rl/baseline_presets.py).
# Preset names are accepted directly as well.
resolve_preset() {
  case "$1" in
    supervised) echo pure_extrinsic ;;
    bfm_lcl)    echo dyncs_zsrl ;;
    metabfm)    echo metabfm_alphabeta ;;
    pure_extrinsic|intrinsic_meta_rl|vanilla_zsrl|dyncs_zsrl|metabfm_alpha|metabfm_alphabeta)
                echo "$1" ;;
    *) echo "unknown method: $1" >&2; exit 2 ;;
  esac
}

usage() {
  sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2
  exit 2
}
