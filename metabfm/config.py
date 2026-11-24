import os

# Root for checkpoints, eval results, and logs. Override with METABFM_DIR.
PROJECT_DIR = os.environ.get(
    "METABFM_DIR", os.path.expanduser("~/metabfm_cache")
)
RL_CKPT_DIR = os.path.join(PROJECT_DIR, "rl_ckpts")
# Replay buffers grow large (tens of GB per run); point METABFM_BUFFER_DIR
# at a local disk with room to spare.
BUFFER_DIR = os.environ.get(
    "METABFM_BUFFER_DIR", os.path.join(PROJECT_DIR, "buffers")
)
