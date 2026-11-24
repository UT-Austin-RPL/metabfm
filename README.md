# MetaBFM

This repository contains code for the paper [Meta-Reinforcement Learning with Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2611.10553) (NeurIPS 2026). It is built on [AMAGO](https://github.com/UT-Austin-RPL/amago).

## Installation

```bash
git clone https://github.com/UT-Austin-RPL/metabfm.git && cd metabfm
conda env create -f environment.yml
conda activate metabfm
pip install flash-attn --no-build-isolation
```

## Running experiments

Each command trains or evaluates one method in one domain on one GPU:

```bash
bash scripts/<domain>.sh <method> train     # train, logging validation to wandb
bash scripts/<domain>.sh <method> eval      # evaluate the trained run's last checkpoints on held-out tasks
bash scripts/<domain>.sh <method> dry_run   # build everything, write the config, and exit
```

- `<domain>`: `wind`, `colorgrid`, or `dmc_walker`
- `<method>`:
  - `supervised`: Supervised Meta-RL (in-context adaptation to training rewards)
  - `vanilla_zsrl`: BFM (zero-shot RL to new rewards)
  - `bfm_lcl`: BFM+LCL (BFM + latent context learning, to generalize across environment dynamics)
  - `metabfm`: MetaBFM (BFM meta-trained to explore and adapt in-context)
  - `metabfm_alpha`: MetaBFM with only α varying
  - `intrinsic_meta_rl`: Supervised Meta-RL + intrinsic curiosity reward

Run `eval` after `train` with the same arguments. Extra flags are passed through to the launcher (`scripts/train_<domain>.py`) and override the script's settings. Examples:

```bash
bash scripts/wind.sh metabfm train
N_REWARDS=10 bash scripts/colorgrid.sh metabfm train      # number of training reward functions (default 10000)
N_SEEDS=n10 bash scripts/dmc_walker.sh metabfm train      # n1, n10, or n10k training bodies (default n10k)
```

Checkpoints and eval results are written to `$METABFM_DIR` (default `~/metabfm_cache`), and replay buffers to `$METABFM_BUFFER_DIR` (default `$METABFM_DIR/buffers`). Runs log to wandb under your default entity; set `WANDB_ENTITY` / `WANDB_PROJECT`, or pass `--no_log`.

## Citation

```bibtex
@inproceedings{grigsby2026metabfm,
  title     = {Meta-Reinforcement Learning with Zero-Shot Reinforcement Learning},
  author    = {Grigsby, Jake and Agarwal, Siddhant and Lei, Yu and Varveropoulos, Leonidas and Zhu, Yuke},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
