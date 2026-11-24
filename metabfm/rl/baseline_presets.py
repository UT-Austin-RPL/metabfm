"""Baseline preset → gin ladder mapping for the hybrid α-β ZSRL agent.

Each preset is an ordered list of gin config names (relative to
``metabfm/rl/configs/``) that compose into a complete TDJepa training
configuration.  Launchers append their env overlay (``env/wind``,
``env/coloredgrid``, ...) after the preset ladder.

All six presets run through the same TDJepa code path:

==================  ===========  =================  ==================  ==========  ========
Baseline            dynamics     α                  β                   intrinsic   policy
==================  ===========  =================  ==================  ==========  ========
pure_extrinsic      noop         pinned_intrinsic   pinned_extrinsic    noop        tformer
intrinsic_meta_rl   latent_idm   pinned_intrinsic   varying             full        tformer
vanilla_zsrl        noop         off                off                 noop        ff
dyncs_zsrl          latent_idm   off                off                 noop        ff
metabfm_alpha       latent_idm   varying            off                 full        tformer
metabfm_alphabeta   latent_idm   varying            varying             full        tformer
==================  ===========  =================  ==================  ==========  ========

* ``pure_extrinsic`` (supervised meta-RL): AMAGO actor-critic on Q_I
  trained from extrinsic rewards only (α≡1, β≡1).
* ``intrinsic_meta_rl``: intrinsic-curiosity meta-RL with a family of
  policies parameterized by β.
* ``vanilla_zsrl`` / ``dyncs_zsrl``: zero-shot RL (α≡β≡0); the intrinsic
  stack and extrinsic encoder are skipped.  ``dyncs_zsrl`` (BFM+LCL) adds
  the ``latent_idm`` dynamics learner.
* ``metabfm_alpha``: α-conditioned ZSRL (α∈[0, 1], β≡0), blending SF with
  intrinsic Q_I.
* ``metabfm_alphabeta``: the full MetaBFM agent (α and β both vary).

Inference is Thompson sampling with a σ=10 BLR posterior
(``inference/ts_sigma10``) for every preset.
"""

from __future__ import annotations


BASELINE_PRESETS: dict[str, list[str]] = {
    "pure_extrinsic": [
        "agent/tdjepa",
        "dynamics/noop",
        "alpha/pinned_intrinsic",
        "beta/pinned_extrinsic",
        "intrinsic/noop",
        "policy/tformer",
        "inference/ts_sigma10",
    ],
    "intrinsic_meta_rl": [
        "agent/tdjepa",
        "dynamics/latent_idm",
        "alpha/pinned_intrinsic",
        "beta/varying",
        "intrinsic/full",
        "policy/tformer",
        "inference/ts_sigma10",
    ],
    "vanilla_zsrl": [
        "agent/tdjepa",
        "dynamics/noop",
        "alpha/off",
        "beta/off",
        "intrinsic/noop",
        "policy/ff",
        "inference/ts_sigma10",
    ],
    "dyncs_zsrl": [
        "agent/tdjepa",
        "dynamics/latent_idm",
        "alpha/off",
        "beta/off",
        "intrinsic/noop",
        "policy/ff",
        "inference/ts_sigma10",
    ],
    "metabfm_alpha": [
        "agent/tdjepa",
        "dynamics/latent_idm",
        "alpha/varying",
        "beta/off",
        "intrinsic/full",
        "policy/tformer",
        "inference/ts_sigma10",
    ],
    "metabfm_alphabeta": [
        "agent/tdjepa",
        "dynamics/latent_idm",
        "alpha/varying",
        "beta/varying",
        "intrinsic/full",
        "policy/tformer",
        "inference/ts_sigma10",
    ],
}
