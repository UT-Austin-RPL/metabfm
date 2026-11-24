"""
Batched Bayesian Linear Regression (BLR) in PyTorch.

PyTorch port of the core of opti-bfm/utils/blr.py with a leading batch
dimension, so N parallel actors each keep an independent posterior.  Used
for Thompson sampling (``sample``) and the kappa-gated update (``cov_norm``).

  eta : (N, d)     — natural parameter (precision @ mean)
  R   : (N, d, d)  — upper-triangular Cholesky factor of the precision
                      P = R^T R,  Sigma = P^{-1},  mu = cho_solve(R, eta)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def _cholesky_rank1_update(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """In-place-safe Cholesky rank-1 update: R' such that R'^T R' = R^T R + x x^T.

    R : (..., d, d)  upper-triangular
    x : (..., d)

    Implements the standard sequential column algorithm (Golub & Van Loan §12.5.4)
    in O(d^2) per batch element, vectorised across the batch dims.
    """
    R = R.clone()
    x = x.clone()
    d = R.shape[-1]
    for k in range(d):
        r_kk = R[..., k, k]
        x_k = x[..., k]
        new_diag = torch.sqrt(r_kk * r_kk + x_k * x_k)
        c = new_diag / r_kk
        s = x_k / r_kk
        R[..., k, k] = new_diag
        if k + 1 < d:
            R[..., k, k + 1 :] = (
                R[..., k, k + 1 :] + s.unsqueeze(-1) * x[..., k + 1 :]
            ) / c.unsqueeze(-1)
            x[..., k + 1 :] = (
                c.unsqueeze(-1) * x[..., k + 1 :] - s.unsqueeze(-1) * R[..., k, k + 1 :]
            )
    return R


@dataclass
class BLR:
    """Batched Bayesian Linear Regression posterior.

    All tensors carry a leading batch dimension N (number of parallel actors).
    """

    eta: torch.Tensor  # (N, d)
    R: torch.Tensor  # (N, d, d)  upper-triangular Cholesky of precision
    lam: float = 1.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def create_LSQ(
        dim: int,
        lam: float = 1.0,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> BLR:
        """Initialise N independent priors with precision lam * I and mean 0."""
        R = (
            math.sqrt(lam)
            * torch.eye(dim, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        eta = torch.zeros(batch_size, dim, device=device)
        return BLR(eta=eta, R=R, lam=lam)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def rank1_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float = 1.0,
        decay: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """In-place rank-1 posterior update with observation r = x^T theta + N(0, sigma^2).

        x : (N, d)   feature vector per actor
        y : (N,)      scalar reward per actor
        mask : (N,) bool  optional per-actor gate (e.g. the kappa threshold)
        """
        x = x / sigma
        y = y / sigma
        if decay != 1.0:
            sqrt_decay = math.sqrt(decay)
            self.R = self.R * sqrt_decay
            self.eta = self.eta * decay
        R_new = _cholesky_rank1_update(self.R, x)
        eta_new = self.eta + x * y.unsqueeze(-1)
        if mask is not None:
            m = mask.view(-1, 1, 1)
            self.R = torch.where(m, R_new, self.R)
            self.eta = torch.where(mask.unsqueeze(-1), eta_new, self.eta)
        else:
            self.R = R_new
            self.eta = eta_new

    def batch_update(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sigma: float = 1.0,
    ) -> None:
        """Update each actor with a batch of observations (for warm-starting).

        X : (N, M, d)  M observations per actor
        y : (N, M)      rewards
        """
        X = X / sigma
        y = y / sigma
        prec = self.R.transpose(-1, -2) @ self.R + torch.einsum("nmi,nmj->nij", X, X)
        self.R = torch.linalg.cholesky(prec).transpose(-1, -2).contiguous()
        self.eta = self.eta + torch.einsum("nm,nmd->nd", y, X)

    # ------------------------------------------------------------------
    # Posterior queries
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self.eta.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.eta.shape[0]

    @property
    def mean(self) -> torch.Tensor:
        """Posterior mean mu = P^{-1} eta, shape (N, d)."""
        return torch.cholesky_solve(
            self.eta.unsqueeze(-1), self.R.transpose(-1, -2)
        ).squeeze(-1)

    @property
    def prec(self) -> torch.Tensor:
        """Precision matrix P = R^T R, shape (N, d, d)."""
        return self.R.transpose(-1, -2) @ self.R

    def cov_norm(self, x: torch.Tensor) -> torch.Tensor:
        """sqrt(x^T Sigma x) per actor.  x: (N, d) -> (N,)."""
        # Sigma = (R^T R)^{-1},  so ||x||_Sigma = ||R^{-T} x||_2
        # Solve R^T v = x  (R is upper-tri so R^T is lower-tri)
        v = torch.linalg.solve_triangular(
            self.R.transpose(-1, -2), x.unsqueeze(-1), upper=False
        ).squeeze(-1)
        return v.norm(dim=-1)

    def half_cov_mul(self, x: torch.Tensor) -> torch.Tensor:
        """Sigma^{1/2} x  =  R^{-1} x   (one possible square root).  x: (N, d)."""
        return torch.linalg.solve_triangular(
            self.R, x.unsqueeze(-1), upper=True
        ).squeeze(-1)

    # ------------------------------------------------------------------
    # Sampling / acquisition
    # ------------------------------------------------------------------

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Thompson Sampling: draw one z per actor from N(mu, Sigma).  Returns (N, d)."""
        eps = torch.randn(
            self.batch_size, self.dim, device=self.eta.device, generator=generator
        )
        return self.mean + self.half_cov_mul(eps)
