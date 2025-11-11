"""
Shared loss calculation functions for B2SC models.

This module contains the loss calculation logic that is shared between
training and validation phases for both GMVAE and BulkEncoder models.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from dataclasses import dataclass


def zinb_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    model=None,
    pi: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Zero-Inflated Negative Binomial (ZINB) loss.

    Implements the ZINB distribution as a mixture of a point mass at zero and a
    Negative Binomial distribution:

    P(Y = y) = {
        pi0 + (1 - pi0) * NB(0; pi, r)     if y = 0
        (1 - pi0) * NB(y; pi, r)           if y > 0
    }

    where NB is parameterized by success probability pi and dispersion r.

    This implementation follows the original B2SC paper formula.

    Args:
        y_true: Observed counts [batch_size, n_genes], non-negative integers
        y_pred: Predicted values [batch_size, n_genes], unused (kept for API compatibility)
        model: Model instance to extract ZINB parameters (prob_extra_zero, over_disp)
        pi: NB success probability in (0,1). If None, extracted from model
        r: NB dispersion parameter > 0. If None, extracted from model
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Scalar ZINB negative log-likelihood loss

    Note:
        - pi here is the Negative Binomial probability parameter (not zero-inflation)
        - Zero-inflation probability is extracted from model.prob_extra_zero
        - Falls back to MSE loss if NaN/Inf detected
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Extract ZINB parameters from model if not provided
    if pi is None or r is None:
        if model is not None:
            pi = getattr(
                model, "prob_extra_zero", torch.tensor(0.1, device=y_true.device)
            )
            r = getattr(model, "over_disp", torch.tensor(1.0, device=y_true.device))
        else:
            pi = torch.tensor(0.1, device=y_true.device)
            r = torch.tensor(1.0, device=y_true.device)

    # Ensure same device
    device = y_true.device
    pi = pi.to(device)
    r = r.to(device)

    # Clamp parameters for numerical stability (minimal guardrails)
    y_true = y_true.clamp(min=0.0)
    pi = pi.clamp(min=eps, max=1.0 - eps)
    r = r.clamp(min=eps)

    # Negative Binomial part: log P(Y = y | NB(pi, r))
    nb_case = (
        -torch.lgamma(r + eps)
        + torch.lgamma(y_true + r + eps)
        - torch.lgamma(y_true + 1.0)
        + r * torch.log(pi + eps)
        + y_true * torch.log(1.0 - (pi + eps))
    )

    # Zero-inflated part: log P(Y = 0 | ZINB)
    # For y=0: -log[pi0 + (1-pi0) * NB(0)]
    # For y>0: 0 (doesn't contribute)
    zero_nb = torch.pow(pi, r)
    zero_case = torch.where(
        y_true < eps,
        -torch.log(
            zero_nb + (1.0 - zero_nb) * torch.exp(-r * torch.log(1.0 - pi + eps)) + eps
        ),
        torch.tensor(0.0, device=device),
    )

    # Total negative log-likelihood
    result = -torch.mean(zero_case + nb_case)

    # Minimal safety check: fallback to MSE if numerical issues occur
    if torch.isnan(result) or torch.isinf(result):
        print("⚠️  NaN/Inf in ZINB loss, using fallback MSE")
        return torch.nn.functional.mse_loss(y_pred, y_true)

    return result


@dataclass
class GMVAELossComponents:
    """Container for GMVAE loss components."""

    total_loss: torch.Tensor
    # Raw (unweighted) losses
    reconstruction_loss_raw: torch.Tensor
    kl_loss_raw: torch.Tensor
    fraction_loss_raw: torch.Tensor
    zinb_loss_raw: torch.Tensor
    # Weighted losses (what contributes to total)
    reconstruction_loss_weighted: torch.Tensor
    kl_loss_weighted: torch.Tensor
    fraction_loss_weighted: torch.Tensor
    zinb_loss_weighted: torch.Tensor
    recon_warmup_weight: float

    # Backward compatibility (deprecated)
    @property
    def reconstruction_loss(self):
        return self.reconstruction_loss_raw

    @property
    def kl_loss(self):
        return self.kl_loss_raw

    @property
    def fraction_loss(self):
        return self.fraction_loss_raw

    @property
    def zinb_loss(self):
        return self.zinb_loss_raw


@dataclass
class BulkEncoderLossComponents:
    """Container for BulkEncoder loss components."""

    total_loss: torch.Tensor
    mus_loss: torch.Tensor
    logvars_loss: torch.Tensor
    pis_loss: torch.Tensor
    uniform_penalty: torch.Tensor


def calculate_kl_loss(
    mus: torch.Tensor,
    logvars: torch.Tensor,
    kl_scale_divisor: float = 1e6,
    kl_clamp_max: float = 1000.0,
) -> torch.Tensor:
    """
    Calculate KL divergence between posterior q(z|x) and standard Normal prior N(0,I).

    Paper Formula (from original B2SC implementation):
    KL(q(z|x) || N(0,I)) = 0.5 * Σ(-1 - log(σ²) + μ² + σ²) / scale_divisor

    This is the standard VAE KL divergence with closed-form solution for Gaussian posterior
    against standard Normal prior. The scale divisor (typically 1e10) keeps KL loss on the
    same order of magnitude as reconstruction loss.

    Where:
    - q(z|x) = N(μ, σ²) is the posterior (encoder output)
    - p(z) = N(0, I) is the standard Normal prior
    - μ (mus) are the learned means
    - σ² = exp(logvars) are the learned variances

    Args:
        mus: Mean parameters [batch_size, n_components, latent_dim]
        logvars: Log variance parameters [batch_size, n_components, latent_dim]
        kl_scale_divisor: Scale factor to divide KL by (paper uses 1e10)
        kl_clamp_max: Maximum KL value for numerical stability

    Returns:
        kl_loss: Scalar KL divergence loss (non-negative by definition)
    """
    # Standard VAE KL divergence: 0.5 * mean(-1 - log(σ²) + μ² + σ²)
    # where σ² = exp(logvars)
    # Using mean (not sum) makes loss independent of batch size
    kl_divergence = 0.5 * torch.mean(-1 - logvars + mus.pow(2) + logvars.exp())

    # Scale by divisor to keep KL on reasonable scale relative to reconstruction loss
    kl_divergence = kl_divergence / kl_scale_divisor
    # print(f"KL divergence: {kl_divergence}")
    # Clamp for numerical stability (KL is always non-negative by definition)
    kl_divergence = torch.clamp(kl_divergence, min=0.0, max=kl_clamp_max)

    return kl_divergence


class GMVAELossCalculator:
    """Handles GMVAE loss calculations for both training and validation."""

    def __init__(self, config, model, reconstruction_warmup_fn):
        """
        Initialize the loss calculator.

        Args:
            config: B2SC configuration object
            model: GMVAE model instance (for ZINB parameters)
            reconstruction_warmup_fn: Function to calculate reconstruction warmup weight
        """
        self.config = config
        self.model = model
        self._reconstruction_warmup_weight = reconstruction_warmup_fn

    def calculate_losses(
        self,
        data: torch.Tensor,
        reconstructed: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        pis: torch.Tensor,
        proportion_tensor: torch.Tensor,
        epoch: int,
        kl_weight: float,
    ) -> GMVAELossComponents:
        """
        Calculate all GMVAE losses.

        Args:
            data: Input data tensor
            reconstructed: Reconstructed data from model
            mus: Mean parameters from model
            logvars: Log variance parameters from model
            pis: Mixture probabilities from model
            proportion_tensor: Expected cell type proportions
            epoch: Current epoch number
            kl_weight: Weight for KL loss

        Returns:
            GMVAELossComponents: All loss components
        """
        # Basic loss calculations
        proportion_tensor_reshaped = proportion_tensor.to(pis.device)
        fraction_loss = F.mse_loss(pis.mean(0), proportion_tensor_reshaped)
        reconstruction_loss = F.mse_loss(reconstructed, data)

        # ZINB loss calculation
        zinb_loss_val = zinb_loss(data, reconstructed, model=self.model)

        # KL loss calculation - skip if weight is 0
        if self.config.gmvae_kl_loss_weight == 0.0:
            # print("🔄 Skipping KL loss calculation (weight = 0)")
            kl_loss_raw = torch.tensor(0.0, device=mus.device)
        else:
            # Calculate KL divergence using paper formula (standard VAE KL)
            kl_loss_raw = calculate_kl_loss(
                mus=mus,
                logvars=logvars,
                kl_scale_divisor=self.config.gmvae_hyperparameters.get(
                    "kl_scale_divisor", 1e6
                ),
                kl_clamp_max=self.config.gmvae_hyperparameters.get(
                    "kl_clamp_max", 1000.0
                ),
            )

        # Store raw losses before any weighting
        reconstruction_loss_raw = reconstruction_loss
        fraction_loss_raw = fraction_loss
        zinb_loss_raw = zinb_loss_val

        # Apply sigmoid warm-up to reconstruction loss
        recon_warmup_weight = self._reconstruction_warmup_weight(epoch)
        reconstruction_loss_with_warmup = reconstruction_loss * recon_warmup_weight

        # Calculate weighted losses (what actually contributes to total loss)
        recon_loss_weighted = (
            self.config.gmvae_recon_loss_weight * reconstruction_loss_with_warmup
        )
        kl_loss_weighted = self.config.gmvae_kl_loss_weight * kl_weight * kl_loss_raw

        fraction_loss_weighted = self.config.gmvae_fraction_loss_weight * fraction_loss
        zinb_loss_weighted = self.config.gmvae_zinb_loss_weight * zinb_loss_val

        # Calculate total weighted loss with safety checks
        try:
            total_loss = (
                recon_loss_weighted
                + kl_loss_weighted
                + fraction_loss_weighted
                + zinb_loss_weighted
            )

            # Final safety check for total loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(
                    "⚠️  NaN/Inf detected in total loss, using fallback loss with same weights"
                )
                total_loss = (
                    recon_loss_weighted
                    + kl_loss_weighted
                    + fraction_loss_weighted
                    + zinb_loss_weighted
                )
                # If fallback still has NaN/Inf, use consistent unweighted fallback
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(
                        "⚠️  Fallback loss also has NaN/Inf, using unweighted fallback"
                    )
                    # Use raw losses without any weighting for consistency
                    safe_recon = torch.clamp(reconstruction_loss_raw, min=0, max=1e6)
                    safe_kl = torch.clamp(kl_loss_raw, min=0, max=1e3)
                    safe_fraction = torch.clamp(fraction_loss_raw, min=0, max=1e3)
                    total_loss = safe_recon + safe_kl + safe_fraction

        except Exception as e:
            print(
                f"⚠️  Error calculating total loss: {e}, using fallback with same weights"
            )
            total_loss = (
                recon_loss_weighted
                + kl_loss_weighted
                + fraction_loss_weighted
                + zinb_loss_weighted
            )
            # If fallback still has issues, use minimal safe loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("⚠️  Exception fallback also has NaN/Inf, using minimal safe loss")
                total_loss = reconstruction_loss_with_warmup + torch.tensor(
                    0.1, device=reconstruction_loss_with_warmup.device
                )

        return GMVAELossComponents(
            total_loss=total_loss,
            # Raw (unweighted) losses
            reconstruction_loss_raw=reconstruction_loss_raw,
            kl_loss_raw=kl_loss_raw,
            fraction_loss_raw=fraction_loss_raw,
            zinb_loss_raw=zinb_loss_raw,
            # Weighted losses (what contributes to total)
            reconstruction_loss_weighted=recon_loss_weighted,
            kl_loss_weighted=kl_loss_weighted,
            fraction_loss_weighted=fraction_loss_weighted,
            zinb_loss_weighted=zinb_loss_weighted,
            recon_warmup_weight=recon_warmup_weight,
        )


class BulkEncoderLossCalculator:
    """Handles BulkEncoder loss calculations for both training and validation."""

    def __init__(self, config):
        """
        Initialize the loss calculator.

        Args:
            config: B2SC configuration object
        """
        self.config = config

    def calculate_losses(
        self,
        pred_mus: torch.Tensor,
        pred_logvars: torch.Tensor,
        pred_pis: torch.Tensor,
        target_mus: torch.Tensor,
        target_logvars: torch.Tensor,
        target_pis: torch.Tensor,
    ) -> BulkEncoderLossComponents:
        """
        Calculate all BulkEncoder losses.

        Args:
            pred_mus: Predicted mu parameters
            pred_logvars: Predicted log variance parameters
            pred_pis: Predicted mixture probabilities
            target_mus: Target mu parameters from GMVAE
            target_logvars: Target log variance parameters from GMVAE
            target_pis: Target mixture probabilities (cell type proportions)

        Returns:
            BulkEncoderLossComponents: All loss components
        """
        # Calculate individual losses
        mus_loss = F.mse_loss(pred_mus, target_mus, reduction="mean")
        logvars_loss = F.mse_loss(pred_logvars, target_logvars, reduction="mean")
        pis_loss = F.mse_loss(pred_pis, target_pis, reduction="mean")

        # Uniform penalty - restored
        C = pred_pis.shape[-1]  # Number of components
        uniform = torch.full_like(pred_pis, 1.0 / C)
        uniform_penalty = F.cosine_similarity(
            pred_pis.view(-1, C), uniform.view(-1, C), dim=-1
        ).mean()  # Average over batch

        # Calculate total weighted loss
        total_loss = (
            self.config.bulk_mus_loss_weight * mus_loss
            + self.config.bulk_logvars_loss_weight * logvars_loss
            + self.config.bulk_pis_loss_weight * pis_loss
            + self.config.bulk_uniform_penalty_weight * uniform_penalty  # RESTORED
        )

        return BulkEncoderLossComponents(
            total_loss=total_loss,
            mus_loss=mus_loss,
            logvars_loss=logvars_loss,
            pis_loss=pis_loss,
            uniform_penalty=uniform_penalty,
        )


def prepare_bulk_encoder_targets(
    batch_size: int,
    sc_mus: torch.Tensor,
    sc_logvars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare target tensors for BulkEncoder training/validation.

    Args:
        batch_size: Size of the current batch
        sc_mus: Single-cell mu parameters from GMVAE
        sc_logvars: Single-cell log variance parameters from GMVAE

    Returns:
        Tuple of (target_mus, target_logvars) expanded for batch
    """
    target_mus = sc_mus.unsqueeze(0).expand(batch_size, -1, -1)
    target_logvars = sc_logvars.unsqueeze(0).expand(batch_size, -1, -1)
    return target_mus, target_logvars
