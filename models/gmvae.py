"""
Gaussian Mixture Variational Autoencoder model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureVAE(nn.Module):
    """
    Gaussian Mixture Variational Autoencoder for single-cell RNA-seq data.

    This model encodes single-cell data into a mixture of Gaussian distributions
    in latent space and decodes back to gene expression space.
    """

    def __init__(
        self, input_dim, hidden_dim, latent_dim, n_components, use_weight_norm=False
    ):
        """
        Initialize GMVAE model.

        Args:
            input_dim: Number of genes (input features)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components (cell types)
            use_weight_norm: If True, apply weight normalization to hidden layers
        """
        print(
            "Initializing GaussianMixtureVAE with input_dim:",
            input_dim,
            "hidden_dim:",
            hidden_dim,
            "latent_dim:",
            latent_dim,
            "n_components:",
            n_components,
        )
        super(GaussianMixtureVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.use_weight_norm = use_weight_norm

        # Encoder
        self.fc0 = nn.Linear(input_dim, hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, n_components * latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, n_components * latent_dim)  # logvar layer
        # Mixture weights
        self.fc_pi = nn.Linear(hidden_dim, n_components)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.05)

        # Parameters for ZINB
        self.prob_extra_zero = nn.Parameter(
            torch.rand(1)
        )  # Initialize to a random value
        self.over_disp = nn.Parameter(torch.rand(1))  # Initialize to a random value

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        # Apply weight normalization (conditional)
        # Only normalize hidden layers, NOT output layers for μ, log σ², pis, or reconstruction
        if self.use_weight_norm:
            self.fc0 = nn.utils.parametrizations.weight_norm(self.fc0)
            self.fc1 = nn.utils.parametrizations.weight_norm(self.fc1)
            self.fc3 = nn.utils.parametrizations.weight_norm(self.fc3)
            self.fc4 = nn.utils.parametrizations.weight_norm(self.fc4)
            # NOT applied to fc21, fc22, fc_pi, fc5

    def encode(self, x):
        """
        Encode input to latent parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            mu: Mean parameters of shape (batch_size, n_components, latent_dim)
            logvar: Log variance parameters of shape (batch_size, n_components, latent_dim)
            pis: Mixture weights of shape (batch_size, n_components)
        """
        h0 = F.relu(self.fc0(x))
        h0 = self.dropout(h0)
        h1 = F.relu(self.fc1(h0))
        h1 = self.dropout(h1)

        mu = self.fc21(h1)  # Shape: (batch_size, n_components * latent_dim)
        logvar = self.fc22(h1)  # Shape: (batch_size, n_components * latent_dim)

        logits = self.fc_pi(h1)  # These are logits for the softmax
        logits = (
            logits - logits.max(dim=-1, keepdim=True).values
        )  # Normalize for numerical stability
        pis = F.softmax(logits, dim=-1)

        mu = mu.reshape(-1, self.n_components, self.latent_dim)
        logvar = logvar.reshape(-1, self.n_components, self.latent_dim)

        # Clamp logvar to prevent extreme values and gradient explosion
        # Range [-10, 10] ensures std is in [exp(-5), exp(5)] ≈ [0.007, 148]
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)

        return mu, logvar, pis

    def reparameterize_with_labels(self, mus, logvars, labels):
        """
        Reparameterization trick with cell type labels (supervised).

        For each label (cell type) in the batch, this method:
        1. Selects all cells of that type from the batch
        2. Averages their component parameters to create a representative latent vector
        3. Samples from this representative distribution

        This creates component-level prototypes that are consistent with the
        BulkEncoder training targets (which are also component-level averages).

        Args:
            mus: Mean parameters of shape (batch_size, n_components, latent_dim)
            logvars: Log variance parameters of shape (batch_size, n_components, latent_dim)
            labels: Cell type labels of shape (batch_size,) with values 0, 1, 2, etc.

        Returns:
            zs: Sampled latent vectors of shape (batch_size, latent_dim)
        """
        zs = []
        for label in labels:
            k = label.item()  # Get the cell type index (0, 1, 2, etc.)

            # Extract the mu and logvar for component k from ALL cells in batch
            # This gives us all batch cells' predictions for this component
            selected_mus = mus[:, k, :]  # Shape: (batch_size, latent_dim)
            selected_logvars = logvars[:, k, :]  # Shape: (batch_size, latent_dim)

            # Create component-level representative by averaging across the batch
            mean_mus = selected_mus.mean(dim=0)  # Shape: (latent_dim,)
            mean_logvars = selected_logvars.mean(dim=0)  # Shape: (latent_dim,)

            # Reparameterization trick with averaged (representative) parameters
            std = torch.exp(0.5 * mean_logvars)
            eps = torch.randn_like(std)
            z = mean_mus + eps * std  # Shape: (latent_dim,)
            zs.append(z)

        zs = torch.stack(zs)  # Shape: (batch_size, latent_dim)
        return zs

    def decode(self, zs):
        """
        Decode latent vectors to gene expression.

        Args:
            zs: Latent vectors of shape (batch_size, latent_dim)

        Returns:
            reconstructed: Reconstructed gene expression of shape (batch_size, input_dim)
        """
        h3 = F.relu(self.fc3(zs))
        h3 = self.dropout(h3)
        h4 = F.relu(self.fc4(h3))
        h4 = self.dropout(h4)
        reconstructed = F.relu(self.fc5(h4))
        return reconstructed

    def forward(self, x, labels):
        """
        Forward pass through the model (supervised with cell type labels).

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            labels: Cell type labels of shape (batch_size,) with values 0, 1, 2, etc.

        Returns:
            reconstructed: Reconstructed gene expression
            mus: Mean parameters
            logvars: Log variance parameters
            pis: Mixture weights
            zs: Sampled latent vectors
        """
        assert(len(labels) == x.shape[0])
        mus, logvars, pis = self.encode(x.view(-1, self.input_dim))
        # Use the true cell type labels to select mixture components
        zs = self.reparameterize_with_labels(mus, logvars, labels)
        reconstructed = self.decode(zs)
        return reconstructed, mus, logvars, pis, zs

    def reparameterize_with_proportion(self, mus, logvars, pi):
        """
        Reparameterization for bulk generation using mixture proportions.

        Args:
            mus: Mean parameters of shape (n_components, latent_dim)
            logvars: Log variance parameters of shape (n_components, latent_dim)
            pi: Mixture proportions of shape (n_components,)

        Returns:
            z: Sampled latent vector of shape (latent_dim,)
            k: Selected component index (torch.Tensor scalar, dtype=torch.long)
                Represents which mixture component (cell type) was selected.
                Example: If n_components=3 (e.g., T_cell, B_cell, NK_cell),
                k could be tensor(0), tensor(1), or tensor(2) corresponding
                to the selected cell type index.
        """
        # Ensure pi is normalized and valid for multinomial sampling
        pi = torch.clamp(pi, min=1e-8)
        pi = pi / pi.sum()

        # Randomly sample a cluster from the multinomial distribution
        k = torch.multinomial(pi, 1).squeeze()

        # Reparameterize the mus and logvars
        selected_mus = mus[k, :]
        selected_logvars = logvars[k, :]

        # Clamp logvars to prevent exp() from exploding (redundant safety check)
        # Note: logvars are already clamped in encode(), but we keep this for safety
        selected_logvars = torch.clamp(selected_logvars, min=-20.0, max=20.0)

        # Reparameterization trick with numerical stability
        std = torch.exp(0.5 * selected_logvars)

        # Check for NaN/Inf in std
        if torch.isnan(std).any() or torch.isinf(std).any():
            print(
                f"⚠️  NaN/Inf in std calculation: logvars range [{selected_logvars.min():.6f}, {selected_logvars.max():.6f}]"
            )
            std = torch.clamp(std, min=1e-8, max=100.0)  # Prevent extreme values

        eps = torch.randn_like(std)
        z = selected_mus + eps * std  # Shape: (latent_dim,)

        # Final safety check
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(
                f"⚠️  NaN/Inf in final z: mus range [{selected_mus.min():.6f}, {selected_mus.max():.6f}], std range [{std.min():.6f}, {std.max():.6f}]"
            )
            z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))

        return z, k

    def decode_bulk(self, mus, logvars, pis):
        """
        Decode bulk sample to single cell.

        Args:
            mus: Mean parameters from bulk encoder
            logvars: Log variance parameters from bulk encoder
            pis: Mixture proportions from bulk encoder

        Returns:
            reconstructed: Generated single cell expression
            k: Selected cell type
        """
        z, k = self.reparameterize_with_proportion(mus, logvars, pis)
        reconstructed = self.decode(z)
        return reconstructed, k
