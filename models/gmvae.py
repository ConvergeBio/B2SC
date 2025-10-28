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

    def __init__(self, input_dim, hidden_dim, latent_dim, n_components):
        """
        Initialize GMVAE model.

        Args:
            input_dim: Number of genes (input features)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components (cell types)
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

        return mu, logvar, pis

    def reparameterize_with_labels(self, mus, logvars, labels):
        """
        Reparameterization trick with specific cluster labels.

        Args:
            mus: Mean parameters of shape (batch_size, n_components, latent_dim)
            logvars: Log variance parameters of shape (batch_size, n_components, latent_dim)
            labels: Cluster labels for each sample

        Returns:
            zs: Sampled latent vectors of shape (batch_size, latent_dim)
        """
        zs = []
        for label in labels:
            k = label.item()
            # Extract the mu and logvar for the specified cluster k
            selected_mus = mus[:, k, :]
            selected_logvars = logvars[:, k, :]

            # Representative value for the cluster (e.g., mean across the batch)
            mean_mus = selected_mus.mean(dim=0)  # Shape: (latent_dim,)
            mean_logvars = selected_logvars.mean(dim=0)  # Shape: (latent_dim,)

            # Reparameterization trick
            std = torch.exp(0.5 * mean_logvars)
            eps = torch.randn_like(std)
            z = mean_mus + eps * std  # Shape: (latent_dim,)
            zs.append(z)
        zs = torch.stack(zs)

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
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            labels: Cluster labels for each sample

        Returns:
            reconstructed: Reconstructed gene expression
            mus: Mean parameters
            logvars: Log variance parameters
            pis: Mixture weights
            zs: Sampled latent vectors
        """
        mus, logvars, pis = self.encode(x.view(-1, self.input_dim))
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
            k: Selected component index
        """
        # Randomly sample a cluster from the multinomial distribution
        k = torch.multinomial(pi, 1).squeeze()

        # Reparameterize the mus and logvars
        selected_mus = mus[k, :]
        selected_logvars = logvars[k, :]

        # Reparameterization trick
        std = torch.exp(0.5 * selected_logvars)
        eps = torch.randn_like(std)
        z = selected_mus + eps * std  # Shape: (latent_dim,)
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
