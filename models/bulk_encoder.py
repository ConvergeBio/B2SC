"""
Bulk encoder model for B2SC pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BulkEncoder(nn.Module):
    """
    Bulk encoder that predicts GMVAE latent parameters from bulk RNA-seq data.

    This model takes bulk RNA-seq expression (sum of single cells) and predicts
    the mixture parameters (mu, logvar, pi) that would generate those cells.
    """

    def __init__(self, input_dim=5000, hidden_dim=600, latent_dim=300, n_components=10):
        """
        Initialize BulkEncoder model.

        Args:
            input_dim: Number of genes (input features)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components (cell types)
        """
        print(
            "Initializing bulkEncoder with input_dim:",
            input_dim,
            "hidden_dim:",
            hidden_dim,
            "latent_dim:",
            latent_dim,
            "n_components:",
            n_components,
        )
        super(BulkEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components

        self.fc0 = nn.Linear(input_dim, hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(p=0.05)

        # The output layers for mu, logvar, and pi
        self.fc_mu = nn.Linear(hidden_dim, n_components * latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, n_components * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        """
        Forward pass through the bulk encoder.

        Args:
            x: Bulk RNA-seq data of shape (batch_size, input_dim)

        Returns:
            mu: Mean parameters of shape (batch_size, n_components, latent_dim)
            logvar: Log variance parameters of shape (batch_size, n_components, latent_dim)
            pi: Mixture proportions of shape (batch_size, n_components)
        """
        h0 = F.relu(self.fc0(x))
        h0 = self.dropout(h0)

        h1 = F.relu(self.fc1(h0))
        h = self.dropout(h1)

        mu = self.fc_mu(h).view(-1, self.n_components, self.latent_dim)
        logvar = self.fc_logvar(h).view(-1, self.n_components, self.latent_dim)
        pi_logits = self.fc_pi(h)
        pi = F.softmax(pi_logits, dim=-1)

        return mu, logvar, pi
