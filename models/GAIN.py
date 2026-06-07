"""
GAIN (Generative Adversarial Imputation Networks) Implementation

Adapted from the original TensorFlow implementation:
https://github.com/jsyoon0823/GAIN

Reference: J. Yoon, J. Jordon, M. van der Schaar,
"GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.

Converted from Jupyter notebook to Python module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GAIN(nn.Module):
    """
    Generative Adversarial Imputation Networks (GAIN).

    GAIN consists of:
    - Generator: imputes missing values
    - Discriminator: distinguishes real from imputed values
    - Hint mechanism: provides partial information to discriminator
    """

    def __init__(self, dim, h_dim=None, device='cpu'):
        """
        Args:
            dim: feature dimension
            h_dim: hidden dimension (default: same as dim)
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.dim = dim
        self.h_dim = h_dim if h_dim is not None else dim
        self.device = device

        # Generator weights
        self.G_W1 = nn.Parameter(torch.tensor(self._xavier_init([dim*2, self.h_dim]), device=device))
        self.G_b1 = nn.Parameter(torch.zeros(self.h_dim, device=device))
        self.G_W2 = nn.Parameter(torch.tensor(self._xavier_init([self.h_dim, self.h_dim]), device=device))
        self.G_b2 = nn.Parameter(torch.zeros(self.h_dim, device=device))
        self.G_W3 = nn.Parameter(torch.tensor(self._xavier_init([self.h_dim, dim]), device=device))
        self.G_b3 = nn.Parameter(torch.zeros(dim, device=device))

        # Discriminator weights
        self.D_W1 = nn.Parameter(torch.tensor(self._xavier_init([dim*2, self.h_dim]), device=device))
        self.D_b1 = nn.Parameter(torch.zeros(self.h_dim, device=device))
        self.D_W2 = nn.Parameter(torch.tensor(self._xavier_init([self.h_dim, self.h_dim]), device=device))
        self.D_b2 = nn.Parameter(torch.zeros(self.h_dim, device=device))
        self.D_W3 = nn.Parameter(torch.tensor(self._xavier_init([self.h_dim, dim]), device=device))
        self.D_b3 = nn.Parameter(torch.zeros(dim, device=device))

    def _xavier_init(self, size):
        """Xavier initialization for weights."""
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return np.random.normal(size=size, scale=xavier_stddev)

    def generator(self, x, m):
        """
        Generator network: imputes missing values.

        Args:
            x: input data with noise in missing entries [batch_size, dim]
            m: mask (1=observed, 0=missing) [batch_size, dim]

        Returns:
            imputed values in [0, 1]
        """
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)
        h2 = F.relu(torch.matmul(h1, self.G_W2) + self.G_b2)
        out = torch.sigmoid(torch.matmul(h2, self.G_W3) + self.G_b3)
        return out

    def discriminator(self, x, h):
        """
        Discriminator network: distinguishes real from imputed.

        Args:
            x: input data (real or imputed) [batch_size, dim]
            h: hint vector [batch_size, dim]

        Returns:
            probability that each value is real
        """
        inputs = torch.cat([x, h], dim=1)
        h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        h2 = F.relu(torch.matmul(h1, self.D_W2) + self.D_b2)
        out = torch.sigmoid(torch.matmul(h2, self.D_W3) + self.D_b3)
        return out

    def forward(self, x, m):
        """
        Forward pass: generate imputed values.

        Args:
            x: input data with noise in missing entries
            m: mask (1=observed, 0=missing)

        Returns:
            imputed data
        """
        g_sample = self.generator(x, m)
        imputed = m * x + (1 - m) * g_sample
        return imputed


class GAINTrainer:
    """Trainer for GAIN model."""

    def __init__(self, gain_model, alpha=10, lr_g=0.001, lr_d=0.001):
        """
        Args:
            gain_model: GAIN model instance
            alpha: weight for MSE loss in generator
            lr_g: learning rate for generator
            lr_d: learning rate for discriminator
        """
        self.model = gain_model
        self.alpha = alpha

        # Separate optimizers for generator and discriminator
        self.optimizer_G = torch.optim.Adam([
            self.model.G_W1, self.model.G_b1,
            self.model.G_W2, self.model.G_b2,
            self.model.G_W3, self.model.G_b3
        ], lr=lr_g)

        self.optimizer_D = torch.optim.Adam([
            self.model.D_W1, self.model.D_b1,
            self.model.D_W2, self.model.D_b2,
            self.model.D_W3, self.model.D_b3
        ], lr=lr_d)

    def discriminator_loss(self, m, new_x, h):
        """
        Discriminator loss: classify real vs imputed values.

        Args:
            m: mask
            new_x: data with noise in missing entries
            h: hint vector

        Returns:
            discriminator loss
        """
        # Generate imputed data
        g_sample = self.model.generator(new_x, m)
        hat_x = new_x * m + g_sample * (1 - m)

        # Discriminate
        d_prob = self.model.discriminator(hat_x, h)

        # Loss: binary cross-entropy
        d_loss = -torch.mean(m * torch.log(d_prob + 1e-8) +
                            (1 - m) * torch.log(1. - d_prob + 1e-8))
        return d_loss

    def generator_loss(self, x, m, new_x, h):
        """
        Generator loss: fool discriminator + reconstruction.

        Args:
            x: original data
            m: mask
            new_x: data with noise in missing entries
            h: hint vector

        Returns:
            generator loss, MSE on observed, MSE on missing
        """
        # Generate imputed data
        g_sample = self.model.generator(new_x, m)
        hat_x = new_x * m + g_sample * (1 - m)

        # Discriminate
        d_prob = self.model.discriminator(hat_x, h)

        # Adversarial loss: fool discriminator
        g_loss_adv = -torch.mean((1 - m) * torch.log(d_prob + 1e-8))

        # MSE loss on observed entries (for stability)
        mse_train_loss = torch.mean((m * new_x - m * g_sample)**2) / torch.mean(m)

        # Total generator loss
        g_loss = g_loss_adv + self.alpha * mse_train_loss

        # MSE on missing entries (for evaluation)
        mse_test_loss = torch.mean(((1 - m) * x - (1 - m) * g_sample)**2) / torch.mean(1 - m)

        return g_loss, mse_train_loss, mse_test_loss

    def train_step(self, x_mb, m_mb, p_hint=0.9):
        """
        Single training step.

        Args:
            x_mb: mini-batch of data [batch_size, dim]
            m_mb: mini-batch of masks [batch_size, dim]
            p_hint: hint probability

        Returns:
            d_loss, g_loss, mse_train, mse_test
        """
        batch_size = x_mb.shape[0]
        dim = x_mb.shape[1]

        # Sample noise for missing entries
        z_mb = self._sample_Z(batch_size, dim)

        # Generate hint vector
        h_mb = self._sample_hint(m_mb, p_hint)

        # Create corrupted input (noise in missing entries)
        new_x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

        # Convert to tensors if not already
        if not isinstance(x_mb, torch.Tensor):
            x_mb = torch.tensor(x_mb, dtype=torch.float32, device=self.model.device)
            m_mb = torch.tensor(m_mb, dtype=torch.float32, device=self.model.device)
            new_x_mb = torch.tensor(new_x_mb, dtype=torch.float32, device=self.model.device)
            h_mb = torch.tensor(h_mb, dtype=torch.float32, device=self.model.device)

        # Train discriminator
        self.optimizer_D.zero_grad()
        d_loss = self.discriminator_loss(m_mb, new_x_mb, h_mb)
        d_loss.backward()
        self.optimizer_D.step()

        # Train generator
        self.optimizer_G.zero_grad()
        g_loss, mse_train, mse_test = self.generator_loss(x_mb, m_mb, new_x_mb, h_mb)
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item(), mse_train.item(), mse_test.item()

    def fit(self, train_data, train_mask, epochs=5000, batch_size=128,
            p_hint=0.9, verbose=True, log_interval=100):
        """
        Train GAIN model.

        Args:
            train_data: training data [n_samples, dim]
            train_mask: training masks [n_samples, dim]
            epochs: number of training iterations
            batch_size: mini-batch size
            p_hint: hint probability
            verbose: whether to print progress
            log_interval: print every N iterations

        Returns:
            training history
        """
        n_samples = train_data.shape[0]
        history = {'d_loss': [], 'g_loss': [], 'mse_train': [], 'mse_test': []}

        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for it in iterator:
            # Sample mini-batch
            idx = np.random.choice(n_samples, batch_size, replace=False)
            x_mb = train_data[idx]
            m_mb = train_mask[idx]

            # Train step
            d_loss, g_loss, mse_train, mse_test = self.train_step(x_mb, m_mb, p_hint)

            # Log
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)
            history['mse_train'].append(mse_train)
            history['mse_test'].append(mse_test)

            # Print progress
            if verbose and it % log_interval == 0:
                print(f'Iter: {it}')
                print(f'Train_loss: {np.sqrt(mse_train):.4f}')
                print(f'Test_loss: {np.sqrt(mse_test):.4f}')
                print()

        return history

    def impute(self, data, mask):
        """
        Impute missing values.

        Args:
            data: data with missing values [n_samples, dim]
            mask: mask (1=observed, 0=missing) [n_samples, dim]

        Returns:
            imputed data
        """
        self.model.eval()

        with torch.no_grad():
            # Sample noise for missing entries
            z = self._sample_Z(data.shape[0], data.shape[1])

            # Create input
            new_x = mask * data + (1 - mask) * z

            # Convert to tensors
            if not isinstance(new_x, torch.Tensor):
                new_x = torch.tensor(new_x, dtype=torch.float32, device=self.model.device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=self.model.device)
            else:
                mask_t = mask

            # Impute
            imputed = self.model(new_x, mask_t)

            # Convert back to numpy if needed
            if not isinstance(data, torch.Tensor):
                imputed = imputed.cpu().numpy()

        return imputed

    def _sample_Z(self, m, n):
        """Sample noise for missing entries."""
        return np.random.uniform(0., 0.01, size=[m, n])

    def _sample_hint(self, m, p_hint):
        """
        Generate hint vector.

        Args:
            m: mask [batch_size, dim]
            p_hint: probability of revealing a missing entry in hint

        Returns:
            hint vector [batch_size, dim]
        """
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()

        batch_size, dim = m.shape

        # Sample binary hint
        h = np.random.uniform(0., 1., size=[batch_size, dim])
        h = (h > p_hint).astype(float)

        # Combine with mask (observed entries always hinted)
        h = m * h

        return h


# Utility functions for data preprocessing
def normalize_data(data):
    """
    Normalize data to [0, 1] per feature.

    Args:
        data: [n_samples, n_features]

    Returns:
        normalized data, min_vals, max_vals
    """
    min_vals = np.min(data, axis=0)
    data = data - min_vals
    max_vals = np.max(data, axis=0)
    data = data / (max_vals + 1e-6)

    return data, min_vals, max_vals


def denormalize_data(data, min_vals, max_vals):
    """Denormalize data back to original scale."""
    return data * max_vals + min_vals


def introduce_missingness(data, p_miss=0.2):
    """
    Introduce random missingness.

    Args:
        data: complete data [n_samples, n_features]
        p_miss: probability of missing per entry

    Returns:
        mask (1=observed, 0=missing)
    """
    n_samples, n_features = data.shape

    # Random missingness per feature
    mask = np.zeros((n_samples, n_features))
    for i in range(n_features):
        A = np.random.uniform(0., 1., size=[n_samples])
        mask[:, i] = (A > p_miss).astype(float)

    return mask


# Example usage
if __name__ == '__main__':
    """
    Example usage of GAIN for imputation.

    # Load your data
    data = np.loadtxt('your_data.csv', delimiter=',', skiprows=1)

    # Normalize
    data, min_vals, max_vals = normalize_data(data)

    # Introduce missingness
    mask = introduce_missingness(data, p_miss=0.2)

    # Split train/test
    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    train_mask = mask[:n_train]
    test_data = data[n_train:]
    test_mask = mask[n_train:]

    # Create model
    gain = GAIN(dim=data.shape[1], device='cpu')
    trainer = GAINTrainer(gain, alpha=10)

    # Train
    history = trainer.fit(train_data, train_mask, epochs=5000,
                         batch_size=128, p_hint=0.9, verbose=True)

    # Impute test data
    imputed_test = trainer.impute(test_data, test_mask)

    # Denormalize
    imputed_test = denormalize_data(imputed_test, min_vals, max_vals)

    # Evaluate
    mse = np.mean(((1 - test_mask) * test_data - (1 - test_mask) * imputed_test)**2)
    rmse = np.sqrt(mse / (1 - test_mask).mean())
    print(f'Test RMSE: {rmse:.4f}')
    """
    pass
