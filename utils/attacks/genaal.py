import os
import random
from typing import Callable, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.data_preparer import DataPreparer


class VAEGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default structure
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 128, 256]

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU(inplace=True))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU(inplace=True))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_tilde = self.decode(z)
        return x_tilde, mu, logvar


class SIDS(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()

        # Default structure
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs  # P(benign)


class GenAALAttack:
    def __init__(
        self,
        attack_samples: pd.DataFrame,
        vae_hidden: Optional[List[int]] = None,
        sids_hidden: Optional[List[int]] = None,
        latent_dim: int = 32,
        lambda_kl: float = 1e-3,
        lambda_l2: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_label: float = 1.0,
        pretrain_lr: float = 1e-4,
        gen_lr: float = 2e-4,
        sid_lr: float = 5e-3,
        batch_size: int = 128,
        seed: Optional[int] = None,
        snapshot_folder: str = "snapshots",
        snapshot_name: str = "genaal_snapshot.pth",
        ephemeral: bool = False,
    ):
        self.lambda_kl = lambda_kl
        self.lambda_l2 = lambda_l2
        self.lambda_recon = lambda_recon
        self.lambda_label = lambda_label

        self.pretrain_lr = pretrain_lr
        self.gen_lr = gen_lr
        self.sid_lr = sid_lr
        self.batch_size = batch_size

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Prepare data
        self.data_preparer = DataPreparer(
            df=attack_samples, encoder_type="label", scaler_type="minmax"
        )
        self.feature_dim = len(self.data_preparer.df.columns) - 1

        # Create networks
        self.generator = VAEGenerator(
            input_dim=self.feature_dim,
            hidden_dims=vae_hidden,
            latent_dim=latent_dim,
        )
        self.sids = SIDS(
            input_dim=self.feature_dim,
            hidden_dims=sids_hidden,
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.sids.to(self.device)

        # Create optimizers
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr)
        self.opt_sids = torch.optim.Adam(
            self.sids.parameters(), lr=self.sid_lr)

        # Load snapshot
        self.snapshot_path = os.path.join(snapshot_folder, snapshot_name)
        self.ephemeral = ephemeral
        if not self.ephemeral:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)

        self.al_iteration = 0

        self._load_networks(self.snapshot_path)

    def _load_networks(self, path: str) -> None:
        if not self.ephemeral and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.sids.load_state_dict(checkpoint["sids_state_dict"])
            self.opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
            self.opt_sids.load_state_dict(checkpoint["opt_sids_state_dict"])
            self.al_iteration = checkpoint.get("al_iteration", 0)

            # Fix optimizer state device
            for opt in (self.opt_gen, self.opt_sids):
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
        else:
            self.al_iteration = 0

        self.generator.to(self.device)
        self.sids.to(self.device)

    def _save_networks(self, path: str) -> None:
        if self.ephemeral:
            return
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "sids_state_dict": self.sids.state_dict(),
            "opt_gen_state_dict": self.opt_gen.state_dict(),
            "opt_sids_state_dict": self.opt_sids.state_dict(),
            "al_iteration": self.al_iteration,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def pretrain_vae(self, df: pd.DataFrame, epochs: int = 500) -> None:
        X_encoded, _ = self.data_preparer.scale_and_encode(df)
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)

        dataset = TensorDataset(X_tensor)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Fresh optimizer for pretraining
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.pretrain_lr)

        self.generator.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for (xb,) in dl:
                xb = xb.to(self.device)
                x_tilde, mu, logvar = self.generator(xb)

                # KL term
                kl = self._kl_divergence(mu, logvar)

                # L2 term
                diff = (x_tilde - xb).view(xb.size(0), -1)
                l2_term = diff.pow(2).sum(dim=1).mean()

                # Reconstruction term
                recon_bce = F.binary_cross_entropy(x_tilde, xb)

                loss = (
                    self.lambda_kl * kl
                    + self.lambda_l2 * l2_term
                    + self.lambda_recon * recon_bce
                )

                self.opt_gen.zero_grad()
                loss.backward()
                self.opt_gen.step()
                epoch_loss += loss.item() * xb.size(0)

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(
                    f"VAE pretrain epoch {ep+1}/{epochs} loss={epoch_loss / len(dataset):.6f}"
                )

    def train_sids(
        self,
        labeled_X: torch.Tensor | np.ndarray,
        labeled_y: torch.Tensor | np.ndarray,
        epochs: int = 20,
    ) -> None:
        if isinstance(labeled_X, np.ndarray):
            labeled_X = torch.from_numpy(labeled_X).float()
        if isinstance(labeled_y, np.ndarray):
            labeled_y = torch.from_numpy(labeled_y).float()

        dataset = TensorDataset(labeled_X, labeled_y)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.opt_sids = torch.optim.Adam(
            self.sids.parameters(), lr=self.sid_lr)
        criterion = nn.BCELoss()

        self.sids.train()
        for ep in range(epochs):
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                probs = self.sids(xb)  # P(benign)
                loss = criterion(probs, yb)

                self.opt_sids.zero_grad()
                loss.backward()
                self.opt_sids.step()
                total_loss += loss.item() * xb.size(0)

    def train_generator_with_sids(
        self,
        train_X: torch.Tensor | np.ndarray,
        gen_epochs: int = 10,
    ) -> None:
        if isinstance(train_X, np.ndarray):
            train_X = torch.from_numpy(train_X).float()

        dataset = TensorDataset(train_X)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Use gen_lr after pretraining
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr)

        self.generator.train()
        self.sids.eval()
        for p in self.sids.parameters():
            p.requires_grad = False

        eps = 1e-8

        for ep in range(gen_epochs):
            epoch_loss = 0.0
            for (xb,) in dl:
                xb = xb.to(self.device)

                # VAE forward
                x_tilde, mu, logvar = self.generator(xb)

                # L2 term
                diff = (x_tilde - xb).view(xb.size(0), -1)
                l2_term = diff.pow(2).sum(dim=1).mean()

                # Reconstruction term
                recon_bce = F.binary_cross_entropy(x_tilde, xb)

                # S-IDS prediction
                s_probs = self.sids(x_tilde)

                # Adversarial label term
                target_ones = torch.ones_like(s_probs)
                label_loss = F.binary_cross_entropy(
                    torch.clamp(s_probs, min=eps, max=1.0 - eps), target_ones
                )

                # KL divergence
                kl = self._kl_divergence(mu, logvar)

                # Total GAN loss
                loss = (
                    self.lambda_kl * kl
                    + self.lambda_l2 * l2_term
                    + self.lambda_recon * recon_bce
                    + self.lambda_label * label_loss
                )

                self.opt_gen.zero_grad()
                loss.backward()
                self.opt_gen.step()
                epoch_loss += loss.item() * xb.size(0)

        # Unfreeze S-IDS
        for p in self.sids.parameters():
            p.requires_grad = True
        self.sids.train()

    def query_blackbox(
        self,
        x_t: torch.Tensor | np.ndarray,
        blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
        training_phase: bool = True,
    ) -> torch.Tensor:
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).float()

        # Decode and unscale
        df_features = self.data_preparer.unscale_and_decode(
            x_t.detach().cpu().numpy()
        )

        preds = np.asarray(blackbox_predict(df_features))

        if training_phase:
            self.query_stats["n_queries"] += np.int64(len(preds))
            self.query_stats["benign_queries"] += np.sum(preds == 1)
            self.query_stats["malicious_queries"] += np.sum(preds == 0)

        return torch.tensor(preds, dtype=torch.float32, device=self.device)

    def fit(
        self,
        df: pd.DataFrame,
        blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
        label_query: int = 15,
        max_iterations: int = 3,
        candidate_pool_k: int = 500,
        nquery: int = 5,
        pretrain_epochs: int = 500,
        sids_epochs: int = 20,
        gen_epochs: int = 10,
    ) -> None:
        self.query_stats = {
            "n_queries": 0,
            "malicious_queries": 0,
            "benign_queries": 0,
        }

        # VAE pretraining
        if self.al_iteration == 0:
            print("Pretraining VAE")
            self.pretrain_vae(df, epochs=pretrain_epochs)

        # Prepare encoded feature tensor
        X_encoded, _ = self.data_preparer.scale_and_encode(df)
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)

        total_n = X_tensor.size(0)
        label_query = min(label_query, total_n)

        # Initialize L_0 and U_0
        print(f"Initializing L0 with {label_query} random queried samples")
        init_indices = random.sample(range(total_n), label_query)
        init_samples = X_tensor[init_indices]

        # Query IDS for original samples
        init_labels = self.query_blackbox(
            init_samples.to(self.device), blackbox_predict
        ).detach().cpu()

        Lx = init_samples.clone()
        Ly = init_labels.clone()

        # U0 = rest
        mask = torch.ones(total_n, dtype=torch.bool)
        mask[init_indices] = False
        Ux = X_tensor[mask]

        # Active learning iterations
        while self.al_iteration < max_iterations:
            k = self.al_iteration
            print(f"Active Learning Iteration {k+1}/{max_iterations}")

            # Train S-IDS on current labeled set L_k
            self.train_sids(Lx, Ly, epochs=sids_epochs)

            # Check S-IDS agreement with blackbox on L_k
            with torch.no_grad():
                bb_labels = self.query_blackbox(
                    Lx, blackbox_predict, training_phase=False).cpu().numpy()
                sids_probs = self.sids(Lx.to(self.device)).cpu().numpy()
                sids_labels = (sids_probs > 0.5).astype(int)
            bb_acc = (sids_labels == bb_labels).mean()
            print(f"S-IDS / blackbox agreement on L_k: {bb_acc:.4f}")

            # Train generator G_k
            if k == 0:
                gen_train_X = Ux.clone()
            else:
                malicious_mask = (Ly == 0)
                gen_train_X = Lx[malicious_mask]

            self.train_generator_with_sids(gen_train_X, gen_epochs=gen_epochs)

            # Evaluate success on L_k
            adv_Lx = self.generate(Lx)
            adv_labels_bb = self.query_blackbox(
                adv_Lx, blackbox_predict, training_phase=False).cpu().numpy()
            orig_labels_bb = self.query_blackbox(
                Lx, blackbox_predict, training_phase=False).cpu().numpy()

            success_bb = ((orig_labels_bb == 0) & (adv_labels_bb == 1)).mean()

            with torch.no_grad():
                orig_probs_sids = self.sids(Lx.to(self.device)).cpu().numpy()
                adv_probs_sids = self.sids(
                    adv_Lx.to(self.device)).cpu().numpy()
            orig_labels_sids = (orig_probs_sids > 0.5).astype(int)
            adv_labels_sids = (adv_probs_sids > 0.5).astype(int)
            success_sids = ((orig_labels_sids == 0) &
                            (adv_labels_sids == 1)).mean()

            diff = (adv_Lx - Lx).view(Lx.size(0), -1)
            mean_pert = diff.norm(p=2, dim=1).mean().item()

            print(
                f"Local success rate (L_k) - Blackbox: {success_bb:.4f}, S-IDS: {success_sids:.4f}, Mean L2: {mean_pert:.6f}"
            )

            # Stop if unlabeled pool exhausted
            if Ux.size(0) == 0:
                print("Unlabeled pool U_k is empty. Stopping active learning.")
                break

            # Sample candidate pool S_k in U_k
            pool_size = min(candidate_pool_k, Ux.size(0))
            pool_indices = random.sample(range(Ux.size(0)), pool_size)
            S_pool = Ux[pool_indices]

            # Generate adversarial examples for S_k and compute perturbations
            self.generator.eval()
            with torch.no_grad():
                x_adv_pool, _, _ = self.generator(S_pool.to(self.device))

            diff_pool = (x_adv_pool - S_pool.to(self.device)
                         ).view(pool_size, -1)
            pert_norms = diff_pool.norm(p=2, dim=1)

            # Top nquery smallest perturbations
            topk = min(nquery, pool_size)
            chosen = torch.argsort(pert_norms)[:topk]
            chosen_indices_in_U = [pool_indices[int(c.item())] for c in chosen]

            # Corresponding original and adversarial samples
            query_original = Ux[chosen_indices_in_U]
            query_adv = x_adv_pool[chosen].detach().cpu()

            # Query black-box IDS
            q_orig_labels = self.query_blackbox(
                query_original, blackbox_predict
            ).cpu()

            q_adv_labels = self.query_blackbox(
                query_adv, blackbox_predict
            ).cpu()

            # Update L and U
            Lx = torch.cat([Lx, query_original, query_adv], dim=0)
            Ly = torch.cat([Ly, q_orig_labels, q_adv_labels], dim=0)

            # Remove queried originals from U_k
            mask = torch.ones(Ux.size(0), dtype=torch.bool)
            mask[chosen_indices_in_U] = False
            Ux = Ux[mask]

            self.al_iteration += 1

        # Final S-IDS training on full L_k
        print("Final S-IDS training on aggregated L_k")
        self.train_sids(Lx, Ly, epochs=sids_epochs)

        self._save_networks(self.snapshot_path)

    def generate(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        self.generator.eval()
        with torch.no_grad():
            x = x.to(self.device)
            x_tilde, _, _ = self.generator(x)
        return x_tilde.cpu()

    def generate_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        X_encoded, _ = self.data_preparer.scale_and_encode(df)
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        adv_tensor = self.generate(X_tensor)
        adv_np = adv_tensor.numpy()
        adv_df = self.data_preparer.unscale_and_decode(adv_np)
        return adv_df

    def evaluate_success_rate(
        self,
        X: torch.Tensor | np.ndarray,
        orig_labels_from_blackbox: np.ndarray,
        blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
    ) -> float:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        adv_np = self.generate(X).numpy()

        adv_df = self.data_preparer.unscale_and_decode(adv_np)
        adv_labels_np = np.asarray(blackbox_predict(adv_df)).astype(np.int64)
        orig_np = np.asarray(orig_labels_from_blackbox).astype(np.int64)

        success_mask = (orig_np == 0) & (adv_labels_np == 1)
        if success_mask.size == 0:
            return 0.0
        return float(success_mask.sum()) / float(success_mask.size)
