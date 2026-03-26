from typing import Callable, Optional, Sequence
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from utils.data_preparer import DataPreparer


class IDSGANGenerator(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 noise_dim: int,
                 n_hidden_layers: int = 3,
                 n_neurons: int = 128):
        super().__init__()

        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons

        self.model = self.__build_model()

    def __build_model(self) -> nn.Sequential:
        layers = []

        input_dim = self.feature_dim + self.noise_dim
        hidden_dim = self.n_neurons

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, self.feature_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        input = torch.cat([x, noise], dim=1)
        return self.model(input)


class IDSGANDiscriminator(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 n_hidden_layers: int = 1,
                 n_neurons: int = 128):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons

        self.model = self.__build_model()

    def __build_model(self) -> nn.Sequential:
        layers = []

        # Input layer
        layers.append(nn.Linear(self.feature_dim, self.n_neurons))
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(nn.LeakyReLU(0.2))

        # Output layer
        layers.append(nn.Linear(self.n_neurons, 1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


class IDSGANAttack:
    def __init__(self,
                 noise_dim: int,
                 attack_samples: pd.DataFrame,
                 n_hidden_layers_generator: int = 3,
                 n_neurons_generator: int = 128,
                 learning_rate_generator: float = 0.0001,
                 steps_generator: int = 1,
                 n_hidden_layers_discriminator: int = 1,
                 n_neurons_discriminator: int = 128,
                 learning_rate_discriminator: float = 0.0001,
                 steps_discriminator: int = 5,
                 weight_clip: float = 0.01,
                 binary_threshold: float = 0.5,
                 batch_size: int = 2048,
                 seed: Optional[int] = None,
                 snapshot_folder: str = "snapshots",
                 snapshot_name: str = "idsgan_snapshot.pth",
                 ephemeral: bool = False
                 ):
        self.noise_dim = noise_dim

        self.n_hidden_layers_generator = n_hidden_layers_generator
        self.n_hidden_layers_discriminator = n_hidden_layers_discriminator
        self.learning_rate_generator = learning_rate_generator
        self.steps_generator = steps_generator

        self.n_neurons_generator = n_neurons_generator
        self.n_neurons_discriminator = n_neurons_discriminator
        self.learning_rate_discriminator = learning_rate_discriminator
        self.steps_discriminator = steps_discriminator

        self.weight_clip = weight_clip
        self.binary_threshold = binary_threshold
        self.batch_size = batch_size
        self.seed = seed
        self.attack_samples = attack_samples
        self.snapshot_path = os.path.join(
            snapshot_folder, snapshot_name)
        self.ephemeral = ephemeral

        if not self.ephemeral:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare the data. Use one-hot encoding for categorical features and min-max scaling for numerical features.
        self.data_preparer = DataPreparer(
            df=attack_samples, encoder_type="onehot", scaler_type="minmax")
        # exclude target column
        self.feature_dim = len(self.data_preparer.df.columns) - 1

        # Create models
        self.generator = IDSGANGenerator(
            feature_dim=self.feature_dim,
            noise_dim=self.noise_dim,
            n_hidden_layers=self.n_hidden_layers_generator,
            n_neurons=self.n_neurons_generator
        )

        self.discriminator = IDSGANDiscriminator(
            feature_dim=self.feature_dim,
            n_hidden_layers=self.n_hidden_layers_discriminator,
            n_neurons=self.n_neurons_discriminator
        )

        # Create optimizers
        self.optimizer_G = optim.RMSprop(
            self.generator.parameters(), lr=self.learning_rate_generator)
        self.optimizer_D = optim.RMSprop(
            self.discriminator.parameters(), lr=self.learning_rate_discriminator)

        self.mse_loss = nn.MSELoss()

        self.load_networks(self.snapshot_path)

    def load_networks(self,
                      path: str) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if not self.ephemeral and os.path.exists(path):
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(
                checkpoint['discriminator_state_dict'])

            self.optimizer_G.load_state_dict(
                checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(
                checkpoint['optimizer_D_state_dict'])

            self.epoch = checkpoint['epoch']

            for state in self.optimizer_G.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            for state in self.optimizer_D.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            self.epoch = 0

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def save_networks(self, path: str) -> None:
        if self.ephemeral:
            return

        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(checkpoint, path)

    def query_blackbox(self,
                       x_t: torch.Tensor,
                       blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
                       training_phase: bool = True) -> torch.Tensor:
        # Auxiliary function to query black-box model. It should return 1 for normal and 0 for attack.
        generated_samples = self.data_preparer.unscale_and_decode(
            x_t.detach().cpu().numpy())
        preds = blackbox_predict(generated_samples)
        preds = np.asarray(preds)

        if training_phase:
            self.query_stats["n_queries"] += np.int64(len(preds))
            self.query_stats["benign_queries"] += np.sum(preds == 1)
            self.query_stats["malicious_queries"] += np.sum(preds == 0)

        return torch.tensor(preds, dtype=torch.float32, device=self.device)

    def _generate_functional_mask(self,
                                  functional_features: Optional[list[str]] = None) -> Optional[torch.Tensor]:
        if functional_features is None:
            return None

        feature_names = self.data_preparer.columns

        # Add numerical features
        functional_feature_names = [
            f for f in feature_names if f in functional_features]

        # Add one-hot encoded features
        ohe = self.data_preparer.encoders["onehot"]
        ohe_cols = ohe.get_feature_names_out()

        for col in functional_features:
            matching_ohe_cols = [
                fname for fname in ohe_cols if fname.startswith(f"{col}_")]
            functional_feature_names.extend(matching_ohe_cols)

        functional_mask = [
            1 if fname in functional_feature_names else 0 for fname in feature_names]
        functional_mask = np.array(functional_mask, dtype=bool)
        functional_mask_t = torch.tensor(
            functional_mask, dtype=torch.bool, device=self.device)
        return functional_mask_t

    def _generate_loaders(self, df: pd.DataFrame):
        X, _ = self.data_preparer.scale_and_encode(df)
        loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        return loader

    def query_generator(self,
                        x_t: torch.Tensor,
                        functional_features: Optional[list[str]] = None) -> torch.Tensor:
        x_t = x_t.to(self.device).float()
        z = torch.rand((x_t.size(0), self.noise_dim),
                       device=self.device)  # Random noise
        g_out = self.generator(x_t, z)

        # Clamp to [0,1]
        g_out = torch.clamp(g_out, 0.0, 1.0)

        # Apply functional mask if provided
        func_mask_t = self._generate_functional_mask(functional_features)
        if func_mask_t is not None:
            g_out = torch.where(func_mask_t, x_t, g_out)

        # Filter one-hot encoded features to ensure valid encoding
        ohe = self.data_preparer.encoders.get("onehot", None)

        if ohe is not None:
            ohe_inputs = ohe.feature_names_in_
            ohe_categories = ohe.categories_

            for i, col in enumerate(ohe_inputs):
                categories = ohe_categories[i]
                ohe_col_names = [f"{col}_{cat}" for cat in categories]
                ohe_col_indices = [
                    self.data_preparer.df.columns.get_loc(name) for name in ohe_col_names]

                # Get the generated values for these columns
                ohe_values = g_out[:, ohe_col_indices]

                # Find the index of the max value for each sample
                max_indices = torch.argmax(ohe_values, dim=1)

                # Create a zero tensor and set the max indices to 1
                one_hot = torch.zeros_like(ohe_values)
                one_hot[torch.arange(ohe_values.size(0)), max_indices] = 1.0

                # Update the generated output with valid one-hot encoding
                g_out[:, ohe_col_indices] = one_hot

        return g_out

    def train(self,
              attack_df: pd.DataFrame,
              normal_df: pd.DataFrame,
              epochs: int,
              blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
              functional_features: Optional[list[str]] = None,
              print_every: int = 1) -> None:

        self.query_stats = {
            "n_queries": 0,
            "malicious_queries": 0,
            "benign_queries": 0,
        }

        # Get loaders
        attack_loader = self._generate_loaders(attack_df)
        normal_loader = self._generate_loaders(normal_df)

        # Calculate how many batches should be sampled from each loader per iteration
        num_attack_batches = len(attack_loader)
        num_normal_batches = len(normal_loader)

        if num_attack_batches > num_normal_batches:
            samples_per_iteration = [
                1, max(1, num_attack_batches // max(1, num_normal_batches))]
        else:
            samples_per_iteration = [
                max(1, num_normal_batches // max(1, num_attack_batches)), 1]
        total_steps = max(num_attack_batches, num_normal_batches)

        # Main training loop
        while self.epoch < epochs:
            iter_att = iter(attack_loader)
            iter_norm = iter(normal_loader)

            for _ in range(total_steps):

                # Prepare batches
                normal_batches = []
                attack_batches = []

                for _ in range(samples_per_iteration[0]):
                    try:
                        batch = next(iter_norm)
                    except StopIteration:
                        iter_norm = iter(normal_loader)
                        batch = next(iter_norm)
                    normal_batches.append(batch)

                for _ in range(samples_per_iteration[1]):
                    try:
                        batch = next(iter_att)
                    except StopIteration:
                        iter_att = iter(attack_loader)
                        batch = next(iter_att)
                    attack_batches.append(batch)

                x_norm = torch.cat(normal_batches, dim=0).to(
                    self.device).float()
                x_att = torch.cat(attack_batches, dim=0).to(
                    self.device).float()

                #
                # Train Discriminator
                #
                self.discriminator.train()
                for _ in range(self.steps_discriminator):
                    with torch.no_grad():
                        x_adv_batch = self.query_generator(
                            x_att, functional_features
                        )

                    # Query IDS on both normal and adversarial malicious
                    concat = torch.cat([x_norm, x_adv_batch], dim=0)
                    y_ids = self.query_blackbox(concat, blackbox_predict)

                    y_norm_ids = y_ids[:x_norm.size(0)]
                    y_adv_ids = y_ids[x_norm.size(0):]

                    d_norm = self.discriminator(x_norm)
                    d_adv = self.discriminator(x_adv_batch)

                    loss_D_norm = self.mse_loss(d_norm, y_norm_ids)
                    loss_D_adv = self.mse_loss(d_adv, y_adv_ids)
                    loss_D = loss_D_norm + loss_D_adv

                    self.optimizer_D.zero_grad()
                    loss_D.backward()
                    self.optimizer_D.step()

                #
                # Train Generator
                #
                self.generator.train()
                for _ in range(self.steps_generator):
                    x_adv = self.query_generator(
                        x_att, functional_features
                    )

                    d_adv = self.discriminator(x_adv)
                    # We want IDS ≈ NORMAL => target 1.0
                    target_normal = torch.ones_like(d_adv, device=self.device)

                    loss_G = self.mse_loss(d_adv, target_normal)

                    self.optimizer_G.zero_grad()
                    loss_G.backward()
                    self.optimizer_G.step()

            # Epoch evaluation
            self.epoch += 1
            if (self.epoch) % print_every == 0 or self.epoch == 1:
                self.discriminator.eval()
                self.generator.eval()
                with torch.no_grad():
                    orig_detected = 0
                    adv_detected = 0
                    total_samples = 0

                    for batch in attack_loader:
                        # Get original predictions
                        preds_orig = self.query_blackbox(
                            batch, blackbox_predict, training_phase=False)

                        # Generate adversarial samples and get predictions
                        adv = self.query_generator(batch, functional_features)
                        preds_adv = self.query_blackbox(
                            adv, blackbox_predict, training_phase=False)
                        # Calculate detected samples
                        orig_detected += torch.sum(preds_orig <= 0.5).item()
                        adv_detected += torch.sum(preds_adv <= 0.5).item()
                        total_samples += batch.size(0)

                    # Calculate detection rates and EIR
                    orig_DR = orig_detected / total_samples if total_samples > 0 else 0
                    adv_DR = adv_detected / total_samples if total_samples > 0 else 0
                    EIR = 1 - (adv_DR / (orig_DR)) if orig_DR > 0 else 1

                    print(
                        f"Epoch {self.epoch}/{epochs}: original detection rate={orig_DR:.4f} adversarial detection rate={adv_DR:.4f} EIR={EIR:.4f}")

        self.save_networks(self.snapshot_path)

    def generate_samples(self,
                         attack_df: pd.DataFrame,
                         functional_features: Optional[list[str]] = None,
                         ) -> pd.DataFrame:
        self.generator.eval()

        attack_loader = self._generate_loaders(attack_df)

        samples = np.empty((0, self.feature_dim), dtype=np.float32)

        with torch.no_grad():
            for x_batch in attack_loader:
                batch_samples = self.query_generator(
                    x_batch.to(self.device), functional_features)
                samples = np.vstack((samples, batch_samples.cpu().numpy()))

        samples_df = self.data_preparer.unscale_and_decode(samples)

        return samples_df

    def evaluate(self,
                 attack_df: pd.DataFrame,
                 blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
                 functional_features: Optional[list[str]] = None,
                 ) -> dict:

        self.generator.eval()

        attack_loader = self._generate_loaders(attack_df)

        total = 0
        fooled = 0
        distance = 0.0

        with torch.no_grad():
            for x_batch in attack_loader:
                # Generate adversarial samples
                adv = self.query_generator(
                    x_batch.to(self.device), functional_features)

                # Query black-box model on generated samples
                preds = self.query_blackbox(
                    adv, blackbox_predict, training_phase=False)

                # Calculate distance between original and generated samples
                batch_orig = x_batch.cpu().numpy()
                batch_gen = adv.cpu().numpy()
                distance += np.sum(np.linalg.norm(batch_orig -
                                   batch_gen, axis=1))

                # Calculate fooled samples
                fooled += np.sum((preds > 0.5).astype(int))
                total += len(preds)

        distance /= total if total > 0 else 0.0
        attack_success_rate = fooled / total if total > 0 else 0.0

        return {
            "total_samples": total,
            "fooled_samples": fooled,
            "attack_success_rate": attack_success_rate,
            "average_distance": distance
        }
