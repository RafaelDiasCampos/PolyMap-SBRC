import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..attacks.idsgan import IDSGANAttack
from ..attacks.genaal import GenAALAttack
from ..attacks.polytope import PolytopeAttack
from ..network_trainer import NetworkTrainer
from ..data_preparer import DataPreparer

#
# General Attack helpers
#


def blackbox_predict_template(network_trainer: NetworkTrainer, data_preparer: DataPreparer, input_data: pd.DataFrame, normal_target: str) -> np.ndarray:
    predicted = network_trainer.predict(input_data, data_preparer)

    result = np.where(predicted == normal_target, 1, 0)
    return result


def get_attack_datasets(dataset: dict, frac: float = 0.2, train_size: float = 0.6, random_state: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    attack_df = dataset["dataPreparer"].df.sample(
        frac=frac, random_state=random_state)

    X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(
        attack_df.iloc[:, :-1].to_numpy(),
        attack_df.iloc[:, -1].to_numpy(),
        train_size=train_size,
        random_state=random_state
    )

    attack_train_df = dataset["dataPreparer"].unscale_and_decode(
        X_attack_train, y_attack_train)

    attack_test_df = dataset["dataPreparer"].unscale_and_decode(
        X_attack_test, y_attack_test)

    return attack_train_df, attack_test_df

#
# IDSGAN
#


def create_and_train_idsgan(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset: dict, dataset_name: str, network_name: str, functional_features: list[str],
                            noise_dim: int = 32,
                            n_hidden_layers_generator: int = 2,
                            n_neurons_generator: int = 64,
                            learning_rate_generator: float = 0.0002,
                            steps_generator: int = 1,
                            n_hidden_layers_discriminator: int = 2,
                            n_neurons_discriminator: int = 64,
                            learning_rate_discriminator: float = 0.0002,
                            steps_discriminator: int = 1,
                            weight_clip: float = 0.01,
                            batch_size: int = 64,
                            idsgan_epochs: int = 100) -> IDSGANAttack:

    # Create the IDSGAN attack instance
    attack_samples = pd.concat([train_df, test_df], ignore_index=True)

    idsgan = IDSGANAttack(
        attack_samples=attack_samples,
        noise_dim=noise_dim,
        n_hidden_layers_generator=n_hidden_layers_generator,
        n_neurons_generator=n_neurons_generator,
        learning_rate_generator=learning_rate_generator,
        steps_generator=steps_generator,
        n_hidden_layers_discriminator=n_hidden_layers_discriminator,
        n_neurons_discriminator=n_neurons_discriminator,
        learning_rate_discriminator=learning_rate_discriminator,
        steps_discriminator=steps_discriminator,
        weight_clip=weight_clip,
        batch_size=batch_size,
        snapshot_folder=f"snapshots/{dataset_name}",
        snapshot_name=f"idsgan_snapshot_{network_name}.pth",
        ephemeral=True
    )

    # Create datasets with only attack samples and normal samples
    attack_df = train_df[train_df['Label'] !=
                         dataset["normal_label"]]
    normal_df = train_df[train_df['Label'] ==
                         dataset["normal_label"]]

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    # Train the IDSGAN
    idsgan.train(
        attack_df=attack_df,
        normal_df=normal_df,
        epochs=idsgan_epochs,
        blackbox_predict=blackbox_predict,
        functional_features=functional_features,
    )

    return idsgan


def evaluate_idsgan(idsgan: IDSGANAttack, test_df: pd.DataFrame, dataset: dict, network_name: str, functional_features: list[str]) -> dict:
    # Keep only attack samples in the test set
    attack_test_df = test_df[test_df['Label'] !=
                             dataset["normal_label"]]

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    generated_samples = idsgan.generate_samples(
        attack_test_df, functional_features)
    blackbox_results = blackbox_predict(generated_samples)

    original_predictions = blackbox_predict(attack_test_df)

    # Samples originally detected as attacks
    original_detected = np.sum(original_predictions == 0)
    # Samples still detected as attacks after perturbation
    adversarial_detected = np.sum(blackbox_results == 0)

    accuracy = 1 - (adversarial_detected /
                    original_detected) if original_detected > 0 else 1

    encoded_samples, _ = dataset["dataPreparer"].scale_and_encode(
        generated_samples)
    original_encoded, _ = dataset["dataPreparer"].scale_and_encode(
        attack_test_df)

    distance = np.mean(np.linalg.norm(
        encoded_samples - original_encoded, axis=1))

    results = {
        "accuracy": accuracy,
        "average_distance": distance
    }

    results["query_stats"] = idsgan.query_stats

    return results

#
# GenAAL
#


def create_and_train_genaal(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset: dict, dataset_name: str, network_name: str,
                            latent_dim: int = 16,
                            vae_hidden: int = 64,
                            sids_hidden: int = 64,
                            lambda_kl: float = 0.1,
                            lambda_recon: float = 1.0,
                            lambda_l2: float = 0.1,
                            lambda_label: float = 0.5,
                            gen_lr: float = 0.0002,
                            sid_lr: float = 0.0002,
                            batch_size: int = 64,
                            max_iterations: int = 1000,
                            candidate_pool_k: int = 50,
                            nquery: int = 10000,
                            pretrain_epochs: int = 10,
                            sids_epochs: int = 10,
                            gen_epochs: int = 10,
                            label_query: int = 15) -> GenAALAttack:

    # Create the GenAAL attack instance
    attack_samples = pd.concat([train_df, test_df], ignore_index=True)

    genaal = GenAALAttack(
        attack_samples=attack_samples,
        latent_dim=latent_dim,
        vae_hidden=vae_hidden,
        sids_hidden=sids_hidden,
        lambda_kl=lambda_kl,
        lambda_recon=lambda_recon,
        lambda_l2=lambda_l2,
        lambda_label=lambda_label,
        gen_lr=gen_lr,
        sid_lr=sid_lr,
        batch_size=batch_size,
        snapshot_folder=f"snapshots/{dataset_name}",
        snapshot_name=f"genaal_snapshot_{network_name}.pth",
        ephemeral=True
    )

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    genaal.fit(
        df=train_df,
        blackbox_predict=blackbox_predict,
        label_query=label_query,
        max_iterations=max_iterations,
        candidate_pool_k=candidate_pool_k,
        nquery=nquery,
        pretrain_epochs=pretrain_epochs,
        sids_epochs=sids_epochs,
        gen_epochs=gen_epochs
    )

    return genaal


def evaluate_genaal(genaal: GenAALAttack, test_df, dataset: dict, network_name: str):
    # Keep only attack samples in the test set
    attack_test_df = test_df[test_df['Label'] != dataset["normal_label"]]

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    generated_samples = genaal.generate_samples(attack_test_df)
    blackbox_results = blackbox_predict(generated_samples)

    original_predictions = blackbox_predict(attack_test_df)

    # Samples originally detected as attacks
    original_detected = np.sum(original_predictions == 0)
    # Samples still detected as attacks after perturbation
    adversarial_detected = np.sum(blackbox_results == 0)

    accuracy = 1 - (adversarial_detected /
                    original_detected) if original_detected > 0 else 1

    generated_encoded, _ = dataset["dataPreparer"].scale_and_encode(
        generated_samples)
    original_encoded, _ = dataset["dataPreparer"].scale_and_encode(
        attack_test_df)

    distance = np.mean(np.linalg.norm(
        generated_encoded - original_encoded, axis=1))

    results = {
        "accuracy": accuracy,
        "average_distance": distance
    }

    results["query_stats"] = genaal.query_stats

    return results

#
# Polytopes
#


def create_and_train_polytope(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset: dict, network_name: str,
                              batch_size: int = 64,
                              random_state: int = None,
                              n_rays: int = 50,
                              step_size: float = 0.01) -> PolytopeAttack:

    # Create the Polytope attack instance
    attack_samples = pd.concat([train_df, test_df], ignore_index=True)

    # Separate normal samples
    normal_samples = attack_samples[attack_samples['Label']
                                    == dataset["normal_label"]]

    polytope = PolytopeAttack(
        attack_samples=attack_samples,
        batch_size=batch_size,
        random_state=random_state,
    )

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    polytope.fit(
        normal_samples=normal_samples,
        blackbox_predict=blackbox_predict,
        n_rays=n_rays,
        step_size=step_size
    )

    return polytope


def evaluate_polytope(polytope: PolytopeAttack, test_df: pd.DataFrame, dataset: dict, network_name: str, functional_features: list[str] = None, move_inside: float = 0.01):
    # Keep only attack samples in the test set
    attack_test_df = test_df[test_df['Label'] != dataset["normal_label"]]

    # Create blackbox predict function
    def blackbox_predict(input_data): return blackbox_predict_template(
        dataset[f"{network_name}_trainer"],
        dataset["dataPreparer"],
        input_data,
        dataset["normal_label"]
    )

    generated_samples = polytope.generate_samples(
        attack_test_df, move_inside=move_inside)
    blackbox_results = blackbox_predict(generated_samples)

    original_predictions = blackbox_predict(attack_test_df)

    # Samples originally detected as attacks
    original_detected = np.sum(original_predictions == 0)
    # Samples still detected as attacks after perturbation
    adversarial_detected = np.sum(blackbox_results == 0)

    accuracy = 1 - (adversarial_detected /
                    original_detected) if original_detected > 0 else 1

    generated_encoded, _ = dataset["dataPreparer"].scale_and_encode(
        generated_samples)
    original_encoded, _ = dataset["dataPreparer"].scale_and_encode(
        attack_test_df)

    distance = np.mean(np.linalg.norm(
        generated_encoded - original_encoded, axis=1))

    results = {
        "accuracy": accuracy,
        "average_distance": distance
    }

    results["query_stats"] = polytope.query_stats

    return results
