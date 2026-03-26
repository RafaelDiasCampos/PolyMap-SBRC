from ..neural_networks import FNN, SNN
from ..data_preparer import DataPreparer
from ..network_trainer import NetworkTrainer

network_types = ["fnn", "snn"]


def load_networks(dataset: dict, dataset_name: str, batch_size: int,
                  n_hidden_layers: int, n_neurons: int, dropout_prob_fnn: float,
                  dropout_prob_snn: float, learning_rate_fnn: float,
                  learning_rate_snn: float, random_state: int) -> None:

    # Prepare the data
    dataset["dataPreparer"] = DataPreparer(
        dataset["dataset"], batch_size=batch_size, random_state=random_state)
    train_loader, test_loader, _ = dataset["dataPreparer"].get_loaders()

    in_features = len(dataset["dataPreparer"].df.columns) - 1
    out_features = dataset["dataset"]['Label'].nunique()

    # Create the neural networks
    fnn = FNN(in_features, out_features, n_hidden_layers,
              n_neurons, dropout_prob=dropout_prob_fnn)
    snn = SNN(in_features, out_features, n_hidden_layers,
              n_neurons, dropout_prob=dropout_prob_snn)

    # Create the network trainers
    fnnTrainer = NetworkTrainer(
        fnn, train_loader, test_loader, learning_rate=learning_rate_fnn, snapshot_folder=f"snapshots/{dataset_name}")
    snnTrainer = NetworkTrainer(
        snn, train_loader, test_loader, learning_rate=learning_rate_snn, snapshot_folder=f"snapshots/{dataset_name}")

    dataset["fnn_trainer"] = fnnTrainer
    dataset["snn_trainer"] = snnTrainer

    return dataset
