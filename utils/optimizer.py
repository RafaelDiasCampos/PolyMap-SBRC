import optuna
import pandas as pd
from .data_preparer import DataPreparer
from .network_trainer import NetworkTrainer
from .neural_networks import NeuralNetwork


class Optimizer:
    def __init__(
        self,
        df: pd.DataFrame,
        network: NeuralNetwork,
        n_hidden_layers: list,
        n_neurons: list,
        batch_size: list,
        learning_rate: list,
        dropout_prob: list,
        random_state: int = 42
    ):
        self.df = df
        self.network = network
        self.random_state = random_state

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob

        self.random_state = random_state

        self.in_features = len(df.columns) - 1
        self.out_features = df['type'].nunique()

    def objective(self, epochs: int, trial: optuna.Trial) -> float:
        # Sample parameters
        n_hidden_layers = trial.suggest_int(
            'n_hidden_layers', self.n_hidden_layers[0], self.n_hidden_layers[-1])
        n_neurons = trial.suggest_categorical('n_neurons', self.n_neurons)
        batch_size = trial.suggest_categorical('batch_size', self.batch_size)
        learning_rate = trial.suggest_float(
            'learning_rate', self.learning_rate[0], self.learning_rate[-1], log=True)
        dropout_prob = trial.suggest_float(
            'dropout_prob', self.dropout_prob[0], self.dropout_prob[-1])

        # Prepare the data
        dataPreparer = DataPreparer(
            self.df, batch_size=batch_size, random_state=self.random_state)
        train_loader, test_loader, val_loader = dataPreparer.get_loaders()

        # Initialize the model
        model = self.network(
            in_features=self.in_features,
            out_features=self.out_features,
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            dropout_prob=dropout_prob
        )

        # Initialize the trainer
        trainer = NetworkTrainer(
            network=model,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            ephemeral=True  # Ephemeral to avoid saving/loading during optimization
        )

        trainer.train_network(
            epochs=epochs,
            verbose=False
        )

        # Evaluate the model
        val_accuracy = trainer.validate()
        return val_accuracy

    def optimize(self, n_trials: int = 20, epochs: int = 100) -> None:
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(lambda trial: self.objective(
            epochs, trial), n_trials=n_trials, show_progress_bar=True)

        print("Best hyperparameters:", self.study.best_params)
        print("Best validation accuracy:", self.study.best_value)
