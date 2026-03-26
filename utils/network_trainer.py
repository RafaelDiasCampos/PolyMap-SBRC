import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from .data_preparer import DataPreparer


class NetworkTrainer:
    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 0.001,
        ephemeral: bool = False,
        snapshot_folder: str = "snapshots"
    ):
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate)

        self.ephemeral = ephemeral

        self.model_name = network.__class__.__name__
        self.snapshot_path = os.path.join(
            snapshot_folder, f"{self.model_name}_snapshot.pth")

        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)

        # Load the network if a snapshot exists
        self.load_network(self.snapshot_path)

    def load_network(
        self,
        path: str
    ) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(path) and not self.ephemeral:
            checkpoint = torch.load(path)

            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            self.epoch = 0

        self.network.to(self.device)

    def save_network(
        self,
        path: str
    ) -> None:
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)
        print(f"Network saved to {path}")

    def get_test_stats(
        self
    ):
        criterion = nn.NLLLoss()

        self.network.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.network(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_test_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total

        return avg_test_loss, accuracy

    def train_network(
        self,
        epochs: int,
        verbose: bool = True
    ) -> None:
        criterion = nn.NLLLoss()

        last_epoch = self.epoch + epochs

        while self.epoch < last_epoch:
            self.network.train()
            train_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.network(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            if verbose and (self.epoch % 10 == 0 or self.epoch == last_epoch - 1):
                avg_test_loss, accuracy = self.get_test_stats()
                print(
                    f"Epoch [{self.epoch}/{last_epoch}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.epoch += 1

        if not self.ephemeral:
            self.save_network(self.snapshot_path)

    def predict_test(
        self
    ) -> tuple:
        self.network.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.network(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        return all_preds, all_targets

    def get_confusion_matrix(
        self,
        data_preparer: DataPreparer
    ) -> tuple:

        all_preds, all_targets = self.predict_test()

        conf_matrix = confusion_matrix(all_targets, all_preds)
        class_names = data_preparer.le_y.classes_

        return conf_matrix, class_names

    def plot_confusion_matrix(
        self,
        data_preparer: DataPreparer,
        title: str | None = None
    ) -> None:
        conf_matrix, class_names = self.get_confusion_matrix(data_preparer)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(
            title if title else f'Confusion Matrix for {self.model_name}')
        plt.show()

    def get_classification_report(
        self,
        data_preparer: DataPreparer,
        output_dict: bool = False
    ) -> str | dict:
        all_preds, all_targets = self.predict_test()
        class_names = data_preparer.le_y.classes_

        report = classification_report(
            all_targets, all_preds, target_names=class_names, zero_division=0, output_dict=output_dict)

        return report

    def validate(
        self,
    ) -> float:
        if self.val_loader is None:
            raise ValueError("Validation loader is not provided.")

        self.network.eval()
        criterion = nn.NLLLoss()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.network(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def predict(
        self,
        df: pd.DataFrame,
        data_preparer: DataPreparer
    ) -> np.ndarray:
        inputs, _ = data_preparer.scale_and_encode(df)
        inputs_tensor = torch.tensor(
            inputs, dtype=torch.float32).to(self.device)

        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_classes = data_preparer.le_y.inverse_transform(
            predicted.cpu().numpy())

        return predicted_classes

    def predict_accuracy(
        self,
        df: pd.DataFrame,
        data_preparer: DataPreparer
    ) -> float:
        predicted_classes = self.predict(df, data_preparer)

        correct_predictions = np.sum(
            predicted_classes == df[data_preparer.df.columns[-1]].values)
        accuracy = correct_predictions / len(df) * 100

        return accuracy
