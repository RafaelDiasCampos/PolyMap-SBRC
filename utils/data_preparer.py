import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


class DataPreparer:
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: float = 42,
        encoder_type: str = "label",
        scaler_type: str = "standard"
    ):
        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.encoder_type = encoder_type
        self.scaler_type = scaler_type

        if self.encoder_type not in ["label", "onehot"]:
            raise ValueError("encoder_type must be either 'label' or 'onehot'")

        if self.scaler_type not in ["standard", "minmax"]:
            raise ValueError(
                "scaler_type must be either 'standard' or 'minmax'")

        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

        self.encoders = {}
        self.original_columns = df.columns.tolist()
        self.n_columns = len(X.columns)

        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(
            include=["object", "category", "string"]).columns
        self.numerical_cols = X.select_dtypes(include=["number"]).columns

        if self.encoder_type == "label":
            # Label encode categorical columns in X
            for col in self.categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
            self.columns = X.columns  # keep original columns
        else:
            # One-hot encode categorical columns in X
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_ohe = ohe.fit_transform(X[self.categorical_cols].astype(str))
            ohe_cols = ohe.get_feature_names_out(self.categorical_cols)
            X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_cols, index=X.index)
            X = pd.concat(
                [X.drop(columns=self.categorical_cols), X_ohe_df], axis=1)
            self.encoders["onehot"] = ohe
            self.categorical_cols
            self.columns = np.concatenate(
                [self.numerical_cols, ohe_cols])  # update columns

        # Scale columns in X
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Label encode y (target)
        self.le_y = LabelEncoder()
        y = self.le_y.fit_transform(y.astype(str))
        self.encoders["target"] = self.le_y

        # Create processed DataFrame
        self.df = pd.DataFrame(X, columns=X.columns)
        self.df[df.columns[-1]] = y

    def split_train_test_val(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Extract features and labels
        X = self.df.iloc[:, :-1].to_numpy()
        Y = self.df.iloc[:, -1].to_numpy()

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y)

        # Further split training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=(self.validation_size/(1 - self.test_size)), random_state=self.random_state, stratify=y_train)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def get_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader]:
        X_train, X_test, X_val, y_train, y_test, y_val = self.split_train_test_val()

        # Calculate class weights for WeightedRandomSampler
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1. / class_counts
        sample_weights = class_weights[y_train.astype(int)]

        # Create a WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # Convert to torch tensors
        def to_tensor(x, y):
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        X_train_t, y_train_t = to_tensor(X_train, y_train)
        X_test_t, y_test_t = to_tensor(X_test, y_test)
        X_val_t, y_val_t = to_tensor(X_val, y_val)

        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, val_loader

    def scale_and_encode(
        self,
        df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if df.shape[1] == self.n_columns + 1:
            X = df.iloc[:, :-1].copy()
            Y = df.iloc[:, -1].copy()
        elif df.shape[1] == self.n_columns:
            X = df.copy()
            Y = None
        else:
            raise ValueError(
                "Input DataFrame has incorrect number of columns.")

        if self.encoder_type == "label":
            # Encode categorical columns
            for col in self.categorical_cols:
                le = self.encoders[col]
                X[col] = le.transform(X[col].astype(str))
        else:
            # One-hot encode categorical columns
            ohe = self.encoders["onehot"]
            X_ohe = ohe.transform(X[self.categorical_cols].astype(str))
            ohe_cols = ohe.get_feature_names_out(self.categorical_cols)
            X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_cols, index=X.index)
            X = pd.concat(
                [X.drop(columns=self.categorical_cols), X_ohe_df], axis=1)

        # Reorder columns to match training data
        X = X[self.columns]

        # Scale columns in X
        X_all = self.scaler.transform(X)

        if Y is not None:
            Y = self.le_y.transform(Y.astype(str))
            Y = Y.astype(float)  # ensure consistent numeric dtype
            return X_all, Y

        return X_all, None

    def unscale_and_decode(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> pd.DataFrame:
        # Recreate DataFrame with correct column names
        X_df = pd.DataFrame(X, columns=self.columns)

        # Unscale columns in X
        X_df = pd.DataFrame(self.scaler.inverse_transform(
            X_df), columns=self.columns)

        if self.encoder_type == "label":
            # Decode categorical columns
            for col in self.categorical_cols:
                le = self.encoders[col]
                X_df[col] = le.inverse_transform(X_df[col].astype(int))
        else:
            # Decode one-hot encoded columns
            ohe = self.encoders["onehot"]
            ohe_cols = ohe.get_feature_names_out(self.categorical_cols)
            X_ohe = X_df[ohe_cols]
            X_decoded = ohe.inverse_transform(X_ohe)
            X_decoded_df = pd.DataFrame(
                X_decoded, columns=self.categorical_cols, index=X_df.index)
            X_df = pd.concat(
                [X_df.drop(columns=ohe_cols), X_decoded_df], axis=1)

        # Decode Y if provided
        if Y is not None:
            Y_decoded = self.le_y.inverse_transform(Y.astype(int))
            X_df[self.df.columns[-1]] = Y_decoded

        return X_df
