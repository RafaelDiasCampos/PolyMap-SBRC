import pandas as pd
import os
from scipy.io import arff


class DataLoader:
    def __init__(
        self,
        data_folder: str,
        low_memory: bool = False
    ):
        self.data_folder = data_folder
        self.data_path = os.path.join(data_folder, 'data.pkl')
        self.low_memory = low_memory

    def data_parser(
        self,
        original_columns: list[str] = None
    ) -> pd.DataFrame:
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(
                f"The folder '{self.data_folder}' does not exist.")

        if os.path.exists(self.data_path):
            return pd.read_pickle(self.data_path)

        # Get the name of all CSV files in the data folder
        files = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith('.csv') or file_name.endswith('.arff'):
                file_path = os.path.join(self.data_folder, file_name)
                files.append(file_path)

        df = pd.DataFrame()

        for file_path in files:
            if file_path.endswith('.csv'):
                temp_df = pd.read_csv(
                    file_path, low_memory=self.low_memory, names=original_columns)
            elif file_path.endswith('.arff'):
                temp_df = pd.DataFrame(arff.loadarff(file_path)[0])
            df = pd.concat([df, temp_df], ignore_index=True)

        df.to_pickle(self.data_path)

        return df

    def load_data(
        self,
        columns: list[str] = None,
        original_columns: list[str] = None
    ) -> pd.DataFrame:
        df = self.data_parser(original_columns=original_columns)

        if columns is not None:
            df = df[columns]

        return df
