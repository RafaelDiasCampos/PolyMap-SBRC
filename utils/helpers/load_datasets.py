from sklearn import datasets
from ..data_loader import DataLoader
import pandas as pd
import numpy as np


def load_ton_iot() -> dict:
    dataLoader = DataLoader("dataset/ton_iot")
    columns = ["src_port",
               "dst_port",
               "proto",
               "duration",
               "src_bytes",
               "dst_bytes",
               "conn_state",
               "missed_bytes",
               "src_pkts",
               "src_ip_bytes",
               "dst_pkts",
               "dst_ip_bytes",
               "type"]

    dataset = {}
    dataset["dataset"] = dataLoader.load_data(columns=columns)
    dataset["dataset"].rename(columns={"type": "Label"}, inplace=True)
    dataset["normal_label"] = "normal"
    # dataset["functional_features"] = np.array(["proto", "conn_state"])
    dataset["functional_features"] = np.array([])
    dataset["attack_frac"] = 0.2
    dataset["move_inside"] = 0.15

    return dataset


def load_bot_iot() -> dict:
    dataLoader = DataLoader("dataset/bot_iot")
    columns = ["sport",
               "dport",
               "proto",
               "dur",
               "sbytes",
               "dbytes",
               "state",
               "spkts",
               "TnBPSrcIP",
               "dpkts",
               "TnBPDstIP",
               "category"]

    dataset = {}
    dataset["dataset"] = dataLoader.load_data(columns=columns)
    dataset["dataset"].rename(columns={"category": "Label"}, inplace=True)

    # Convert sport and dport rows to np.int32 -> Some values in these fields are 0x0303
    cols = ['sport', 'dport']

    for c in cols:
        s = pd.to_numeric(dataset["dataset"][c], errors='coerce')
        s = s.where((s.notna()) & (np.floor(s) == s) & (s.between(1, 65535)))
        dataset["dataset"][c] = s.astype('Int64')

    dataset["dataset"] = dataset["dataset"].dropna(subset=cols)
    dataset["dataset"][cols] = dataset["dataset"][cols].astype(np.int32)
    dataset["normal_label"] = "Normal"
    dataset["functional_features"] = np.array([])
    dataset["attack_frac"] = 0.03
    dataset["move_inside"] = 0.3

    return dataset


def load_ctu_13() -> dict:
    dataLoader = DataLoader("dataset/ctu-13")
    columns = ["Sport",
               "Dport",
               "Proto",
               "Dur",
               "SrcBytes",
               "TotBytes",
               "State",
               "TotPkts",
               "Dir",
               "sTos",
               "dTos",
               "Label"]

    object_columns = [
        "Proto",
        "Dir",
        "State",
        "Label"]

    dataset = {}
    dataset["dataset"] = dataLoader.load_data(columns=columns)

    # Group unique labels into "Normal" and "Botnet"
    unique_labels = dataset["dataset"]['Label'].unique()
    for label in unique_labels:
        if "Background" in label:
            dataset["dataset"].loc[dataset["dataset"]
                                   ["Label"] == label, "Label"] = "Normal"
        elif "Normal" in label:
            dataset["dataset"].loc[dataset["dataset"]
                                   ["Label"] == label, "Label"] = "Normal"
        elif "Botnet" in label:
            dataset["dataset"].loc[dataset["dataset"]
                                   ["Label"] == label, "Label"] = "Botnet"

    # Convert every non-object column to numeric
    for col in dataset["dataset"].columns:
        if col not in object_columns:
            dataset["dataset"][col] = pd.to_numeric(
                dataset["dataset"][col], errors='coerce')

    # Convert object columns to string type
    for col in object_columns:
        dataset["dataset"][col] = dataset["dataset"][col].astype(str)
    dataset["dataset"].replace([np.inf, -np.inf], np.nan, inplace=True)

    df = dataset["dataset"]

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    str_cols = df.select_dtypes(include=["string", "object"]).columns
    df[str_cols] = df[str_cols].fillna("0")

    dataset["normal_label"] = "Normal"
    dataset["functional_features"] = np.array([])
    dataset["attack_frac"] = 0.05
    dataset["move_inside"] = 0.008

    return dataset

def load_nsl_kdd() -> dict:
    dataLoader = DataLoader("dataset/nsl-kdd")
    columns = ["protocol_type",
               "duration",
               "src_bytes",
               "dst_bytes",
               "flag",
               "service",
               "dst_host_count",
               "count",
               "srv_count",
               "serror_rate",
               "class"]

    dataset = {}
    dataset["dataset"] = dataLoader.load_data(columns=columns)
    dataset["dataset"].rename(columns={"class": "Label"}, inplace=True)

    str_columns = ["protocol_type", "flag", "service", "Label"]
    dataset["dataset"] = dataset["dataset"].astype(
        {col: 'string' for col in str_columns})
    dataset["normal_label"] = "normal"
    # dataset["functional_features"] = np.array(["protocol_type", "flag", "service"])
    dataset["functional_features"] = np.array([])
    dataset["attack_frac"] = 0.2
    dataset["move_inside"] = 0.35

    return dataset


def load_all_datasets(copies: int = None, random_state: int = None, frac: float = 1.0) -> dict:
    datasets = {}
    
    try:
        datasets["ton_iot"] = load_ton_iot()
    except Exception as e:
        print(f"Error loading TON IoT dataset: {e}")
        
    try:
        datasets["bot_iot"] = load_bot_iot()
    except Exception as e:
        print(f"Error loading Bot IoT dataset: {e}")
        
    try:
        datasets["ctu_13"] = load_ctu_13()
    except Exception as e:
        print(f"Error loading CTU-13 dataset: {e}")
        
    try:
        datasets["nsl_kdd"] = load_nsl_kdd()
    except Exception as e:
        print(f"Error loading NSL-KDD dataset: {e}")
        
    if frac < 1.0:
        for dataset_name, dataset in datasets.items():
            dataset["dataset"] = dataset["dataset"].sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    # Optionally create shuffled copies of each dataset
    if copies is not None:
        if random_state is None:
            random_state = np.random.randint(0, 10000)

        dataset_names = list(datasets.keys())
        for i in range(copies):
            for name in dataset_names:
                dataset = datasets[name]
                new_name = f"{name}_copy_{i+1}"
                dataset_copy = dataset["dataset"].sample(
                    frac=1.0, random_state=random_state + i).reset_index(drop=True)

                datasets[new_name] = {}
                for key in dataset:
                    if key == "dataset":
                        datasets[new_name]["dataset"] = dataset_copy
                    else:
                        datasets[new_name][key] = dataset[key]

    return datasets
