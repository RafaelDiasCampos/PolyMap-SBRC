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


def load_bot_iot_full() -> dict:
    dataLoader = DataLoader("dataset/bot_iot_full")
    original_columns = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'smac',
                        'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']

    columns = ["sport",
               "dport",
               "proto",
               "dur",
               "sbytes",
               "dbytes",
               "state",
               "spkts",
               "dpkts",
               "category"]

    dataset = {}

    # Load data from the bot_iot dataset
    dataset["dataset"] = load_bot_iot()["dataset"].copy()

    # Load Full Data
    full_data = dataLoader.load_data(
        columns=columns, original_columns=original_columns)
    full_data.rename(columns={"category": "Label"}, inplace=True)

    # Convert sport and dport rows to np.int32 -> Some values in these fields are 0x0303
    cols = ['sport', 'dport']

    for c in cols:
        s = pd.to_numeric(full_data[c], errors='coerce')
        s = s.where((s.notna()) & (np.floor(s) == s) & (s.between(1, 65535)))
        full_data[c] = s.astype('Int64')

    full_data = full_data.dropna(subset=cols)
    full_data[cols] = full_data[cols].astype(np.int32)

    labels_to_add = [
        "Normal",
        "Theft"
    ]

    # Drop labels from the dataset
    dataset["dataset"] = dataset["dataset"][~dataset["dataset"]
                                            ["Label"].isin(labels_to_add)]

    # Add new labels to the dataset
    full_data = full_data[full_data["Label"].isin(labels_to_add)]
    dataset["dataset"] = pd.concat([dataset["dataset"], full_data])
    del full_data

    dataset["normal_label"] = "Normal"
    dataset["functional_features"] = np.array([])
    dataset["attack_frac"] = 0.2
    dataset["move_inside"] = 0.001

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


def load_all_datasets(copies: int = None, random_state: int = None) -> dict:
    datasets = {
        "ton_iot": load_ton_iot(),
        "bot_iot": load_bot_iot(),
        "ctu_13": load_ctu_13(),
        # "bot_iot_full": load_bot_iot_full(),
        "nsl_kdd": load_nsl_kdd()
    }

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
