import pandas as pd
import numpy as np

def load_nasa_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)

    # Create column names
    columns = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, df.shape[1]-1)]
    df.columns = columns

    return df


def add_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df


def create_failure_label(df, threshold=30):
    df["failure"] = np.where(df["RUL"] <= threshold, 1, 0)
    return df


def prepare_features(df):
    # Drop unnecessary columns
    df = df.drop(["engine_id", "cycle", "max_cycle", "RUL"], axis=1)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    return X, y