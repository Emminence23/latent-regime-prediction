"""Load the AQI sensor data and make the train/test split used in the report."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "DD1(Oct).csv"
LABEL = "Ref. O3(ppb)"
ALL_FEATURES = ["Time", "Temp(C)", "RH(%)", "o3op1(mV)", "o3op2(mV)"]


def load_data(path=DATA_PATH, train_frac=0.8):
    """Return a dict with raw and standardised train/test frames.

    Keeps rows with Valid == 1, turns "DD-MM-YYYY HH:MM" into the hour of day,
    splits by row order (first 80% train) and standardises with train stats.
    """
    df = pd.read_csv(path)
    df = df.loc[df["Valid"] == 1, ["Time", LABEL] + ALL_FEATURES[1:]].copy()
    df["Time"] = df["Time"].map(
        lambda s: int(s.replace("-", " ").replace(":", " ").split()[-2])
    )
    n_train = int(len(df) * train_frac)
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    means, stds = train.mean(), train.std()
    return {
        "train": train,
        "test": test,
        "train_std": (train - means) / stds,
        "test_std": (test - means) / stds,
        "y_mean": means[LABEL],
        "y_sd": stds[LABEL],
    }
