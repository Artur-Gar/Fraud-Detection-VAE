from typing import Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


Mode = Literal["supervised", "unsupervised"]

def _load_credit_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, int, float]:
    df = pd.read_csv(csv_path)
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    n = len(df)
    fraud_rate = y.mean()
    return X, y, n, fraud_rate


def _global_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    main_frac: float,
    calib_frac: float,
    test_frac: float,
    random_state: int,
):
    assert abs(main_frac + calib_frac + test_frac - 1.0) < 1e-6, \
        "MAIN_FRAC + CALIB_FRAC + TEST_FRAC must sum to 1."

    # First: main vs temp
    test_size_temp = 1.0 - main_frac
    X_main, X_temp, y_main, y_temp = train_test_split(
        X,
        y,
        test_size=test_size_temp,
        stratify=y,
        random_state=random_state,
    )

    # Second: split temp into calib vs test
    calib_in_temp = calib_frac / (calib_frac + test_frac)
    test_size = (1.0 - calib_in_temp)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_main, X_calib, X_test, y_main, y_calib, y_test


def _fit_and_transform_scaler(X_train: np.ndarray, *others: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(X_train)

    transformed = [scaler.transform(arr).astype(np.float32) for arr in others]
    return transformed


### High-level dispatcher
def prepare_splits(
    mode: Mode,
    csv_path: str,
    main_frac: float,
    calib_frac: float,
    test_frac: float,
    val_frac: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepare data splits for BOTH:
      - Unsupervised VAE + conformal (only legit for training, legit-only calib)
      - Supervised baseline classifier + conformal (all classes, full calib)

    """

    # 1) Load
    X, y, n, fraud_rate = _load_credit_data(csv_path)
    print(f"Loaded: n={n}, fraud_rate={fraud_rate:.5f}")

    # 2) Global stratified split
    X_main, X_calib, X_test, y_main, y_calib, y_test = _global_stratified_split(
        X,
        y,
        main_frac=main_frac,
        calib_frac=calib_frac,
        test_frac=test_frac,
        random_state=random_state
    )

    print("\nGlobal splits:")
    print(f"  D_main : {len(X_main)} (fraud_rate={y_main.mean():.5f})")
    print(f"  D_calib: {len(X_calib)} (fraud_rate={y_calib.mean():.5f})")
    print(f"  D_test : {len(X_test)} (fraud_rate={y_test.mean():.5f})")
    
    result: Dict[str, Any] = {
        "mode": mode,
        "y_calib": y_calib,
        "y_test": y_test
    }

    if mode == "unsupervised":
        X_main_legit = X_main[y_main == 0]

        X_train_vae, X_val_vae = train_test_split(
            X_main_legit,
            test_size=val_frac,
            random_state=random_state
        )

        X_calib_legit = X_calib[y_calib == 0]

        X_train_vae_s, X_val_vae_s, X_calib_legit_s, X_test_s = _fit_and_transform_scaler(
            X_train_vae,
            X_train_vae,
            X_val_vae,
            X_calib_legit,
            X_test
        )

        result.update({
            "X_train_vae_s": X_train_vae_s,
            "X_val_vae_s": X_val_vae_s,
            "X_calib_legit_s": X_calib_legit_s,
            "X_test_s": X_test_s,
        })
        return result

    elif mode == "supervised":
        X_train_sup, X_val_sup, y_train_sup, y_val_sup = train_test_split(
            X_main,
            y_main,
            test_size=val_frac,
            stratify=y_main,
            random_state=random_state,
        )

        # separate scaler, fit on supervised train (all classes)
        X_train_sup_s, X_val_sup_s, X_calib_s, X_test_s = _fit_and_transform_scaler(
            X_train_sup,
            X_train_sup,
            X_val_sup,
            X_calib,
            X_test,
        )

        result.update({
            "X_train_sup_s": X_train_sup_s,
            "X_val_sup_s": X_val_sup_s,
            "X_calib_s": X_calib_s,
            "X_test_s": X_test_s,
            "y_train_sup": y_train_sup,
            "y_val_sup": y_val_sup,
        })
        return result

class FraudDataset(Dataset):
    """
    - X float32
    - y float32 with shape (n, 1)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loader(X, y, batch_size: int, shuffle: bool):
    ds = FraudDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_loader(X, y=None, batch_size=64, shuffle=True):
    X = torch.tensor(X, dtype=torch.float32)

    if y is None:
        # unsupervised (VAE)
        return DataLoader(
            X,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

    # supervised (baseline)
    y = torch.tensor(y, dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    dataset = list(zip(X, y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )