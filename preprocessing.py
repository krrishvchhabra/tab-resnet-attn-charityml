import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

NUMERICAL_COLS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_COLS = [
    'workclass', 'education_level', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 'native-country'
]
TARGET_COL = 'income'

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]

def load_and_preprocess_data(data_path="data/census.csv", batch_size=256, test_size=0.2, val_size=0.1, random_state=42):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Place census.csv in data/")

    data = pd.read_csv(data_path)

    # 1. Log-transform skewed continuous features
    for col in ['capital-gain', 'capital-loss']:
        data[col] = np.log1p(data[col])

    # 2. Split into features and label
    y = data[TARGET_COL].apply(lambda x: 1 if x.strip() == '>50K' else 0).values

    # 3. Categorical encoding (mapping to integer indices for PyTorch Embedding layers)
    cat_dims = []
    x_cat = np.zeros((len(data), len(CATEGORICAL_COLS)), dtype=np.int64)
    for i, col in enumerate(CATEGORICAL_COLS):
        data[col] = data[col].astype('category')
        x_cat[:, i] = data[col].cat.codes.values
        cat_dims.append(len(data[col].cat.categories))

    # 4. Numerical scaling
    scaler = MinMaxScaler()
    x_num = scaler.fit_transform(data[NUMERICAL_COLS].values)

    # 5. Train/Val/Test Split (70% train, 10% val, 20% test)
    # First split off test set
    x_num_tv, x_num_test, x_cat_tv, x_cat_test, y_tv, y_test = train_test_split(
        x_num, x_cat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then split train into train & val
    val_ratio = val_size / (1.0 - test_size)
    x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(
        x_num_tv, x_cat_tv, y_tv, test_size=val_ratio, random_state=random_state, stratify=y_tv
    )

    # 6. PyTorch DataLoaders
    train_dataset = TabularDataset(x_num_train, x_cat_train, y_train)
    val_dataset = TabularDataset(x_num_val, x_cat_val, y_val)
    test_dataset = TabularDataset(x_num_test, x_cat_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        'num_features_dim': len(NUMERICAL_COLS),
        'cat_dims': cat_dims,
        'cat_cols': CATEGORICAL_COLS,
        'num_cols': NUMERICAL_COLS,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset)
    }

    return train_loader, val_loader, test_loader, metadata