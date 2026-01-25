import os
import pandas as pd
import numpy as np

def load_data(root_path, dataset_name):
    """
    加载 UCR 数据集 (Train & Test)
    """
    train_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.tsv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found for {dataset_name} in {root_path}")

    df_train = pd.read_csv(train_path, sep='\t', header=None)
    df_test = pd.read_csv(test_path, sep='\t', header=None)

    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values

    # 简单的 NaN 补 0 处理 (和你 Notebook 里的一致)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    return X_train, y_train, X_test, y_test
