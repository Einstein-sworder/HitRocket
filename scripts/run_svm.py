import os
import sys
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.hitrocket import HITRocketTransform
from src.utils import load_data

def run_svm_experiment():
    DATA_DIR = os.path.join(project_root, 'data', 'sample_data')
    RESULTS_DIR = os.path.join(project_root, 'results', 'svm_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        return

    datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    results = []

    print(f"🚀 Start SVM Experiment...")

    for ds in datasets:
        print(f"Processing: {ds}")
        try:
            X_train, y_train, X_test, y_test = load_data(DATA_DIR, ds)

            X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

            model = HITRocketTransform(dilations=[1,2,4,8,16], bias_quantiles=[0.25, 0.5, 0.75])

            with torch.no_grad():
                feats_train = model(X_train_t).numpy()
                feats_test = model(X_test_t).numpy()

            scaler = StandardScaler()
            feats_train = scaler.fit_transform(feats_train)
            feats_test = scaler.transform(feats_test)

            # 使用 LinearSVC (比 SVC(kernel='linear') 快很多)
            clf = LinearSVC(dual=False, max_iter=1000)
            clf.fit(feats_train, y_train)
            acc = accuracy_score(y_test, clf.predict(feats_test))

            print(f"   Accuracy: {acc:.4f}")
            results.append({'Dataset': ds, 'SVM_Acc': acc})

        except Exception as e:
            print(f"   Error: {e}")

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'summary_svm.csv'), index=False)

if __name__ == '__main__':
    run_svm_experiment()
