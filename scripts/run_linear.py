import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --- 关键：将项目根目录加入路径，以便导入 src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.hitrocket import HITRocketTransform
from src.utils import load_data

def run_experiment():
    # 配置
    DATA_DIR = os.path.join(project_root, 'data', 'sample_data') # 默认跑 sample，跑全量改这里
    RESULTS_DIR = os.path.join(project_root, 'results', 'linear_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 检查数据
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return

    datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"🚀 Start Linear Experiment on {len(datasets)} datasets...")

    # 模型参数 (参考你的 Notebook)
    dilations = [1, 2, 4, 8, 16, 32]
    bias_quantiles = [0.25, 0.5, 0.75]

    results = []

    for ds in datasets:
        print(f"\n➡️  Processing: {ds}")
        try:
            # 1. 加载数据
            X_train, y_train, X_test, y_test = load_data(DATA_DIR, ds)

            # 2. 特征提取 (HIT-Rocket)
            # 注意：需转为 Tensor 并增加通道维度 (N, 1, L)
            import torch
            X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

            # 使用 GPU 如果可用
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = HITRocketTransform(dilations, bias_quantiles, num_samples=5, hadamard_order=16)
            model.to(device)
            X_train_t, X_test_t = X_train_t.to(device), X_test_t.to(device)

            t_start = time.time()
            with torch.no_grad():
                feats_train = model(X_train_t).cpu().numpy()
                feats_test = model(X_test_t).cpu().numpy()
            transform_time = time.time() - t_start

            # 3. 标准化
            scaler = StandardScaler()
            feats_train = scaler.fit_transform(feats_train)
            feats_test = scaler.transform(feats_test)

            # 4. 分类器 (Ridge)
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf.fit(feats_train, y_train)
            preds = clf.predict(feats_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            print(f"   ✅ Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {transform_time:.3f}s")

            results.append({
                'Dataset': ds, 
                'Accuracy': acc, 
                'F1': f1, 
                'Time': transform_time
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 保存结果
    df_res = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'summary_linear.csv')
    df_res.to_csv(save_path, index=False)
    print(f"\n💾 Results saved to: {save_path}")

if __name__ == '__main__':
    run_experiment()
