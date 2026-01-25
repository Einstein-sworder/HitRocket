import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard
from scipy.fftpack import dct

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.utils import load_data

# 对比核心类 (直接内嵌在此，因为它只用于对比实验)
class KernelComparison:
    def __init__(self, kernel_len=16, num_kernels=16):
        self.kernel_len = kernel_len
        self.num_kernels = num_kernels

    def get_kernels(self, method):
        if method == 'Gaussian Random':
            return torch.randn(self.num_kernels, 1, self.kernel_len)
        elif method == 'MiniROCKET':
            k = np.random.choice([-1, 2], size=(self.num_kernels, 1, self.kernel_len))
            return torch.FloatTensor(k)
        elif method == 'Hadamard':
            h = hadamard(self.kernel_len)
            return torch.FloatTensor(h[:self.num_kernels]).unsqueeze(1)
        # 简单演示，其他方法可按需添加
        return torch.randn(self.num_kernels, 1, self.kernel_len)

    def run(self, X):
        X_t = torch.FloatTensor(X).unsqueeze(1)
        results = {}
        methods = ['Gaussian Random', 'MiniROCKET', 'Hadamard']

        for m in methods:
            w = self.get_kernels(m)
            t0 = time.time()
            # 简单卷积模拟
            _ = F.conv1d(X_t, w, padding=0)
            t_cost = (time.time() - t0) * 1000 # ms
            results[m] = t_cost

        return results

def main():
    DATA_DIR = os.path.join(project_root, 'data', 'sample_data')
    FIG_DIR = os.path.join(project_root, 'results', 'figures')
    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR): return

    datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    print("🚀 Running Kernel Comparison Benchmark...")
    all_res = []

    comparator = KernelComparison()

    for ds in datasets[:3]: # 演示只跑前3个
        X_train, _, _, _ = load_data(DATA_DIR, ds)
        res = comparator.run(X_train)
        res['Dataset'] = ds
        all_res.append(res)
        print(f"   Processed {ds}")

    df = pd.DataFrame(all_res)
    print(df)

    # 简单绘图
    if not df.empty:
        df.set_index('Dataset').plot(kind='bar')
        plt.title("Kernel Transformation Time Comparison")
        plt.ylabel("Time (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'comparison_plot.png'))
        print(f"\n📊 Plot saved to {FIG_DIR}/comparison_plot.png")

if __name__ == '__main__':
    main()
