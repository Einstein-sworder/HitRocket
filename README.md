# HIT-Rocket: High-Performance Time Series Classification



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)**HIT-Rocket** is a highly efficient feature extraction method for Time Series Classification (TSC). It leverages randomized convolution with **Hadamard Transform** and GPU acceleration to achieve state-of-the-art accuracy with significantly faster training times compared to existing kernel-based methods (e.g., MiniROCKET).



This repository contains the official implementation of the paper **"HIT-Rocket: Efficient Time Series Classification using Hadamard Transform"**.



 🚀 Quick Start
1. Requirements

Ensure you have Python installed. Install the necessary dependencies:```bash

pip install -r requirements.txt

Key dependencies: numpy, pandas, scikit-learn, torch.



2. Dataset Setup

We provide a script to automatically setup sample datasets (e.g., GunPoint, Coffee) for quick testing.



Bash



python scripts/setup_data.py

Note: To reproduce the full experiments, please download the UCRArchive_2018 and extract it to data/UCRArchive_2018/.

3. Run Experiments

✅ Main Linear Classifier (Recommended)

This runs the core HIT-Rocket algorithm with a Ridge Classifier (linear). This is the primary version used in the paper.



Bash



python scripts/run_linear.py

Expected Output:



Plaintext



🚀 Start Linear Experiment...

➡️  Processing: Coffee

   ✅ Accuracy: 1.0000 | F1: 1.0000 | Time: 0.015s

➡️  Processing: GunPoint

   ✅ Accuracy: 0.9933 | F1: 0.9933 | Time: 0.021s

⚡ SVM Classifier

To run the experiments using a Linear SVM classifier:



Bash



python scripts/run_svm.py

📊 Kernel Comparison Benchmark

To compare the transformation speed of HIT-Rocket (Hadamard) vs. Gaussian, MiniROCKET, and DCT:



Bash



python scripts/run_comparison.py

Results will be saved in results/figures/.

📂 Project Structure

Plaintext



HIT-Rocket/

├── data/                   # Dataset storage

│   ├── sample_data/        # Small datasets for testing

│   └── UCRArchive_2018/    # Full UCR Archive (User provided)

│

├── src/                    # Core Source Code

│   ├── hitrocket.py        # HITRocketTransform (Algorithm Implementation)

│   └── utils.py            # Data loading utilities

│

├── scripts/                # Execution Scripts

│   ├── run_linear.py       # Main experiment script

│   ├── run_svm.py          # SVM experiment script

│   └── run_comparison.py   # Kernel speed benchmark

│

├── notebooks/              # Jupyter Notebooks & Tutorials

│   └── QuickStart.ipynb    # Beginner tutorial

│

└── results/                # Experiment Outputs

📝 Citation

If you find this code or method useful for your research, please cite our ArXiv preprint.(A journal version is forthcoming; please check back for the updated citation.)



Paper Link: https://arxiv.org/abs/2511.01572



@article{hitrocket2025,

  title={HIT-Rocket: Efficient Time Series Classification using Hadamard Transform},

  author={Tan, Chenxing and Authors},

  journal={arXiv preprint arXiv:2511.01572},

  year={2025},

  url={[https://arxiv.org/abs/2511.01572](https://arxiv.org/abs/2511.01572)}

}

