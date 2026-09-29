# FAME: Formal Abstract Minimal Explanation for Neural Networks


Official implementation of **FAME (Formal Abstract Minimal Explanations)**, a novel class of abductive explanations grounded in **abstract interpretation**. 

FAME is the first method to scale formal, provably correct explanations to large neural networks (including ResNet architectures on CIFAR-10) by eliminating the sequential "traversal order" bottleneck of prior SAT/SMT-based approaches.



## 🛠️ Installation

This project uses `pyproject.toml` (Setuptools) for dependency management.

### Prerequisites
* **Python:** >= 3.9
* **Deep Learning Frameworks:** 
    * PyTorch >= 2.3.1
    * Keras >= 3.11.3
* **Abstract Interpretation Tools:** 
    * Decomon (refactor branch)
* **Optimization:** CVXPY

### Setup

#### Install the package and all dependencies
pip install --editable .

-----

#### 📖 Usage

To reproduce the experiments and benchmarks presented in the paper, navigate to the notebooks/ directory.

Running Benchmarks
- cd notebooks
- for cifar10 run cifar10_cnn.py
- for gtsrb run gtsrb_fc.py