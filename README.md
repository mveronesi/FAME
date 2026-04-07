

# FAME: Formal Abstract Minimal Explanation for Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10661-b31b1b.svg)](https://arxiv.org/abs/2603.10661)
[![Venue: ICLR 2026](https://img.shields.io/badge/Venue-ICLR%202026-blue.svg)](https://openreview.net/forum?id=VJkNqJJAhV)

Official implementation of **FAME (Formal Abstract Minimal Explanations)**, a novel class of abductive explanations grounded in **abstract interpretation**. 

FAME is the first method to scale formal, provably correct explanations to large neural networks (including ResNet architectures on CIFAR-10) by eliminating the sequential "traversal order" bottleneck of prior SAT/SMT-based approaches.

---

## 🚀 Key Contributions

* **Scalable Formal XAI:** Leverages **LiRPA (Linear Relaxation based Perturbation Analysis)** to handle high-dimensional models where exact solvers fail.
* **Batch Freeing Mechanism:** A recursive refinement procedure that discards multiple irrelevant features simultaneously, formulated as an efficient knapsack-style optimization.
* **Provable Quality Guarantees:** Includes a procedure to measure the "worst-case gap" between abstract minimal explanations and true minimal abductive explanations.
* **GPU Accelerated:** Implemented in PyTorch, allowing formal reasoning to benefit from modern hardware acceleration.

---

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
```bash
# Clone the repository
git clone https://github.com/ducoffeM/FAME.git
cd FAME

# Install the package and all dependencies
pip install .

# Optional: Install development dependencies (pytest, black, tox)
pip install ".[dev]"
````
-----

## 📖 Usage

To reproduce the experiments and benchmarks presented in the paper, navigate to the notebooks/ directory. Experiments are categorized by model type using the following naming convention: FAME_benchmark_{model_name}.ipynb.

Running Benchmarks

1\. Start your Jupyter environment:

```bash
jupyter notebook
```
2\. Open the notebook corresponding to the experiment you wish to replicate:

- CIFAR-10 (CNN/ResNet): notebooks/FAME_CIFAR10_cnn.ipynb

- MNIST (MLP): notebooks/FAME_MNIST_10x2.ipynb

3\. Follow the internal cells to load the pre-trained weights and execute the FAME explanation generation.

-----

## 📑 Citation

If you use this code or our method in your research, please cite our ICLR 2026 paper:

```bibtex
@article{boumazouza2026fame,
  title={FAME: Formal Abstract Minimal Explanation for Neural Networks},
  author={Boumazouza, Ryma and Elsaleh, Raya and Ducoffe, Melanie and Bassan, Shahaf and Katz, Guy},
  journal={arXiv preprint arXiv:2603.10661},
  year={2026}
}
```

-----

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/MIT) file for details.

## ✉️ Contact

For questions regarding the paper or code, please contact:

  * **Ryma Boumazouza** 
  * **Mélanie Ducoffe** 


