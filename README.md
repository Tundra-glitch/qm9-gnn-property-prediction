# Machine Learning Surrogate Model for Quantum Chemistry using Graph Neural Networks

## 🧪 Overview

This project develops a Graph Neural Network (GNN) as a surrogate model for predicting quantum-chemical properties of molecules in the QM9 dataset.

The model learns a mapping from molecular graph structure to properties computed using Density Functional Theory (DFT), enabling fast approximation of expensive quantum simulations.

---

## ⚛️ Scientific Motivation

Quantum chemistry methods such as Density Functional Theory (DFT) provide accurate molecular properties but are computationally expensive.

This project explores how Graph Neural Networks can serve as **machine learning surrogate potentials**, enabling:

- Fast molecular property prediction
- Scalable quantum chemistry approximation
- Foundation for reaction pathway modeling and molecular dynamics

---

## 🧬 Dataset: QM9

The QM9 dataset contains ~134,000 small organic molecules with DFT-computed properties:

- Dipole moment
- HOMO / LUMO energies
- Electronic spatial extent
- Enthalpy, internal energy, free energy

Each molecule is represented as a graph:
- Nodes → atoms
- Edges → chemical bonds

---

## 🧠 Model Architecture

A Graph Convolutional Network (GCN) is used:

- 2-layer message passing neural network
- ReLU activations
- Global mean pooling
- Fully connected regression head

---

## ⚙️ Training Setup

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 1e-3
- Train set: 10,000 molecules
- Test set: 2,000 molecules

---

## 📊 Results

The model successfully learns molecular property trends:

- Stable convergence observed
- Generalization to unseen molecules demonstrated
- Baseline surrogate performance established

---

## 📈 Training Dynamics

Loss decreases smoothly over training epochs, indicating:
- Stable optimization
- Effective message passing representation learning

---

## 🧰 Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- QM9 dataset

---

## 🚀 How to Run

```bash
pip install torch torch-geometric rdkit
python train.py
