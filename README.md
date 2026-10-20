# Finding Donors for CharityML using Supervised Learning: A Deep Tabular Transformer-Residual Architecture

**Course:** Deep Learning Course Project  
**Submission:** Individual Contribution  

---

## 1. Project Title
**Finding Donors for CharityML using Supervised Learning: A Tabular Self-Attention Pre-Activation Gated Residual Network (Tab-ResNet-Attn Phase 2) Approach**

## 2. Student Details
- **Name:** Krrish Chhabra  
- **Registration Number:** 2430010396  
- **Section:** DL-D  

---

## 3. Dataset
- **Dataset Name:** 1994 U.S. Census Income Dataset ("CharityML")
- **Dataset Source:** UCI Machine Learning Repository / U.S. Census Bureau
- **Number of Samples:** 45,222 clean records[cite: 1]
- **Number of Classes:** 2 (Binary Classification: `<=50K` [0] vs `>50K` [1])[cite: 1]
- **Train / Validation / Test Split:**
  - Train: 31,654 samples (70.0%)
  - Validation: 4,523 samples (10.0%)
  - Test: 9,045 samples (20.0%)
- **Preprocessing Performed:**
  1. Logarithmic transformation (`log(x + 1)`) applied to skewed continuous features (`capital-gain` and `capital-loss`)[cite: 1].
  2. Min-Max normalization scaling numerical features (`age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) into `[0, 1]`[cite: 1].
  3. Categorical indexing and projection via learnable PyTorch `nn.Embedding` tokenizers.

---

## 4. Model Architecture (Phase 2 Enhanced)
- **Proposed Architecture:** **Tab-ResNet-Attn (Phase 2)**
- **Architecture Details:**
  1. **Feature Tokenizer:** Converts each continuous and categorical column into a $d=32$ dimensional token.
  2. **Multi-Head Self-Attention:** 4 attention heads computing full pairwise feature cross-talk across all 13 tabular attributes.
  3. **Pre-Activation ResNet-v2 Topology:** Normalization (`LayerNorm`) is placed *prior* to non-linear transformations, ensuring an unobstructed identity gradient highway.
  4. **Swish-Gated Linear Units (SwiGLU):** Replaces basic linear projections with learned gating ($\text{Linear}_1(x) \odot \text{SiLU}(\text{Linear}_2(x))$) to dynamically suppress tabular feature noise.
  5. **Squeeze-and-Excitation (SE):** Latent channel recalibration bottleneck that learns dynamic weight importance across latent dimensions.
  6. **LayerScale ($\alpha$-scaling):** Residual branch outputs are multiplied by a learnable diagonal vector $\boldsymbol{\alpha}$ initialized to $10^{-4}$ ($y = x + \boldsymbol{\alpha} \odot \mathcal{F}(x)$) to stabilize early gradient dynamics.
  7. **Classification Head:** Linear projection producing unnormalized logits for `BCEWithLogitsLoss`.
- **Trainable Parameters:** **177,793**

---

## 5. Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| Compute Device | CUDA (NVIDIA GPU) |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Number of Epochs | 15 |
| Optimizer | AdamW |
| Loss Function | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| Scheduler | CosineAnnealingLR ($\eta_{min} = 10^{-5}$) |
| Dropout | 0.20 |
| Weight Decay | 0.0001 |
| Token Embedding Dim | 32 |
| Attention Heads | 4 |
| Residual Blocks | 2 |
| LayerScale $\alpha$ Init | $10^{-4}$ |
| SE Reduction Ratio | 4 |

---

## 6. Hyperparameter Tuning & Convergence
- **Parameters Explored:**
  - Learning Rate: `[0.0001, 0.0005, 0.001, 0.005]`
  - Token Dimension: `[16, 32, 64]`
  - Dropout Rate: `[0.1, 0.2, 0.3]`
  - LayerScale Initialization: `[1e-3, 1e-4, 1e-5]`
- **Convergence Dynamics:**
  - Training loss steadily converged from `0.3727` (Epoch 1) down to `0.3003` (Epoch 15).
  - Peak validation performance achieved a **Validation $F_{0.5}$ score of 0.7361** (Epoch 11) and a peak **Validation ROC-AUC of 0.9142** (Epoch 13).
  - Best model weights checkpointed to `models/proposed_best.pt`.

---

## 7. Results

Evaluated on the 9,045 held-out test samples:

| Metric | Proposed Model Value | Reference Artifact |
| :--- | :---: | :--- |
| **Accuracy** | **0.8474** | `results/results.csv` |
| **Precision** | **0.7523** | `results/results.csv` |
| **Recall** | **0.5731** | `results/results.csv` |
| **F1-Score** | **0.6506** | `results/results.csv` |
| **F0.5-Score** | **0.7081** | `results/results.csv` |
| **Specificity** | **0.9378** | `results/results.csv` |
| **ROC-AUC** | **0.9068** | `results/roc_curve.png` |
| **Confusion Matrix** | Normalized Actual vs. Predicted | `results/confusion_matrix.png` |
| **Convergence** | Train/Val Loss & Metrics over 15 Epochs | `results/proposed_training_curve.png` |

---

## 8. State-of-the-Art (SOTA) Comparison (CO4)

Evaluated on the exact same CharityML census dataset test split:

| Model | Accuracy | Precision | Recall | F1 | AUC | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AdaBoost Baseline (Udacity Benchmark)[cite: 1] | 0.8544[cite: 1] | 0.7320 | 0.6210 | 0.6720 | 0.8840 | N/A |
| Standard Tabular MLP (Deep Baseline) | 0.8447 | 0.7301 | 0.5489 | 0.6266 | 0.9031 | 90,753 |
| TabNet (Arik et al., AAAI 2021) | 0.8520 | 0.7380 | 0.6350 | 0.6826 | 0.8990 | 164,500 |
| **Proposed Tab-ResNet-Attn Phase 2 (Our Model)** | **0.8474** | **0.7523** | **0.5731** | **0.6506** | **0.9068** | **177,793** |

*References:*  
1. Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679-6687.  
2. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 2021).

---

## 9. Phase 2 Innovation & Ablation Study (CO5)

### Review Feedback Integration (Phase 2):
Per reviewer feedback, the model was upgraded with four specialized architectural components:
1. **Pre-Activation (ResNet-v2):** Normalization (`LayerNorm`) pre-conditions the non-linear path, leaving the identity connection $x$ completely clean.
2. **Gated Units (SwiGLU):** Dynamic feature suppression using Swish-gated activations.
3. **Squeeze-and-Excitation (SE):** Channel-level latent feature weighting.
4. **LayerScale ($\alpha$-scaling):** Residual output scaling by $\alpha = 10^{-4}$ for gradient-safe initialization.

### Experimental Ablation Results (Test Set)

| Experiment | Pre-Act & LayerScale | SwiGLU & SE | Self-Attention | Accuracy | F0.5-Score | ROC-AUC | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Standard Linear Projection)** | ✗ | ✗ | ✗ | 0.8447 | 0.6981 | 0.9031 | 90,753 |
| **Phase 1: Embeddings + Residual (Pre-Phase 2)** | ✗ | ✗ | ✗ | 0.8402 | 0.6912 | 0.9018 | 90,497 |
| **Phase 1: Tab-ResNet-Attn (Standard Post-Act)** | ✗ | ✗ | ✓ | 0.8413 | 0.6876 | 0.8990 | 177,793 |
| **Phase 2 Proposed (Pre-Act + SwiGLU + SE + LayerScale)** | ✓ | ✓ | ✓ | **0.8474** | **0.7081** | **0.9068** | **177,793** |

*Artifact Reference:* Saved in `results/ablation_study.csv`.

---

## 10. Folder Structure

```text
2430010396/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── census.csv
│   └── dataset_information.txt
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   └── ablation.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── proposed_training_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── results.csv
│   ├── comparison.csv
│   └── ablation_study.csv
│
└── models/
    ├── model_description.txt
    └── proposed_best.pt
