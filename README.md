# Privacy-Preserving Federated Vision Transformers for Pneumonia Detection on Chest X-Rays

This repository contains an academic implementation of a **Privacy-Preserving Federated Learning framework** using **Vision Transformers (ViT)** for pneumonia classification from chest X-ray images.  
The system is evaluated under **non-identically distributed (non-IID) client data** to reflect realistic clinical data heterogeneity.

In addition to standard federated learning, **Differentially Private Federated Learning (DP-FL)** is implemented in selected experiments using **DP-SGD with explicit privacy budgets (ε)** to provide formal client-level privacy guarantees.

The complete implementation is provided as a single, well-documented Jupyter notebook developed and executed in Google Colab.

---

## Project Overview

Medical imaging datasets such as chest X-rays contain sensitive patient information, making centralized data aggregation challenging due to privacy, security, and regulatory constraints. Federated Learning (FL) enables collaborative model training across multiple institutions while keeping raw patient data local.

In this project, a Vision Transformer-based classifier is trained under three progressively constrained settings:
1. Centralized training (baseline)
2. Federated learning with non-IID client data
3. Differentially Private Federated Learning (DP-FL) with explicit privacy budgets

The objective is to analyze **performance, robustness, and privacy–utility trade-offs** when applying ViT models in decentralized and privacy-preserving medical AI pipelines.

---

## Key Features

- Vision Transformer (ViT-B/16) for chest X-ray classification  
- Federated learning with simulated multi-client setup  
- Non-IID data partitioning to mimic real hospital data distributions  
- Differentially Private Federated Learning using **DP-SGD (Opacus)**  
- Explicit privacy budget (ε) computation using a **Rényi Differential Privacy accountant**  
- Quantitative analysis of privacy–utility trade-offs  
- Fully reproducible end-to-end pipeline in a single notebook  

---

## Dataset

A publicly available chest X-ray dataset is used for binary classification between **Pneumonia** and **Normal** cases.

> **Note:**  
> The dataset is **not included** in this repository due to size and licensing constraints.  
> Users must obtain the dataset separately from its official public source and update dataset paths accordingly.

---

## Methodology

### Vision Transformer Model
- ViT-B/16 architecture with patch-based tokenization  
- Pretrained backbone adapted for binary pneumonia classification  
- Global self-attention enables modeling of long-range spatial dependencies in lung regions  

### Federated Learning Setup
- Multiple clients are simulated, each holding a local dataset  
- Client data distributions are intentionally **non-IID**  
- Local models are trained independently and aggregated using **FedAvg**  
- Raw medical images never leave the client  

### Differential Privacy (Selected Experiments)
- Differential Privacy is implemented using **DP-SGD via the Opacus framework**
- Per-sample gradient clipping with fixed ℓ2 norm
- Gaussian noise injection calibrated to target privacy budgets
- Explicit privacy budgets are computed:
  - ε = 8.0 (moderate privacy)
  - ε = 3.0 and ε = 1.0 (strong privacy)
- Privacy accounting is performed using a **Rényi Differential Privacy (RDP) accountant**

These mechanisms provide **formal client-level differential privacy guarantees** for the DP-FL experiments under the stated assumptions.

---

## Evaluation

Model performance is evaluated on a held-out test set using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Special emphasis is placed on **recall for the pneumonia class**, as false negatives represent missed clinical cases. Performance is compared across centralized, federated, and DP-FL settings to highlight privacy–utility trade-offs.

---

## Notebook Structure

The notebook is organized into the following sections:

1. Project Overview  
2. Environment Setup and Dependencies  
3. Dataset Description and Preprocessing  
4. Non-IID Client Data Partitioning  
5. Vision Transformer Model Architecture  
6. Federated Learning Framework  
7. Federated Training Procedure  
8. Model Evaluation Strategy  
9. Experimental Results and Observations  
10. Limitations and Future Directions  
11. Privacy Considerations  
12. Disclaimer  

---

## How to Run

1. Open the notebook in **Google Colab** or a local Jupyter environment  
2. Install required dependencies as specified in the notebook  
3. Download the dataset from its public source and update the dataset path  
4. Run all cells sequentially from top to bottom  

---

## Limitations

- Federated learning is simulated rather than deployed across real hospital networks  
- Privacy guarantees apply only to DP-FL experiments with explicit ε computation  
- The number of clients and communication rounds is limited by computational resources  
- Evaluation is performed on a single pediatric chest X-ray dataset  

---

## Disclaimer

This project is intended solely for **academic and educational purposes**.  
It is **not** a clinical decision-support system and must **not** be used for medical diagnosis or treatment.

---

## License

This repository is shared for research and educational use only.  
Please review dataset licenses separately before use.
