# 💳 Credit Card Fraud Detection Using Deep Learning

An end-to-end deep learning project for detecting fraudulent credit card transactions using multiple neural network architectures. This project implements and compares **Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Autoencoder** models to identify fraudulent transactions in highly imbalanced financial datasets.

The project also explores different loss functions, validation-based decision threshold optimization, model evaluation, and Explainable AI techniques.

## 📌 Project Overview

Credit card fraud detection is a challenging classification problem due to the extremely imbalanced nature of transaction datasets, where fraudulent transactions represent only a very small percentage of all transactions.

This project investigates how different deep learning architectures and loss functions perform in identifying fraudulent transactions while minimizing false positives and false negatives.

### Key Highlights

- Implementation of four deep learning architectures.
- Comparison of Binary Cross-Entropy, Weighted Binary Cross-Entropy, and Focal Loss.
- Class imbalance handling using class weights.
- Training, validation, and independent testing.
- Validation-based decision threshold optimization.
- Evaluation using Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.
- Confusion matrix visualization and model performance comparison.
- Explainable AI using SHAP and LIME.
- Genuine-transaction-based Autoencoder anomaly detection.

## 🧠 Deep Learning Architectures

The project implements and evaluates the following architectures:

| Model | Description |
|---|---|
| MLP | Fully connected neural network for binary classification. |
| 1D CNN | Extracts local patterns from transaction feature vectors. |
| LSTM | Evaluates recurrent neural network architecture on feature sequences. |
| Autoencoder | Learns genuine transaction representations and detects anomalies using reconstruction error. |

### Loss Function Comparison

To investigate the impact of class imbalance, multiple loss functions are evaluated.

**1. Binary Cross-Entropy (BCE)**

Used as the baseline classification loss.

**2. Weighted Binary Cross-Entropy**

Assigns higher importance to fraudulent transactions through class weights.

**3. Focal Loss**

Focuses training on difficult-to-classify examples by reducing the contribution of well-classified examples.

The MLP and CNN architectures are evaluated using all three loss functions, while the LSTM is trained using standard Binary Cross-Entropy.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

- Total transactions: 284,807
- Genuine transactions: 284,315
- Fraudulent transactions: 492
- Total features: 30
- Target column: `Class`

### Dataset Features

- `Time`: Time elapsed between the first transaction and the current transaction.
- `Amount`: Transaction amount.
- `V1`–`V28`: Anonymized features obtained through Principal Component Analysis (PCA).
- `Class`: Transaction label.
  - `0` – Genuine transaction
  - `1` – Fraudulent transaction

**Class Distribution:**

- Genuine transactions: 99.8273%
- Fraudulent transactions: 0.1727%

The dataset demonstrates the significant class imbalance involved in real-world fraud detection problems.

## ⚙️ Methodology

The project follows the following workflow:

1. **Data Loading:** Load and inspect the credit card transaction dataset.
2. **Exploratory Data Analysis:** Examine missing values, duplicate records, and class distribution.
3. **Data Splitting:** Divide the dataset into stratified training, validation, and testing subsets.
4. **Feature Scaling:** Apply StandardScaler to the `Time` and `Amount` features using training data only.
5. **Class Imbalance Handling:** Calculate balanced class weights for fraudulent and genuine transactions.
6. **Model Development:** Build and train MLP, CNN, LSTM, and Autoencoder architectures.
7. **Model Training:** Train classification models using different loss functions.
8. **Threshold Optimization:** Select classification thresholds using validation predictions.
9. **Model Evaluation:** Evaluate all models on the untouched test dataset.
10. **Explainability:** Apply SHAP and LIME to understand model predictions.

### Dataset Splitting

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

Stratified splitting is used to preserve the fraud-to-genuine transaction ratio across all subsets.

## 🔍 Model Evaluation

The models are evaluated using metrics suitable for highly imbalanced classification problems.

| Metric | Purpose |
|---|---|
| Precision | Measures how many predicted fraudulent transactions are actually fraudulent. |
| Recall | Measures how many actual fraudulent transactions are correctly identified. |
| F1-Score | Harmonic mean of Precision and Recall. |
| PR-AUC | Measures performance across precision-recall thresholds. |
| ROC-AUC | Measures the ability to distinguish between genuine and fraudulent transactions. |
| Confusion Matrix | Shows true positives, true negatives, false positives, and false negatives. |

PR-AUC, Recall, and F1-Score receive particular attention because accuracy alone can be misleading for highly imbalanced datasets.

## 🧩 Explainable AI

To improve model interpretability, the project incorporates:

- **SHAP (SHapley Additive exPlanations):** Helps understand feature contributions to model predictions.
- **LIME (Local Interpretable Model-Agnostic Explanations):** Explains individual transaction predictions through locally interpretable approximations.

These techniques help provide insights into why a transaction may be classified as fraudulent.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Explainable AI | SHAP, LIME |
| Development Environment | Google Colab |

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection_With_LSTM.ipynb
│
├── README.md
│
└── creditcard.csv                 # Dataset (download separately)
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow shap lime
```

### 4. Download the Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

[Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 5. Run the Notebook

Open the notebook using Google Colab:

1. Upload `Credit_Card_Fraud_Detection_With_LSTM.ipynb`.
2. Run the notebook cells sequentially.
3. Upload the downloaded `creditcard.csv` file when prompted.
4. Execute the training, evaluation, visualization, and explainability sections.

**Google Colab:** [Open Google Colab](https://colab.research.google.com/)

## ⚠️ Important Methodological Limitation

The supplied dataset does not contain customer or card identifiers required to construct genuine per-customer transaction histories.

Therefore, the LSTM model treats the 30 input features as a sequence of 30 steps, with one feature value per step.

This is an exploratory LSTM architecture comparison rather than a true temporal transaction prediction model. Additionally, the anonymized PCA components (`V1`–`V28`) do not have an established temporal or spatial ordering.

LSTM results should therefore be interpreted with this limitation in mind.

## 🔮 Future Enhancements

- Experiment with SMOTE, ADASYN, and Random Undersampling.
- Explore Random Search and Bayesian Optimization for hyperparameter tuning.
- Implement stratified k-fold cross-validation.
- Investigate Transformer-based architectures.
- Develop genuine sequential transaction models using customer-level transaction histories.
- Build a real-time fraud detection application with an interactive dashboard.
- Explore ensemble learning techniques to improve fraud detection performance.

## 🎯 Project Objective

The primary objective is to investigate and compare different deep learning architectures and training strategies for credit card fraud detection, particularly under extreme class imbalance.

By combining neural network architectures, loss function comparisons, threshold optimization, and explainable AI, this project aims to provide a comprehensive experimental framework for understanding deep learning-based fraud detection.
