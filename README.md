# 💳Credit Card Fraud Detection Using Deep Learning Models

## Project Overview

This project focuses on detecting fraudulent credit card transactions using deep learning techniques. It explores multiple neural network architectures and loss functions to address the highly imbalanced nature of credit card transaction data.

Four deep learning architectures are implemented and evaluated:

1. Multi-Layer Perceptron (MLP)
2. 1D Convolutional Neural Network (CNN)
3. Long Short-Term Memory Network (LSTM)
4. Autoencoder

The project also incorporates class-weighted Binary Cross-Entropy, Focal Loss, validation-based threshold optimization, and model explainability techniques.

## Dataset

**Dataset:** European Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset information:

- Total transactions: 284,807
- Genuine transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target column: `Class`

Target labels:
- `0` – Genuine transaction
- `1` – Fraudulent transaction

The dataset contains anonymized PCA-transformed features (V1–V28), along with Time and Amount.

**Note:** The dataset CSV is not included in this repository. Download `creditcard.csv` from Kaggle and upload it when prompted by the notebook.

## Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- LIME

## Deep Learning Models Implemented

### 1. Multi-Layer Perceptron (MLP)

A feedforward neural network used for binary classification of genuine and fraudulent transactions.

Experiments:
- Standard Binary Cross-Entropy (BCE)
- Class-Weighted BCE
- Focal Loss

### 2. 1D Convolutional Neural Network (CNN)

A one-dimensional convolutional neural network used to learn patterns from transaction feature vectors.

Experiments:
- Standard BCE
- Class-Weighted BCE
- Focal Loss

### 3. Long Short-Term Memory (LSTM)

An LSTM network was added as an additional deep learning architecture for comparison.

The 30 input features are reshaped into a feature sequence containing 30 steps, with one feature value per step.

**Important limitation:** The supplied dataset does not contain customer or card identifiers needed to construct genuine per-customer transaction histories. Therefore, this is an exploratory feature-sequence LSTM experiment, not a true temporal transaction-history model. The ordering of the anonymized PCA components does not have an established temporal meaning.

The LSTM was trained using standard Binary Cross-Entropy.

### 4. Autoencoder

An Autoencoder was trained using genuine transactions to learn their reconstruction patterns.

Transactions with higher reconstruction errors were treated as potential anomalies.

The reconstruction error acts as an anomaly score rather than a calibrated fraud probability.

## Methodology

The project follows these steps:

1. Dataset upload and loading
2. Dataset inspection and class imbalance analysis
3. Stratified training, validation and test splitting
4. Feature scaling using training data only
5. Calculation of class weights
6. Implementation of deep learning architectures
7. Training MLP and CNN using different loss functions
8. Training LSTM using standard BCE
9. Training Autoencoder on genuine transactions
10. Validation-based decision threshold optimization
11. Final evaluation on the untouched test set
12. Performance comparison and visualization
13. Model explainability analysis

### Dataset Splitting

The dataset was divided into:

- Training set: 60%
- Validation set: 20%
- Test set: 20%

A stratified split was used to preserve the class distribution. StandardScaler was fitted only on the training data and then applied to validation and test data.

## Experimental Results

The models were evaluated on a held-out test set containing 56,962 transactions:

- Genuine transactions: 56,863
- Fraudulent transactions: 99

### Model Performance Comparison

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP - BCE | 88.17% | 82.83% | 85.42% | 84.40% | 97.94% |
| MLP - Weighted BCE | 87.50% | 77.78% | 82.35% | 72.68% | 97.54% |
| MLP - Focal Loss | 87.78% | 79.80% | 83.60% | 84.02% | 97.81% |
| CNN - BCE | 84.38% | 81.82% | 83.08% | 80.67% | 96.80% |
| CNN - Weighted BCE | 69.09% | 76.77% | 72.73% | — | — |
| CNN - Focal Loss | 88.76% | 79.80% | 84.04% | 80.62% | 96.97% |
| LSTM - BCE | 85.26% | 81.82% | 83.51% | 73.65% | 92.86% |
| Autoencoder | 38.30% | 54.55% | 45.00% | — | — |

Note: The dashes indicate metrics not reproduced in the extracted results table. Refer to the executed notebook for the complete evaluation output.

### Best Performing Model

The MLP trained using standard Binary Cross-Entropy achieved the best overall F1-score and PR-AUC in this experiment.

**MLP - BCE:**

- Precision: 88.17%
- Recall: 82.83%
- F1-score: 85.42%
- PR-AUC: 84.40%
- ROC-AUC: 97.94%

### LSTM Results

The LSTM model achieved:

- Precision: 85.26%
- Recall: 81.82%
- F1-score: 83.51%
- PR-AUC: 73.65%
- ROC-AUC: 92.86%

The LSTM provides an additional architecture comparison, although its feature-sequence representation has the methodological limitation described above.

### Confusion Matrix – MLP with BCE

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,852 | 11 |
| Actual Fraud | 17 | 82 |

Interpretation:

- 82 fraudulent transactions were correctly detected.
- 17 fraudulent transactions were missed.
- 11 genuine transactions were incorrectly flagged as fraudulent.

## Experimental Observations

- MLP with standard BCE achieved the best overall F1-score and PR-AUC.
- CNN with Focal Loss achieved the highest precision among the evaluated configurations.
- LSTM achieved competitive precision, recall and F1-score.
- Class-weighted BCE did not consistently outperform standard BCE.
- The Autoencoder produced more false-positive predictions than the supervised classifiers.
- Validation-based threshold optimization was used instead of tuning thresholds on the test set.

These findings demonstrate that different deep learning architectures and loss functions behave differently under severe class imbalance.

## Model Explainability

SHAP and LIME are included as explainability techniques in the project workflow.

### SHAP

SHAP (SHapley Additive exPlanations) explains feature contributions to model predictions.

### LIME

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the model locally.

These are explainability techniques, not separate prediction models.

## Visualizations and Outputs

The executed notebook contains:

- Dataset class distribution graph
- Model training logs
- Training and validation loss curves
- Confusion matrices
- Model performance comparison graph
- LSTM training output
- Model evaluation reports
- Explainability analysis sections

## How to Run the Project

1. Clone or download this repository.
2. Open the Jupyter Notebook in Google Colab.
3. Download `creditcard.csv` from Kaggle.
4. Run the notebook cells in order.
5. Upload the CSV file when prompted.
6. Allow the deep learning models to train.
7. View the evaluation metrics, graphs and model comparison results.

## Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection_With_LSTM.ipynb
├── Credit_Card_Fraud_Detection_With_LSTM_Report.pdf
└── README.md
```

## Conclusion

This project demonstrates the implementation and comparison of four deep learning architectures for credit card fraud detection.

MLP, CNN, LSTM and Autoencoder models were evaluated using appropriate performance metrics, with particular attention to the highly imbalanced nature of the dataset.

Among the evaluated configurations, MLP with standard Binary Cross-Entropy achieved the best overall performance, with an F1-score of 85.42% and PR-AUC of 84.40%.

The project also explores imbalance-aware loss functions, threshold optimization and model explainability.
