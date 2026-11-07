# Credit Card Fraud Detection Using Deep Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using deep learning techniques. It explores different neural network architectures and loss functions to address the highly imbalanced nature of credit card transaction data.

The project implements and evaluates three deep learning architectures:

- Multi-Layer Perceptron (MLP)
- 1D Convolutional Neural Network (CNN)
- Autoencoder

It also investigates different loss functions and validation-based decision threshold optimization to improve fraud detection performance.

## Dataset

**Dataset:** European Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset details:

- Total transactions: 284,807
- Genuine transactions: 284,315
- Fraudulent transactions: 492
- Total input features: 30
- Target column: `Class`

The `Class` column represents:
- `0` – Genuine transaction
- `1` – Fraudulent transaction

The dataset contains anonymized PCA-transformed features (V1–V28), along with Time and Amount.

**Note:** The dataset CSV file is not included in this repository. Download `creditcard.csv` from Kaggle and upload it when prompted by the notebook.

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

Three MLP configurations were evaluated:
- Standard Binary Cross-Entropy (BCE)
- Class-Weighted Binary Cross-Entropy
- Focal Loss

### 2. 1D Convolutional Neural Network (CNN)

A 1D CNN architecture was implemented to learn patterns from the transaction feature vectors.

Three CNN configurations were evaluated:
- Standard Binary Cross-Entropy (BCE)
- Class-Weighted Binary Cross-Entropy
- Focal Loss

### 3. Autoencoder

An Autoencoder was trained using genuine transactions to learn their reconstruction patterns.

Transactions with higher reconstruction errors were treated as potential anomalies.

## Methodology

The project follows these steps:

1. Dataset upload and loading
2. Dataset inspection and class imbalance analysis
3. Stratified training, validation and test splitting
4. Feature scaling using training data only
5. Calculation of class weights
6. Implementation of deep learning architectures
7. Training MLP and CNN with different loss functions
8. Training the Autoencoder on genuine transactions
9. Decision threshold optimization using validation predictions
10. Final evaluation on the untouched test set
11. Performance comparison and visualization
12. Model explainability using SHAP and LIME

The dataset was divided into:
- Training set: 60%
- Validation set: 20%
- Test set: 20%

The random seed was set to 42 for reproducibility.

## Experimental Results

All models were evaluated on the held-out test set containing 56,962 transactions, including 99 fraudulent transactions.

### Model Performance Comparison

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP - BCE | 88.17% | 82.83% | 85.42% | 84.40% | 97.94% |
| MLP - Weighted BCE | 87.50% | 77.78% | 82.35% | 72.68% | 97.54% |
| MLP - Focal Loss | 87.78% | 79.80% | 83.60% | 84.02% | 97.81% |
| CNN - BCE | 86.81% | 79.80% | 83.16% | 79.96% | 96.75% |
| CNN - Weighted BCE | 65.71% | 69.70% | 67.65% | 64.74% | 95.50% |
| CNN - Focal Loss | 88.37% | 76.77% | 82.16% | 79.64% | 97.08% |
| Autoencoder | 43.67% | 69.70% | 53.70% | 48.69% | 96.18% |

### Best Performing Model

The MLP trained using standard Binary Cross-Entropy achieved the best overall F1-score and PR-AUC in this experiment.

**MLP - BCE results:**

- Precision: 88.17%
- Recall: 82.83%
- F1-score: 85.42%
- PR-AUC: 84.40%
- ROC-AUC: 97.94%

### Confusion Matrix – MLP with BCE

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,852 | 11 |
| Actual Fraud | 17 | 82 |

Interpretation:

- 82 fraudulent transactions were correctly detected.
- 17 fraudulent transactions were missed.
- 11 genuine transactions were incorrectly flagged as fraudulent.

### Experimental Observations

- MLP with standard BCE achieved the best overall F1-score and PR-AUC.
- MLP with Focal Loss also achieved competitive performance.
- CNN with Focal Loss achieved the highest precision among the evaluated models.
- Class-weighted BCE did not outperform standard BCE in these experimental settings.
- The Autoencoder detected anomalies but produced more false-positive predictions.
- Validation-based threshold optimization was used instead of selecting thresholds using the test set.

These results show that the tested imbalance-aware loss functions did not consistently improve performance over the standard BCE baseline.

## Model Explainability

SHAP and LIME were used to interpret model predictions.

### SHAP

SHAP (SHapley Additive exPlanations) was used to examine feature contributions to the predictions of the best-performing MLP model.

The MLP model selected for SHAP analysis was chosen based on validation PR-AUC.

### LIME

LIME (Local Interpretable Model-agnostic Explanations) was used to explain an individual transaction prediction by examining the influence of its features.

These techniques provide model interpretability and are not separate prediction models.

## Visualizations and Outputs

The executed notebook includes:

- Dataset class distribution graph
- Model training outputs
- Training and validation loss curves
- Confusion matrices for all evaluated models
- Model performance comparison graph
- SHAP summary plot
- LIME individual prediction explanation

The notebook contains the code and corresponding executed outputs.

## How to Run the Project

1. Clone or download this GitHub repository.
2. Open the `.ipynb` notebook using Google Colab.
3. Download the `creditcard.csv` dataset from Kaggle.
4. Run the notebook cells in sequence.
5. Upload the CSV file when the notebook requests it.
6. Allow the models to train and evaluate.
7. View the performance metrics, graphs, confusion matrices, SHAP and LIME explanations.

## Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── Credit_Card_Fraud_Detection_Report.pdf
└── README.md
```

## Conclusion

This project demonstrates the application of deep learning techniques for credit card fraud detection under severe class imbalance.

MLP, CNN and Autoencoder architectures were implemented and evaluated using different loss functions and performance metrics.

Among the evaluated configurations, MLP with standard Binary Cross-Entropy achieved the best overall performance, with an F1-score of 85.42% and PR-AUC of 84.40%.

The project also incorporated SHAP and LIME to improve the interpretability of model predictions.
