# 💳 Credit Card Fraud Detection Using Deep Learning

A deep learning project that detects fraudulent credit card transactions
using **MLP, 1D CNN, LSTM, and Autoencoder** architectures. The project
compares different loss functions, evaluates model performance under
class imbalance, and uses SHAP and LIME for explainability.

## Project Overview

Credit card fraud detection is a challenging classification problem
because fraudulent transactions represent a very small percentage of all
transactions.

A model that predicts every transaction as genuine can achieve high
accuracy while failing to detect fraud.

The objective of this project is to compare different deep learning
approaches and identify models that can effectively detect fraudulent
transactions while maintaining a reasonable balance between precision
and recall.

## Dataset

**Dataset:** European Credit Card Fraud Detection

**Source:** [Kaggle -- Credit Card Fraud
Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset statistics:

  Description                   Value
  ------------------------- ---------
  Total transactions          284,807
  Genuine transactions        284,315
  Fraudulent transactions         492
  Input features                   30
  Target column               `Class`

### Features

-   `Time`: Seconds elapsed since the first transaction.
-   `Amount`: Transaction amount.
-   `V1` to `V28`: Anonymized features obtained through Principal
    Component Analysis (PCA).
-   `Class`: Target variable (0 = Genuine, 1 = Fraud).

The dataset is highly imbalanced, with approximately 0.173% fraudulent
transactions.

## Technologies Used

-   Python
-   Google Colab
-   TensorFlow / Keras
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   SHAP
-   LIME

## Deep Learning Models

Four deep learning architectures are implemented:

### 1. Multi-Layer Perceptron (MLP)

A feedforward neural network designed for tabular data.

Architecture:

`Input → Dense(64) → Dropout → Dense(32) → Dropout → Sigmoid Output`

### 2. 1D Convolutional Neural Network (CNN)

Uses one-dimensional convolutional layers to examine patterns among the
input features.

### 3. Long Short-Term Memory (LSTM)

A recurrent neural network architecture included for experimental
comparison.

### 4. Autoencoder

An unsupervised anomaly detection model trained on genuine transactions.
It identifies potentially fraudulent transactions using reconstruction
error.

### Logistic Regression Baseline

A class-weighted Logistic Regression model is included as a conventional
reference baseline to compare deep learning performance against a
simpler approach.

## Loss Functions

The following loss functions are compared:

-   **Binary Cross-Entropy (BCE):** Standard binary classification loss.
-   **Class-Weighted BCE:** Assigns greater importance to the minority
    fraud class.
-   **Focal Loss:** Reduces the influence of easy examples and focuses
    training on difficult predictions.

The experiments show that standard BCE performed strongly, while class
weighting did not improve performance with the tested settings.

## Methodology

The project follows these steps:

1.  Load and explore the credit card transaction dataset.
2.  Check missing values and duplicate records.
3.  Split the data into training, validation and testing sets using
    stratified sampling.
4.  Standardize `Time` and `Amount` using a scaler fitted only on
    training data.
5.  Train the deep learning models with different loss functions.
6.  Select classification thresholds using validation predictions.
7.  Select the main model using validation PR-AUC.
8.  Evaluate the selected model on the held-out test set.
9.  Perform five-seed evaluation.
10. Conduct duplicate-removal and chronological experiments.
11. Apply SHAP and LIME for model explainability.

### Data Splitting

The main experiment uses:

-   Training: 60%
-   Validation: 20%
-   Testing: 20%

The chronological experiment sorts transactions by `Time` and uses the
first 60% for training, the next 20% for validation and the final 20%
for testing.

## Evaluation Metrics

The project uses:

-   Precision
-   Recall
-   F1-score
-   Precision-Recall AUC (PR-AUC)
-   ROC-AUC

PR-AUC is emphasized because the dataset contains very few fraudulent
transactions.

## Results

### Main Stratified Experiment

The MLP with standard BCE was selected using validation PR-AUC.

  Metric        Result
  ----------- --------
  Precision     88.17%
  Recall        82.83%
  F1-score      85.42%
  PR-AUC        84.40%
  ROC-AUC       97.94%

Test-set confusion matrix:

                     Predicted Genuine   Predicted Fraud
  ---------------- ------------------- -----------------
  Actual Genuine                56,852                11
  Actual Fraud                      17                82

The model correctly detected 82 of the 99 fraudulent transactions in the
test set.

### Five-Seed Evaluation

To examine performance variation, the main stratified experiment was
repeated using five random seeds:

`42, 1, 2, 3, 4`

  Model                     Mean PR-AUC   Mean Recall   Mean Precision   Mean F1
  --------------------- --------------- ------------- ---------------- ---------
  MLP -- BCE              0.827 ± 0.050         0.808            0.887     0.845
  MLP -- Focal            0.826 ± 0.048         0.802            0.878     0.837
  CNN -- BCE              0.824 ± 0.044         0.806            0.859     0.831
  Logistic Regression     0.757 ± 0.056         0.776            0.887     0.827
  LSTM -- BCE             0.754 ± 0.069         0.790            0.873     0.829
  MLP -- Weighted BCE     0.754 ± 0.049         0.794            0.874     0.831
  CNN -- Weighted BCE     0.720 ± 0.061         0.725            0.808     0.762
  Autoencoder             0.511 ± 0.066         0.596            0.491     0.535

MLP BCE achieved the highest mean PR-AUC and F1-score. MLP Focal and CNN
BCE also performed competitively. The differences should not be treated
as statistically conclusive based on five runs.

### Chronological Experiment

The chronological experiment evaluates how models perform when
predicting later transactions.

  Model            Precision   Recall   F1-score   PR-AUC
  -------------- ----------- -------- ---------- --------
  MLP -- BCE          94.23%   65.33%     77.17%   80.53%
  MLP -- Focal        89.83%   70.67%     79.10%   81.59%
  CNN -- BCE          81.67%   65.33%     72.59%   76.80%
  LSTM -- BCE         91.07%   68.00%     77.86%   75.84%
  Autoencoder         35.96%   42.67%     39.02%   23.45%

The chronological experiment shows a reduction in fraud recall compared
with the random stratified experiment.

### Duplicate Analysis

The dataset contained 1,081 exact duplicate rows.

After removing duplicates, the MLP BCE results changed:

  Metric        Original   Deduplicated
  ----------- ---------- --------------
  Precision       88.17%         89.29%
  Recall          82.83%         78.95%
  F1-score        85.42%         83.80%
  PR-AUC          84.40%         81.90%

This experiment highlights the importance of checking duplicate
transactions when evaluating fraud detection models. The original and
deduplicated runs use different dataset sizes, so this comparison should
be interpreted with that caveat.

## Explainability

### SHAP

SHAP was used to understand the global influence of input features on
model predictions.

The most influential features in the generated SHAP summary plot were:

1.  V14
2.  V4
3.  V7
4.  V3
5.  V12

Since the PCA features are anonymized, their real-world meanings cannot
be inferred.

### LIME

LIME was used to explain two individual predictions:

-   **True Positive:** A fraudulent transaction correctly classified as
    fraud, with approximately 85% predicted fraud probability.
-   **False Negative:** A fraudulent transaction incorrectly classified
    as genuine, with approximately 100% predicted genuine probability.

These explanations help illustrate why the model made different
predictions for individual transactions.

## Key Findings

-   MLP BCE achieved the strongest overall results in the main
    stratified experiment.
-   MLP BCE had the highest mean PR-AUC and F1-score in the five-seed
    evaluation.
-   MLP Focal performed best in chronological test F1-score.
-   Class-weighted BCE did not improve results with the tested weights.
-   The Autoencoder was less effective than the supervised models in
    these experiments.
-   Chronological testing showed lower fraud recall, highlighting the
    challenge of detecting future fraud patterns.
-   SHAP and LIME provided global and individual prediction
    explanations.

## Limitations

1.  The stratified test set contains only 99 fraud transactions, so
    small metric differences should not be treated as statistically
    conclusive.
2.  CNN and LSTM use pseudo-sequences created from PCA features rather
    than genuine customer transaction histories. The dataset does not
    contain customer or card identifiers.
3.  `Time` represents elapsed seconds, not a calendar timestamp.
4.  PCA features are anonymized and cannot be interpreted directly.
5.  The Autoencoder produces reconstruction-error anomaly scores, not
    calibrated fraud probabilities.
6.  No external validation on another dataset or banking system has been
    performed.
7.  SHAP and LIME explain model behavior; they do not establish
    causation.
8.  Real-world deployment would require probability calibration,
    cost-sensitive threshold selection, monitoring for data drift, and
    security and privacy reviews.

## Conclusion

This project compares four deep learning architectures for credit card
fraud detection under severe class imbalance.

The MLP trained with standard Binary Cross-Entropy is recommended as the
primary model because it achieved the strongest overall performance in
the main stratified experiment and the highest mean PR-AUC and F1-score
across five seeds.

However, MLP Focal and CNN BCE also performed competitively. The
chronological experiment demonstrates that performance can decline when
predicting later transactions, so the random-split results should not be
interpreted as guaranteed real-world performance.

## How to Run the Project

1.  Clone or download this repository.
2.  Open [Google Colab](https://colab.research.google.com/).
3.  Open the project notebook (`.ipynb`) in Colab.
4.  Download `creditcard.csv` from the Kaggle dataset linked above.
5.  Run the notebook cells from top to bottom.
6.  Upload the dataset CSV file when prompted.
7.  Review the generated evaluation tables, visualizations and
    explainability outputs.

## Repository Structure

``` text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection_Final_Updated.ipynb
├── README.md
├── Credit_Card_Fraud_All_Results.csv
└── coding-workspace.png        # Optional screenshot, if included
```

`Credit_Card_Fraud_All_Results.csv` combines the main model comparison,
original-versus-deduplicated results, five-seed summary, five-seed raw
results, and chronological results. The `Section` column identifies the
experiment for each row. Some columns are blank for sections where that
metric does not apply.

The dataset is obtained separately from Kaggle and is not included in
this repository.

## Author

Developed as a Deep Learning Lab project focused on comparing neural
network architectures and evaluating fraud detection under highly
imbalanced data.

------------------------------------------------------------------------

**If you find this project useful, feel free to explore the notebook and
its experimental results.**
