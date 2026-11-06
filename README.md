# Credit Card Fraud Detection Using Deep Learning

## Project Overview
This project detects fraudulent credit card transactions using deep learning techniques. It addresses the highly imbalanced nature of the European Credit Card Fraud Detection dataset.

## Deep Learning Models
- Multi-Layer Perceptron (MLP)
- 1D Convolutional Neural Network (CNN)
- Autoencoder

## Techniques Implemented
- Binary Cross-Entropy (BCE)
- Weighted Binary Cross-Entropy
- Focal Loss
- Validation-based threshold optimization
- SHAP and LIME explainability (optional analysis)

## Dataset
European Credit Card Fraud Detection dataset.

The dataset contains 284,807 transactions, including 492 fraudulent transactions. Download `creditcard.csv` from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Upload the CSV file when prompted by the notebook. The dataset itself is not included in this repository.

## Best Experimental Result

The MLP model trained with Binary Cross-Entropy achieved:

| Metric | Result |
|---|---:|
| Precision | 88.17% |
| Recall | 82.83% |
| F1-score | 85.42% |
| PR-AUC | 84.40% |
| ROC-AUC | 97.94% |

These are results from the held-out test set in the reported experiment.

## How to Run
1. Download the notebook.
2. Open it using Google Colab.
3. Upload `creditcard.csv` when prompted.
4. Run the notebook cells in order.

## Tools and Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SHAP
- LIME
- Google Colab

## Author
B.Ravi kiran
