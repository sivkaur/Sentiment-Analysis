
# Sentiment Analysis

This repository holds the implementation and evaluation of various classifiers for sentiment analysis. The classifiers explored include Naive Bayes, Support Vector Machine (SVM), and Random Forest, and are evaluated based on accuracy, precision, recall, and F1-score to find the optimal model for sentiment classification.

## Classifier Performance Metrics

### Naive Bayes:
- **Accuracy**: 63.77%
- **Precision**: 66.20%
- **Recall**: 63.77%
- **F1-Score**: 63.44%

While Naive Bayes serves as a baseline model with moderate precision and recall, it shows limited improvement after hyperparameter tuning, indicating its simplicity and speed are the primary advantages.

### Support Vector Machine (SVM):
- **Accuracy**: 69.51% (Original), 70.31% (Tuned)
- **Precision**: 72.00% (Original), 71.00% (Tuned)
- **Recall**: 70.00% (Original and Tuned)
- **F1-Score**: 69.00% (Original), 70.00% (Tuned)

SVM classifiers are effective in high-dimensional spaces and have shown improvement post-tuning, especially in accuracy and F1-score, which signifies its capability to discern complex patterns within textual data.

### Random Forest:
- **Accuracy**: 70.25% (Original), 70.71% (Tuned)
- **Precision**: 71.00% (Original and Tuned)
- **Recall**: 70.00% (Original), 71.00% (Tuned)
- **F1-Score**: 70.00% (Original), 71.00% (Tuned)

Random Forest stands out as the top performer with the highest accuracy and F1-score, demonstrating robustness against overfitting and the ability to capture diverse patterns.

## Comparative Analysis

Based on the evaluation, Random Forest is the most recommended model, particularly after hyperparameter tuning, due to its strong predictive performance and balanced sensitivity and specificity. It is suitable for complex text classification challenges, while SVM is notable where interpretability of a linear model is a priority. Despite lower scores, Naive Bayes could still be the choice for scenarios prioritizing computational simplicity.

## Repository Structure

- `training_classifiers.py`: Script to train classifiers and tune hyperparameters.
- `sivleen_kaur_testing_classifiers.py`: Script to test classifiers and generate performance metrics.

## Installation and Usage

Please see the [installation section](#installation) and [usage instructions](#usage) for details on how to set up and run the classifiers.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To train and tune the classifiers:

```bash
python training_classifiers.py
```

To test the classifiers and evaluate their performance:

```bash
python sivleen_kaur_testing_classifiers.py
```
