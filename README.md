# Logistic regression-Based Detection of Fake Bills

A machine learning project that classifies banknotes as **Genuine** or **Fake** using six geometric measurements and Logistic Regression. The project uses Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

---

## 1. Project Overview

The objective of this project is to develop a machine learning model capable of distinguishing genuine banknotes from counterfeit banknotes based on their geometric characteristics.

The project uses the Fake Bills dataset and applies data preprocessing, exploratory data analysis, model training, prediction and performance evaluation.

### Team Members

| Student Name | Assigned Role | GitHub Handle |
|---|---|---|
| Sanchit Malhotra | Execution Lead | SanchitMalhotra11 |
| Arpit | Documentation Lead | arpitiitrpr0307 |
| Rajguru Gourav Baban | Video Lead | 2026esb1490-code |
| Kavisha Jaiswara | PPT Lead | kavishajaiswara |

All team members contributed to project development, data analysis, testing and documentation.

---

## 2. Dataset Overview

- **Dataset:** Fake Bills (Banknote Authentication)
- **Source:** Kaggle
- **Dataset Link:** https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills
- **Total Observations:** 1,500
- **Total Input Features:** 6
- **Target Variable:** `is_genuine`
- **File Format:** CSV (semicolon-separated)

### Input Features

The dataset contains the following six geometric measurements:

| Feature | Description |
|---|---|
| `diagonal` | Diagonal measurement of the banknote |
| `height_left` | Height measurement on the left side |
| `height_right` | Height measurement on the right side |
| `margin_low` | Lower margin measurement |
| `margin_up` | Upper margin measurement |
| `length` | Length measurement of the banknote |

### Target Variable

`is_genuine` represents the classification label:

- `1` or `True` = Genuine banknote
- `0` or `False` = Fake banknote

The original dataset is already included in the repository as `fake_bills.csv`. No separate dataset download is required.

---

## 3. Repository Structure

The repository is organized as follows:

```text
AI-101-Project/
│
├── Logistic regression.ipynb
│
├── fake_bills.csv
├── clean_bills.csv
├── predictions.csv
├── logreg_model.pkl
│
├── src/
│   ├── __init__.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   └── test_utils.py
│
├── docs/
│   └── Fake_Bill_Detection.pdf
│
├── Plots_Results/
│   └── Plots_Results/
│       └── Previous visualization images
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Description of Important Files

| File | Purpose |
|---|---|
| `Logistic regression.ipynb` | Main notebook containing data analysis, model training and evaluation |
| `fake_bills.csv` | Original dataset |
| `clean_bills.csv` | Processed dataset |
| `predictions.csv` | Stores actual labels, predicted labels and prediction probabilities |
| `logreg_model.pkl` | Saved trained Logistic Regression pipeline |
| `src/utils.py` | Contains reusable Python utility functions |
| `tests/test_utils.py` | Contains automated unit tests |
| `requirements.txt` | Lists the required Python dependencies |
| `docs/Fake_Bill_Detection.pdf` | Project presentation/documentation |

---

## 4. Installation and Setup Guide

Follow these steps in order to download, configure and run the project on your local computer.

### Prerequisites

Before starting, ensure that the following software is installed:

1. **Python 3.9 or later:** https://www.python.org/downloads/
2. **Git:** https://git-scm.com/downloads
3. **A modern web browser:** To access Jupyter Notebook.

During Windows Python installation, enable the option **Add Python to PATH**.

### Step 1: Clone the repository

Open Command Prompt (Windows) or Terminal (macOS/Linux) and execute:

```bash
git clone https://github.com/SanchitMalhotra11/AI-101-Project.git

cd AI-101-Project
```

This downloads the complete project repository to your computer.

### Step 2: Create a virtual environment

A virtual environment isolates the project's Python dependencies from other Python installations.

Execute:

```bash
python -m venv venv
```

Activate the environment.

**Windows (Command Prompt):**

```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

After successful activation, `(venv)` should appear at the beginning of your terminal prompt.

### Step 3: Install dependencies

First, upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all required libraries:

```bash
python -m pip install -r requirements.txt
```

This installs the libraries required for data processing, visualization, machine learning and running Jupyter Notebook.

### Step 4: Verify the dataset

The dataset is already included in the repository.

Ensure that `fake_bills.csv` is present in the main project directory:

```text
AI-101-Project/
    fake_bills.csv
```

The notebook reads the dataset using:

```python
df = pd.read_csv("fake_bills.csv", sep=";")
```

No additional dataset download or folder creation is required.

### Step 5: Launch Jupyter Notebook

Ensure that your virtual environment is activated.

Execute:

```bash
jupyter notebook
```

Jupyter Notebook will open in your default web browser.

If it does not open automatically, copy the local URL displayed in the terminal and paste it into your browser.

### Step 6: Run the main notebook

Open:

`Logistic regression.ipynb`

Execute the notebook sequentially from beginning to end.

For a fresh execution, select:

**Kernel → Restart Kernel and Clear All Outputs**

Then select:

**Run → Run All**

Wait until all cells finish executing successfully.

### Step 7: Verify the results

After successful execution, verify the generated predictions, trained model and visualization files in the project directory.

---

## 5. Project Methodology

The project follows a structured machine learning workflow.

### 5.1 Data Loading

The original dataset is loaded using Pandas.

```python
df = pd.read_csv("fake_bills.csv", sep=";")
```

### 5.2 Data Cleaning

The following operations are performed:

- Standardization of column names.
- Removal of duplicate observations.
- Removal of rows with missing target labels.
- Conversion of Boolean target labels into numerical values.

The original feature values are retained at this stage. Missing feature values are handled after the train-test split using a training-fitted imputer.

### 5.3 Train-Test Split

The dataset is divided into:

- 80% training data.
- 20% testing data.

A stratified split is used to maintain the class distribution in both subsets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 5.4 Data Preprocessing

A Scikit-learn Pipeline is used to prevent data leakage.

The preprocessing and model-training sequence is:

1. **Median Imputation:** Missing feature values are replaced using medians learned from the training data.
2. **Standard Scaling:** Features are standardized using the training data's mean and standard deviation.
3. **Logistic Regression:** The classification model is trained on the processed training data.

The complete pipeline is fitted using training data only. The same fitted preprocessing transformations are automatically applied to the testing data.

### 5.5 Model Training

Logistic Regression is used as the classification algorithm.

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The trained pipeline is saved using Python's pickle module.

### 5.6 Prediction

The trained model predicts whether each banknote is genuine or fake.

It also generates the probability of a banknote being genuine.

### 5.7 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

Additional visualizations include:

- Sigmoid curve.
- Two-dimensional decision boundary.
- Learning curve.
- Precision-Recall curve.
- Feature correlation heatmap.

---

## 6. Expected Outputs

After executing the notebook, the following files are generated or updated in the project root directory.

| Output File | Description |
|---|---|
| `logreg_model.pkl` | Saved trained machine learning pipeline |
| `predictions.csv` | Actual labels, predicted labels and probabilities |
| `confusion_matrix.png` | Confusion matrix visualization |
| `roc_curve.png` | ROC curve |
| `logistic_sigmoid_curve.png` | Logistic Regression sigmoid visualization |
| `decision_boundary.png` | Two-dimensional decision boundary |
| `learning_curve.png` | Training and cross-validation learning curves |
| `precision_recall_curve.png` | Precision-Recall visualization |
| `feature_correlation.png` | Feature correlation heatmap |

The previously committed images in `Plots_Results/Plots_Results/` are existing visualization files. The current notebook saves its generated plots in the project root directory.

---

## 7. Testing and Validation

The project includes automated unit tests for the reusable functions implemented in `src/utils.py`.

### Run Automated Tests

From the project root directory, with the virtual environment activated, execute:

```bash
python -m unittest discover -s tests -v
```

### Test Coverage

The existing test suite checks:

1. Accuracy, Precision, Recall and F1-score calculations.
2. Correct return of evaluation metric keys.
3. ROC-AUC calculation.
4. Perfect prediction handling.
5. Confusion matrix generation.
6. ROC curve generation.
7. CSV loading and semicolon delimiter handling.
8. Boolean target conversion.
9. Missing target error handling.
10. Model saving and loading.

The tests execute independently and do not require retraining the complete machine learning model.

### Example: Using Utility Functions

The project provides reusable functions through `src/utils.py`.

For example:

```python
from src.utils import evaluate_model

metrics = evaluate_model(
    y_test,
    y_pred,
    y_score=y_prob
)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

---

## 8. Links and Resources

- **GitHub Repository:** https://github.com/SanchitMalhotra11/AI-101-Project
- **Dataset:** https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills
- **Project Documentation:** [Fake_Bill_Detection.pdf](docs/Fake_Bill_Detection.pdf)
- **Video Demonstration:** To be added.

---

## 9. AI Usage Disclosure

AI tools were used for assistance with:

- Project development and Python programming guidance.
- Debugging and improving code.
- Preparing presentation materials.
- Guidance on GitHub repository management.
- Video editing guidance and documentation.

The team reviewed and worked on the implementation, analysis and project documentation.

---

## 10. Conclusion

This project demonstrates the application of machine learning to counterfeit banknote detection using geometric measurements.

By combining data preprocessing, exploratory analysis, Logistic Regression, automated testing and performance evaluation, the project provides a practical example of a complete machine learning workflow.

The implementation also uses a Scikit-learn Pipeline to ensure that preprocessing parameters are learned from training data and applied consistently to unseen testing data.
