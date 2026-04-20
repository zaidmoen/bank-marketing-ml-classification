# 🏦 Bank Marketing ML Classification

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blue?style=for-the-badge&logo=scikit-learn)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

> **Predicting term deposit subscriptions using Machine Learning on real-world bank marketing data.**

</div>

---

## 📌 Project Overview

This project applies **end-to-end Machine Learning** to the classic **Bank Marketing Dataset** to classify whether a client will subscribe to a term deposit (`yes` / `no`). The work is divided into two structured phases:

| Phase | Description | Notebook |
|-------|-------------|----------|
| **Phase 1** | Data Preprocessing & Exploratory Data Analysis (EDA) | `Part1_Preprocessing.ipynb` |
| **Phase 2** | Model Building, Training & Evaluation | `Part2_Modeling.ipynb` |

---

## 🗂️ Repository Structure

```
bank-marketing-ml-classification/
│
├── 📓 Part1_Preprocessing.ipynb   # EDA, cleaning, feature engineering
├── 📓 Part2_Modeling.ipynb        # Model training, evaluation, comparison
├── 📊 bank.csv                    # Raw dataset (11,162 records × 17 features)
├── 📄 ML_Project.pdf              # Full project report
├── 🐍 scripts/
│   └── generate_part2_notebook.py # Utility script for notebook generation
├── .gitignore
└── README.md
```

---

## 📊 Dataset Overview

The dataset contains **11,162 client records** collected from direct phone call marketing campaigns by a Portuguese bank.

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age of the client |
| `job` | Categorical | Type of job (admin, blue-collar, entrepreneur, …) |
| `marital` | Categorical | Marital status (married, single, divorced) |
| `education` | Categorical | Education level (primary, secondary, tertiary) |
| `default` | Binary | Has credit in default? |
| `balance` | Numeric | Average yearly bank balance (in euros) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numeric | Last contact day of the month |
| `month` | Categorical | Last contact month |
| `duration` | Numeric | Last contact duration (seconds) |
| `campaign` | Numeric | Number of contacts during this campaign |
| `pdays` | Numeric | Days since last contact from previous campaign |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign |
| `deposit` | **Target** | Has the client subscribed? (**yes** / **no**) |

---

## 🔬 Phase 1 — Preprocessing & EDA

### ✅ Key Steps

- **Data Exploration**: Loaded and inspected 11,162 records × 17 features
- **Quality Assurance**: Confirmed zero missing values and zero duplicates
- **Univariate Analysis**: Distribution plots for all numerical and categorical features
- **Multivariate Analysis**: Correlation heatmap and feature-target relationships
- **Feature Engineering**:
  - Label encoding for binary categorical features
  - One-hot encoding for multi-class categorical features
  - Standard scaling of numerical features
- **Train/Test Split**: Stratified 80/20 split to preserve class balance

### 📈 Key Findings

- The dataset is **imbalanced** — majority of clients did not subscribe (~88%)
- `duration` (call duration) shows the strongest correlation with the target
- Clients with no prior campaign contact had higher subscription rates
- `poutcome = success` strongly predicts a positive outcome

---

## 🤖 Phase 2 — Model Building & Evaluation

### Models Trained

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier |
| **Decision Tree** | Interpretable tree-based model |
| **Random Forest** | Ensemble of decision trees |
| **K-Nearest Neighbors (KNN)** | Distance-based classifier |
| **Support Vector Machine (SVM)** | Maximum margin classifier |
| **Naive Bayes** | Probabilistic classifier |

### 📐 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision & Recall** — Performance on the minority class
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of TP, TN, FP, FN
- **ROC-AUC** — Area under the ROC curve

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:
- Python **3.10+**
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/zaidmoen/bank-marketing-ml-classification.git
cd bank-marketing-ml-classification

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch Jupyter
jupyter notebook
```

### Running the Notebooks

Run the notebooks **in order**:

```bash
# Step 1: Preprocessing & EDA
jupyter notebook Part1_Preprocessing.ipynb

# Step 2: Model Training & Evaluation
jupyter notebook Part2_Modeling.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Data visualization & EDA |
| **Scikit-Learn** | ML models, preprocessing, evaluation |
| **Jupyter Notebook** | Interactive development environment |

---

## 📄 Project Report

A detailed written report is available in [`ML_Project.pdf`](./ML_Project.pdf) covering:
- Problem statement and motivation
- Methodology and experimental setup
- Results and model comparison
- Conclusions and future recommendations

---

## 👤 Author

<div align="center">

**Zaid Moen**

[![GitHub](https://img.shields.io/badge/GitHub-zaidmoen-181717?style=for-the-badge&logo=github)](https://github.com/zaidmoen)

</div>

---

<div align="center">
<sub>⭐ If you found this project useful, please consider giving it a star!</sub>
</div>
