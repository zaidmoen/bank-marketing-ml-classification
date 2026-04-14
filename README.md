# 🏦 Bank Marketing ML Classification

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blue)
![Python](https://img.shields.io/badge/Python-3.14+-green)
![Data Analysis](https://img.shields.io/badge/Data-Preprocessing-orange)

This project focuses on predicting whether a client will subscribe to a term deposit based on a series of marketing campaigns. Using the **Bank Marketing Dataset**, we apply various Machine Learning techniques to classify customer behavior.

## 📖 Project Overview

The goal of this project is to build a robust classification model that helps the bank optimize its marketing strategy by identifying potential subscribers.

### Phase 1: Preprocessing & EDA (Completed)
- **Data Exploration**: Analyzed 11,162 records with 17 features.
- **Quality Check**: Confirmed zero missing values and no duplicates.
- **Feature Engineering**: Encoded categorical variables and standardized numerical features.
- **Stratification**: Split the data into stratified training and testing sets to maintain class balance.

## 📊 Dataset Structure

| Feature | Description |
| :--- | :--- |
| `age` | Age of the client |
| `job` | Type of job |
| `marital` | Marital status |
| `education` | Education level |
| `balance` | Average yearly balance |
| `housing` | Has housing loan? |
| `loan` | Has personal loan? |
| `deposit` | **Target Variable** (Yes/No) |

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- Jupyter Notebook / JupyterLab
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zaidmoen/bank-marketing-ml-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Part1_Preprocessing.ipynb
   ```

## 🛠️ Built With
- **Pandas** - Data manipulation
- **Scikit-Learn** - Machine learning utilities
- **Seaborn/Matplotlib** - Data visualization

---
**Developed by:** [Zaid Moen](https://github.com/zaidmoen)
