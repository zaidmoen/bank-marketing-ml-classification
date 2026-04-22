# Bank Marketing ML Classification

This project uses the Bank Marketing dataset to classify the `poutcome` target into four classes:

- `failure`
- `other`
- `success`
- `unknown`

The work is organized into three parts:

| Part | Description | File |
|---|---|---|
| 1 | Data exploration and preprocessing | `Part1_Preprocessing.ipynb` |
| 2 | K-Means clustering, supervised modeling, tuning, and evaluation | `Part2_Modeling.ipynb` |
| 3 | Written report and discussion in student style | `Part3_Report.md` |

## Repository Structure

```text
nai/
|-- Part1_Preprocessing.ipynb
|-- Part2_Modeling.ipynb
|-- Part3_Report.md
|-- bank.csv
|-- ML_Project.pdf
|-- README.md
`-- scripts/
    `-- generate_part2_notebook.py
```

## Dataset Summary

- Total rows: `11,162`
- Total columns: `17`
- Main target used in modeling: `poutcome`
- Secondary label in the dataset: `deposit`

The dataset contains both numerical and categorical features. In the preprocessing stage, the data was checked for quality, encoded, scaled, and split into train and test sets.

## Part 1 Summary

`Part1_Preprocessing.ipynb` includes:

- dataset inspection
- feature type analysis
- missing value and duplicate checks
- exploratory plots
- encoding and scaling
- train/test split

## Part 2 Summary

`Part2_Modeling.ipynb` includes:

- K-Means clustering with elbow method
- comparison between clusters and real labels
- MLP classification
- Gradient Boosting classification
- hyperparameter tuning with stratified k-fold cross validation
- confusion matrices and metric comparison plots

Main result from Part 2:

- K-Means gave limited agreement with the true classes
- Gradient Boosting gave the best final supervised result in the current experiments

## Part 3 Summary

`Part3_Report.md` is the written report for the project. It covers:

- the experiment setup
- the main results
- the effect of hyperparameter tuning
- confusion matrix analysis
- challenges and possible improvements

## Running the Project

Open the files in this order:

1. `Part1_Preprocessing.ipynb`
2. `Part2_Modeling.ipynb`
3. `Part3_Report.md`

## Requirements

The project was built with Python and common machine learning libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

## Assignment Brief

The original assignment handout is in `ML_Project.pdf`.
