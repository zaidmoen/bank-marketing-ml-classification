import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent.parent
PART1_NOTEBOOK = ROOT / "Part1_Preprocessing.ipynb"
PART2_NOTEBOOK = ROOT / "Part2_Modeling.ipynb"
FINAL_NOTEBOOK = ROOT / "Final_Project_All_in_One.ipynb"
CELL_COUNTER = 0


def next_cell_id() -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"cell-{CELL_COUNTER:02d}"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def load_metadata() -> dict:
    for notebook_path in (PART2_NOTEBOOK, PART1_NOTEBOOK):
        if notebook_path.exists():
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            return notebook.get("metadata", {})

    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Machine Learning Project
            ## Final All-in-One Notebook for `poutcome` Classification

            **Prepared by:** Zaid Mayyalleh  
            **Course:** Machine Learning  
            **Dataset:** `bank.csv`

            This notebook combines the full project in one place:

            1. data exploration and preprocessing,
            2. K-Means clustering,
            3. supervised learning,
            4. hyperparameter tuning,
            5. final evaluation,
            6. discussion and report answers.
            """
        ),
        markdown_cell(
            """
            ## Assignment Coverage Map

            The notebook is organized to match the assignment directly:

            - **Section 1**: imports and setup
            - **Section 2**: data loading and exploration
            - **Section 3**: preprocessing for `poutcome`
            - **Section 4**: K-Means clustering
            - **Section 5**: supervised models
            - **Section 6**: hyperparameter tuning with stratified k-fold
            - **Section 7**: final evaluation and visualizations
            - **Section 8**: discussion, challenges, and improvements
            """
        ),
        markdown_cell(
            """
            ### 1. Imports and setup

            I start by loading the libraries needed for analysis, visualization, preprocessing, clustering, model training, tuning, and evaluation.
            """
        ),
        code_cell(
            """
            import warnings

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            from IPython.display import display

            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import (
                accuracy_score,
                adjusted_rand_score,
                classification_report,
                confusion_matrix,
                normalized_mutual_info_score,
                precision_recall_fscore_support,
            )
            from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import LabelEncoder, StandardScaler

            warnings.filterwarnings("ignore")

            sns.set_theme(style="whitegrid", palette="deep")
            plt.rcParams["figure.figsize"] = (9, 5)
            plt.rcParams["axes.titlesize"] = 13
            plt.rcParams["axes.labelsize"] = 11
            """
        ),
        markdown_cell(
            """
            ### 2. Load the dataset

            In this section I load the bank marketing dataset and check its basic shape before moving to detailed analysis.
            """
        ),
        code_cell(
            """
            df = pd.read_csv("bank.csv")

            print(f"Rows: {df.shape[0]}")
            print(f"Columns: {df.shape[1]}")

            display(df.head())
            """
        ),
        markdown_cell(
            """
            ### 3. Dataset structure and target distribution

            Since the project target is `poutcome`, I inspect the feature types and the class distribution of this target first.
            """
        ),
        code_cell(
            """
            dataset_summary = pd.DataFrame(
                {
                    "data_type": df.dtypes.astype(str),
                    "non_null_count": df.notnull().sum(),
                    "unique_values": df.nunique(),
                }
            )
            display(dataset_summary)

            categorical_features = df.select_dtypes(include="object").columns.tolist()
            numerical_features = df.select_dtypes(include=np.number).columns.tolist()

            print("Categorical features:", categorical_features)
            print("Numerical features:", numerical_features)

            poutcome_summary = pd.DataFrame(
                {
                    "count": df["poutcome"].value_counts(),
                    "percentage": (df["poutcome"].value_counts(normalize=True) * 100).round(2),
                }
            )
            display(poutcome_summary)

            plt.figure(figsize=(7, 4))
            sns.countplot(data=df, x="poutcome", order=poutcome_summary.index, palette="Set2")
            plt.title("Class Distribution for poutcome")
            plt.xlabel("poutcome")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            The target is clearly imbalanced because the `unknown` class is much larger than the other classes. This means accuracy alone will not be enough to judge the models.
            """
        ),
        markdown_cell(
            """
            ### 4. Data quality checks

            Before preprocessing, I check missing values, duplicate rows, and the summary statistics of the numerical columns.
            """
        ),
        code_cell(
            """
            print(f"Duplicate rows: {df.duplicated().sum()}")

            missing_table = pd.DataFrame(
                {
                    "missing_values": df.isnull().sum(),
                    "missing_percentage": (df.isnull().mean() * 100).round(2),
                }
            )
            display(missing_table)

            display(df[numerical_features].describe().round(2))
            """
        ),
        markdown_cell(
            """
            ### 5. Exploratory analysis

            I explore the numerical features and a few useful categorical features to understand the dataset better before modeling.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 4, figsize=(18, 8))
            axes = axes.flatten()

            for idx, column in enumerate(numerical_features):
                sns.histplot(df[column], kde=True, ax=axes[idx], color="#4c72b0")
                axes[idx].set_title(f"Distribution of {column}")

            axes[-1].axis("off")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(8, 6))
            sns.heatmap(df[numerical_features].corr(), annot=True, cmap="Blues", fmt=".2f")
            plt.title("Correlation Heatmap for Numerical Features")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            sns.countplot(data=df, y="job", order=df["job"].value_counts().index, ax=axes[0], color="#4c72b0")
            axes[0].set_title("Job Distribution")
            axes[0].set_xlabel("Count")
            axes[0].set_ylabel("Job")

            sns.countplot(data=df, x="contact", order=df["contact"].value_counts().index, ax=axes[1], palette="Set3")
            axes[1].set_title("Contact Type Distribution")
            axes[1].set_xlabel("Contact")
            axes[1].set_ylabel("Count")

            sns.countplot(data=df, x="month", order=df["month"].value_counts().index, ax=axes[2], palette="Set2")
            axes[2].set_title("Month Distribution")
            axes[2].set_xlabel("Month")
            axes[2].tick_params(axis="x", rotation=45)
            axes[2].set_ylabel("Count")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Short observations from EDA

            A few points are clear from the exploration:

            - the dataset is clean in terms of missing values and duplicates,
            - the target is imbalanced,
            - several numerical features are skewed and should be scaled,
            - the data contains many categorical features, so encoding is necessary before modeling.
            """
        ),
        markdown_cell(
            """
            ### 6. Preprocessing for supervised learning

            The assignment asks to classify `poutcome`, so I use `poutcome` as the target and remove `deposit` from the input features.  
            I apply one-hot encoding to categorical variables and standard scaling to the numerical columns.
            """
        ),
        code_cell(
            """
            X = df.drop(columns=["poutcome", "deposit"]).copy()
            y = df["poutcome"].copy()

            categorical_cols = X.select_dtypes(include="object").columns.tolist()
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

            print("Categorical input columns:", categorical_cols)
            print("Numerical input columns:", numeric_cols)
            print("Encoded feature count:", X_encoded.shape[1])
            """
        ),
        code_cell(
            """
            X_train, X_test, y_train_raw, y_test_raw = train_test_split(
                X_encoded,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            scaler = StandardScaler()
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train_raw)
            y_test = label_encoder.transform(y_test_raw)
            class_names = list(label_encoder.classes_)

            print("Train shape:", X_train.shape)
            print("Test shape:", X_test.shape)

            class_mapping = pd.DataFrame(
                {"class_id": range(len(class_names)), "poutcome": class_names}
            )
            display(class_mapping)
            """
        ),
        code_cell(
            """
            train_test_distribution = pd.DataFrame(
                {
                    "full_dataset_%": (y.value_counts(normalize=True) * 100).round(2),
                    "train_%": (y_train_raw.value_counts(normalize=True) * 100).round(2),
                    "test_%": (y_test_raw.value_counts(normalize=True) * 100).round(2),
                }
            ).reindex(class_names)

            display(train_test_distribution)
            """
        ),
        markdown_cell(
            """
            The train and test sets keep almost the same class percentages, which confirms that the stratified split worked correctly.
            """
        ),
        markdown_cell(
            """
            ### 7. Unsupervised learning with K-Means

            For K-Means, the assignment says to remove both `poutcome` and `deposit`.  
            I prepare a separate feature matrix for clustering, scale it, and then test multiple values of `k` using the elbow method.
            """
        ),
        code_cell(
            """
            cluster_X = df.drop(columns=["poutcome", "deposit"]).copy()
            cluster_cat_cols = cluster_X.select_dtypes(include="object").columns.tolist()
            cluster_num_cols = cluster_X.select_dtypes(include=np.number).columns.tolist()

            cluster_encoded = pd.get_dummies(cluster_X, columns=cluster_cat_cols, drop_first=False)
            cluster_encoded[cluster_num_cols] = StandardScaler().fit_transform(
                cluster_encoded[cluster_num_cols]
            )

            inertias = []
            k_values = list(range(2, 9))

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                kmeans.fit(cluster_encoded)
                inertias.append(kmeans.inertia_)

            elbow_table = pd.DataFrame({"k": k_values, "inertia": inertias})
            display(elbow_table)

            plt.figure(figsize=(7, 4))
            plt.plot(k_values, inertias, marker="o")
            plt.title("Elbow Method for K-Means")
            plt.xlabel("Number of clusters")
            plt.ylabel("Inertia")
            plt.xticks(k_values)
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=20)
            cluster_labels = kmeans_final.fit_predict(cluster_encoded)

            ari = adjusted_rand_score(df["poutcome"], cluster_labels)
            nmi = normalized_mutual_info_score(df["poutcome"], cluster_labels)

            print(f"Adjusted Rand Index: {ari:.3f}")
            print(f"Normalized Mutual Information: {nmi:.3f}")

            cluster_vs_class = pd.crosstab(
                pd.Series(cluster_labels, name="cluster"),
                df["poutcome"],
            )
            display(cluster_vs_class)
            """
        ),
        code_cell(
            """
            pca = PCA(n_components=2, random_state=42)
            cluster_2d = pca.fit_transform(cluster_encoded)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            sns.scatterplot(
                x=cluster_2d[:, 0],
                y=cluster_2d[:, 1],
                hue=cluster_labels,
                palette="tab10",
                s=16,
                alpha=0.45,
                linewidth=0,
                ax=axes[0],
            )
            axes[0].set_title("K-Means Clusters in 2D PCA Space")
            axes[0].set_xlabel("PCA 1")
            axes[0].set_ylabel("PCA 2")

            sns.scatterplot(
                x=cluster_2d[:, 0],
                y=cluster_2d[:, 1],
                hue=df["poutcome"],
                palette="Set2",
                s=16,
                alpha=0.45,
                linewidth=0,
                ax=axes[1],
            )
            axes[1].set_title("True poutcome Labels in 2D PCA Space")
            axes[1].set_xlabel("PCA 1")
            axes[1].set_ylabel("PCA 2")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### K-Means discussion

            The elbow method shows a gradual decrease in inertia, but not a very sharp bend.  
            I used `k = 4` to match the number of real target classes. The clustering scores are not strong, which shows that K-Means is limited here because it is unsupervised and the target classes are imbalanced.
            """
        ),
        markdown_cell(
            """
            ### 8. Supervised learning models

            To classify `poutcome`, I use two models:

            - **MLPClassifier** as the neural network model
            - **GradientBoostingClassifier** as the ensemble model

            I first train simple default versions to get baseline results before tuning.
            """
        ),
        code_cell(
            """
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            baseline_models = {
                "MLP default": MLPClassifier(
                    random_state=42,
                    max_iter=250,
                    early_stopping=True,
                    n_iter_no_change=15,
                ),
                "Gradient Boosting default": GradientBoostingClassifier(random_state=42),
            }

            baseline_rows = []

            for model_name, model in baseline_models.items():
                model.fit(X_train, y_train)
                baseline_pred = model.predict(X_test)
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    y_test,
                    baseline_pred,
                    average="macro",
                    zero_division=0,
                )

                baseline_rows.append(
                    {
                        "model": model_name,
                        "test_accuracy": accuracy_score(y_test, baseline_pred),
                        "macro_precision": macro_p,
                        "macro_recall": macro_r,
                        "macro_f1": macro_f1,
                    }
                )

            baseline_results = pd.DataFrame(baseline_rows).sort_values(
                by="test_accuracy",
                ascending=False,
            )
            display(baseline_results.round(4))
            """
        ),
        markdown_cell(
            """
            ### 9. Hyperparameter tuning with stratified k-fold cross validation

            I use `GridSearchCV` with `f1_macro` because the class distribution is imbalanced.  
            This helps me give more attention to the smaller classes instead of letting the largest class dominate the score.
            """
        ),
        code_cell(
            """
            mlp = MLPClassifier(
                random_state=42,
                max_iter=250,
                early_stopping=True,
                n_iter_no_change=15,
            )

            mlp_param_grid = {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "learning_rate_init": [0.001, 0.01],
                "batch_size": [32, 64],
            }

            mlp_search = GridSearchCV(
                estimator=mlp,
                param_grid=mlp_param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
            )

            mlp_search.fit(X_train, y_train)

            print("Best MLP parameters:", mlp_search.best_params_)
            print(f"Best MLP CV score: {mlp_search.best_score_:.4f}")
            """
        ),
        code_cell(
            """
            gb = GradientBoostingClassifier(random_state=42)

            gb_param_grid = {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            }

            gb_search = GridSearchCV(
                estimator=gb,
                param_grid=gb_param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
            )

            gb_search.fit(X_train, y_train)

            print("Best Gradient Boosting parameters:", gb_search.best_params_)
            print(f"Best Gradient Boosting CV score: {gb_search.best_score_:.4f}")
            """
        ),
        code_cell(
            """
            mlp_tuning_table = (
                pd.DataFrame(mlp_search.cv_results_)[
                    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
                ]
                .sort_values(by=["rank_test_score", "mean_test_score"])
                .head(5)
            )

            gb_tuning_table = (
                pd.DataFrame(gb_search.cv_results_)[
                    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
                ]
                .sort_values(by=["rank_test_score", "mean_test_score"])
                .head(5)
            )

            print("Top MLP parameter combinations")
            display(mlp_tuning_table)

            print("Top Gradient Boosting parameter combinations")
            display(gb_tuning_table)
            """
        ),
        markdown_cell(
            """
            ### 10. Final evaluation on train and test sets

            In this section I evaluate the best tuned model from each algorithm, compare train vs test performance, and inspect the class-level metrics.
            """
        ),
        code_cell(
            """
            def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, class_names):
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)

                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    y_test,
                    test_pred,
                    average="macro",
                    zero_division=0,
                )
                micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                    y_test,
                    test_pred,
                    average="micro",
                    zero_division=0,
                )

                report_df = pd.DataFrame(
                    classification_report(
                        y_test,
                        test_pred,
                        target_names=class_names,
                        output_dict=True,
                        zero_division=0,
                    )
                ).T
                report_df.loc["micro avg"] = [micro_p, micro_r, micro_f1, len(y_test)]

                summary = {
                    "model": model_name,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "macro_precision": macro_p,
                    "macro_recall": macro_r,
                    "macro_f1": macro_f1,
                    "micro_precision": micro_p,
                    "micro_recall": micro_r,
                    "micro_f1": micro_f1,
                }

                cm = confusion_matrix(y_test, test_pred)
                return summary, report_df, cm


            tuned_models = {
                "MLP tuned": mlp_search.best_estimator_,
                "Gradient Boosting tuned": gb_search.best_estimator_,
            }

            evaluation_rows = []
            detailed_reports = {}
            confusion_matrices = {}

            for model_name, model in tuned_models.items():
                summary, report_df, cm = evaluate_model(
                    model_name,
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    class_names,
                )
                evaluation_rows.append(summary)
                detailed_reports[model_name] = report_df
                confusion_matrices[model_name] = cm

            comparison_df = pd.DataFrame(evaluation_rows).sort_values(
                by="test_accuracy",
                ascending=False,
            )
            display(comparison_df.round(4))
            """
        ),
        code_cell(
            """
            for model_name, report_df in detailed_reports.items():
                print(model_name)
                display(
                    report_df.loc[
                        class_names + ["macro avg", "weighted avg", "micro avg"]
                    ].round(4)
                )
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax,
                )
                ax.set_title(model_name)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")

            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plot_df = comparison_df.melt(
                id_vars="model",
                value_vars=["test_accuracy", "macro_f1", "micro_f1"],
                var_name="metric",
                value_name="score",
            )

            plt.figure(figsize=(8, 4.5))
            sns.barplot(data=plot_df, x="metric", y="score", hue="model", palette="Set1")
            plt.ylim(0, 1)
            plt.title("Model Performance Comparison")
            plt.ylabel("Score")
            plt.xlabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            gb_best = gb_search.best_estimator_
            feature_importance = (
                pd.DataFrame(
                    {
                        "feature": X_train.columns,
                        "importance": gb_best.feature_importances_,
                    }
                )
                .sort_values(by="importance", ascending=False)
                .head(15)
            )

            display(feature_importance)

            plt.figure(figsize=(9, 5))
            sns.barplot(data=feature_importance, x="importance", y="feature", palette="viridis")
            plt.title("Top 15 Feature Importances from Gradient Boosting")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### 11. Direct answers for the report section

            The assignment also asks for a written discussion.  
            Instead of keeping it in a separate file, I answer the report questions directly here in the same notebook.
            """
        ),
        code_cell(
            """
            best_model_row = comparison_df.iloc[0]
            best_model_name = best_model_row["model"]

            print(f"Best model: {best_model_name}")
            print(f"Test accuracy: {best_model_row['test_accuracy']:.4f}")
            print(f"Macro F1-score: {best_model_row['macro_f1']:.4f}")
            print()

            print("Effect of tuning")
            print(
                f"MLP: baseline macro F1 = {baseline_results.loc[baseline_results['model'] == 'MLP default', 'macro_f1'].iloc[0]:.4f}, "
                f"tuned macro F1 = {comparison_df.loc[comparison_df['model'] == 'MLP tuned', 'macro_f1'].iloc[0]:.4f}"
            )
            print(
                f"Gradient Boosting: baseline macro F1 = {baseline_results.loc[baseline_results['model'] == 'Gradient Boosting default', 'macro_f1'].iloc[0]:.4f}, "
                f"tuned macro F1 = {comparison_df.loc[comparison_df['model'] == 'Gradient Boosting tuned', 'macro_f1'].iloc[0]:.4f}"
            )
            """
        ),
        markdown_cell(
            """
            ### 12. Final discussion

            **Which model performed best?**  
            The best model was the tuned Gradient Boosting model. It achieved the highest test accuracy and the strongest macro F1-score among the tested supervised models.

            **How did tuning affect the models?**  
            Tuning improved Gradient Boosting slightly and gave only a small improvement for MLP. This means the default settings were already reasonable, but some gains were still possible through grid search.

            **What do the confusion matrices show?**  
            The `unknown` class was the easiest one because it had the most samples and was classified very well by both tuned models.  
            The `other` class was the hardest one and was often confused with `failure` and `success`.

            **What challenges appeared in the project?**

            - strong class imbalance,
            - weak recall for smaller classes,
            - limited match between K-Means clusters and true labels,
            - tuning cost in terms of runtime.

            **What can be improved later?**

            - try class balancing methods,
            - test more ensemble models such as Random Forest or XGBoost,
            - expand the tuning grid,
            - try feature selection and more advanced feature engineering.
            """
        ),
        markdown_cell(
            """
            ## Final conclusion

            This notebook completes the full machine learning workflow for classifying `poutcome`:

            - data exploration and preprocessing,
            - unsupervised clustering,
            - supervised classification,
            - tuning and evaluation,
            - written discussion and final conclusions.

            Overall, supervised learning performed much better than unsupervised clustering for this task, and the tuned Gradient Boosting model gave the best final result in this project.
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": load_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    FINAL_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {FINAL_NOTEBOOK.name}")


if __name__ == "__main__":
    main()
