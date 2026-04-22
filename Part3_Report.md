# Machine Learning Project
## Part 3: Report and Discussion for `poutcome` Classification

**Prepared by:** Zaid Mayyalleh  
**Course:** Machine Learning  
**Dataset:** `bank.csv`

### 1. Introduction

In this project I used the Bank Marketing dataset to classify the `poutcome` target into 4 classes: `failure`, `other`, `success`, and `unknown`.

The dataset has **11,162 rows** and **17 columns**. During preprocessing I found that the data has **0 missing values** and **0 duplicate rows**, so I did not need to do extra cleaning for null values. For the modeling part, I used all columns except `poutcome` and `deposit` as input features. Then I applied one-hot encoding to the categorical columns, standard scaling to the numerical columns, and a stratified 80/20 train-test split.

One important point from the beginning is that the target is imbalanced:

| Class | Count | Percentage |
|---|---:|---:|
| `unknown` | 8326 | 74.59% |
| `failure` | 1228 | 11.00% |
| `success` | 1071 | 9.60% |
| `other` | 537 | 4.81% |

This imbalance affected the results a lot, especially for the small `other` class.

### 2. Experiments I Did

I worked on the project in two main directions:

1. **Unsupervised learning with K-Means**  
   I removed both `poutcome` and `deposit`, encoded the categorical columns, scaled the numerical columns, and tested several values of `k` using the elbow method.

2. **Supervised learning**  
   I trained two models to classify `poutcome`:
   - Multi-Layer Perceptron (MLP)
   - Gradient Boosting

For the supervised part I first trained default models, then I used **GridSearchCV** with **5-fold stratified cross validation** and the **macro F1-score** as the scoring metric, because the classes are not balanced.

### 3. K-Means Results

For K-Means, I tested `k` values from 2 to 8. The elbow was not very sharp, but I finally used **k = 4** so I could compare the clusters with the 4 real classes.

The clustering scores were:

- **Adjusted Rand Index (ARI): 0.308**
- **Normalized Mutual Information (NMI): 0.338**

These values are not strong, so the clustering did not match the true classes very well. This makes sense because K-Means is an unsupervised method and it does not directly learn the real labels. Also, the large `unknown` class made the clustering harder.

### 4. Supervised Model Results

#### 4.1 Default Models

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| MLP default | 0.8957 | 0.6147 |
| Gradient Boosting default | 0.9046 | 0.6293 |

#### 4.2 Best Hyperparameters

- **MLP best parameters:** `batch_size=32`, `hidden_layer_sizes=(128,)`, `learning_rate_init=0.01`
- **MLP best CV macro F1:** `0.5810`
- **Gradient Boosting best parameters:** `n_estimators=150`, `max_depth=3`, `learning_rate=0.1`
- **Gradient Boosting best CV macro F1:** `0.6125`

#### 4.3 Tuned Models

| Model | Train Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| MLP tuned | 0.9027 | 0.8948 | 0.6461 | 0.6264 | 0.6177 | 0.8948 |
| Gradient Boosting tuned | 0.9346 | 0.9060 | 0.6500 | 0.6492 | 0.6323 | 0.9060 |

### 5. Which Model Performed Best?

The best model in my experiments was **Gradient Boosting tuned**.

I say this because it gave:

- the **highest test accuracy**: `0.9060`
- the **highest macro F1-score**: `0.6323`
- better class balance than MLP, especially between `failure` and `success`

The train accuracy for Gradient Boosting tuned was `0.9346`, while the test accuracy was `0.9060`. This gap is not too large, so the model shows some overfitting, but it is still acceptable.

### 6. Effect of Hyperparameter Tuning

Hyperparameter tuning helped, but the improvement was small.

- For **MLP**, the macro F1 improved from `0.6147` to `0.6177`, but the test accuracy slightly decreased from `0.8957` to `0.8948`.
- For **Gradient Boosting**, the test accuracy improved from `0.9046` to `0.9060`, and the macro F1 improved from `0.6293` to `0.6323`.

So, tuning was more useful for Gradient Boosting than for MLP in this project, but in both cases the gains were limited.

### 7. Confusion Matrix Analysis

From the confusion matrices, I noticed some clear patterns:

1. The **`unknown`** class was the easiest class for both models. In the tuned results, both models predicted **1666 out of 1666** `unknown` samples correctly.
2. The **`other`** class was the hardest class. In the tuned Gradient Boosting model, only **9 out of 107** `other` samples were classified correctly.
3. The `other` class was usually confused with **`failure`** and **`success`**.
4. There was also confusion between **`failure`** and **`success`**. In the tuned Gradient Boosting model:
   - `49` failure samples were predicted as success
   - `42` success samples were predicted as failure

This tells me that the model learned the dominant class very well, but it still struggled to separate the smaller classes from each other.

### 8. Challenges I Faced

During this project, the main challenges were:

- **Class imbalance**, because `unknown` is much larger than the other classes
- **Weak performance on the `other` class**, because it has the smallest number of samples
- **Unsupervised learning mismatch**, since K-Means is not designed to directly predict labeled classes
- **Time cost of tuning**, especially for the neural network model

### 9. Possible Improvements

If I continue this work later, I would try these ideas:

- use class balancing methods such as oversampling or class weights
- test stronger tree-based models such as Random Forest or XGBoost
- expand the hyperparameter search space
- try feature selection or feature engineering to reduce overlap between classes
- compare more evaluation strategies, not only one train-test split

### 10. Conclusion

In general, the project showed that supervised learning works much better than unsupervised clustering for this task. K-Means gave weak agreement with the real labels, while Gradient Boosting and MLP gave much better classification results.

Among all the models I tested, **Gradient Boosting tuned** was the best overall. Still, the project also showed that high accuracy alone can be misleading when the data is imbalanced. Because of that, the macro metrics and confusion matrices were important for understanding the real performance.

The main plots, confusion matrices, and comparison charts are available in [`Part2_Modeling.ipynb`](./Part2_Modeling.ipynb).
