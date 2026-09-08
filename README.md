# Task 6: K-Nearest Neighbors (KNN) Classification

## 📌 Objective

Understand and implement the K-Nearest Neighbors (KNN) algorithm for classification problems using the Iris dataset.

## 🛠 Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📂 Dataset

Iris Dataset (loaded via `sklearn.datasets.load_iris`) — 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica).

## 🔍 Approach

1. Loaded and explored the Iris dataset.
2. Split data into training (80%) and testing (20%) sets.
3. Normalized features using `StandardScaler`.
4. Trained a `KNeighborsClassifier` with K=5 as a baseline.
5. Experimented with K values from 1 to 20 to find the optimal K.
6. Evaluated the best model using accuracy score, classification report, and confusion matrix.
7. Visualized decision boundaries using 2 features for interpretability.

## 📊 Results

- Baseline accuracy (K=5): **XX%**
- Best K found: **K = X**
- Best accuracy: **XX%**

### Accuracy vs K

![Accuracy vs K](screenshots/accuracy_vs_k.png)

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

### Decision Boundary

![Decision Boundary](screenshots/decision_boundary.png)

## 📖 What I Learned

- Instance-based (lazy) learning
- Role of Euclidean distance in classification
- Importance of feature normalization in distance-based algorithms
- Effect of K on bias-variance tradeoff
- How to visualize decision boundaries for classifiers

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/knn_classification.ipynb
```
