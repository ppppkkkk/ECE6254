import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np


iris_data = sklearn.datasets.load_iris(as_frame=True)
x = iris_data.data
y = iris_data.target
# only choose 2 class
x = x[y != 2]
y = y[y != 2]
X_temp, X_test, y_temp, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2 / 9, random_state=42)
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

# Precetron
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train_normalized, y_train)
predictions = clf.predict(X_test_normalized)
perceptron_test_error = np.mean(predictions != y_test)
print(f"Perceptron Test Error {perceptron_test_error:.4f}")

# KNN
best_val_error = float('inf')
best_k = 1

for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_normalized, y_train)
    val_predictions = knn.predict(X_val_normalized)
    val_error = np.mean(val_predictions != y_val)

    if val_error < best_val_error:
        best_val_error = val_error
        best_k = k

# 使用测试集上的最佳K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_normalized, y_train)
knn_predictions = knn_best.predict(X_test_normalized)
knn_test_error = np.mean(knn_predictions != y_test)
print(f"KNN Test Error with k={best_k}: {knn_test_error:.4f}")


