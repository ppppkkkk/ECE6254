from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target


def manual_standardization(data):
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    std_dev = np.sqrt(variance)
    standardized_data = (data - mean) / std_dev
    return standardized_data


X_standardized_manual = manual_standardization(X)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardized_sklearn = scaler.fit_transform(X)

comparison = np.allclose(X_standardized_manual, X_standardized_sklearn)
print(f"they are the same? {comparison}")

