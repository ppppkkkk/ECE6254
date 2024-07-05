import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(w, X, Y):
    m = len(Y)
    h = sigmoid(X.dot(w))
    grad = (1/m) * X.T.dot(h - Y)
    return grad


def gradient_descent(X, Y, alpha, num_iterations):
    m, n = X.shape
    w = np.zeros(n)  # Initialization
    for i in range(num_iterations):
        grad = gradient(w, X, Y)
        w = w - alpha * grad
    return w


X = np.array([
    [1, 1],
    [1, 0]
])

Y = np.array([1, -1])

alpha = 0.1  # Learning rate
num_iterations = 1000  # Iteration times

optimal_w = gradient_descent(X, Y, alpha, num_iterations)

print("Optimal w:", optimal_w)