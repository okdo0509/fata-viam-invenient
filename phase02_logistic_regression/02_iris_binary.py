import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def softmax(matrix):
    max_values = np.max(matrix, axis=1, keepdims=True)
    matrix = matrix - max_values  # For numerical stability
    matrix = np.exp(matrix)
    sum_exp = np.sum(matrix, axis=1, keepdims=True)
    return matrix / sum_exp
def loss_function(y_pred_matrix, y_true):
    y_pred = np.clip(y_pred_matrix, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))
def diff_loss_w(y, x, y_pred,N):
    return np.dot(x.T, y_pred - y) / N
def diff_loss_b(y, y_pred):
    return np.mean(y_pred - y, axis=0)

iris_data = pd.read_csv("C:\\Users\\옥유준\\Downloads\\archive\\Iris.csv")

X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
y = iris_data['Species']
y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).to_numpy()
y = pd.get_dummies(y).to_numpy()


np.random.seed(42)
indices = np.random.permutation(len(X))
split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]


X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

weights = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])  # Shape (4, 3)
bias = np.array([0.0,0.0,0.0])

for epoch in range(100):
    n = 0.1
    z = np.dot(X_train, weights) + bias
    y_pred = softmax(z)
    gradient_W = diff_loss_w(y_train, X_train, y_pred, len(X_train))
    gradient_b = diff_loss_b(y_train, y_pred)
    weights -= n * gradient_W
    bias -= n * gradient_b
    print(f"Epoch {epoch+1}/ loss {loss_function(y_pred, y_train)}")

# Evaluate on test data
test_z = np.dot(X_test, weights) + bias
test_y_pred = softmax(test_z)
test_loss = loss_function(test_y_pred, y_test)
test_accuracy = np.argmax(test_y_pred, axis=1)
print(f"test accuracy: {np.mean(test_accuracy == np.argmax(y_test, axis=1))}")
print(f"test loss: {test_loss}")