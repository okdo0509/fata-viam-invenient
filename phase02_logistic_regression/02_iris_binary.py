import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(num):
    return 1/(1+np.exp(-num))
def diff_sigmoid(num):
    s = sigmoid(num)
    return s*(1-s)
def loss_function(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return np.mean(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
def diff_loss_w(y_i, x_i, y_pred):
    return (y_pred - y_i) * x_i
def diff_loss_b(y_i, y_pred):
    return (y_pred - y_i)

iris_data = pd.read_csv("C:\\Users\\옥유준\\Downloads\\archive\\Iris.csv")

X = iris_data[['SepalLengthCm', 'SepalWidthCm']].iloc[:100].to_numpy()
y = iris_data['Species'].iloc[:100].apply(lambda x: 1 if x == 'Iris-versicolor' else 0).to_numpy()

np.random.seed(42)
indices = np.random.permutation(len(X))
split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

weights = np.array([100.0, 100.0])
bias = 100.0

for epoch in range(100):
    n = 2
    m = []
    for i in range(len(X_train)):
        m.append(np.dot(weights, X_train[i]) + bias)
    y_pred = np.array([sigmoid(m[i]) for i in range(len(m))])
    gradient_W = np.mean(np.array([diff_loss_w(y_train[i], X_train[i], y_pred[i]) for i in range(len(X_train))]), axis=0)
    gradient_b = np.mean(np.array([diff_loss_b(y_train[i], y_pred[i]) for i in range(len(X_train))]), axis=0)
    print(f"Epoch {epoch+1}/ loss {loss_function(y_train, y_pred)}")
    weights -= n * gradient_W
    bias -= n * gradient_b

# Evaluate on test data
test_m = [np.dot(weights, X_test[i]) + bias for i in range(len(X_test))]
test_y_pred = [sigmoid(test_m[i]) for i in range(len(test_m))]
test_loss = loss_function(y_test, np.array(test_y_pred))
test_accuracy = np.where(np.array(test_y_pred) >= 0.5, 1, 0)
print(f"test accuracy: {np.mean(test_accuracy == y_test)}")