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
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1)) #shape(N,)

def active_function(x):
    return np.where(x > 0, x, 0.01 * (np.exp(x) - 1)) # Leaky ReLU

def diff_Z_2(y, x, y_pred,N):
    return np.dot(x.T, y_pred - y) / N

def diff_A_1(weights_2, dZ_2):
    return np.dot(dZ_2, weights_2.T)

def diff_Z_1(dA_1, Z_1):
    dZ_1 = dA_1 * np.where(Z_1 > 0, 1, 0.01 * np.exp(Z_1))
    return dZ_1

#데이터 셋 받기
data = pd.read_csv("C:\\Users\\옥유준\\fata-viam-invenient\\phase03_Neural_Network\\vehicle.csv")

X = data[['COMPACTNESS', 'CIRCULARITY', 'DISTANCE CIRCULARITY', 'RADIUS RATIO', 'PR AXIS ASPECT RATIO', 'MAX LENGTH ASPECT RATIO', 'SCATTER RATIO', 'ELONGATEDNESS', 'PR AXISRECTANGULAR', 'LENGTHRECTANGULAR', 'MAJORVARIANCE', 'MINORVARIANCE', 'GYRATIONRADIUS', 'MAJORSKEWNESS', 'MINORSKEWNESS', 'MINORKURTOSIS', 'MAJORKURTOSIS', 'HOLLOWS RATIO']].to_numpy()
Y = data['target'].map({1: 0, 2: 1, 3: 2, 4: 3}).to_numpy()
Y = pd.get_dummies(Y).to_numpy()

np.random.seed(42)
indices = np.random.permutation(len(X))
split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

mean = np.mean(X[train_indices], axis=0)
std = np.std(X[train_indices], axis=0)
X = (X - mean) / std  # Standardize the features

X_train, X_test = X[train_indices], X[test_indices] #shape(N,18)
Y_train, Y_test = Y[train_indices], Y[test_indices] #shape(N,4)

weights_1 = np.random.randn(X_train.shape[1], 10) * 0.01  # Shape (18, 10)
weights_2 = np.random.randn(10, 4) * 0.01  # Shape (10, 4)
bias_1 = np.zeros((1, 10))  # Shape (1, 10)
bias_2 = np.zeros((1, 4))  # Shape (1, 4)

for epoch in range(100):
    n = 1.5
    Z_1 = np.dot(X_train, weights_1) + bias_1 #shape(N,10)
    A_1 = active_function(Z_1) #shape(N,10)
    Z_2 = np.dot(A_1, weights_2) + bias_2 #shape(N,4)
    Y_pred = softmax(Z_2) #shape(N,4)
    Loss = loss_function(Y_pred, Y_train)
    dZ_2 = Y_pred - Y_train #shape(N,4)
    dW_2 = np.dot(A_1.T, dZ_2) / len(X_train)
    db_2 = np.mean(dZ_2, axis=0, keepdims=True) #shape (1, 4)
    dA_1 = np.dot(dZ_2, weights_2.T) #shape(N,10)
    dZ_1 = dA_1 * np.where(Z_1 > 0, 1, 0.01 * np.exp(Z_1)) #shape(N,10)
    dW_1 = np.dot(X_train.T, dZ_1) / len(X_train)
    db_1 = np.mean(dZ_1, axis=0, keepdims=True) #shape (1, 10)
    weights_1 -= n * dW_1
    bias_1 -= n * db_1
    weights_2 -= n * dW_2
    bias_2 -= n * db_2

#    print(f"Epoch {epoch+1}/ loss {Loss}")

test_Z_1 = np.dot(X_test, weights_1) + bias_1
test_A_1 = active_function(test_Z_1)
test_Z_2 = np.dot(test_A_1, weights_2) + bias_2
test_Y_pred = softmax(test_Z_2)
test_loss = loss_function(test_Y_pred, Y_test)
test_accuracy = np.argmax(test_Y_pred, axis=1)

Z_1 = np.dot(X_train, weights_1) + bias_1 #shape(N,10)
A_1 = active_function(Z_1) #shape(N,10)
Z_2 = np.dot(A_1, weights_2) + bias_2 #shape(N,4)
Y_pred = softmax(Z_2) #shape(N,4)

print(f"test accuracy: {np.mean(test_accuracy == np.argmax(Y_test, axis=1))}, train accuracy: {np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_train, axis=1))}")

wrong_prediction = [0, 0, 0, 0]
Actual_class_count = [0, 0, 0, 0] 
for i in range(len(X_test)): # Initialize a list to count wrong predictions for each class
    if np.argmax(test_Y_pred[i]) != np.argmax(Y_test[i]):
        wrong_prediction[np.argmax(test_Y_pred[i])] += 1
        Actual_class_count[np.argmax(Y_test[i])] += 1
        print(f"Predicted: {np.argmax(test_Y_pred[i])}, Actual: {np.argmax(Y_test[i])}")

print("Wrong predictions by class:")
for cls, count in enumerate(wrong_prediction):
    print(f"Class {cls}: {count}")

for cls, count in enumerate(Actual_class_count):
    print(f"Actual Class {cls}: {count}")