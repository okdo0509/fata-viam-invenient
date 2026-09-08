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

train_data = [[1,1],[2,1],[3,4],[6,5],[4,2],[12,25],[11,28],[15,30],[18,37],[19,35]]
train_target = [0,0,0,0,0,1,1,1,1,1]
test_data = [[-10,-10],[2,5],[3,5],[6,10],[4,8],[10,18],[11,25],[8,15],[20,40],[14,16]]
test_target = [0,0,0,0,0,1,1,1,1,1]

weights = np.array([10.0,-3.0])
bias = -45.0
train_data = np.array(train_data)
train_target = np.array(train_target)
test_data = np.array(test_data)
test_target = np.array(test_target)

for epoch in range(10):
    n = 0.1
    m = []
    for i in range(len(train_data)):
        m.append(np.dot(weights, train_data[i]) + bias)
    y_pred = np.array([sigmoid(m[i]) for i in range(len(m))])
    gradient_w = np.mean(np.array([diff_loss_w(train_target[i], train_data[i], y_pred[i]) for i in range(len(train_data))]), axis=0)
    gradient_b = np.mean(np.array([diff_loss_b(train_target[i], y_pred[i]) for i in range(len(train_data))]), axis=0)
    print(f"Epoch {epoch+1}/ loss {loss_function(train_target, y_pred)}")
    weights -= n * gradient_w
    bias -= n * gradient_b

# Evaluate on test data
test_m = [np.dot(weights, test_data[i]) + bias for i in range(len(test_data))]
test_y_pred = [sigmoid(test_m[i]) for i in range(len(test_m))]
test_loss = loss_function(test_target, np.array(test_y_pred))
test_accuracy = np.where(np.array(test_y_pred) >= 0.5, 1, 0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {np.mean(test_accuracy == test_target)}")
print(f"Test Probabilities: {dict(enumerate(test_y_pred))}")