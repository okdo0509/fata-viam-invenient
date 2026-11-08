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

n_1 = 1  # Learning rate for the first layer
n_2 = 1  # Learning rate for the second layer
pre_Loss = float('inf')  # Initialize previous loss to infinity
pre_weights_1 = weights_1.copy()
pre_gradient_W_1 = np.zeros_like(weights_1)
pre_weights_2 = weights_2.copy()
pre_gradient_W_2 = np.zeros_like(weights_2)
pre_Efficent_1 = 0
pre_Efficent_2 = 0
pre_seta_1 = 0
pre_seta_2 = 0
pre_n_1 = n_1
pre_n_2 = n_2
Tau = 0.31
n_min_1 = 0.7
n_min_2 = 0.7

for epoch in range(100):
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

    if epoch > 0:
        delta_W_1 = weights_1 - pre_weights_1
        delta_W_2 = weights_2 - pre_weights_2
        delta_gradient_W_1 = dW_1 - pre_gradient_W_1
        delta_gradient_W_2 = dW_2 - pre_gradient_W_2
        r_L = (-Loss + pre_Loss) / pre_Loss
        k_1 = np.dot(delta_W_1.flatten(), delta_gradient_W_1.flatten()) / np.linalg.norm(delta_W_1)**2
        k_2 = np.dot(delta_W_2.flatten(), delta_gradient_W_2.flatten()) / np.linalg.norm(delta_W_2)**2
        r_seta_1 = np.linalg.norm(delta_W_1) / (np.linalg.norm(pre_weights_1) + 1e-8)
        r_seta_2 = np.linalg.norm(delta_W_2) / (np.linalg.norm(pre_weights_2) + 1e-8)
        Efficent_1 = r_L/ (r_seta_1 + 1e-8)
        Efficent_2 = r_L/ (r_seta_2 + 1e-8)
        delta_r_L = Loss - pre_Loss
        delta_r_seta_1 = r_seta_1 - pre_seta_1
        delta_r_seta_2 = r_seta_2 - pre_seta_2
        delta_n_1 = np.log(n_1) - np.log(pre_n_1)
        delta_n_2 = np.log(n_2) - np.log(pre_n_2)
        direction_1 = np.sign(delta_n_1)
        direction_2 = np.sign(delta_n_2)

        q_t_1 = 0
        q_t_2 = 0

        if pre_Efficent_1 is not None and pre_Efficent_2 is not None:
            delta_E_1 = Efficent_1 - pre_Efficent_1
            delta_E_2 = Efficent_2 - pre_Efficent_2
            if (abs(Efficent_1) + abs(pre_Efficent_1)) > 1e-8:
                q_t_1 = direction_1 * delta_E_1 / ((abs(Efficent_1) + abs(pre_Efficent_1)) + 1e-8)
            if (abs(Efficent_2) + abs(pre_Efficent_2)) > 1e-8:
                q_t_2 = direction_2 * delta_E_2 / ((abs(Efficent_2) + abs(pre_Efficent_2)) + 1e-8)

        if r_L > 0 and Efficent_1 > 0.005:   
            proposal_n_1 = n_1 * np.exp(q_t_1) * 1.05
        elif r_L < 0 and Efficent_1 < -0.01:
            proposal_n_1 = n_1 * n_min_1
        else:
            proposal_n_1 = n_1
        if r_L > 0 and Efficent_2 > 0.005:
            proposal_n_2 = n_2 * np.exp(q_t_2) * 1.05
        elif r_L < 0 and Efficent_2 < -0.01:
            proposal_n_2 = n_2 * n_min_2
        else:
            proposal_n_2 = n_2

        cur_1 = 2/k_1 * 0.5  if k_1 > 0 else float("inf")
        cur_2 = 2/k_2 * 0.5  if k_2 > 0 else float("inf")

        r_U_1 = n_1 * np.linalg.norm(dW_1) / (np.linalg.norm(weights_1)+1e-8)
        r_U_2 = n_2 * np.linalg.norm(dW_2) / (np.linalg.norm(weights_2)+1e-8)

        n_update_1 = Tau * (np.linalg.norm(weights_1)+1e-8) / (np.linalg.norm(dW_1)+1e-8)
        n_update_2 = Tau * (np.linalg.norm(weights_2)+1e-8) / (np.linalg.norm(dW_2)+1e-8)

        s_1 = min(1, n_update_1/proposal_n_1, cur_1/proposal_n_1)
        s_2 = min(1, n_update_2/proposal_n_2, cur_2/proposal_n_2)
        
        pre_n_1 = n_1
        pre_n_2 = n_2
        actual_n_1 = min(proposal_n_1, n_update_1, cur_1)
        actual_n_2 = min(proposal_n_2, n_update_2, cur_2)

        cosine_1 = np.dot(dW_1.flatten().T,pre_gradient_W_1.flatten())/(np.linalg.norm(dW_1)*np.linalg.norm(pre_gradient_W_1))
        cosine_2 = np.dot(dW_2.flatten().T,pre_gradient_W_2.flatten())/(np.linalg.norm(dW_2)*np.linalg.norm(pre_gradient_W_2))
        
        pre_weights_1 = weights_1.copy()
        pre_gradient_W_1 = dW_1.copy()
        pre_weights_2 = weights_2.copy()
        pre_gradient_W_2 = dW_2.copy()
        pre_Loss = Loss.copy()
        pre_seta_1 = r_seta_1.copy()
        pre_seta_2 = r_seta_2.copy()
        pre_Efficent_1 = Efficent_1.copy()
        pre_Efficent_2 = Efficent_2.copy()

        weights_1 -= (actual_n_1) * dW_1
        bias_1 -= (actual_n_1) * db_1
        weights_2 -= (actual_n_2) * dW_2
        bias_2 -= (actual_n_2) * db_2

        if s_1 < 1:
            n_1 = n_1
        else:
            n_1 = proposal_n_1
        if s_2 < 1:
            n_2 = n_2
        else:
            n_2 = proposal_n_2

        pre_n_1 = n_1
        pre_n_2 = n_2

        print(f"Epoch {epoch+1}/100 loss {Loss}")
    else:
        pre_weights_1 = weights_1.copy()
        pre_gradient_W_1 = dW_1.copy()
        pre_weights_2 = weights_2.copy()
        pre_gradient_W_2 = dW_2.copy()
        pre_Loss = Loss

        weights_1 -= n_1 * dW_1
        bias_1 -= n_1 * db_1
        weights_2 -= n_2 * dW_2
        bias_2 -= n_2 * db_2


        


test_Z_1 = np.dot(X_test, weights_1) + bias_1
test_A_1 = active_function(test_Z_1)
test_Z_2 = np.dot(test_A_1, weights_2) + bias_2
test_Y_pred = softmax(test_Z_2)
test_loss = loss_function(test_Y_pred, Y_test)
test_accuracy = np.argmax(test_Y_pred, axis=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {np.mean(test_accuracy == np.argmax(Y_test, axis=1))}")

