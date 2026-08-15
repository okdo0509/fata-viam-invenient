import numpy as np
import matplotlib.pyplot as plt

def f(num):
    return 1/(1+np.exp(num))
def diff_f_W(num):
    return -f(num)*(1-f(num))
def diff_m_w(y_i,x_i):
    return y_i*(x_i/np.linalg.norm(weights)-(np.dot(weights,x_i)+bias)*weights/(np.linalg.norm(weights)**3))
def diff_m_b(y_i,x_i):
    return (y_i/np.linalg.norm(weights))

class_data = [[1,1],[2,1],[3,4],[6,5],[4,2],[12,25],[11,28],[15,30],[18,37],[19,35]]
class_target = [0,0,0,0,0,1,1,1,1,1]

weights = np.array([10.0,-3.0])
bias = -45.0
class_data = np.array(class_data)
class_target = np.array(class_target)
class_y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])

for epoch in range(10):
    n = 2
    m = []
    for i in range(len(class_data)):
        m.append(class_y[i] * (np.dot(weights, class_data[i]) + bias) / np.linalg.norm(weights))
    gradient_w = np.sum(np.array([diff_f_W(m[i]) * diff_m_w(class_y[i], class_data[i]) for i in range(len(class_data))]), axis=0)
    gradient_b = np.sum(np.array([diff_f_W(m[i]) * diff_m_b(class_y[i], class_data[i]) for i in range(len(class_data))]), axis=0)
    print(f"Epoch {epoch+1}/ loss {np.sum(np.array([f(m[i]) for i in range(len(m))]))/len(m)}")
    weights -= n * gradient_w
    bias -= n * gradient_b