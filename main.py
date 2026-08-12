import numpy as np
import matplotlib.pyplot as plt

clase_data = [[1,1],[2,1],[3,4],[6,5],[4,2],[12,25],[11,28],[15,30],[18,37],[19,35]]
clase_target = [0,0,0,0,0,1,1,1,1,1]

weights = np.array([10,-3])
bias = -45
clase_data = np.array(clase_data)
clase_target = np.array(clase_target)
best_b = 0
best_w0 = 0
best_w1 = 0
min_count = 10
for nb in range(-10,11,1):
    distances = []
    answer = []
    for data in clase_data:
        result = (np.dot(weights, data) + bias+nb) / np.sqrt(np.sum((weights)**2))
        distances.append(result)
        if result > 0:
            answer.append(1)
        else:
            answer.append(0)
    count = np.sum(np.array(answer) != clase_target)
    if count < min_count:
        min_count = count
        best_b = nb

min_count = 10
for nw0 in range(-50,51,1):
    weights = np.array([nw0,-3])
    distances = []
    answer = []
    for data in clase_data:
        result = (np.dot(weights, data) + bias) / np.sqrt(np.sum((weights)**2))
        distances.append(result)
        if result > 0:
            answer.append(1)
        else:
            answer.append(0)
    count = np.sum(np.array(answer) != clase_target)
    if count < min_count:
        min_count = count
        best_nw0 = nw0

min_count = 10
for nw1 in range(-50,51,1):
    weights = np.array([10,nw1])
    distances = []
    answer = []
    for data in clase_data:
        result = (np.dot(weights, data) + bias) / np.sqrt(np.sum((weights)**2))
        distances.append(result)
        if result > 0:
            answer.append(1)
        else:
            answer.append(0)
    count = np.sum(np.array(answer) != clase_target)
    if count < min_count:
        min_count = count
        best_nw1 = nw1

print(bias+best_b,best_nw0, best_nw1)
bias = bias+best_b
distances = []
answer = []
weights = np.array([best_nw0,best_nw1])
for data in clase_data:
    result = (np.dot(weights, data) + bias) / np.sqrt(np.sum((weights)**2))
    distances.append(result)
    if result > 0:
        answer.append(1)
    else:
        answer.append(0)
count = np.sum(np.array(answer) != clase_target)
print(distances)
print(count)

