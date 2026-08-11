import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clase_data = [[1,1],[2,1],[3,4],[6,5],[4,2],[12,25],[11,28],[15,30],[18,37],[19,35]]
clase_target = [0,0,0,0,0,1,1,1,1,1]

clase_data = np.array(clase_data)
clase_target = np.array(clase_target)

plt.scatter(clase_data[:, 0], clase_data[:, 1], c=clase_target, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Clase Data')
plt.show()