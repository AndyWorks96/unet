import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(0)
xlsx1_filePath = './data/二次降维20.xlsx'

# xlsx2_filePath = 'C:/Users/RONG/Desktop/PythonBasic/data_B.xlsx'
data_1 = pd.read_excel(xlsx1_filePath)
df1 = pd.DataFrame(data_1[data_1.columns[1:21]])
df = pd.DataFrame(np.random.random((100, 5)), columns=["a", "b", "c", "d", "e"])
# 计算每对变量之间的相关性
corr_matrix = df1.corr()
from matplotlib import  pyplot
pyplot.figure(figsize=(10, 10))

plt.xticks( fontweight='bold')
plt.yticks( fontweight='bold')
p1 = sns.heatmap(corr_matrix)

plt.show()
