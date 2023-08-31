import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(0)
xlsx1_filePath = './FeaturesFusion20.xlsx'

# xlsx2_filePath = 'C:/Users/RONG/Desktop/PythonBasic/data_B.xlsx'
data_1 = pd.read_excel(xlsx1_filePath)
df1 = pd.DataFrame(data_1[data_1.columns[0:20]])
df = pd.DataFrame(np.random.random((100, 5)), columns=["a", "b", "c", "d", "e"])
# 计算每对变量之间的相关性
corr_matrix = df1.corr()
from matplotlib import  pyplot
pyplot.figure(figsize=(10, 10),dpi=80)
plt.subplots_adjust(left=0.25, bottom=0.35)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
cc = corr_matrix.columns[0:2]
p1 = sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,cmap='YlGnBu', center=0, annot=True)
# p1 = sns.heatmap(corr_matrix, xticklabels=[], yticklabels=[],cmap='YlGnBu', center=0, annot=True)

plt.show()
