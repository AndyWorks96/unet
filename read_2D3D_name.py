import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(0)
xlsx1_filePath = './排序特征2D+3D.xlsx'

data_1 = pd.read_excel(xlsx1_filePath)
d1 = data_1['Radiomics of 3D-CNN'].T
d1 = d1[0:54]
d1 = [d1]
print(d1)