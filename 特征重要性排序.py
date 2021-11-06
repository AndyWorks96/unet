import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')
xlsx1_filePath = './data/Molecular_Descriptor(1).xlsx'

# xlsx2_filePath = 'C:/Users/RONG/Desktop/PythonBasic/data_B.xlsx'
data_1 = pd.read_excel(xlsx1_filePath)
# data_2 = pd.read_excel(xlsx2_filePath)
rows_1,__ = data_1.shape
# rows_2,__ = data_2.shape
# data_1.insert(0,'label',[0]*rows_1)
# data_2.insert(0,'label',[1]*rows_2)
# data = pd.concat([data_1,data_2])
data = data_1
data = shuffle(data)
data = data.fillna(0)
X = data[data.columns[1:]]
xx = data[data.columns[1:1]]
y = data['label']
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X) #new knowledge
X = pd.DataFrame(X)
X.columns = colNames

#LASSO method
alphas = np.logspace(-5,-2,50)
print(alphas)
model_lassoCV = LassoCV(alphas = alphas, cv = 10, max_iter = 100000).fit(X,y) #cv, cross-validation

print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_,index = X.columns) #new knowledge
# print(coef)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)))

