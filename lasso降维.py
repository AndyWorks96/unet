import pandas as pd
import sklearn
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,Lasso,LassoLarsCV
from sklearn.metrics import accuracy_score,roc_auc_score,mean_squared_error,r2_score
from sklearn.model_selection import RepeatedKFold,train_test_split,LeaveOneOut,GridSearchCV,permutation_test_score,cross_val_score
from sklearn import svm
from matplotlib.pyplot import MultipleLocator
from scipy.stats import pearsonr,ttest_ind,levene
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

xlsx1_filePath = './data/allPatient.xlsx'

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
X = data[data.columns[0:100]]
# xx = data[data.columns[1:1]]
y = data['label']
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X) #new knowledge
X = pd.DataFrame(X)
X.columns = colNames

#LASSO method
alphas = np.logspace(-3,-1.5,50)
print(alphas)
model_lassoCV = LassoCV(alphas = alphas, cv = 4, max_iter = 100000).fit(X,y) #cv, cross-validation

import matplotlib.pyplot as plt

MSEs = (model_lassoCV.mse_path_)

MSEs_mean = np.apply_along_axis(np.mean,1,MSEs)
MSEs_std = np.apply_along_axis(np.std,1,MSEs)

plt.figure() #dpi =300
plt.errorbar(model_lassoCV.alphas_,MSEs_mean ,yerr=MSEs_std,fmt="o",
             ms=3,mfc="r",mec="r", ecolor="lightblue" ,elinewidth=2, capsize=4,capthick=1)
plt.semilogx()
plt.axvline(model_lassoCV.alpha_,color = 'black',ls="--")
plt.xlabel('Lambda')
plt.ylabel('MSE')
ax = plt.gca()
y_major_locator = MultipleLocator
plt.show()


print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_,index = X.columns) #new knowledge
# print(coef)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)))
index = coef[coef!=0].index
X = X[index]
coefs = model_lassoCV.path(X,y,alphas = alphas , cv = 4,max_iter=100000)[1].T
plt.figure()
plt.semilogx(model_lassoCV.alphas_, coefs, '-')
plt.axvline(model_lassoCV.alpha_, color='black', ls="--")
plt.xlabel('Lambda')
plt.ylabel('Cofficients')
plt.show()

x_values = np.arange(len(index))
y_values = coef[coef != 0]
plt.bar(x_values, y_values, color='lightblue', edgecolor='black', alpha=0.8)
plt.xticks(x_values, index, rotation='45', ha='right', va='top')
plt.xlabel("feature")
plt.ylabel("weight")
plt.show()