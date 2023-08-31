from numpy import loadtxt
# from xgboost import XGBClassifier
import numpy as np
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoCV
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import svm
from matplotlib.pyplot import MultipleLocator

data_1 = pd.read_excel('./AllFeaturesFusion200_960.xlsx')

data_1 = data_1.fillna(0)
data_1 = data_1.loc[(data_1.sum(axis=1) != 0), (data_1.sum(axis=0) != 0)]
# X = data_1[data_1.columns[0:132]]
X = data_1[data_1.columns[0:339]]

# X = data_1[data_1.columns[0:131]]
y = data_1[data_1.columns[339]]
colNames = X.columns
X = X.astype(np.float64)

X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
# kf = KFold(n_splits=4)
# kf = StratifiedKFold(n_splits=5)
kf = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=0)

kf.get_n_splits(X)
# kf.get_n_splits(y)
averagePre = 0
random_state = np.random.RandomState(0)
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
box_array = []
# 1*10 -4
alphas = np.logspace(-3, -2,100)
# print(alphas)
model_lassocv = LassoCV(alphas = alphas,cv=5,max_iter=100000,selection= 'random').fit(X,y)
coef = pd.Series(model_lassocv.coef_,index=X.columns)
# print(coef)
print(model_lassocv.alpha_)
print("Lasso picked " + str(sum(coef!=0))+" variables and eliminated the other "+str(sum(coef==0)))
index = coef[coef != 0].index

X = X[index]
# print(X.head())
print(coef[coef != 0])

df1 = pd.DataFrame(X[X.columns[0:34]])
# df = pd.DataFrame(np.random.random((100, 5)), columns=["a", "b", "c", "d", "e"])
# 计算每对变量之间的相关性
corr_matrix = df1.corr()
from matplotlib import  pyplot
pyplot.figure(figsize=(10, 10))

plt.xticks( fontweight='bold')
plt.yticks( fontweight='bold')
p1 = sns.heatmap(corr_matrix)

plt.show()