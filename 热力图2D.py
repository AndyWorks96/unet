# from xgboost import XGBClassifier
from sklearn.linear_model import LassoCV
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.metrics import roc_curve, auc
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
data_1 = pd.read_excel('./allPatientsRadio_dwi.xlsx')
data_2 = pd.read_excel('./allPatientsRadio_T2.xlsx')

data_1 = data_1.fillna(0)
X = data_1[data_1.columns[0:100]]
# y = data_1[data_1.columns[703:704]]
y = data_1['zLabel']
X_2 = data_2[data_2.columns[0:100]]
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

colNames2 = X_2.columns
X_2 = X_2.astype(np.float64)
X_2 = StandardScaler().fit_transform(X_2)
X_2 = pd.DataFrame(X_2)
X_2.columns = colNames2
X = pd.concat([X, X_2], axis=1)
d = {**dict.fromkeys(X.select_dtypes(np.number).columns, 0),
     **dict.fromkeys(X.select_dtypes(exclude=np.number).columns, 'tmp')}

X = X.fillna(d)

kf = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=0)

kf.get_n_splits(X)
# kf.get_n_splits(y)
averagePre = 0
random_state = np.random.RandomState(0)
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
box_array = []
# 1*10 -4
alphas = np.logspace(-3,-1.6,100)
# print(alphas)
model_lassocv = LassoCV(alphas = alphas,cv=5,max_iter=100000).fit(X,y)
coef = pd.Series(model_lassocv.coef_,index=X.columns)
# print(coef)
print(model_lassocv.alpha_)
print("Lasso picked " + str(sum(coef!=0))+" variables and eliminated the other "+str(sum(coef==0)))
index = coef[coef != 0].index

X = X[index]
# X = X[X.columns[9:]]
# print(X.head())
print(coef[coef != 0])
# 计算每对变量之间的相关性

corr_matrix = X.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask,k=1)] = True

from matplotlib import  pyplot
pyplot.figure(figsize=(10, 10),dpi=80)
plt.subplots_adjust(left=0.25, bottom=0.35)
# plt.xticks(fontsize=8)






x_label = ['T2_original_shape_Flatness',
'T2_3DCNN_287',
'ADC_3DCNN_444',
'T2_original_firstorder_Median',
'ADC_original_gldm_DependenceEntropy',
'ADC_original_gldm_HighGrayLevelEmphasis',
'ADC_original_firstorder_Energy',
'T2_original_shape_Maximum2DDiameterColumn',
'T2_3DCNN_384',
'ADC_original_glrlm_RunLengthNonUniformityNormalized',
'T2_3DCNN_326',
'ADC_3DCNN_52',
'ADC_3DCNN_221',
'ADC_3DCNN_220',
'ADC_3DCNN_18',
'ADC_original_glszm_SmallAreaHighGrayLevelEmphasis',
'ADC_original_shape_SurfaceArea',
'T2_3DCNN_189',
'ADC_3DCNN_142',
'ADC_3DCNN_235']
p1 = sns.heatmap(corr_matrix,mask=mask, vmax=0.3, xticklabels=x_label, yticklabels=x_label,cmap='YlGnBu',
                 center=0, annot=True)
plt.xticks( rotation='45',ha='right', va='top', size=10)
plt.yticks(fontsize=10)
plt.show()


# import numpy as np
#
# # 随机生成一个200行10列的数据集
# data_new = np.random.randn(200,10)
# #求出这个数据集的相关系数矩阵
# corr = np.corrcoef(data_new,rowvar=False)
# #以corr的形状生成一个全为0的矩阵
# mask = np.zeros_like(corr)
# #将mask的对角线及以上设置为True
# #这部分就是对应要被遮掉的部分
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#     sns.heatmap(corr, mask=mask, vmax=0.3, annot=True,cmap="RdBu_r")

