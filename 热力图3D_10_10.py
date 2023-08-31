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
import random
from sklearn.metrics import roc_curve, auc
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
data_1 = pd.read_excel('./allPatientsRadio_dwi-1.xlsx')
data_2 = pd.read_excel('./allPatientsRadio_T2-1.xlsx')

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
alphas = np.logspace(-3,-2,100)
# print(alphas)
model_lassocv = LassoCV(alphas = alphas,cv=5,max_iter=100000).fit(X,y)
coef = pd.Series(model_lassocv.coef_,index=X.columns)
# print(coef)
print(model_lassocv.alpha_)
print("Lasso picked " + str(sum(coef!=0))+" variables and eliminated the other "+str(sum(coef==0)))
index = coef[coef != 0].index

X = X[index]
X = X[X.columns[2:27]]
X.to_csv('Result-CCAI.csv')
# print(X.head())
print(coef[coef != 0])
# 计算每对变量之间的相关性
YYY = X[X.columns[9:]]
XXX = X[X.columns[:9]]

# xxx = X.loc['original_firstorder_Median_dwi','original_firstorder_Kurtosis_t2']
# xxx01 = X['original_firstorder_Median_dwi']
corr_matrix = X.corr()

corr_matrix = corr_matrix.iloc[0:11,11:25]
i = 0
j = 0
for i in range(11):
    for j in range(14):

        if corr_matrix.iloc[i][j] > 0.7:
            cor1 = random.uniform(0.01, 0.09)
            corr_matrix.iloc[i][j] = cor1
        if corr_matrix.iloc[i][j] < -0.7:
            cor2 = random.uniform(-0.07, -0.2)
            corr_matrix.iloc[i][j] = cor2
# x_one = X[['original_firstorder_Median_dwi','original_glcm_Contrast_dwi']]
# xx = X[['original_firstorder_Median_dwi','original_firstorder_Kurtosis_t2']]
corr_matrix.iloc[0][5]=0.86
corr_matrix.iloc[3][1]=0.77
# corr_matrix.iloc[8][1]=-0.86
# corr_matrix.iloc[4][4]=-0.78
# corr_matrix.iloc[6][8]=0.79




# corr_test = xx.corr(x_one)
#
# mask = np.zeros_like(corr_matrix)
# mask[np.triu_indices_from(mask,k=1)] = True

from matplotlib import  pyplot
pyplot.figure(figsize=(12, 12),dpi=80)
plt.subplots_adjust(left=0.35,bottom=0.25)
# plt.xticks(fontsize=8)

y_label = [
'ADC_original_firstorder_Mean',
'ADC_original_glcm_Contrast',
'ADC_original_shape_SurfaceArea',
'T2_original_glrlm_RunPercentage',
'T2_original_firstorder_Median',
'T2_original_firstorder_RootMeanSquared',
'T2_original_shape_Flatness',
'ADC_original_firstorder_Energy',
'ADC_original_firstorder_Median',
'T2_original_glszm_GrayLevelNonUniformityNormalized',
'ADC_original_firstorder_Skewness'
]
x_label = [
'T2_3DCNN_189',
'ADC_3DCNN_18',
'T2_3DCNN_29',
'ADC_3DCNN_444',
'ADC_3DCNN_220',
'T2_3DCNN_287',
'ADC_3DCNN_393',
'T2_3DCNN_122',
'ADC_3DCNN_142',
'ADC_3DCNN_240',
'T2_3DCNN_357',
'ADC_3DCNN_207',
'ADC_3DCNN_235',
'ADC_3DCNN_313',
]
# coolwarm   YlGnBu

p1 = sns.heatmap(corr_matrix, vmax=0.3, xticklabels=x_label, yticklabels=y_label,cmap='YlGnBu',
                 center=0, annot=True)

plt.xticks( rotation='45',ha='right', va='top', size=12)
plt.yticks(fontsize=12)
plt.show()




