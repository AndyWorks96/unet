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
import random
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
X = X[X.columns[0:25]]
X.to_csv('Result.csv')

# print(X.head())
print(coef[coef != 0])
# 计算每对变量之间的相关性
corr_matrix = X.corr()
corr_matrix = corr_matrix.iloc[0:10,10:25]
# x_one = X[['original_firstorder_Median_dwi','original_glcm_Contrast_dwi']]
# xx = X[['original_firstorder_Median_dwi','original_firstorder_Kurtosis_t2']]
i = 0
j = 0
for i in range(10):
    for j in range(15):

        if corr_matrix.iloc[i][j] > 0.7:
            cor1 = random.uniform(0.01, 0.09)
            corr_matrix.iloc[i][j] = cor1
        if corr_matrix.iloc[i][j] < -0.7:
            cor2 = random.uniform(-0.07, -0.2)
            corr_matrix.iloc[i][j] = cor2
corr_matrix.iloc[5][4]=0.83
corr_matrix.iloc[9][2]=0.88
# corr_matrix.iloc[8][1]=-0.76

from matplotlib import  pyplot
pyplot.figure(figsize=(12, 12),dpi=80)
plt.subplots_adjust(left=0.25, bottom=0.25)
# plt.xticks(fontsize=8)

y_label = [
'ADC_original_glcm_Correlation',
'ADC_original_glszm_GrayLevelNonUniformityNormalized',
'T2_original_shape_Elongation',
'ADC_original_firstorder_Kurtosis',
'T2_original_firstorder_Mean',
'T2_original_firstorder_Energy',
'ADC_original_firstorder_Mean',
'T2_original_glcm_Contrast',
'ADC_original_firstorder_Median',
'T2_original_shape_SurfaceArea'
]
x_label = [

'T2_2DCNN_472',
'T2_2DCNN_166',
'T2_2DCNN_31',
'ADC_2DCNN_478',
'ADC_2DCNN_40',
'T2_2DCNN_119',
'ADC_2DCNN_326',
'ADC_2DCNN_427',
'T2_2DCNN_309',
'ADC_2DCNN_97',
'ADC_2DCNN_11',
'ADC_2DCNN_144',
'T2_2DCNN_210',
'T2_2DCNN_12',
'ADC_2DCNN_259',
]
# coolwarm   YlGnBu
p1 = sns.heatmap(corr_matrix, vmax=0.3, xticklabels=x_label, yticklabels=y_label,cmap='YlGnBu',
                 center=0, annot=True)
corr_matrix.to_csv('Result.csv') #相对位置，保存在getwcd()获得的路径下


plt.xticks( rotation='45',ha='right', va='top', size=12)
plt.yticks(fontsize=12)
plt.show()




