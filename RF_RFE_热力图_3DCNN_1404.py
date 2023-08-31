from numpy import loadtxt
from sklearn.svm import SVR
# from xgboost import XGBClassifier
import numpy as np
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV,RFE
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import seaborn as sns
data_1 = pd.read_excel('./AllFeaturesFusion1404_960.xlsx')

data_1 = data_1.fillna(0)
data_1 = data_1.loc[(data_1.sum(axis=1) != 0), (data_1.sum(axis=0) != 0)]
# X = data_1[data_1.columns[0:132]]
X = data_1[data_1.columns[0:1543]]

# X = data_1[data_1.columns[0:131]]
y = data_1[data_1.columns[1543]]
colNames = X.columns
X = X.astype(np.float64)

X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
# 固定特征重要性排序
rf = RandomForestClassifier(n_estimators=500,min_samples_leaf=10,n_jobs=-1,random_state=0)
rf.fit(X,y)

# 得到特征重要度分数
importances_values = rf.feature_importances_
importances = pd.DataFrame(importances_values, columns=["importance"])
feature_data = pd.DataFrame(X.columns, columns=["feature"])
importance = pd.concat([feature_data, importances], axis=1)
# 倒叙排序
importance = importance.sort_values(["importance"], ascending=True)
importance["importance"] = (importance["importance"] * 1000).astype(int)
importance = importance.sort_values(["importance"])
fea_15 = importance.iloc[319:339]
# importance.iloc[319:339].plot(kind="barh",figsize=(10,8),x="feature", y="importance",
                 # legend = False,color='#2E8AFB')

#  #2E8AFB   #576D8D
# lll = fea_15.iloc[337:339]
# print(fea_15[0])
# importance.set_index('feature', inplace=True)
# importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))

# plt.margins(0.2)
# plt.subplots_adjust(left=0.38)

# # 去掉上、右边框
# ax=plt.gca()  #gca:get current axis得到当前轴
# #设置图片的右边框和上边框为不显示
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# # 隐藏刻度线
# plt.tick_params(left=False)
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlabel("Relative feature importance", fontsize=20)
# plt.ylabel("")
#
# plt.title("Predictive power of top 20 fused features", fontsize=20)
# plt.grid(b=True, axis='x')

# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

plt.show()


print("Features sorted by their score:")
feature_importances = sorted(zip(X.columns, map(lambda x:round(x,4), rf.feature_importances_)),key=lambda x: x[1],reverse=True)
# print(sorted(zip(map(lambda x:round(x,4),rf.feature_importances_),X.columns)))

print(sorted(zip(X.columns, map(lambda x:round(x,4), rf.feature_importances_)),key=lambda x: x[1],reverse=True))


# kf = KFold(n_splits=4)
# kf = StratifiedKFold(n_splits=5)
kf = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=0)

kf.get_n_splits(X)
# kf.get_n_splits(y)
averagePre = 0
random_state = np.random.RandomState(0)
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
box_array = []


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


rf = RandomForestClassifierWithCoef(n_estimators=500, min_samples_leaf=5, n_jobs=-1)
rfecv = RFECV(estimator=rf, step=1, scoring='accuracy', cv=2,min_features_to_select=120)
selector = rfecv.fit(X, y)

# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=5, step=1)
# selector = selector.fit(X, y)
# print(selector.suppot_)
# print(selector.ranking_)

print('RFECV 选择出的特征个数 ：', rfecv.n_features_)  # RFECV选择的特征个数
print('特征优先级 ： ', rfecv.ranking_)

X = rfecv.transform(X)
X = pd.DataFrame(X)
# X.columns = colNames
fig = plt.figure()
corr_matrix = corr_matrix.iloc[0:9,9:20]
# corr_matrix = X.corr()
p1 = sns.heatmap(corr_matrix, xticklabels=[], yticklabels=[],cmap='YlGnBu',
                 center=0, annot=False)
plt.xticks( rotation='45',ha='right', va='top', size=1)
plt.yticks(fontsize=1)
plt.show()
# fig.savefig('./')