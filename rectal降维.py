from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import StandardScaler

xlsx1_filePath = './data/xgb_excel_raw_224.xlsx'
xlsx2_filePath = './data/xgb_excel_raw_224.xlsx'
# xlsx2_filePath = 'C:/Users/RONG/Desktop/PythonBasic/data_B.xlsx'
data_1 = pd.read_excel(xlsx1_filePath)
data_2 = pd.read_excel(xlsx2_filePath)
rows_1,__ = data_1.shape

data = data_1
# data = shuffle(data)
data = data.fillna(0)
data = data.loc[(data.sum(axis=1) != 0), (data.sum(axis=0) != 0)]
X = data[data.columns[1:142]]
# xx = data[data.columns[352:352]]
y = data_2['zLabel']
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X) #new knowledge
X = pd.DataFrame(X)
X.columns = colNames

# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
# make_friedman1 =make_friedman1()
# estimator = SVR(kernel="linear")
estimator = RandomForestRegressor(n_estimators=20, random_state=0)
selector = RFE(estimator, n_features_to_select=128, step=10)
selector = selector.fit(X, y)
supp = selector.support_
for i in range(141):
    if selector.support_.data[i] is True:
        print(colNames[i])
print(selector.ranking_)
# ranking = selector.ranking_.reshape(digits.images[0].shape)
# print(ranking)
# plt.matshow(selector.ranking_, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()