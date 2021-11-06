import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import warnings
from sklearn.ensemble import RandomForestRegressor
# from xgboost import plot_importance
from matplotlib import pyplot
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
# from xgboost import XGBRegressor
xlsx1_filePath = './data/二次降维20.xlsx'

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
X = data[data.columns[1:21]]

y = data['pIC50']
colNames = X.columns
X = X.astype(np.float64)
# X = StandardScaler().fit_transform(X) #new knowledge
X = pd.DataFrame(X)
X.columns = colNames
names = colNames
# model = XGBRegressor(min_samples_split=6, n_estimators=100)
rf = RandomForestRegressor()
rf.fit(X, y)
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
# model.fit(X,y)
# plot_importance(model)
# pyplot.show()
# feature_important = pd.Series(model.feature_importances_, index = housing.feature_names).sort_values(ascending=False)
# plt.bar(feature_important.index, feature_important.data)
# plt.show()


# from sklearn.datasets import load_boston



# boston_data = load_boston()
# X = boston_data['data']
# names = boston_data["feature_names"]
# y = boston_data['target']
#     # dir(boston_data) ⇒ 查看其支持的属性为 ['DESCR', 'data', 'feature_names', 'target']
# rf = RandomForestRegressor()
# rf.fit(X, y)
#
#
# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))