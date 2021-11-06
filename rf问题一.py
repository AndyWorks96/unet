from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
# make_friedman1 =make_friedman1()
# estimator = SVR(kernel="linear")
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
estimator = RandomForestRegressor(n_estimators=20,random_state=0)
selector = RFE(estimator, n_features_to_select=30, step=1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
ranking = selector.ranking_.reshape(digits.images[0].shape)
print(ranking)
# plt.matshow(selector.ranking_, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()