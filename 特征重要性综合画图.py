# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
from sklearn.linear_model import LassoCV
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False # 解决坐标轴的负号显示问题
## 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import *
from sklearn.tree import *
import warnings
warnings.filterwarnings('ignore')

## 数据导入
traindf = pd.read_excel("./data/Molecular_Descriptor.xlsx",sheet_name="training")

## 导入数据的待预测值
traindfy = pd.read_excel("./data/ERα_activity.xlsx",sheet_name="training")

train_x = traindf.iloc[:,1:]
# train_x = traindf.iloc[]
# train_y = traindfy
train_y = traindfy.iloc[:,2]

# train_y.plot(kind = "hist",bins = 50,figsize = (10,6))
# plt.show()

## 训练数据的标准化处理
train_xs = StandardScaler().fit_transform(train_x)
# train_xs[0:5,:]

## 可视化其中的几个变量与因变量Y的散点图，看一下数据的分布
train_xname = train_x.columns.values
# plt.figure(figsize=(50,50))

# for ii in np.arange(20):
#     plt.subplot(8,8,ii+1)
#     plt.scatter(train_xs[:,ii],train_y)
#     plt.title(train_xname[ii])
# plt.tight_layout()
# plt.show()

## 数据的分布情况各种各样

## 计算每个字变量与Y的相关系数，对数据进行分析
varcorr = []
for ii in range(len(train_xname)):
    corrii = np.corrcoef(train_xs[:,ii],train_y.values)
    varcorr.append(corrii[0,1])

## 条形图可视化
varcorrdf = pd.DataFrame({"varname":train_xname,"mycorr":varcorr})
# varcorrdf.head()
print(varcorrdf)

varcorrdf.isna().sum()   ## 出现了大约225个相关系数的缺失值，分析为什么

varcorrdf.varname[varcorrdf.mycorr.isna()]

traindf[varcorrdf.varname[varcorrdf.mycorr.isna()]].apply(np.unique)  ## 这些列变量的数据的取值都只有0
## 可以首先判断出这225列的变量是不重要的可以直接剔除

## 提出不重要的225个变量
index = varcorrdf.varname[~varcorrdf.mycorr.isna()].index.values
train_xs = train_xs[:,index]

train_xname = train_xname[index]

# 1*10 -4
alphas = np.logspace(-3, -2,100)
# print(alphas)
model_lassocv = LassoCV(alphas = alphas,cv=5,max_iter=100000,selection= 'random').fit(train_xs,train_y)
coef = pd.Series(model_lassocv.coef_,index=train_xname)
# print(coef)

print(model_lassocv.alpha_)
print("Lasso picked " + str(sum(coef!=0))+" variables and eliminated the other "+str(sum(coef==0)))

index = coef[coef != 0].index

X = train_xs[index]


train_xname[1:5]

## 再次通过可视化检查数据
plt.figure(figsize=(20,20))

# for ii in np.arange(64):
#     plt.subplot(8,8,ii+1)
#     plt.scatter(train_xs[:,ii],train_y)
#     plt.title(train_xname[ii])
# plt.tight_layout()
# plt.show()

varcorrdf.mycorr.plot(kind = "hist",bins = 30,figsize = (12,7))
# plt.show()

## 相关系数的分布大部分在0的附近


## 基于统计方法选择

## （1）删除低方差的特征
# from sklearn.feature_selection import VarianceThreshold
# VTH = VarianceThreshold(threshold = 1)
# train_xs = VTH.fit_transform(train_xs)
# print(train_xs.shape)
## 挑出了166个

## 选择K个最高得分的变量，回归可使用mutual_info_regression等
# from sklearn.feature_selection import SelectKBest, mutual_info_regression
# ## 通过方差分析的F值选择K个变量
# KbestF = SelectKBest(mutual_info_regression, k=50)
# train_xs = KbestF.fit_transform(train_xs,train_y.values)
# print(train_xs.shape)

# train_xname[KbestF.get_support()]   ## 选择出的较重要特征
## 该种特征选择方法还有其他的方式可以使用

## 基于机器学习的方法

## 这里我们使用基于随机森林回归模型的特征选择方法

## 根据特征的重要性权重选择特征
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

## 利用随机森林模型进行特征的选择
rfc = RandomForestRegressor(n_estimators=100,random_state=0)
rfc = rfc.fit(train_xs,train_y.values) # 使用模型拟合数据
## 定义从模型中进行特征选择的选择器
sfm = SelectFromModel(estimator=rfc, ## 进行特征选择的模型
                      prefit = True, ## 对模型进行预训练
                      max_features = 30,##选择的最大特征数量
                     )
## 将模型选择器作用于数据特征
sfm_train_x = sfm.transform(train_xs)
print(sfm_train_x.shape)

train_xname[sfm.get_support()]

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
## 将数据切分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(
    train_xs,train_y.values,test_size = 0.2,random_state = 10)
print("X_train.shape :",X_train.shape)
print("X_test.shape :",X_test.shape)

## 使用随机森林进行回归模型的建立
rfr1 = RandomForestRegressor(n_estimators=100,random_state = 1)
rfr1 = rfr1.fit(X_train,y_train)
## 计算在训练和和测试集上的预测均方根误差
rfr1_lab = rfr1.predict(X_train)
rfr1_pre = rfr1.predict(X_test)
print("训练数据集上的均方根误差:",mean_squared_error(y_train,rfr1_lab))
print("测试数据集上的均方根误差:",mean_squared_error(y_test,rfr1_pre))

print("训练数据集上的均方根误差:",mean_absolute_error(y_train,rfr1_lab))
print("测试数据集上的均方根误差:",mean_absolute_error(y_test,rfr1_pre))

## 针对该模型还有很多参数可以调整，从而可以获得更好的预测效果
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus']=False
## 可视化出在训练集和测试集上的预测效果
plt.figure(figsize=(16,8))
plt.subplot(1,2,1) ## 训练数据结果可视化
rmse = round(mean_squared_error(y_train,rfr1_lab),4)
index = np.argsort(y_train)
plt.plot(np.arange(len(index)),y_train[index],"red",
         linewidth=1.8, label = "原数据样本")
plt.plot(np.arange(len(index)),rfr1_lab[index],"bo",
         markersize=2.6,label = "预测结果")
plt.text(200,8,s = "均方根误差值:"+str(rmse))
plt.legend()
plt.grid()
plt.xlabel("索引")
plt.ylabel("Label")
plt.title("回归模型-训练集")

plt.subplot(1,2,2)   ## 测试数据结果可视化
rmse = round(mean_squared_error(y_test,rfr1_pre),4)
index = np.argsort(y_test)
plt.plot(np.arange(len(index)),y_test[index],"red",
         linewidth=1.8, label = "原数据样本")
plt.plot(np.arange(len(index)),rfr1_pre[index],"bo",
         markersize=2.6,label = "预测结果")
plt.text(50,8,s = "均方根误差值:"+str(rmse))
plt.legend()
plt.grid()
plt.xlabel("索引")
plt.ylabel("Label")
plt.title("回归模型-测试集")
plt.tight_layout()
plt.show()

## 从图像上看，对因变量的预测效果还是挺不错的


## 使用条形图可视化每个变量的重要性
importances = pd.DataFrame({"feature":train_xname,
                            "importance":rfr1.feature_importances_})
importances = importances.sort_values("importance",ascending = True)
importances.iloc[0:40,:].plot(kind="barh",figsize=(10,10),x = "feature",y = "importance",
                 legend = False,color='b')
plt.xlabel("特征得分")
plt.ylabel("")
plt.title("特征重要性排序")
plt.grid()
plt.show()

## 重要性得分较大的差不多也就20个左右，也一定程度上说明的出题方要求的那20个自变量也不是随便写的。