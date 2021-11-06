from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100,random_state=0)
cross_val_score(regressor, boston.data, boston.target, cv=10,scoring = "neg_mean_squared_error")
# sorted(sklearn.metrics.SCORERS.keys())#sklearn当中的模型评估指标（打分）列表
# scoring='neg_mean_squared_error'表示返回负的均方误差，如果不加上这个参数，返回的则是R**2
