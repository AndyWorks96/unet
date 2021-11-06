from numpy import loadtxt
from xgboost import XGBClassifier
r
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_excel('./data/allPatient.xlsx')
print(data)
