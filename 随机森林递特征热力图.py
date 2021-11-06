from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
# svc = SVC(kernel="linear", C=1)
rf = RandomForestRegressor(n_estimators=20,random_state=0)
rfe = RFE(estimator=rf, n_features_to_select=5, step=1)
rfe.fit(X, y)
# supp = rfe.support_.
# print(rfe.fit(X,y).fea)
d1 = digits.images[0]
dd = digits.images[0].shape
ranking = rfe.ranking_.reshape(digits.images[0].shape)
print(rfe.ranking_[1])
# ranking = rfe.ranking_.reshape(8*8)
# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()