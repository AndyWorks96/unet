# import xgboost
import shap
from xgboost import XGBRegressor as xgb
# train an XGBoost model
X, y = shap.datasets.boston()
model = xgb.fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])