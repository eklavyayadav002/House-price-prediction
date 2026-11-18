import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("House Price Prediction (target is in $100,000s)")
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))
print("R2:", round(r2_score(y_test, pred), 3))
print("Example prediction:", round(float(model.predict(X_test.iloc[[0]])[0]), 3))
