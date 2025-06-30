# This project forecasts sales using ARIMA and ML models (Linear Regression, Random Forest, XGBoost), 
# combining time series analysis with feature engineering for accurate short-term predictions.
import numpy as np # Used for numerical operations
import pandas as pd # Used for data manipulation and analysis
import matplotlib.pyplot as plt # Used for plotting graphs
import seaborn as sns # Used for statistical data visualization
from statsmodels.tsa.seasonal import seasonal_decompose # Used for decomposing time series data
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # Used for simple exponential smoothing
from statsmodels.tsa.stattools import adfuller # Used for checking stationarity of time series data
from statsmodels.tsa.arima.model import ARIMA # Used for ARIMA modeling
from pmdarima import auto_arima # Used for automatic ARIMA model selection
from sklearn.metrics import mean_absolute_error, mean_squared_error # Used for evaluating model performance
from sklearn.linear_model import LinearRegression # Used for linear regression modeling
from sklearn.ensemble import RandomForestRegressor # Used for random forest regression
from xgboost import XGBRegressor # Used for XGBoost regression
from sklearn.model_selection import TimeSeriesSplit # Used for 
import joblib # 
import warnings
warnings.filterwarnings("ignore")

# 🔹 Step 1: Load Data
print("\nStep 1: Load Data")
df = pd.read_csv('D:/Study/SkilledScore/Projects/Sales Forecasting with Time Series Analysis/sales_data.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.sort_index()
print(df.head())

# 🔹 Step 2: Explore and Visualize the Data
print("\nStep 2: Visualize")
df_daily = df.resample('D').sum()

plt.figure(figsize=(14, 5))
plt.plot(df_daily.index, df_daily['sales_units'])
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.grid(True)
plt.tight_layout()
plt.show()

# Rolling mean
df_daily['rolling_mean_30'] = df_daily['sales_units'].rolling(window=30).mean()
plt.figure(figsize=(14, 5))
plt.plot(df_daily['sales_units'], label='Daily Sales', alpha=0.5)
plt.plot(df_daily['rolling_mean_30'], label='30-Day Rolling Mean', color='red')
plt.title("Sales with Rolling Trend")
plt.legend()
plt.tight_layout()
plt.show()

# 🔹 Step 3: Decomposition
print("\nStep 3: Decompose Time Series")
decomposition = seasonal_decompose(df_daily['sales_units'], model='additive', period=365)
decomposition.plot()
plt.suptitle("Time Series Decomposition", fontsize=16)
plt.tight_layout()
plt.show()

# Check Stationarity
df_clean = df_daily['sales_units'].dropna()
result = adfuller(df_clean)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# 🔹 Step 4: Forecasting Models (ETS, ARIMA)
print("\nStep 4: Forecast Models")
train = df_clean[df_clean.index < '2025-01-01']

# SES Model
ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(steps=30)

# Auto ARIMA
auto_model = auto_arima(train, seasonal=False, trace=True, suppress_warnings=True)
arima_forecast = auto_model.predict(n_periods=30)
forecast_dates = pd.date_range(start='2025-01-02', periods=30, freq='D')

plt.figure(figsize=(12, 5))
plt.plot(train[-365:], label='Train (Last Year)')
plt.plot(forecast_dates, ses_forecast, label='SES Forecast', color='green')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='orange')
plt.title("30-Day Forecast: SES vs ARIMA")
plt.legend()
plt.tight_layout()
plt.show()

# 🔹 Step 5: Evaluation
print("\nStep 5: Evaluation")
test = df_clean.loc['2024-12-02':'2024-12-31']
train_eval = df_clean[df_clean.index < '2024-12-02']

model_eval = ARIMA(train_eval, order=(1,1,1)).fit()
forecast_eval = model_eval.forecast(steps=30)

mae = mean_absolute_error(test, forecast_eval)
rmse = mean_squared_error(test, forecast_eval) ** 0.5
mape = np.mean(np.abs((test - forecast_eval) / test)) * 100

print(f'MAE:  {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')

plt.figure(figsize=(12, 5))
plt.plot(train_eval[-100:], label='Train (last 100 days)')
plt.plot(test, label='Actual', color='black')
plt.plot(test.index, forecast_eval, label='Forecast', color='red')
plt.title("Evaluation: ARIMA Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.show()

# 🔹 Step 6: Time Series Cross-Validation
print("\nStep 6: Time Series Cross-Validation")
ts_cv = TimeSeriesSplit(n_splits=5)
mae_scores, rmse_scores = [], []

for train_idx, test_idx in ts_cv.split(df_clean):
    train_cv, test_cv = df_clean.iloc[train_idx], df_clean.iloc[test_idx]
    model_cv = ARIMA(train_cv, order=(1, 1, 1)).fit()
    forecast_cv = model_cv.forecast(steps=len(test_cv))
    mae_scores.append(mean_absolute_error(test_cv, forecast_cv))
    rmse_scores.append(np.sqrt(mean_squared_error(test_cv, forecast_cv)))

print("\nTime Series Cross-Validation Results:")
print(f"Average MAE:  {np.mean(mae_scores):.2f}")
print(f"Average RMSE: {np.mean(rmse_scores):.2f}")

# 🔹 Step 7: ML-Based Forecasting
print("\nStep 7: ML-Based Forecasting")
data_ml = pd.DataFrame(df_clean)
data_ml.columns = ['sales_units']

# Feature engineering
data_ml['dayofweek'] = data_ml.index.dayofweek
data_ml['month'] = data_ml.index.month
data_ml['is_weekend'] = data_ml.index.dayofweek >= 5
data_ml['rolling_mean_7'] = data_ml['sales_units'].rolling(7).mean()
data_ml['rolling_std_7'] = data_ml['sales_units'].rolling(7).std()

# Lag features
for lag in range(1, 8):
    data_ml[f'lag_{lag}'] = data_ml['sales_units'].shift(lag)

data_ml.dropna(inplace=True)

# Train-test split
train_ml = data_ml[:-30]
test_ml = data_ml[-30:]

X_train, y_train = train_ml.drop(columns='sales_units'), train_ml['sales_units']
X_test, y_test = test_ml.drop(columns='sales_units'), test_ml['sales_units']

# Check for low-variance features
print("\n🔍 Checking feature variances and removing near-constant features...")
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=1e-4)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)


print("X_train_selected.shape:", X_train_selected.shape)
print("X_test_selected.shape:", X_test_selected.shape)
print("Feature names (train):", X_train.columns)



# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_selected, y_train)
xgb_model = XGBRegressor(n_estimators=100, random_state=42).fit(X_train_selected, y_train)
lr_model = LinearRegression().fit(X_train_selected, y_train)


# Predict
lr_pred = lr_model.predict(X_test_selected)
rf_pred = rf_model.predict(X_test_selected)
xgb_pred = xgb_model.predict(X_test_selected)

# Ensemble prediction
print("\n🔄 Ensemble Prediction")
ensemble_pred = (lr_pred + rf_pred + xgb_pred) / 3


# Evaluation
print("\nXGBoost")
print(f"MAE: {mean_absolute_error(y_test, xgb_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}")

print("\nEnsemble (RF + LR + XGB)")
print(f"MAE: {mean_absolute_error(y_test, ensemble_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_pred)):.2f}")

print("\nML Forecasting:")
print("Linear Regression")
print(f"MAE: {mean_absolute_error(y_test, lr_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
print("Sample LR predictions:", lr_pred[:5])

print("\nRandom Forest")
print(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")

# Plotting
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(lr_pred, label='Linear Regression Forecast', linestyle='--')
plt.plot(rf_pred, label='Random Forest Forecast', linestyle='--')
plt.title('ML-Based Forecasting (Last 30 Days)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(ensemble_pred, label='Ensemble Forecast', linestyle='--')
plt.title('Ensemble ML-Based Forecasting (Last 30 Days)')
plt.legend()
plt.tight_layout()
plt.show()

# Save models
joblib.dump(lr_model, 'linear_regression_model.joblib')
joblib.dump(rf_model, 'random_forest_model.joblib')
joblib.dump(xgb_model, 'xgboost_model.joblib')
joblib.dump(auto_model, 'auto_arima_model.joblib')
print("\u2705 ML models saved!")
