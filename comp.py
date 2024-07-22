import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


file_path = '/Users/ilaydagurocak/Desktop/kaggle/train.csv'

dtype_dict = {
    'merchant_id': 'category',
    'month_id': 'category',
    'merchant_source_name': 'category',
    'settlement_period': 'category',
    'working_type': 'category',
    'mcc_id': 'category',
    'merchant_segment': 'category',
    'net_payment_count': 'int32'
}
columns_to_use = list(dtype_dict.keys())


chunk_size = 50000


processed_df = pd.DataFrame()
for chunk in pd.read_csv(file_path, usecols=columns_to_use, dtype=dtype_dict, chunksize=chunk_size):
    chunk['merchant_source_name'] = chunk['merchant_source_name'].astype('category').cat.codes
    chunk['settlement_period'] = chunk['settlement_period'].astype('category').cat.codes
    chunk['working_type'] = chunk['working_type'].astype('category').cat.codes
    chunk['mcc_id'] = chunk['mcc_id'].astype('category').cat.codes
    chunk['merchant_segment'] = chunk['merchant_segment'].astype('category').cat.codes
    processed_df = pd.concat([processed_df, chunk])

processed_df['year'] = processed_df['month_id'].apply(lambda x: int(x[:4]))
processed_df['month'] = processed_df['month_id'].apply(lambda x: int(x[4:]))


X = processed_df.drop(['net_payment_count', 'merchant_id', 'month_id'], axis=1)
y = processed_df['net_payment_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tatil_gunleri = {
    '20240101': 1,  # Yılbaşı (1 gün)
    '20240114': 0.5,  # Kurtuluş Günü (0.5 gün)
    '20240409': 3.5,  # Ramazan Bayramı (3.5 gün)
    '20240423': 1,  # Ulusal Egemenlik ve Çocuk Bayramı (1 gün)
    '20240501': 1,  # Emek ve Dayanışma Günü (1 gün)
    '20240519': 1,  # Atatürk'ü Anma, Gençlik ve Spor Bayramı (1 gün)
    '20240615': 4.5,  # Kurban Bayramı (4.5 gün)
    '20240715': 1,  # Demokrasi ve Milli Birlik Günü (1 gün)
    '20240830': 1,  # Zafer Bayramı (1 gün)
    '20241028': 1.5,  # Cumhuriyet Bayramı (1.5 gün)
}


X_train.index = pd.to_datetime(X_train.index)
X_test.index = pd.to_datetime(X_test.index)

X_train['is_holiday'] = X_train.index.strftime('%Y%m%d').isin(tatil_gunleri.keys()).astype(int)
X_test['is_holiday'] = X_test.index.strftime('%Y%m%d').isin(tatil_gunleri.keys()).astype(int)


rf = RandomForestRegressor(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
predictions_rf = best_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, predictions_rf)
print(f"Optimized Random Forest Test MAE with Holidays: {mae_rf}")

xgb = XGBRegressor(random_state=42)
param_grid_xgb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
predictions_xgb = best_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
print(f"Optimized XGBoost Test MAE with Holidays: {mae_xgb}")
