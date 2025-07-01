import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from xgboost import XGBRegressor

class WeatherPredictor:
    def __init__(self):
        self.temp_model = None
        self.precip_model = None
        self.rain_classifier = None
        self.rain_regressor = None
        self.models_dir = 'models'
        self.temp_features = None
        self.precip_features = None
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def prepare_data(self, df):
        # Chỉ sử dụng các cột có sẵn trong dataset thực tế
        feature_cols = [
            'rain (mm)', 'snowfall (cm)', 
            'snow_depth (m)', 'weather_code (wmo code)', 'is_day ()',
            'relative_humidity_2m (%)', 'dew_point_2m (°C)', 'cloud_cover (%)',
            # Thêm các feature khí tượng quan trọng cho lượng mưa
            'pressure_msl (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m',
            'wind_gusts_10m (km/h)', 'vapour_pressure_deficit (kPa)'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Tạo features bổ sung từ time
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        available_features.extend(['hour', 'day', 'month', 'day_of_week'])
        
        X = df[available_features].fillna(0)
        y_temp = df['temperature_2m (°C)'].ffill()
        y_precip = df['precipitation (mm)'].ffill()
        
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_temp_train, y_temp_test = y_temp[:split_idx], y_temp[split_idx:]
        y_precip_train, y_precip_test = y_precip[:split_idx], y_precip[split_idx:]
        
        return X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test, available_features

    def train_temperature_model(self, X_train, y_train):
        print("Training Temperature Random Forest...")
        self.temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.temp_model.fit(X_train, y_train)
        # Lưu lại feature names
        self.temp_features = X_train.columns.tolist()
        joblib.dump(self.temp_model, f'{self.models_dir}/temperature_model.pkl')
        print("Temperature model trained and saved!")

    def train_precipitation_model(self, X_train, y_train):
        print("Training Precipitation XGBoost...")

        self.precip_model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.precip_model.fit(X_train, y_train)

        # Lưu lại danh sách các features dùng để huấn luyện
        self.precip_features = X_train.columns.tolist()

        # Lưu mô hình thành file .pkl
        joblib.dump(self.precip_model, f'{self.models_dir}/precipitation_model.pkl')

        print("Precipitation model trained and saved!")

    def evaluate_models(self, X_test, y_temp_test, y_precip_test):
        results = {}
        
        if self.temp_model:
            y_pred_temp = self.temp_model.predict(X_test)
            results['temperature'] = {
                'rmse': np.sqrt(mean_squared_error(y_temp_test, y_pred_temp)),
                'mae': mean_absolute_error(y_temp_test, y_pred_temp),
                'r2': r2_score(y_temp_test, y_pred_temp)
            }
        
        if self.rain_classifier and self.rain_regressor:
            print("Đang đánh giá mô hình lượng mưa two-stage...")
            y_pred_precip = self.predict_precipitation(X_test)
            results['precipitation'] = {
                'rmse': np.sqrt(mean_squared_error(y_precip_test, y_pred_precip)),
                'mae': mean_absolute_error(y_precip_test, y_pred_precip),
                'r2': r2_score(y_precip_test, y_pred_precip)
            }
        
        return results

    def load_models(self):
        try:
            self.temp_model = joblib.load(f'{self.models_dir}/temperature_model.pkl')
            # Lấy feature names từ model
            self.temp_features = list(self.temp_model.feature_names_in_)
            print("Temperature model loaded!")
        except:
            print("Temperature model not found!")
        
        try:
            self.precip_model = joblib.load(f'{self.models_dir}/precipitation_model.pkl')
            self.precip_features = list(self.precip_model.feature_names_in_)
            print("Precipitation model loaded!")
        except:
            print("Precipitation model not found!")
        
        try:
            self.rain_classifier = joblib.load(f'{self.models_dir}/rain_classifier.pkl')
            print("Rain classifier loaded!")
        except:
            print("Rain classifier not found!")
        
        try:
            self.rain_regressor = joblib.load(f'{self.models_dir}/rain_regressor.pkl')
            print("Rain regressor loaded!")
        except:
            print("Rain regressor not found!")

    def _prepare_X_for_predict(self, X, feature_names):
        # Đảm bảo X chỉ có đúng các cột đã train, thêm cột thiếu với giá trị 0
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        # Loại bỏ mọi cột lạ (chỉ giữ đúng feature_names)
        X = X[feature_names]
        return X

    def predict_precipitation(self, X):
        rain_yes = self.rain_classifier.predict(X)
        y_pred = np.zeros(X.shape[0])
        if np.any(rain_yes == 1):
            X_rain = X[rain_yes == 1]
            y_pred_rain = self.rain_regressor.predict(X_rain)
            y_pred[rain_yes == 1] = y_pred_rain
        return y_pred

    def predict_future(self, df, days=7, start_offset_hours=0):
        # Chỉ sử dụng các cột có sẵn trong dataset thực tế
        feature_cols = [
            'rain (mm)', 'snowfall (cm)', 
            'snow_depth (m)', 'weather_code (wmo code)', 'is_day ()',
            'relative_humidity_2m (%)', 'dew_point_2m (°C)', 'cloud_cover (%)'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Tạo features bổ sung từ time
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        available_features.extend(['hour', 'day', 'month', 'day_of_week'])
        
        latest_data = df[available_features].fillna(0).iloc[-1:]
        
        future_dates = pd.date_range(
            start=df['time'].max() + pd.Timedelta(hours=1 + start_offset_hours),
            periods=days*24,
            freq='H'
        )
        
        future_data = []
        for date in future_dates:
            row = latest_data.copy()
            row['hour'] = date.hour
            row['day'] = date.day
            row['month'] = date.month
            row['day_of_week'] = date.dayofweek
            future_data.append(row.values[0])
        
        future_df = pd.DataFrame(future_data, columns=available_features)
        
        predictions = {}
        
        if self.temp_model:
            X_temp = self._prepare_X_for_predict(future_df.copy(), self.temp_features)
            temp_preds = self.temp_model.predict(X_temp)
            predictions['temperature'] = temp_preds
        
        if self.rain_classifier and self.rain_regressor:
            X_precip = self._prepare_X_for_predict(future_df.copy(), self.precip_features)
            precip_preds = self.predict_precipitation(X_precip)
            predictions['precipitation'] = precip_preds
        
        return predictions, future_dates

def train_models(data_file="C:\\Users\\taqua\\OneDrive\\Chuyên nghành\Kì 4\DAP391m\Project\copy11 final1\data\dataset.csv"):
    # Đọc file CSV với low_memory=False để tránh warning
    df = pd.read_csv(data_file, encoding='latin1', low_memory=False)
    
    # Kiểm tra xem có cột 'time' không
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    else:
        # Nếu không có cột time, tạo một cột time giả dựa trên index
        print("Không tìm thấy cột 'time', tạo cột time giả...")
        df['time'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    
    predictor = WeatherPredictor()
    
    X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test, features = predictor.prepare_data(df)
    
    predictor.train_temperature_model(X_train, y_temp_train)
    predictor.train_precipitation_model(X_train, y_precip_train)

    # --- HUẤN LUYỆN MÔ HÌNH TWO-STAGE CHO LƯỢNG MƯA ---
    print("\nTraining Two-Stage Rain Model...")

    df['rain_yes'] = (df['precipitation (mm)'] > 0).astype(int)
    y_rain_class = df['rain_yes']
    
    # Tập huấn luyện cho rain classifier
    X_rain_class = df.loc[:len(X_train)-1, features]
    y_rain_class_train = y_rain_class[:len(X_train)]

    # Huấn luyện rain classifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBRegressor
    predictor.rain_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    predictor.rain_classifier.fit(X_rain_class, y_rain_class_train)

    # Tập huấn luyện cho rain regressor (chỉ lấy mẫu có mưa)
    df_rain_train = df.loc[:len(X_train)-1].copy()
    df_rain_train = df_rain_train[df_rain_train['rain_yes'] == 1]

    if not df_rain_train.empty:
        X_rain_reg = df_rain_train[features]
        y_rain_reg = df_rain_train['precipitation (mm)']
        predictor.rain_regressor = XGBRegressor(n_estimators=30, learning_rate=0.1, random_state=42)
        predictor.rain_regressor.fit(X_rain_reg, y_rain_reg)
        print("Two-stage rain model trained successfully!")
    else:
        print("⚠️ Không có mẫu nào có mưa trong tập train -> skip mô hình rain_regressor.")

    results = predictor.evaluate_models(X_test, y_temp_test, y_precip_test)
    
    print("\n=== Kết quả đánh giá mô hình ===")
    for target, metrics in results.items():
        print(f"\n{target.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
        if target == 'precipitation':
            # Thống kê số lượng mẫu mưa thực sự trong tập test
            n_rain = (y_precip_test > 0).sum()
            n_total = len(y_precip_test)
            print(f"  Số mẫu test có mưa thực sự: {n_rain}/{n_total}")
    
    return predictor

if __name__ == "__main__":
    predictor = train_models()

    # Lưu mô hình rain_classifier và rain_regressor
    joblib.dump(predictor.rain_classifier, 'models/rain_classifier.pkl')
    joblib.dump(predictor.rain_regressor, 'models/rain_regressor.pkl') 