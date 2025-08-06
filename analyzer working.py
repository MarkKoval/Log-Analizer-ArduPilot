import pandas as pd
import matplotlib
# Використовуємо агресивний бекенд, який не вимагає GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymavlink import mavutil
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np
import io
import base64
import logging
from datetime import datetime, timedelta
from geopy.distance import geodesic


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._parse_log()
        self._preprocess_data()
    
    def _parse_log(self):
        """Оптимізований парсинг бінарного логу"""
        logger.info("Початок парсингу бінарного логу...")
        
        try:
            mlog = mavutil.mavlink_connection(self.file_path)
            data = []
            
            msg_types = [
                'GPS', 'GPS2', 'ATT', 'CTUN', 
                'NKF1', 'BARO', 'BAT', 'MODE'
            ]
            
            start_time = None
            while True:
                msg = mlog.recv_match(type=msg_types, blocking=False)
                if msg is None:
                    break
                
                msg_dict = msg.to_dict()
                msg_type = msg.get_type()
                msg_dict['msg_type'] = msg_type
                
                if 'TimeUS' in msg_dict:
                    timestamp = msg_dict['TimeUS'] / 1e6
                    if start_time is None:
                        start_time = datetime.now() - timedelta(seconds=timestamp)
                    msg_dict['timestamp'] = start_time + timedelta(seconds=timestamp)
                
                data.append(msg_dict)
            
            if not data:
                raise ValueError("Не вдалося отримати дані з логу")
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Помилка парсингу логу: {str(e)}")
            raise

    def _preprocess_data(self):
        # Дросель із CTUN
        """Попередня обробка даних з використанням BARO.Alt"""
        if self.data.empty:
            return

        # Конвертація типів даних
        numeric_cols = ['Alt', 'Spd', 'Roll', 'Pitch', 'Yaw', 'Volt', 'Curr']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Обробка часу
        if 'TimeUS' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['TimeUS'], unit='us')
            self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)

        # Визначення висоти з BARO
        if 'BARO' in self.data['msg_type'].values:
            baro_data = self.data[self.data['msg_type'] == 'BARO']
            if 'Alt' in baro_data.columns:
                self.data['altitude'] = pd.to_numeric(baro_data['Alt'], errors='coerce')

        # Швидкість з GPS
        if 'GPS' in self.data['msg_type'].values:
            gps_data = self.data[self.data['msg_type'] == 'GPS']
            if 'Spd' in gps_data.columns:
                self.data['speed'] = pd.to_numeric(gps_data['Spd'], errors='coerce')

        # Інтерполяція тільки числових даних
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            self.data[numeric_data.columns] = numeric_data.interpolate(method='time')

        # Пройдена дистанція
        if 'GPS' in self.data['msg_type'].values:
            gps_data = self.data[self.data['msg_type'] == 'GPS']
            if 'Lat' in gps_data.columns and 'Lng' in gps_data.columns:
                self.data['lat'] = pd.to_numeric(gps_data['Lat'], errors='coerce') / 1e7
                self.data['lon'] = pd.to_numeric(gps_data['Lng'], errors='coerce') / 1e7


    def get_basic_statistics(self):
        """Основні статистики польоту"""
        stats = {}
        
        if not self.data.empty:
            if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 1:
                duration = self.data.index[-1] - self.data.index[0]
                stats['flight_duration'] = str(duration)
            
            if 'altitude' in self.data.columns:
                stats['max_altitude'] = f"{self.data['altitude'].max():.2f} м"
                stats['min_altitude'] = f"{self.data['altitude'].min():.2f} м"
                stats['avg_altitude'] = f"{self.data['altitude'].mean():.2f} м"
            
            if 'speed' in self.data.columns:
                stats['max_speed'] = f"{self.data['speed'].max() * 3.6:.2f} км/год"
                stats['avg_speed'] = f"{self.data['speed'].mean() * 3.6:.2f} км/год"

            # Обчислення пройденої дистанції
            # Використовуємо лише GPS-повідомлення для координат
            if 'lat' in self.data.columns and 'lon' in self.data.columns:
                gps_points = self.data[['lat', 'lon']].dropna().to_numpy()
                
                distance_km = 0.0
                for i in range(1, len(gps_points)):
                    distance_km += geodesic(gps_points[i - 1], gps_points[i]).km
            
                stats['total_distance_km'] = f"{distance_km:.2f} км"
                stats['total_distance_m'] = f"{distance_km * 1000:.1f} м"


        
        return stats

    def generate_basic_graphs(self):
        """Генерація великих графіків"""
        graphs = {}

        # Розміри графіків у дюймах (ширина, висота)
        FIGURE_SIZE = (16, 9)  # Збільшений розмір для FullHD роздільної здатності

        if not self.data.empty:
            # Графік висоти
            if 'altitude' in self.data.columns:
                valid_alt = self.data['altitude'].dropna()
                if not valid_alt.empty:
                    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

                    # Основна лінія висоти
                    ax.plot(valid_alt.index, valid_alt, 
                           label='Барометрична висота', 
                           color='blue',
                           linewidth=2)

                    # Налаштування графіка
                    ax.set_title('Висота польоту (BARO.Alt)', fontsize=16)
                    ax.set_ylabel('Висота (м)', fontsize=14)
                    ax.set_xlabel('Час', fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=12)

                    # Збільшення розмірів шрифтів
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)

                    # Автоматичне форматування дати
                    fig.autofmt_xdate()

                    graphs['altitude'] = self._fig_to_base64(fig)
                    plt.close(fig)

            # Графік швидкості
            if 'speed' in self.data.columns:
                valid_speed = self.data['speed'].dropna()
                if not valid_speed.empty:
                    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

                    # Конвертація у км/год та побудова графіка
                    speed_kmh = valid_speed * 3.6
                    ax.plot(speed_kmh.index, speed_kmh,
                           label='Швидкість (GPS)',
                           color='green',
                           linewidth=2)

                    # Налаштування графіка
                    ax.set_title('Швидкість польоту', fontsize=16)
                    ax.set_ylabel('Швидкість (км/год)', fontsize=14)
                    ax.set_xlabel('Час', fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=12)

                    # Збільшення розмірів шрифтів
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)

                    # Автоматичне форматування дати
                    fig.autofmt_xdate()

                    graphs['speed'] = self._fig_to_base64(fig)
                    plt.close(fig)

        return graphs

    def perform_ai_analysis(self):
        """Розширений аналіз з використанням ШІ з додатковими перевірками"""
        analysis = {}
        
        # Перевірка наявності необхідних даних
        if self.data.empty or 'altitude' not in self.data.columns or 'speed' not in self.data.columns:
            analysis['error'] = "Недостатньо даних для аналізу (відсутні висота або швидкість)"
            return analysis
        
        try:
            # Підготовка даних
            features = self.data[['altitude', 'speed']].dropna()
            
            if len(features) < 10:  # Мінімальна кількість точок для аналізу
                analysis['warning'] = "Замало даних для точного аналізу (мінімум 10 точок)"
                return analysis
            
            # Нормалізація даних
            features_norm = (features - features.mean()) / features.std()
            
            # 1. Виявлення аномалій
            iso_forest = IsolationForest(
                n_estimators=100,
                contamination='auto',
                random_state=42
            )
            features['anomaly'] = iso_forest.fit_predict(features_norm)
            anomalies = features[features['anomaly'] == -1]
            analysis['anomaly_count'] = len(anomalies)
            analysis['anomaly_percent'] = f"{len(anomalies)/len(features)*100:.1f}%"
            
            # 2. Кластеризація
            kmeans = KMeans(
                n_clusters=min(3, len(features)-1),  # Не більше кластерів ніж точок
                random_state=42
            )
            features['cluster'] = kmeans.fit_predict(features_norm)
            analysis['cluster_info'] = {
                'cluster_sizes': features['cluster'].value_counts().to_dict()
            }
            
            # 3. Візуалізація результатів
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Точки за кластерами
            scatter = ax.scatter(
                features['speed']*3.6,  # Конвертація у км/год
                features['altitude'],
                c=features['cluster'],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            
            # Аномалії
            if not anomalies.empty:
                ax.scatter(
                    anomalies['speed']*3.6,
                    anomalies['altitude'],
                    color='red',
                    marker='x',
                    s=100,
                    linewidths=2,
                    label='Аномалії'
                )
            
            # Центроїди кластерів
            centers = kmeans.cluster_centers_
            columns_used = ['altitude', 'speed']
            stds = features[columns_used].std().values
            means = features[columns_used].mean().values
            centers_original = centers * stds + means

            ax.scatter(
                centers_original[:,1]*3.6,  # Швидкість
                centers_original[:,0],      # Висота
                color='black',
                marker='o',
                s=200,
                alpha=0.8,
                label='Центри кластерів'
            )
            
            ax.set_title('Аналіз режимів польоту (ШІ)')
            ax.set_xlabel('Швидкість (км/год)')
            ax.set_ylabel('Висота (м)')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            
            analysis['clusters_plot'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            # 4. Додаткові метрики
            analysis['metrics'] = {
                'avg_speed_by_cluster': features.groupby('cluster')['speed'].mean().mul(3.6).round(1).to_dict(),
                'avg_altitude_by_cluster': features.groupby('cluster')['altitude'].mean().round(1).to_dict()
            }
            
        except Exception as e:
            analysis['error'] = f"Помилка під час аналізу: {str(e)}"
            logger.error(f"AI analysis failed: {str(e)}")
        
        return analysis

    def _fig_to_base64(self, fig):
        """Конвертація графіка у base64"""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')