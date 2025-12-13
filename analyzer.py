import pandas as pd
from pymavlink import mavutil
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np
import io
import base64
import logging
from datetime import datetime, timedelta
from geopy.distance import geodesic
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._parse_log()
        self._preprocess_data()
    
    def _parse_log(self):
        """Parse binary log file from ArduPilot"""
        logger.info("Starting binary log parsing...")
        
        try:
            mlog = mavutil.mavlink_connection(self.file_path)
            data = []
            
            msg_types = [
                'GPS', 'GPS2', 'ATT', 'CTUN', 
                'NKF1', 'BARO', 'BAT', 'MODE',
                'POWR', 'CURR', 'VIBE'
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
                raise ValueError("No data found in log file")
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Log parsing error: {str(e)}")
            raise

    def estimate_distance_from_speed(self):
        if 'speed' not in self.data.columns:
            return None

        df = self.data[['speed']].dropna()

        if df.empty or len(df) < 2:
            return None

        df = df.copy()
        df['time_diff'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
        df['distance'] = df['speed'] * df['time_diff']  # m/s * s = meters
        total_distance_m = df['distance'].sum()
        return total_distance_m

    def _preprocess_data(self):
        """Preprocess and clean the log data"""
        if self.data.empty:
            return

        # Convert numeric columns
        numeric_cols = ['Alt', 'Spd', 'Roll', 'Pitch', 'Yaw', 'Volt', 'Curr', 
                       'Lat', 'Lng', 'Thr', 'ThrOut', 'DAlt', 'DSAlt', 'SAlt']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Process timestamps
        if 'TimeUS' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['TimeUS'], unit='us')
            self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)

        # Extract altitude from BARO
        if 'BARO' in self.data['msg_type'].unique():
            baro_mask = self.data['msg_type'] == 'BARO'
            self.data.loc[baro_mask, 'altitude'] = pd.to_numeric(
                self.data.loc[baro_mask, 'Alt'], errors='coerce'
            )

        # Extract speed from GPS
        if 'GPS' in self.data['msg_type'].unique():
            gps_mask = self.data['msg_type'] == 'GPS'
            self.data.loc[gps_mask, 'speed'] = pd.to_numeric(
                self.data.loc[gps_mask, 'Spd'], errors='coerce'
            )
            # Convert coordinates from degrees*1e7 to decimal degrees
            self.data.loc[gps_mask, 'lat'] = self.data.loc[gps_mask, 'Lat'] / 1e7
            self.data.loc[gps_mask, 'lon'] = self.data.loc[gps_mask, 'Lng'] / 1e7

        # Extract throttle from CTUN - with existence check
        if 'CTUN' in self.data['msg_type'].unique():
            ctun_mask = self.data['msg_type'] == 'CTUN'
            # Using ThO as throttle
            if 'ThO' in self.data.columns:
                self.data.loc[ctun_mask, 'throttle'] = pd.to_numeric(self.data.loc[ctun_mask, 'ThO'], errors='coerce')
                logger.info("Throttle extracted from CTUN.ThO column")
            elif 'ThD' in self.data.columns:
                self.data.loc[ctun_mask, 'throttle'] = pd.to_numeric(self.data.loc[ctun_mask, 'ThD'], errors='coerce')
                logger.info("Throttle extracted from CTUN.ThD column")

        # DEBUG: Force print throttle info even if empty
        if 'throttle' in self.data.columns:
            print("Throttle column present.")
            print(self.data['throttle'].describe())
        else:
            print("Throttle column MISSING.")

        # Interpolate numeric data
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            self.data[numeric_data.columns] = numeric_data.interpolate(method='time')

        # --- Trim log to first moment when throttle >= 10% ---
        if 'throttle' in self.data.columns:
            throttle_data = self.data['throttle'].dropna()
            
            start_mask = throttle_data >= 40
            end_mask = throttle_data <= 0.5
        
            if start_mask.any() and end_mask.any():
                first_idx = start_mask.idxmax()
                last_idx = end_mask[end_mask.index > first_idx].index[-1]  # guaranteed after first_idx
        
                print(f"Trimming from {first_idx} to {last_idx}")
        
                # Make sure last_idx is really after first_idx
                if last_idx > first_idx:
                    self.data = self.data.loc[first_idx:last_idx]
                else:
                    print("End index is before start index — trimming skipped.")
            else:
                print("Throttle thresholds not found for trimming.")
        else:
            print("Throttle column missing.")

    def get_basic_statistics(self):
        """Calculate basic flight statistics"""
        stats = {}

        if self.data.empty:
            return stats

        # Flight duration
        if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 1:
            duration = self.data.index[-1] - self.data.index[0]
            stats['flight_duration'] = str(duration)

        distance_est = self.estimate_distance_from_speed()
        if distance_est:
            stats['estimated_distance_m'] = f"{distance_est:.1f} м"
            stats['estimated_distance_km'] = f"{distance_est / 1000:.2f} км"

        # Altitude statistics
        if 'altitude' in self.data.columns:
            alt_data = self.data['altitude'].dropna()
            if not alt_data.empty:
                stats['max_altitude'] = f"{alt_data.max():.2f} м"
                stats['min_altitude'] = f"{alt_data.min():.2f} м"
                stats['avg_altitude'] = f"{alt_data.mean():.2f} м"
                stats['altitude_range'] = f"{alt_data.max() - alt_data.min():.2f} м"

        # Speed statistics
        if 'speed' in self.data.columns:
            speed_data = self.data['speed'].dropna()
            if not speed_data.empty:
                stats['max_speed'] = f"{speed_data.max() * 3.6:.2f} км/год"
                stats['min_speed'] = f"{speed_data.min() * 3.6:.2f} км/год"
                stats['avg_speed'] = f"{speed_data.mean() * 3.6:.2f} км/год"

        # Throttle statistics (only if available)
        if 'throttle' in self.data.columns:
            thr_data = self.data['throttle'].dropna()
            if not thr_data.empty:
                stats['max_throttle'] = f"{thr_data.max():.1f}%"
                stats['min_throttle'] = f"{thr_data.min():.1f}%"
                stats['avg_throttle'] = f"{thr_data.mean():.1f}%"

        # Calculate distance traveled
        distance_km = None
        if 'lat' in self.data.columns and 'lon' in self.data.columns:
            gps_points = self.data[
                (self.data['msg_type'] == 'GPS') &
                (self.data['lat'].abs() > 0.001) &
                (self.data['lon'].abs() > 0.001)
            ][['lat', 'lon']].dropna()

            if len(gps_points) > 1:
                points = gps_points.to_numpy()
                distance_km = 0.0

                for i in range(1, len(points)):
                    try:
                        distance_km += geodesic(points[i-1], points[i]).km
                    except Exception:
                        continue

                stats['total_distance_km'] = f"{distance_km:.2f} км"
                stats['total_distance_m'] = f"{distance_km * 1000:.1f} м"

        # If GPS didn't give results - estimate via speed
        if not distance_km or distance_km == 0.0:
            distance_est = self.estimate_distance_from_speed()
            if distance_est:
                stats['total_distance_m'] = f"{distance_est:.1f} м"
                stats['total_distance_km'] = f"{distance_est / 1000:.2f} км"

        return stats

    def generate_basic_graphs(self, include=None):
        """Generate interactive flight data visualizations using Plotly

        Args:
            include: Optional set of graph identifiers to include. Supported
                values: {'altitude', 'speed', 'throttle'}.
        """
        graphs = {}

        if self.data.empty:
            return graphs

        include = include or {'altitude', 'speed', 'throttle'}
        include = set(include)
        
        # Загальні налаштування для всіх графіків
        common_layout = {
            'font': {
                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                'size': 14,
                'color': '#333',
                'weight': 'bold'  # жирний шрифт
            },
            'hovermode': 'x unified',
            'template': 'plotly_white',
            'height': 500,
            'margin': dict(l=50, r=50, b=50, t=80, pad=4),
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
        }
        
        # Altitude plot
        if 'altitude' in include and 'altitude' in self.data.columns:
            alt_data = self.data['altitude'].dropna()
            if not alt_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=alt_data.index,
                    y=alt_data,
                    mode='lines',
                    name='Висота (BARO)',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Висота</b>: %{y:.2f} м<extra></extra>'
                ))
                
                layout = common_layout.copy()
                layout.update({
                    'title': {
                        'text': 'Висота польоту',
                        'font': {
                            'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            'size': 18,
                            'weight': 'bold'  # жирний заголовок
                        }
                    },
                    'yaxis': {
                        'title': {
                            'text': 'Висота (m)',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі Y
                            }
                        }
                    },
                    'xaxis': {
                        'title': {
                            'text': 'Час',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі X
                            }
                        }
                    },
                })
                fig.update_layout(layout)
                
                graphs['altitude'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Speed plot
        if 'speed' in include and 'speed' in self.data.columns:
            speed_data = self.data['speed'].dropna()
            if not speed_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=speed_data.index,
                    y=speed_data * 3.6,
                    mode='lines',
                    name='Швидкість (GPS)',
                    line=dict(color='green', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Швидкість</b>: %{y:.2f} км/год<extra></extra>'
                ))
                
                layout = common_layout.copy()
                layout.update({
                    'title': {
                        'text': 'Швидкість польоту',
                        'font': {
                            'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            'size': 18,
                            'weight': 'bold'  # жирний заголовок
                        }
                    },
                    'yaxis': {
                        'title': {
                            'text': 'Швидкість (км/год)',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі Y
                            }
                        }
                    },
                    'xaxis': {
                        'title': {
                            'text': 'Час',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі X
                            }
                        }
                    },
                })
                fig.update_layout(layout)
                
                graphs['speed'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Throttle plot
        if 'throttle' in include and 'throttle' in self.data.columns:
            thr_data = self.data['throttle'].dropna()
            if not thr_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=thr_data.index,
                    y=thr_data,
                    mode='lines',
                    name='Throttle',
                    line=dict(color='orange', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Throttle</b>: %{y:.1f}%<extra></extra>'
                ))
                
                layout = common_layout.copy()
                layout.update({
                    'title': {
                        'text': 'Відсоток газу (Throttle)',
                        'font': {
                            'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            'size': 18,
                            'weight': 'bold'  # жирний заголовок
                        }
                    },
                    'yaxis': {
                        'title': {
                            'text': 'Відсоток газу (%)',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі Y
                            }
                        }
                    },
                    'xaxis': {
                        'title': {
                            'text': 'Час',
                            'font': {
                                'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                'size': 16,
                                'weight': 'bold'  # жирний текст осі X
                            }
                        }
                    },
                })
                fig.update_layout(layout)
                
                graphs['throttle'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        return graphs
