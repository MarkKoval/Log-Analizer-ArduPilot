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
                'POWR', 'CURR', 'VIBE', 'RCIN',
                'RCOU'
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
                       'Lat', 'Lng', 'Thr', 'ThrOut', 'DAlt', 'DSAlt', 'SAlt',
                       'NSats', 'HDop']

        rc_in_channels = [f'C{i}' for i in range(1, 13)]
        rc_out_channels = [f'C{i}Out' for i in range(1, 13)]
        numeric_cols.extend(rc_in_channels)
        numeric_cols.extend(rc_out_channels)
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

        # Extract RC channels from RCIN and RCOU
        rc_in_channels = [f'C{i}' for i in range(1, 13)]
        rc_out_channels = [f'C{i}Out' for i in range(1, 13)]

        if 'RCIN' in self.data['msg_type'].unique():
            rcin_mask = self.data['msg_type'] == 'RCIN'
            for col in rc_in_channels:
                if col in self.data.columns:
                    self.data.loc[rcin_mask, f'rcin_{col}'] = pd.to_numeric(
                        self.data.loc[rcin_mask, col], errors='coerce'
                    )

        if 'RCOU' in self.data['msg_type'].unique():
            rcou_mask = self.data['msg_type'] == 'RCOU'
            for col in rc_out_channels:
                src_col = col.replace('Out', '')
                if src_col in self.data.columns:
                    self.data.loc[rcou_mask, f'rcout_{src_col}'] = pd.to_numeric(
                        self.data.loc[rcou_mask, src_col], errors='coerce'
                    )

        # Flight mode extraction
        if 'MODE' in self.data['msg_type'].unique():
            mode_mask = self.data['msg_type'] == 'MODE'
            if 'Mode' in self.data.columns:
                self.data.loc[mode_mask, 'flight_mode'] = self.data.loc[mode_mask, 'Mode']
            elif 'ModeNum' in self.data.columns:
                self.data.loc[mode_mask, 'flight_mode'] = self.data.loc[mode_mask, 'ModeNum']

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
        """Calculate extended flight statistics (50+ параметрів)"""
        stats = {}

        if self.data.empty:
            return stats

        def _add_stat(key, value, suffix=""):
            if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
                return
            stats[key] = f"{value}{suffix}" if suffix else value

        # Flight duration
        if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 1:
            duration = self.data.index[-1] - self.data.index[0]
            _add_stat('flight_duration', str(duration))
            _add_stat('log_start_time', self.data.index[0].strftime('%Y-%m-%d %H:%M:%S'))
            _add_stat('log_end_time', self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S'))

        distance_est = self.estimate_distance_from_speed()
        if distance_est:
            _add_stat('estimated_distance_m', f"{distance_est:.1f} м")
            _add_stat('estimated_distance_km', f"{distance_est / 1000:.2f} км")

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

                _add_stat('total_distance_km', f"{distance_km:.2f} км")
                _add_stat('total_distance_m', f"{distance_km * 1000:.1f} м")

        if not distance_km or distance_km == 0.0:
            if distance_est:
                _add_stat('total_distance_m', f"{distance_est:.1f} м")
                _add_stat('total_distance_km', f"{distance_est / 1000:.2f} км")

        # Altitude statistics
        if 'altitude' in self.data.columns:
            alt_data = self.data['altitude'].dropna()
            if not alt_data.empty:
                _add_stat('max_altitude', f"{alt_data.max():.2f} м")
                _add_stat('min_altitude', f"{alt_data.min():.2f} м")
                _add_stat('avg_altitude', f"{alt_data.mean():.2f} м")
                _add_stat('median_altitude', f"{alt_data.median():.2f} м")
                _add_stat('altitude_std', f"{alt_data.std():.2f} м")
                _add_stat('altitude_variance', f"{alt_data.var():.2f} м²")
                _add_stat('altitude_25th', f"{alt_data.quantile(0.25):.2f} м")
                _add_stat('altitude_75th', f"{alt_data.quantile(0.75):.2f} м")
                _add_stat('altitude_range', f"{alt_data.max() - alt_data.min():.2f} м")
                _add_stat('start_altitude', f"{alt_data.iloc[0]:.2f} м")
                _add_stat('end_altitude', f"{alt_data.iloc[-1]:.2f} м")

                if isinstance(alt_data.index, pd.DatetimeIndex) and len(alt_data.index) > 1:
                    dt_seconds = alt_data.index.to_series().diff().dt.total_seconds().replace(0, pd.NA)
                    climb_rate = alt_data.diff() / dt_seconds
                    climb_rate = climb_rate.dropna()
                    if not climb_rate.empty:
                        _add_stat('max_climb_rate', f"{climb_rate.max():.2f} м/с")
                        _add_stat('min_climb_rate', f"{climb_rate.min():.2f} м/с")
                        _add_stat('avg_climb_rate', f"{climb_rate.mean():.2f} м/с")
                        _add_stat('climb_rate_std', f"{climb_rate.std():.2f} м/с")
                        _add_stat('time_above_10m', f"{(alt_data > 10).mean() * 100:.1f}% часу")
                        _add_stat('time_below_5m', f"{(alt_data < 5).mean() * 100:.1f}% часу")

        # Speed statistics
        if 'speed' in self.data.columns:
            speed_data = self.data['speed'].dropna()
            if not speed_data.empty:
                kmh = speed_data * 3.6
                _add_stat('max_speed', f"{kmh.max():.2f} км/год")
                _add_stat('min_speed', f"{kmh.min():.2f} км/год")
                _add_stat('avg_speed', f"{kmh.mean():.2f} км/год")
                _add_stat('median_speed', f"{kmh.median():.2f} км/год")
                _add_stat('speed_std', f"{kmh.std():.2f} км/год")
                _add_stat('speed_variance', f"{kmh.var():.2f} (км/год)²")
                _add_stat('speed_p95', f"{kmh.quantile(0.95):.2f} км/год")
                _add_stat('speed_p05', f"{kmh.quantile(0.05):.2f} км/год")
                _add_stat('speed_range', f"{kmh.max() - kmh.min():.2f} км/год")
                _add_stat('start_speed', f"{kmh.iloc[0]:.2f} км/год")
                _add_stat('end_speed', f"{kmh.iloc[-1]:.2f} км/год")

                if isinstance(speed_data.index, pd.DatetimeIndex) and len(speed_data.index) > 1:
                    dt_seconds = speed_data.index.to_series().diff().dt.total_seconds().replace(0, pd.NA)
                    acceleration = speed_data.diff() / dt_seconds
                    acceleration = acceleration.dropna()
                    if not acceleration.empty:
                        _add_stat('max_acceleration', f"{(acceleration.max() * 3.6):.2f} (км/год)/с")
                        _add_stat('min_acceleration', f"{(acceleration.min() * 3.6):.2f} (км/год)/с")
                        _add_stat('avg_acceleration', f"{(acceleration.mean() * 3.6):.2f} (км/год)/с")
                        _add_stat('acceleration_std', f"{(acceleration.std() * 3.6):.2f} (км/год)/с")
                        _add_stat('time_above_50kmh', f"{(kmh > 50).mean() * 100:.1f}% часу")
                        _add_stat('time_below_10kmh', f"{(kmh < 10).mean() * 100:.1f}% часу")

        # Throttle statistics (only if available)
        if 'throttle' in self.data.columns:
            thr_data = self.data['throttle'].dropna()
            if not thr_data.empty:
                _add_stat('max_throttle', f"{thr_data.max():.1f}%")
                _add_stat('min_throttle', f"{thr_data.min():.1f}%")
                _add_stat('avg_throttle', f"{thr_data.mean():.1f}%")
                _add_stat('median_throttle', f"{thr_data.median():.1f}%")
                _add_stat('throttle_std', f"{thr_data.std():.2f}%")
                _add_stat('throttle_range', f"{thr_data.max() - thr_data.min():.1f}%")
                _add_stat('time_over_80pct', f"{(thr_data > 80).mean() * 100:.1f}% часу")
                _add_stat('time_below_20pct', f"{(thr_data < 20).mean() * 100:.1f}% часу")

        # Battery statistics
        if 'Volt' in self.data.columns:
            volt_data = pd.to_numeric(self.data['Volt'], errors='coerce').dropna()
            if not volt_data.empty:
                _add_stat('max_voltage', f"{volt_data.max():.2f} V")
                _add_stat('min_voltage', f"{volt_data.min():.2f} V")
                _add_stat('avg_voltage', f"{volt_data.mean():.2f} V")
                _add_stat('median_voltage', f"{volt_data.median():.2f} V")
                _add_stat('voltage_std', f"{volt_data.std():.2f} V")
                _add_stat('start_voltage', f"{volt_data.iloc[0]:.2f} V")
                _add_stat('end_voltage', f"{volt_data.iloc[-1]:.2f} V")
                _add_stat('voltage_drop', f"{(volt_data.iloc[0] - volt_data.iloc[-1]):.2f} V")
                _add_stat('voltage_drop_pct', f"{((volt_data.iloc[0] - volt_data.iloc[-1]) / volt_data.iloc[0] * 100):.1f}%")

        if 'Curr' in self.data.columns:
            curr_data = pd.to_numeric(self.data['Curr'], errors='coerce').dropna()
            if not curr_data.empty:
                _add_stat('max_current', f"{curr_data.max():.2f} A")
                _add_stat('avg_current', f"{curr_data.mean():.2f} A")
                _add_stat('median_current', f"{curr_data.median():.2f} A")
                _add_stat('current_std', f"{curr_data.std():.2f} A")
                _add_stat('start_current', f"{curr_data.iloc[0]:.2f} A")
                _add_stat('end_current', f"{curr_data.iloc[-1]:.2f} A")
                if isinstance(curr_data.index, pd.DatetimeIndex) and len(curr_data.index) > 1:
                    dt_seconds = curr_data.index.to_series().diff().dt.total_seconds().fillna(0)
                    charge_consumed = np.trapz(curr_data, dx=dt_seconds) / 3600.0
                    _add_stat('consumed_ah', f"{charge_consumed:.2f} Ah")
                    _add_stat('avg_power_w', f"{(curr_data * pd.to_numeric(self.data.get('Volt', curr_data.index * 0), errors='coerce')).mean():.2f} W")

        # GPS quality
        if 'NSats' in self.data.columns:
            sats = pd.to_numeric(self.data['NSats'], errors='coerce').dropna()
            if not sats.empty:
                _add_stat('gps_satellites_avg', f"{sats.mean():.1f}")
                _add_stat('gps_satellites_min', f"{sats.min():.0f}")
                _add_stat('gps_satellites_max', f"{sats.max():.0f}")

        if 'HDop' in self.data.columns:
            hdop = pd.to_numeric(self.data['HDop'], errors='coerce').dropna()
            if not hdop.empty:
                _add_stat('gps_hdop_avg', f"{hdop.mean():.2f}")
                _add_stat('gps_hdop_min', f"{hdop.min():.2f}")
                _add_stat('gps_hdop_max', f"{hdop.max():.2f}")

        # Attitude statistics
        for axis in ['Roll', 'Pitch', 'Yaw']:
            if axis in self.data.columns:
                axis_data = pd.to_numeric(self.data[axis], errors='coerce').dropna()
                if not axis_data.empty:
                    _add_stat(f'max_{axis.lower()}', f"{axis_data.max():.1f}°")
                    _add_stat(f'min_{axis.lower()}', f"{axis_data.min():.1f}°")
                    _add_stat(f'avg_{axis.lower()}', f"{axis_data.mean():.1f}°")
                    _add_stat(f'median_{axis.lower()}', f"{axis_data.median():.1f}°")
                    _add_stat(f'std_{axis.lower()}', f"{axis_data.std():.1f}°")
                    _add_stat(f'range_{axis.lower()}', f"{axis_data.max() - axis_data.min():.1f}°")
                    if isinstance(axis_data.index, pd.DatetimeIndex) and len(axis_data.index) > 1:
                        dt_seconds = axis_data.index.to_series().diff().dt.total_seconds().replace(0, pd.NA)
                        rate = axis_data.diff() / dt_seconds
                        rate = rate.dropna()
                        if not rate.empty:
                            _add_stat(f'max_{axis.lower()}_rate', f"{rate.max():.2f}°/с")
                            _add_stat(f'min_{axis.lower()}_rate', f"{rate.min():.2f}°/с")

        # Vibration statistics (ArduPilot VIBE message)
        vibe_columns = ['VibeX', 'VibeY', 'VibeZ']
        available_vibes = [col for col in vibe_columns if col in self.data.columns]
        if available_vibes:
            combined = pd.DataFrame({col: pd.to_numeric(self.data[col], errors='coerce') for col in available_vibes})
            combined = combined.dropna(how='all')
            if not combined.empty:
                _add_stat('vibration_peak', f"{combined.max().max():.2f} m/s²")
                _add_stat('vibration_avg', f"{combined.mean().mean():.2f} m/s²")
                for col in available_vibes:
                    axis_series = combined[col].dropna()
                    if not axis_series.empty:
                        _add_stat(f'{col.lower()}_max', f"{axis_series.max():.2f} m/s²")
                        _add_stat(f'{col.lower()}_avg', f"{axis_series.mean():.2f} m/s²")
                        _add_stat(f'{col.lower()}_std', f"{axis_series.std():.2f} m/s²")

        # RC channel statistics (inputs and outputs)
        def _collect_channel_stats(prefix, channels):
            for ch in channels:
                colname = f'{prefix}{ch}'
                if colname in self.data.columns:
                    series = pd.to_numeric(self.data[colname], errors='coerce').dropna()
                    if series.empty:
                        continue
                    _add_stat(f'{colname}_min', f"{series.min():.0f} μs")
                    _add_stat(f'{colname}_max', f"{series.max():.0f} μs")
                    _add_stat(f'{colname}_avg', f"{series.mean():.1f} μs")
                    _add_stat(f'{colname}_median', f"{series.median():.1f} μs")
                    _add_stat(f'{colname}_std', f"{series.std():.1f} μs")
                    _add_stat(f'{colname}_range', f"{series.max() - series.min():.1f} μs")

        _collect_channel_stats('rcin_', [f'C{i}' for i in range(1, 13)])
        _collect_channel_stats('rcout_', [f'C{i}' for i in range(1, 13)])

        # Flight mode statistics
        if 'flight_mode' in self.data.columns:
            mode_series = self.data['flight_mode'].ffill().dropna()
            if not mode_series.empty:
                counts = mode_series.value_counts()
                _add_stat('flight_modes_detected', ', '.join(counts.index.astype(str)))
                mode_changes = int(mode_series.ne(mode_series.shift()).sum() - 1)
                _add_stat('flight_mode_changes', max(mode_changes, 0))
                _add_stat('most_used_mode', counts.idxmax())
                if isinstance(mode_series.index, pd.DatetimeIndex) and len(mode_series.index) > 1:
                    durations = mode_series.to_frame('mode')
                    durations['duration'] = durations.index.to_series().diff().shift(-1).dt.total_seconds().fillna(0)
                    duration_by_mode = durations.groupby('mode')['duration'].sum().sort_values(ascending=False)
                    top_mode, top_duration = duration_by_mode.index[0], duration_by_mode.iloc[0]
                    _add_stat('longest_mode', f"{top_mode} ({top_duration/60:.1f} хв)")
                    _add_stat('mode_duration_breakdown', ', '.join([
                        f"{m}: {t/60:.1f} хв" for m, t in duration_by_mode.items()
                    ]))

        return stats

    def generate_basic_graphs(self, include=None):
        """Generate interactive flight data visualizations using Plotly

        Args:
            include: Optional set of graph identifiers to include. Supported
                values: {'altitude', 'speed', 'throttle', 'attitude', 'battery', 'vibration', 'rc_channels', 'flight_modes'}.
        """
        graphs = {}

        if self.data.empty:
            return graphs

        include = include or {'altitude', 'speed', 'throttle', 'attitude', 'battery', 'vibration', 'rc_channels', 'flight_modes'}
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

        # Attitude plot
        if 'attitude' in include and {'Roll', 'Pitch'} <= set(self.data.columns):
            roll_data = pd.to_numeric(self.data['Roll'], errors='coerce').dropna()
            pitch_data = pd.to_numeric(self.data['Pitch'], errors='coerce').dropna()
            if not roll_data.empty and not pitch_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=roll_data.index,
                    y=roll_data,
                    mode='lines',
                    name='Roll',
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Roll</b>: %{y:.1f}°<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=pitch_data.index,
                    y=pitch_data,
                    mode='lines',
                    name='Pitch',
                    line=dict(color='teal', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Pitch</b>: %{y:.1f}°<extra></extra>'
                ))

                layout = common_layout.copy()
                layout.update({
                    'title': {'text': 'Кути крену та тангажу'},
                    'yaxis': {'title': 'Кути (°)'},
                    'xaxis': {'title': 'Час'},
                })
                fig.update_layout(layout)
                graphs['attitude'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Battery plot
        if 'battery' in include and 'Volt' in self.data.columns:
            volt_data = pd.to_numeric(self.data['Volt'], errors='coerce').dropna()
            curr_data = pd.to_numeric(self.data['Curr'], errors='coerce').dropna() if 'Curr' in self.data.columns else None
            if not volt_data.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(
                    x=volt_data.index,
                    y=volt_data,
                    mode='lines',
                    name='Напруга (V)',
                    line=dict(color='#e74c3c', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Напруга</b>: %{y:.2f} V<extra></extra>'
                ), secondary_y=False)

                if curr_data is not None and not curr_data.empty:
                    fig.add_trace(go.Scatter(
                        x=curr_data.index,
                        y=curr_data,
                        mode='lines',
                        name='Струм (A)',
                        line=dict(color='#8e44ad', width=2, dash='dot'),
                        hovertemplate='<b>Час</b>: %{x}<br><b>Струм</b>: %{y:.2f} A<extra></extra>'
                    ), secondary_y=True)

                layout = common_layout.copy()
                layout.update({
                    'title': {'text': 'Бортове живлення'},
                    'xaxis': {'title': 'Час'},
                    'yaxis': {'title': 'Напруга (V)'},
                })
                fig.update_yaxes(title_text='Струм (A)', secondary_y=True)
                fig.update_layout(layout)
                graphs['battery'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Vibration plot
        if 'vibration' in include:
            vibe_columns = ['VibeX', 'VibeY', 'VibeZ']
            available_vibes = [col for col in vibe_columns if col in self.data.columns]
            if available_vibes:
                combined = pd.DataFrame({col: pd.to_numeric(self.data[col], errors='coerce') for col in available_vibes})
                combined = combined.dropna(how='all')
                if not combined.empty:
                    fig = go.Figure()
                    colors = ['#ff7f50', '#1abc9c', '#9b59b6']
                    for idx, col in enumerate(available_vibes):
                        fig.add_trace(go.Scatter(
                            x=combined.index,
                            y=combined[col],
                            mode='lines',
                            name=col,
                            line=dict(color=colors[idx % len(colors)], width=2),
                            hovertemplate=f'<b>Час</b>: %{{x}}<br><b>{col}</b>: %{{y:.2f}} m/s²<extra></extra>'
                        ))

                    layout = common_layout.copy()
                    layout.update({
                        'title': {'text': 'Вібрації'},
                        'xaxis': {'title': 'Час'},
                        'yaxis': {'title': 'Прискорення (m/s²)'},
                    })
                    fig.update_layout(layout)
                    graphs['vibration'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # RC input channels plot
        if 'rc_channels' in include:
            rc_columns = [col for col in self.data.columns if col.startswith('rcin_C')]
            if rc_columns:
                rc_data = self.data[rc_columns].dropna(how='all')
                if not rc_data.empty:
                    fig = go.Figure()
                    palette = ['#3498db', '#e67e22', '#9b59b6', '#27ae60', '#c0392b', '#2980b9', '#8e44ad', '#16a085']
                    for idx, col in enumerate(sorted(rc_columns)[:8]):
                        fig.add_trace(go.Scatter(
                            x=rc_data.index,
                            y=rc_data[col],
                            mode='lines',
                            name=col.replace('rcin_', '').upper(),
                            line=dict(color=palette[idx % len(palette)], width=1.5),
                            hovertemplate='<b>Час</b>: %{x}<br><b>RC</b>: %{y:.0f} μs<extra></extra>'
                        ))

                    layout = common_layout.copy()
                    layout.update({
                        'title': {'text': 'RC IN (канали керування)'},
                        'xaxis': {'title': 'Час'},
                        'yaxis': {'title': 'ШІМ (μs)'},
                    })
                    fig.update_layout(layout)
                    graphs['rc_channels'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Flight mode timeline
        if 'flight_modes' in include and 'flight_mode' in self.data.columns:
            mode_series = self.data['flight_mode'].ffill().dropna()
            if not mode_series.empty:
                unique_modes = mode_series.dropna().unique()
                mode_to_num = {mode: idx for idx, mode in enumerate(unique_modes)}
                encoded = mode_series.map(mode_to_num)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=encoded.index,
                    y=encoded,
                    mode='lines',
                    step='post',
                    name='Режим',
                    line=dict(color='#2c3e50', width=2),
                    hovertemplate='<b>Час</b>: %{x}<br><b>Режим</b>: %{text}<extra></extra>',
                    text=mode_series.astype(str)
                ))
                layout = common_layout.copy()
                layout.update({
                    'title': {'text': 'Режими польоту'},
                    'xaxis': {'title': 'Час'},
                    'yaxis': {
                        'title': 'Режим',
                        'tickvals': list(mode_to_num.values()),
                        'ticktext': list(mode_to_num.keys())
                    },
                })
                fig.update_layout(layout)
                graphs['flight_modes'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return graphs
