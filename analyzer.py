import logging
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import geodesic
from plotly.subplots import make_subplots
from pymavlink import mavutil
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BehaviorAnalyzer:
    """Lightweight classifier trained from the current log's telemetry."""

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            random_state=42,
            max_iter=300,
        )
        self.scaler = StandardScaler()
        self.trained = False

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        if "altitude" in df.columns:
            altitude = pd.to_numeric(df["altitude"], errors="coerce").interpolate().ffill()
            features["altitude"] = altitude
            features["altitude_rate"] = altitude.diff().fillna(0)

        if "speed" in df.columns:
            features["speed"] = (
                pd.to_numeric(df["speed"], errors="coerce").interpolate().ffill() * 3.6
            )

        if "throttle" in df.columns:
            features["throttle"] = pd.to_numeric(
                df["throttle"], errors="coerce"
            ).interpolate().ffill()

        if "VibeZ" in df.columns:
            features["vibration"] = pd.to_numeric(
                df["VibeZ"], errors="coerce"
            ).interpolate().ffill()

        return features.dropna(axis=1, how="all")

    def _pseudo_labels(self, features: pd.DataFrame) -> pd.Series:
        vertical = features.get("altitude_rate", pd.Series(0, index=features.index))
        speed = features.get("speed", pd.Series(0, index=features.index))
        throttle = features.get("throttle", pd.Series(0, index=features.index))

        labels = pd.Series("ground/idle", index=features.index)
        labels = labels.mask(vertical > 0.6, "takeoff/climb")
        labels = labels.mask(vertical < -0.6, "descent/landing")
        labels = labels.mask((speed > 30) & (vertical.abs() <= 0.6), "cruise")
        labels = labels.mask((speed <= 10) & (throttle.between(5, 40)), "hover/loiter")
        return labels.fillna("ground/idle")

    def train_from_log(self, df: pd.DataFrame) -> None:
        features = self._extract_features(df)
        if features.empty or len(features) < 30:
            logger.info("Not enough data to train the behavior model.")
            return

        labels = self._pseudo_labels(features)
        self.model.fit(self.scaler.fit_transform(features.fillna(0)), labels)
        self.trained = True

    def analyze(self, df: pd.DataFrame) -> dict:
        features = self._extract_features(df)
        if features.empty or not self.trained:
            return {}

        predictions = pd.Series(self.model.predict(self.scaler.transform(features.fillna(0))))
        distribution = predictions.value_counts(normalize=True).sort_values(ascending=False)
        top_behavior = distribution.index[0]
        confidence = distribution.iloc[0] * 100

        return {
            "summary": f"Dominant behavior: {top_behavior} ({confidence:.1f}% confidence).",
            "distribution": distribution.to_dict(),
        }


class LogAnalyzer:
    GRAPH_OPTIONS = {
        "altitude",
        "speed",
        "throttle",
        "attitude",
        "battery",
        "vibration",
        "rc_channels",
        "flight_modes",
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._parse_log()
        self._preprocess_data()
        self.behavior_model = BehaviorAnalyzer()

    def _parse_log(self) -> pd.DataFrame:
        logger.info("Starting binary log parsing for %s", self.file_path)
        msg_types = [
            "GPS",
            "GPS2",
            "ATT",
            "CTUN",
            "NKF1",
            "BARO",
            "BAT",
            "MODE",
            "POWR",
            "CURR",
            "VIBE",
            "RCIN",
            "RCOU",
        ]

        connection = mavutil.mavlink_connection(self.file_path)
        rows = []

        while True:
            msg = connection.recv_match(type=msg_types, blocking=False)
            if msg is None:
                break

            msg_dict = msg.to_dict()
            msg_dict["msg_type"] = msg.get_type()
            rows.append(msg_dict)

        if not rows:
            raise ValueError("No supported telemetry messages were found in this log.")

        return pd.DataFrame(rows)

    def _preprocess_data(self) -> None:
        if self.data.empty:
            return

        numeric_cols = [
            "Alt",
            "Spd",
            "Roll",
            "Pitch",
            "Yaw",
            "Volt",
            "Curr",
            "Lat",
            "Lng",
            "Thr",
            "ThO",
            "ThD",
            "DAlt",
            "DSAlt",
            "SAlt",
            "NSats",
            "HDop",
            "VibeX",
            "VibeY",
            "VibeZ",
        ]
        numeric_cols.extend([f"C{i}" for i in range(1, 13)])
        numeric_cols.extend([f"Servo{i}" for i in range(1, 13)])

        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        if "TimeUS" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(self.data["TimeUS"], unit="us")
            self.data = self.data.set_index("timestamp").sort_index()

        msg_types = set(self.data["msg_type"].dropna().unique())

        if "BARO" in msg_types and "Alt" in self.data.columns:
            mask = self.data["msg_type"] == "BARO"
            self.data.loc[mask, "altitude"] = pd.to_numeric(self.data.loc[mask, "Alt"], errors="coerce")

        if "GPS" in msg_types:
            mask = self.data["msg_type"] == "GPS"
            if "Spd" in self.data.columns:
                self.data.loc[mask, "speed"] = pd.to_numeric(self.data.loc[mask, "Spd"], errors="coerce")
            if "Lat" in self.data.columns and "Lng" in self.data.columns:
                self.data.loc[mask, "lat"] = self.data.loc[mask, "Lat"] / 1e7
                self.data.loc[mask, "lon"] = self.data.loc[mask, "Lng"] / 1e7

        if "CTUN" in msg_types:
            mask = self.data["msg_type"] == "CTUN"
            for source in ("ThO", "ThD", "Thr"):
                if source in self.data.columns:
                    self.data.loc[mask, "throttle"] = pd.to_numeric(
                        self.data.loc[mask, source], errors="coerce"
                    )
                    break

        if "RCIN" in msg_types:
            mask = self.data["msg_type"] == "RCIN"
            for i in range(1, 13):
                source = f"C{i}"
                if source in self.data.columns:
                    self.data.loc[mask, f"rcin_C{i}"] = pd.to_numeric(
                        self.data.loc[mask, source], errors="coerce"
                    )

        if "RCOU" in msg_types:
            mask = self.data["msg_type"] == "RCOU"
            for i in range(1, 13):
                for source in (f"C{i}", f"Servo{i}"):
                    if source in self.data.columns:
                        self.data.loc[mask, f"rcout_C{i}"] = pd.to_numeric(
                            self.data.loc[mask, source], errors="coerce"
                        )
                        break

        if "MODE" in msg_types:
            mask = self.data["msg_type"] == "MODE"
            for source in ("Mode", "ModeNum"):
                if source in self.data.columns:
                    self.data.loc[mask, "flight_mode"] = self.data.loc[mask, source]
                    break

        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            method = "time" if isinstance(self.data.index, pd.DatetimeIndex) else "linear"
            self.data[numeric_data.columns] = numeric_data.interpolate(method=method)

        self._trim_to_active_window()

    def _trim_to_active_window(self) -> None:
        if "throttle" not in self.data.columns:
            return

        throttle = pd.to_numeric(self.data["throttle"], errors="coerce").dropna()
        if throttle.empty:
            return

        start_candidates = throttle[throttle >= 40]
        if start_candidates.empty:
            return

        first_idx = start_candidates.index[0]
        end_candidates = throttle[(throttle.index > first_idx) & (throttle <= 0.5)]
        if end_candidates.empty:
            return

        last_idx = end_candidates.index[-1]
        if last_idx > first_idx:
            logger.info("Trimming active flight window from %s to %s", first_idx, last_idx)
            self.data = self.data.loc[first_idx:last_idx]

    def _time_seconds(self, index: pd.Index) -> np.ndarray:
        if isinstance(index, pd.DatetimeIndex):
            return (index - index[0]).total_seconds().to_numpy()
        return np.arange(len(index), dtype=float)

    def _series(self, column: str) -> pd.Series:
        if column not in self.data.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.data[column], errors="coerce").dropna()

    def _add_stat(self, stats: dict, key: str, value) -> None:
        if value is None:
            return
        if isinstance(value, float) and pd.isna(value):
            return
        stats[key] = value

    def estimate_distance_from_speed(self):
        if "speed" not in self.data.columns or not isinstance(self.data.index, pd.DatetimeIndex):
            return None

        series = pd.to_numeric(self.data["speed"], errors="coerce").dropna()
        if len(series) < 2:
            return None

        seconds = self._time_seconds(series.index)
        distance_m = np.trapz(series.to_numpy(), x=seconds)
        return float(distance_m)

    def _distance_from_gps(self):
        if not {"lat", "lon"} <= set(self.data.columns):
            return None

        gps_points = self.data[
            (self.data["msg_type"] == "GPS")
            & (self.data["lat"].abs() > 0.001)
            & (self.data["lon"].abs() > 0.001)
        ][["lat", "lon"]].dropna()

        if len(gps_points) < 2:
            return None

        distance_km = 0.0
        points = gps_points.to_numpy()
        for idx in range(1, len(points)):
            try:
                distance_km += geodesic(points[idx - 1], points[idx]).km
            except Exception:
                continue
        return distance_km or None

    def get_basic_statistics(self):
        stats = {}
        if self.data.empty:
            return stats

        if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 1:
            duration = self.data.index[-1] - self.data.index[0]
            self._add_stat(stats, "flight_duration", str(duration))
            self._add_stat(stats, "log_start_time", self.data.index[0].strftime("%Y-%m-%d %H:%M:%S"))
            self._add_stat(stats, "log_end_time", self.data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))

        gps_distance_km = self._distance_from_gps()
        speed_distance_m = self.estimate_distance_from_speed()
        if gps_distance_km is not None:
            self._add_stat(stats, "total_distance_km", f"{gps_distance_km:.2f} km")
            self._add_stat(stats, "total_distance_m", f"{gps_distance_km * 1000:.0f} m")
        elif speed_distance_m is not None:
            self._add_stat(stats, "total_distance_km", f"{speed_distance_m / 1000:.2f} km")
            self._add_stat(stats, "total_distance_m", f"{speed_distance_m:.0f} m")

        if speed_distance_m is not None:
            self._add_stat(stats, "estimated_distance_km", f"{speed_distance_m / 1000:.2f} km")
            self._add_stat(stats, "estimated_distance_m", f"{speed_distance_m:.0f} m")

        altitude = self._series("altitude")
        if not altitude.empty:
            self._add_stat(stats, "max_altitude", f"{altitude.max():.2f} m")
            self._add_stat(stats, "min_altitude", f"{altitude.min():.2f} m")
            self._add_stat(stats, "avg_altitude", f"{altitude.mean():.2f} m")
            self._add_stat(stats, "altitude_range", f"{(altitude.max() - altitude.min()):.2f} m")
            self._add_stat(stats, "start_altitude", f"{altitude.iloc[0]:.2f} m")
            self._add_stat(stats, "end_altitude", f"{altitude.iloc[-1]:.2f} m")
            if isinstance(altitude.index, pd.DatetimeIndex) and len(altitude) > 1:
                seconds = altitude.index.to_series().diff().dt.total_seconds().replace(0, pd.NA)
                climb = (altitude.diff() / seconds).dropna()
                if not climb.empty:
                    self._add_stat(stats, "max_climb_rate", f"{climb.max():.2f} m/s")
                    self._add_stat(stats, "min_climb_rate", f"{climb.min():.2f} m/s")

        speed = self._series("speed")
        if not speed.empty:
            kmh = speed * 3.6
            self._add_stat(stats, "max_speed", f"{kmh.max():.2f} km/h")
            self._add_stat(stats, "min_speed", f"{kmh.min():.2f} km/h")
            self._add_stat(stats, "avg_speed", f"{kmh.mean():.2f} km/h")
            self._add_stat(stats, "speed_range", f"{(kmh.max() - kmh.min()):.2f} km/h")
            self._add_stat(stats, "start_speed", f"{kmh.iloc[0]:.2f} km/h")
            self._add_stat(stats, "end_speed", f"{kmh.iloc[-1]:.2f} km/h")

        throttle = self._series("throttle")
        if not throttle.empty:
            self._add_stat(stats, "max_throttle", f"{throttle.max():.1f}%")
            self._add_stat(stats, "min_throttle", f"{throttle.min():.1f}%")
            self._add_stat(stats, "avg_throttle", f"{throttle.mean():.1f}%")
            self._add_stat(stats, "time_over_80pct", f"{(throttle > 80).mean() * 100:.1f}%")
            self._add_stat(stats, "time_below_20pct", f"{(throttle < 20).mean() * 100:.1f}%")

        voltage = self._series("Volt")
        if not voltage.empty:
            self._add_stat(stats, "max_voltage", f"{voltage.max():.2f} V")
            self._add_stat(stats, "min_voltage", f"{voltage.min():.2f} V")
            self._add_stat(stats, "avg_voltage", f"{voltage.mean():.2f} V")
            self._add_stat(stats, "start_voltage", f"{voltage.iloc[0]:.2f} V")
            self._add_stat(stats, "end_voltage", f"{voltage.iloc[-1]:.2f} V")
            self._add_stat(stats, "voltage_drop", f"{(voltage.iloc[0] - voltage.iloc[-1]):.2f} V")

        current = self._series("Curr")
        if not current.empty:
            self._add_stat(stats, "max_current", f"{current.max():.2f} A")
            self._add_stat(stats, "avg_current", f"{current.mean():.2f} A")
            if isinstance(current.index, pd.DatetimeIndex) and len(current) > 1:
                hours = (current.index - current.index[0]).total_seconds() / 3600.0
                self._add_stat(stats, "consumed_ah", f"{np.trapz(current.to_numpy(), x=hours):.2f} Ah")
            if not voltage.empty:
                power = pd.concat(
                    [current.rename("current"), voltage.rename("voltage")],
                    axis=1,
                ).dropna()
                if not power.empty:
                    self._add_stat(
                        stats,
                        "avg_power_w",
                        f"{(power['current'] * power['voltage']).mean():.2f} W",
                    )

        sats = self._series("NSats")
        if not sats.empty:
            self._add_stat(stats, "gps_satellites_avg", f"{sats.mean():.1f}")
            self._add_stat(stats, "gps_satellites_min", f"{sats.min():.0f}")
            self._add_stat(stats, "gps_satellites_max", f"{sats.max():.0f}")

        hdop = self._series("HDop")
        if not hdop.empty:
            self._add_stat(stats, "gps_hdop_avg", f"{hdop.mean():.2f}")
            self._add_stat(stats, "gps_hdop_min", f"{hdop.min():.2f}")
            self._add_stat(stats, "gps_hdop_max", f"{hdop.max():.2f}")

        for axis in ("Roll", "Pitch", "Yaw"):
            series = self._series(axis)
            if not series.empty:
                axis_key = axis.lower()
                self._add_stat(stats, f"max_{axis_key}", f"{series.max():.1f} deg")
                self._add_stat(stats, f"min_{axis_key}", f"{series.min():.1f} deg")
                self._add_stat(stats, f"avg_{axis_key}", f"{series.mean():.1f} deg")

        vibration_columns = [col for col in ("VibeX", "VibeY", "VibeZ") if col in self.data.columns]
        if vibration_columns:
            vibration = pd.DataFrame(
                {col: pd.to_numeric(self.data[col], errors="coerce") for col in vibration_columns}
            ).dropna(how="all")
            if not vibration.empty:
                self._add_stat(stats, "vibration_peak", f"{vibration.max().max():.2f} m/s^2")
                self._add_stat(stats, "vibration_avg", f"{vibration.mean().mean():.2f} m/s^2")

        for prefix in ("rcin_", "rcout_"):
            channels = [col for col in self.data.columns if col.startswith(prefix)]
            for channel in sorted(channels)[:8]:
                series = pd.to_numeric(self.data[channel], errors="coerce").dropna()
                if not series.empty:
                    self._add_stat(stats, f"{channel}_avg", f"{series.mean():.0f} us")
                    self._add_stat(stats, f"{channel}_range", f"{(series.max() - series.min()):.0f} us")

        if "flight_mode" in self.data.columns:
            modes = self.data["flight_mode"].ffill().dropna()
            if not modes.empty:
                counts = modes.value_counts()
                self._add_stat(stats, "flight_modes_detected", ", ".join(counts.index.astype(str)))
                self._add_stat(stats, "flight_mode_changes", max(int(modes.ne(modes.shift()).sum() - 1), 0))
                self._add_stat(stats, "most_used_mode", str(counts.idxmax()))
                if isinstance(modes.index, pd.DatetimeIndex) and len(modes.index) > 1:
                    durations = modes.to_frame("mode")
                    durations["duration_s"] = (
                        durations.index.to_series().diff().shift(-1).dt.total_seconds().fillna(0)
                    )
                    by_mode = durations.groupby("mode")["duration_s"].sum().sort_values(ascending=False)
                    if not by_mode.empty:
                        top_mode = by_mode.index[0]
                        self._add_stat(stats, "longest_mode", f"{top_mode} ({by_mode.iloc[0] / 60:.1f} min)")
                        self._add_stat(
                            stats,
                            "mode_duration_breakdown",
                            ", ".join(f"{mode}: {seconds / 60:.1f} min" for mode, seconds in by_mode.items()),
                        )

        try:
            self.behavior_model.train_from_log(self.data)
            behavior = self.behavior_model.analyze(self.data)
            if behavior:
                stats["behavior_summary"] = behavior["summary"]
                stats["behavior_breakdown"] = ", ".join(
                    f"{label}: {share * 100:.1f}%"
                    for label, share in behavior["distribution"].items()
                )
        except Exception:
            logger.warning("Could not build behavior model:\n%s", traceback.format_exc())

        return stats

    def _apply_layout(self, fig: go.Figure, title: str, y_title: str) -> None:
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=480,
            margin=dict(l=48, r=32, b=48, t=72),
            paper_bgcolor="#fffaf1",
            plot_bgcolor="#fffaf1",
            font=dict(
                family="'Segoe UI Variable', 'Segoe UI', sans-serif",
                size=13,
                color="#1a2233",
            ),
            title=dict(text=title, font=dict(size=22, color="#14213d")),
            xaxis=dict(title="Time", gridcolor="rgba(20, 33, 61, 0.08)", zeroline=False),
            yaxis=dict(title=y_title, gridcolor="rgba(20, 33, 61, 0.08)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    def _render_figure(self, fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def generate_basic_graphs(self, include=None):
        graphs = {}
        if self.data.empty:
            return graphs

        include = set(include or self.GRAPH_OPTIONS) & self.GRAPH_OPTIONS

        if "altitude" in include and "altitude" in self.data.columns:
            altitude = pd.to_numeric(self.data["altitude"], errors="coerce").dropna()
            if not altitude.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=altitude.index,
                        y=altitude,
                        mode="lines",
                        name="BARO altitude",
                        line=dict(color="#1f5f8b", width=2.5),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Altitude</b>: %{y:.2f} m<extra></extra>",
                    )
                )
                self._apply_layout(fig, "Altitude profile", "Altitude (m)")
                graphs["altitude"] = self._render_figure(fig)

        if "speed" in include and "speed" in self.data.columns:
            speed = pd.to_numeric(self.data["speed"], errors="coerce").dropna()
            if not speed.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=speed.index,
                        y=speed * 3.6,
                        mode="lines",
                        name="GPS speed",
                        line=dict(color="#2e7d32", width=2.5),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Speed</b>: %{y:.2f} km/h<extra></extra>",
                    )
                )
                self._apply_layout(fig, "Speed trace", "Speed (km/h)")
                graphs["speed"] = self._render_figure(fig)

        if "throttle" in include and "throttle" in self.data.columns:
            throttle = pd.to_numeric(self.data["throttle"], errors="coerce").dropna()
            if not throttle.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=throttle.index,
                        y=throttle,
                        mode="lines",
                        name="Throttle",
                        line=dict(color="#c56a1a", width=2.5),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Throttle</b>: %{y:.1f}%<extra></extra>",
                    )
                )
                self._apply_layout(fig, "Throttle trace", "Throttle (%)")
                graphs["throttle"] = self._render_figure(fig)

        if "attitude" in include and {"Roll", "Pitch"} <= set(self.data.columns):
            roll = pd.to_numeric(self.data["Roll"], errors="coerce").dropna()
            pitch = pd.to_numeric(self.data["Pitch"], errors="coerce").dropna()
            if not roll.empty and not pitch.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=roll.index,
                        y=roll,
                        mode="lines",
                        name="Roll",
                        line=dict(color="#7b4f9d", width=2.2),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Roll</b>: %{y:.1f} deg<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pitch.index,
                        y=pitch,
                        mode="lines",
                        name="Pitch",
                        line=dict(color="#188977", width=2.2),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Pitch</b>: %{y:.1f} deg<extra></extra>",
                    )
                )
                self._apply_layout(fig, "Attitude trace", "Angle (deg)")
                graphs["attitude"] = self._render_figure(fig)

        if "battery" in include and "Volt" in self.data.columns:
            voltage = pd.to_numeric(self.data["Volt"], errors="coerce").dropna()
            current = (
                pd.to_numeric(self.data["Curr"], errors="coerce").dropna()
                if "Curr" in self.data.columns
                else None
            )
            if not voltage.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=voltage.index,
                        y=voltage,
                        mode="lines",
                        name="Voltage",
                        line=dict(color="#e74c3c", width=2.2),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Voltage</b>: %{y:.2f} V<extra></extra>",
                    ),
                    secondary_y=False,
                )
                if current is not None and not current.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=current.index,
                            y=current,
                            mode="lines",
                            name="Current",
                            line=dict(color="#8e44ad", width=2.0, dash="dot"),
                            hovertemplate="<b>Time</b>: %{x}<br><b>Current</b>: %{y:.2f} A<extra></extra>",
                        ),
                        secondary_y=True,
                    )
                self._apply_layout(fig, "Power system", "Voltage (V)")
                fig.update_yaxes(title_text="Current (A)", secondary_y=True)
                graphs["battery"] = self._render_figure(fig)

        if "vibration" in include:
            columns = [col for col in ("VibeX", "VibeY", "VibeZ") if col in self.data.columns]
            if columns:
                vibration = pd.DataFrame(
                    {col: pd.to_numeric(self.data[col], errors="coerce") for col in columns}
                ).dropna(how="all")
                if not vibration.empty:
                    fig = go.Figure()
                    colors = ["#ff7f50", "#1abc9c", "#9b59b6"]
                    for idx, col in enumerate(columns):
                        fig.add_trace(
                            go.Scatter(
                                x=vibration.index,
                                y=vibration[col],
                                mode="lines",
                                name=col,
                                line=dict(color=colors[idx % len(colors)], width=2),
                                hovertemplate=f"<b>Time</b>: %{{x}}<br><b>{col}</b>: %{{y:.2f}} m/s^2<extra></extra>",
                            )
                        )
                    self._apply_layout(fig, "Vibration profile", "Acceleration (m/s^2)")
                    graphs["vibration"] = self._render_figure(fig)

        if "rc_channels" in include:
            columns = [col for col in self.data.columns if col.startswith("rcin_C")]
            if columns:
                rc = self.data[columns].dropna(how="all")
                if not rc.empty:
                    fig = go.Figure()
                    palette = ["#3498db", "#e67e22", "#9b59b6", "#27ae60", "#c0392b", "#2980b9", "#8e44ad", "#16a085"]
                    for idx, col in enumerate(sorted(columns)[:8]):
                        fig.add_trace(
                            go.Scatter(
                                x=rc.index,
                                y=rc[col],
                                mode="lines",
                                name=col.replace("rcin_", "").upper(),
                                line=dict(color=palette[idx % len(palette)], width=1.5),
                                hovertemplate="<b>Time</b>: %{x}<br><b>RC</b>: %{y:.0f} us<extra></extra>",
                            )
                        )
                    self._apply_layout(fig, "RC input channels", "PWM (us)")
                    graphs["rc_channels"] = self._render_figure(fig)

        if "flight_modes" in include and "flight_mode" in self.data.columns:
            modes = self.data["flight_mode"].ffill().dropna()
            if not modes.empty:
                unique_modes = modes.dropna().unique()
                mode_to_num = {mode: idx for idx, mode in enumerate(unique_modes)}
                encoded = modes.map(mode_to_num)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=encoded.index,
                        y=encoded,
                        mode="lines",
                        line_shape="hv",
                        name="Mode",
                        line=dict(color="#2c3e50", width=2),
                        text=modes.astype(str),
                        hovertemplate="<b>Time</b>: %{x}<br><b>Mode</b>: %{text}<extra></extra>",
                    )
                )
                self._apply_layout(fig, "Flight mode timeline", "Mode")
                fig.update_yaxes(
                    tickvals=list(mode_to_num.values()),
                    ticktext=list(mode_to_num.keys()),
                )
                graphs["flight_modes"] = self._render_figure(fig)

        return graphs
