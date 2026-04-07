import os
import tempfile
import uuid
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from analyzer import LogAnalyzer

MAX_UPLOAD_SIZE_MB = int(os.getenv("LOG_ANALYZER_MAX_UPLOAD_MB", "512"))
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "log_analyzer_uploads"

OPTION_METADATA = {
    "basic": {
        "label": "Flight overview",
        "description": "Core metrics like duration, distance, altitude, and speed.",
    },
    "altitude": {
        "label": "Altitude",
        "description": "BARO altitude profile across the flight.",
    },
    "speed": {
        "label": "Speed",
        "description": "GPS speed over time.",
    },
    "throttle": {
        "label": "Throttle",
        "description": "Throttle output trend and throttle-heavy segments.",
    },
    "attitude": {
        "label": "Attitude",
        "description": "Roll and pitch behavior during the mission.",
    },
    "battery": {
        "label": "Battery",
        "description": "Voltage and current draw from the power system.",
    },
    "vibration": {
        "label": "Vibration",
        "description": "IMU vibration levels across axes.",
    },
    "rc_channels": {
        "label": "RC channels",
        "description": "Input channel behavior throughout the log.",
    },
    "flight_modes": {
        "label": "Flight modes",
        "description": "Mode switches and how long each mode was active.",
    },
}

ANALYSIS_OPTIONS_ORDER = list(OPTION_METADATA.keys())
DEFAULT_ANALYSIS_OPTIONS = set(ANALYSIS_OPTIONS_ORDER)

GRAPH_METADATA = {
    "altitude": {
        "title": "Altitude profile",
        "description": "BARO altitude across the captured flight window.",
    },
    "speed": {
        "title": "Speed trace",
        "description": "Ground speed converted to km/h from GPS data.",
    },
    "throttle": {
        "title": "Throttle trace",
        "description": "Throttle percentage over time.",
    },
    "attitude": {
        "title": "Attitude trace",
        "description": "Roll and pitch angles through the mission.",
    },
    "battery": {
        "title": "Power system",
        "description": "Voltage and current draw from onboard power telemetry.",
    },
    "vibration": {
        "title": "Vibration profile",
        "description": "VibeX, VibeY, and VibeZ compared together.",
    },
    "rc_channels": {
        "title": "RC input channels",
        "description": "Primary RC channels seen in the log.",
    },
    "flight_modes": {
        "title": "Flight mode timeline",
        "description": "Mode changes laid out chronologically.",
    },
}

SUMMARY_METRICS = [
    ("flight_duration", "Duration"),
    ("total_distance_km", "Distance"),
    ("max_altitude", "Max altitude"),
    ("max_speed", "Max speed"),
    ("avg_throttle", "Avg throttle"),
    ("min_voltage", "Min voltage"),
    ("most_used_mode", "Primary mode"),
]

STAT_LABELS = {
    "flight_duration": "Flight duration",
    "log_start_time": "Log start time",
    "log_end_time": "Log end time",
    "estimated_distance_m": "Estimated distance",
    "estimated_distance_km": "Estimated distance (km)",
    "total_distance_km": "Total distance",
    "total_distance_m": "Total distance (m)",
    "max_altitude": "Max altitude",
    "min_altitude": "Min altitude",
    "avg_altitude": "Average altitude",
    "median_altitude": "Median altitude",
    "altitude_std": "Altitude deviation",
    "altitude_variance": "Altitude variance",
    "altitude_25th": "Altitude P25",
    "altitude_75th": "Altitude P75",
    "altitude_range": "Altitude range",
    "start_altitude": "Start altitude",
    "end_altitude": "End altitude",
    "max_climb_rate": "Max climb rate",
    "min_climb_rate": "Min climb rate",
    "avg_climb_rate": "Average climb rate",
    "climb_rate_std": "Climb rate deviation",
    "time_above_10m": "Time above 10 m",
    "time_below_5m": "Time below 5 m",
    "max_speed": "Max speed",
    "min_speed": "Min speed",
    "avg_speed": "Average speed",
    "median_speed": "Median speed",
    "speed_std": "Speed deviation",
    "speed_variance": "Speed variance",
    "speed_p95": "Speed P95",
    "speed_p05": "Speed P05",
    "speed_range": "Speed range",
    "start_speed": "Start speed",
    "end_speed": "End speed",
    "max_acceleration": "Max acceleration",
    "min_acceleration": "Min acceleration",
    "avg_acceleration": "Average acceleration",
    "acceleration_std": "Acceleration deviation",
    "time_above_50kmh": "Time above 50 km/h",
    "time_below_10kmh": "Time below 10 km/h",
    "max_throttle": "Max throttle",
    "min_throttle": "Min throttle",
    "avg_throttle": "Average throttle",
    "median_throttle": "Median throttle",
    "throttle_std": "Throttle deviation",
    "throttle_range": "Throttle range",
    "time_over_80pct": "Time over 80%",
    "time_below_20pct": "Time below 20%",
    "max_voltage": "Max voltage",
    "min_voltage": "Min voltage",
    "avg_voltage": "Average voltage",
    "median_voltage": "Median voltage",
    "voltage_std": "Voltage deviation",
    "start_voltage": "Start voltage",
    "end_voltage": "End voltage",
    "voltage_drop": "Voltage drop",
    "voltage_drop_pct": "Voltage drop %",
    "max_current": "Max current",
    "avg_current": "Average current",
    "median_current": "Median current",
    "current_std": "Current deviation",
    "start_current": "Start current",
    "end_current": "End current",
    "consumed_ah": "Consumed Ah",
    "avg_power_w": "Average power",
    "gps_satellites_avg": "Average satellites",
    "gps_satellites_min": "Min satellites",
    "gps_satellites_max": "Max satellites",
    "gps_hdop_avg": "Average HDOP",
    "gps_hdop_min": "Min HDOP",
    "gps_hdop_max": "Max HDOP",
    "vibration_peak": "Peak vibration",
    "vibration_avg": "Average vibration",
    "flight_modes_detected": "Detected modes",
    "flight_mode_changes": "Mode changes",
    "most_used_mode": "Most used mode",
    "longest_mode": "Longest mode",
    "mode_duration_breakdown": "Mode durations",
    "behavior_summary": "Behavior summary",
    "behavior_breakdown": "Behavior breakdown",
}

STAT_SECTIONS = [
    ("Flight summary", {"flight_duration", "log_start_time", "log_end_time", "estimated_distance_m", "estimated_distance_km", "total_distance_m", "total_distance_km", "behavior_summary", "behavior_breakdown"}),
    ("Altitude", {key for key in STAT_LABELS if key.startswith("altitude") or key.endswith("altitude") or "climb_rate" in key or key in {"time_above_10m", "time_below_5m"}}),
    ("Speed", {key for key in STAT_LABELS if key.startswith("speed") or key.endswith("speed") or "acceleration" in key or key in {"time_above_50kmh", "time_below_10kmh"}}),
    ("Throttle", {key for key in STAT_LABELS if key.startswith("throttle") or key in {"max_throttle", "min_throttle", "avg_throttle", "median_throttle", "time_over_80pct", "time_below_20pct"}}),
    ("Battery and power", {key for key in STAT_LABELS if "voltage" in key or "current" in key or key in {"consumed_ah", "avg_power_w"}}),
    ("GPS quality", {"gps_satellites_avg", "gps_satellites_min", "gps_satellites_max", "gps_hdop_avg", "gps_hdop_min", "gps_hdop_max"}),
    ("Attitude", {key for key in STAT_LABELS if any(axis in key for axis in ("roll", "pitch", "yaw"))}),
    ("Vibration", {key for key in STAT_LABELS if key.startswith("vibe") or key.startswith("vibration")}),
    ("Flight modes", {"flight_modes_detected", "flight_mode_changes", "most_used_mode", "longest_mode", "mode_duration_breakdown"}),
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_ROOT)
app.config["ALLOWED_EXTENSIONS"] = {"bin"}
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "log-analyzer-local-dev")

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def build_upload_path(filename: str) -> Path:
    return UPLOAD_ROOT / Path(filename).name


def original_filename(filename: str) -> str:
    name = Path(filename).name
    prefix, separator, remainder = name.partition("_")
    if separator and len(prefix) == 32:
        try:
            int(prefix, 16)
            return remainder or name
        except ValueError:
            return name
    return name


def format_stat_label(key: str) -> str:
    if key in STAT_LABELS:
        return STAT_LABELS[key]

    normalized = key.replace("rcin_", "RC IN ").replace("rcout_", "RC OUT ")
    normalized = normalized.replace("_", " ")
    return normalized.title()


def build_summary_metrics(stats: dict):
    metrics = []
    for key, label in SUMMARY_METRICS:
        value = stats.get(key)
        if value:
            metrics.append({"label": label, "value": value})
    return metrics


def build_stats_sections(stats: dict):
    grouped_keys = set()
    sections = []

    for title, keys in STAT_SECTIONS:
        items = []
        for key, value in stats.items():
            if key in keys:
                items.append({"label": format_stat_label(key), "value": value})
                grouped_keys.add(key)
        if items:
            sections.append({"title": title, "items": items})

    uncategorized = [
        {"label": format_stat_label(key), "value": value}
        for key, value in stats.items()
        if key not in grouped_keys
    ]
    if uncategorized:
        sections.append({"title": "Additional metrics", "items": uncategorized})

    return sections


def build_graph_sections(graphs: dict):
    ordered = []
    for key in ANALYSIS_OPTIONS_ORDER:
        if key not in graphs:
            continue
        meta = GRAPH_METADATA.get(
            key,
            {"title": format_stat_label(key), "description": "Telemetry chart"},
        )
        ordered.append(
            {
                "key": key,
                "title": meta["title"],
                "description": meta["description"],
                "html": graphs[key],
            }
        )
    return ordered


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash(f"File is too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB} MB.", "error")
    return redirect(url_for("upload_file"))


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        selected_options = set(request.form.getlist("analysis_options")) or DEFAULT_ANALYSIS_OPTIONS

        if "file" not in request.files:
            flash("Choose a .bin log file before starting analysis.", "error")
            return redirect(request.url)

        file = request.files["file"]
        if not file or file.filename == "":
            flash("Choose a .bin log file before starting analysis.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only ArduPilot .bin log files are supported.", "error")
            return redirect(request.url)

        try:
            filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
            file_path = build_upload_path(filename)
            file.save(file_path)

            return redirect(
                url_for(
                    "analysis_page",
                    filename=filename,
                    options=",".join(
                        option for option in ANALYSIS_OPTIONS_ORDER if option in selected_options
                    ),
                )
            )
        except Exception as exc:
            flash(f"Failed to save the uploaded file: {exc}", "error")
            return redirect(request.url)

    analysis_options = [
        {"key": key, **OPTION_METADATA[key]}
        for key in ANALYSIS_OPTIONS_ORDER
    ]
    return render_template(
        "index.html",
        analysis_options=analysis_options,
        max_upload_mb=MAX_UPLOAD_SIZE_MB,
        page_class="page-home",
        title="ArduPilot Log Analyzer",
    )


@app.route("/analysis/<filename>")
def analysis_page(filename):
    file_path = build_upload_path(filename)
    if not file_path.exists():
        flash("The uploaded log file could not be found. Upload it again and rerun the analysis.", "error")
        return redirect(url_for("upload_file"))

    selected_options = {
        option
        for option in request.args.get("options", "").split(",")
        if option in DEFAULT_ANALYSIS_OPTIONS
    }
    if not selected_options:
        selected_options = DEFAULT_ANALYSIS_OPTIONS

    try:
        analyzer = LogAnalyzer(str(file_path))
        basic_stats = analyzer.get_basic_statistics() if "basic" in selected_options else {}
        graphs = analyzer.generate_basic_graphs(include=selected_options)

        selected_option_labels = [
            OPTION_METADATA[key]["label"]
            for key in ANALYSIS_OPTIONS_ORDER
            if key in selected_options
        ]

        return render_template(
            "results.html",
            filename=filename,
            display_filename=original_filename(filename),
            selected_options=selected_option_labels,
            summary_metrics=build_summary_metrics(basic_stats),
            stat_sections=build_stats_sections(basic_stats),
            graph_sections=build_graph_sections(graphs),
            page_class="page-results",
            include_plotly=bool(graphs),
            title="Analysis Results",
        )
    except Exception as exc:
        flash(f"Failed to analyze the log file: {exc}", "error")
        return redirect(url_for("upload_file"))


@app.route("/download/<filename>")
def download_file(filename):
    file_path = build_upload_path(filename)
    if not file_path.exists():
        flash("The source log is no longer available for download.", "error")
        return redirect(url_for("upload_file"))
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        file_path.name,
        as_attachment=True,
        download_name=original_filename(file_path.name),
    )


if __name__ == "__main__":
    app.run(debug=True)
