# ArduPilot Log Analyzer

Web app for reviewing ArduPilot `.bin` flight logs with grouped metrics and interactive charts.

## What It Does

- Uploads a `.bin` log through a Flask UI
- Extracts key telemetry from ArduPilot messages
- Builds grouped flight statistics
- Renders interactive Plotly charts for altitude, speed, throttle, attitude, battery, vibration, RC input, and flight modes
- Adds a lightweight behavior summary based on the current log

## Requirements

- Python 3.8+
- `pip`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Project Structure

```text
app.py
analyzer.py
requirements.txt
static/
  css/styles.css
  js/scripts.js
templates/
  base.html
  index.html
  results.html
```

## Notes

- Upload size defaults to `512 MB`.
- You can change the upload limit with `LOG_ANALYZER_MAX_UPLOAD_MB`.
- You can set `FLASK_SECRET_KEY` if you do not want to use the local default key.
