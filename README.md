# ArduPilot Log Analyzer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) 

![License](https://img.shields.io/badge/license-MIT-green.svg)

A web-based tool for analyzing flight logs from ArduPilot-powered drones, providing both basic statistics and advanced AI-powered insights **(in dev)** .

## Features

- 📊 Basic flight statistics (altitude, speed, duration, distance)
- 🖼️ Interactive flight parameter graphs
- 🤖 AI-powered anomaly detection and flight pattern clustering **(in dev)**
- 🌍 GPS trajectory visualization **(in dev)**
- 🚀 Optimized for large log files
- 🕸️ Web-based interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MarkKoval/Log-Analizer-ArduPilot.git
cd Log-Analizer-ArduPilot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open in browser:
```text
http://localhost:5000
```

## Usage
- Upload your ArduPilot .bin log file
- View basic flight statistics
- Explore interactive graphs
- Analyze flight patterns with AI tools

## Supported Log Parameters
- Altitude (BARO)
- Speed (GPS)
- Throttle
- GPS coordinates
- Battery voltage/current
- Vibration data
- Flight modes

## File Structure
```bash
Log-Analizer-ArduPilot/
├── app.py                # Main Flask application
├── analyzer.py           # Core analysis logic
├── static/               # Static files (CSS, JS)
│   ├── css/
│   └── js/
├── templates/            # HTML templates
│   ├── index.html
│   └── results.html
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Screenshots

<img width="1912" height="954" alt="Image" src="https://github.com/user-attachments/assets/d8a56a6a-e6de-4041-9189-813d395297d0" />

<img width="1912" height="1338" alt="Image" src="https://github.com/user-attachments/assets/fa98d99a-3f01-42e5-ad23-603c74abb825" />

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.