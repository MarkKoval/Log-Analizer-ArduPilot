from flask import Flask, session, render_template, request, redirect, url_for, send_from_directory, flash
from flask_session import Session
import tempfile
import os
import uuid
from werkzeug.utils import secure_filename
from analyzer import LogAnalyzer

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'bin'}
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_sessions')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

Session(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/session-type')
def session_type():
    return str(type(session))  # Ось тут

@app.route('/test-session')
def test_session():
    session['big'] = 'x' * 5000
    return 'Session записана'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Перевірка чи файл був відправлений
        if 'file' not in request.files:
            flash('Не вибрано файл для завантаження')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Якщо користувач не вибрав файл
        if file.filename == '':
            flash('Не вибрано файл')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Генеруємо унікальне ім'я файлу
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Після збереження файлу перенаправляємо на сторінку аналізу
                return redirect(url_for('analysis_page', filename=filename))
                
            except Exception as e:
                flash(f'Помилка збереження файлу: {str(e)}')
                return redirect(request.url)
    
    return render_template('index.html')

@app.route('/analysis/<filename>')
def analysis_page(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        analyzer = LogAnalyzer(filepath)
        
        basic_stats = analyzer.get_basic_statistics()
        graphs = analyzer.generate_basic_graphs()
        
        return render_template('results.html', 
                           basic_stats=basic_stats,
                           graphs=graphs,
                           filename=filename)
    
    except Exception as e:
        flash(f'Помилка аналізу файлу: {str(e)}')
        return redirect(url_for('upload_file'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)