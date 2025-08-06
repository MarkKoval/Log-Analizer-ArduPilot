import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from analyzer import LogAnalyzer
import tempfile
import uuid

app = Flask(__name__)

# Налаштування
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100MB limit
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['ALLOWED_EXTENSIONS'] = {'bin'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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