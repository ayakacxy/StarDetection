import logging
import sys
import os
from docx2pdf import convert
from datetime import datetime
from flask import Flask, jsonify, send_file, send_from_directory, request
from flask_cors import CORS, cross_origin
import torch
import time
import numpy as np
import random
import subprocess
import shutil
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QIcon
from pdf_processor import PDFReader, create_report, split_text, filter_text, count_characters_text, overall_ai_detection_result
from model_inference import calculate_generation_probability
import threading
import ctypes
import requests
import easyocr
import json
from logging.handlers import RotatingFileHandler
# 设置基本路径
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 日志配置
# 记录启动开始时间
start_time = time.time()
log_file = os.path.join(BASE_DIR, 'config', 'log.jsonl')
# 自定义JSON格式化程序
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage()
        }
        return json.dumps(log_record,ensure_ascii=False)

# 配置日志记录
handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8', delay=0)
formatter = JsonFormatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("===== Application started at {} =====".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
# Flask日志配置
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.INFO)
flask_log.addHandler(handler)

# 重定向标准输出和标准错误输出到日志文件
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip() != "":
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

# 设置随机种子以确保结果的可复现性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# Flask应用配置
flask_app = Flask(__name__)
CORS(flask_app, supports_credentials=True)  # 允许所有来源的跨域请求
#Flask的配置禁用彩色日志
flask_app.config['LOGGER_HANDLER_POLICY'] = 'always'
flask_app.logger.handlers = []
flask_app.logger.propagate = True

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'config', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'config', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

flask_app.static_folder = OUTPUT_FOLDER


MODEL_DIR = os.path.join(BASE_DIR, 'config', '.EasyOCR', 'model')

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, model_storage_directory=MODEL_DIR)
    
    if not os.path.exists(image_path):
        logger.error(f"File not found: {image_path}")
        return "File not found"
    
    # 调用EasyOCR进行文字识别
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)


@flask_app.route('/text', methods=['POST'])
@cross_origin()
def text():
    try:
        data = request.json
        language = data.get('language')
        text = data.get('text')

        if not text:
            return jsonify({'error': '未输入文本'}), 400

        segments = split_text(text, language)
        analysis = []

        for segment in segments:
            filtered_segment = filter_text(segment)
            ai_prob = calculate_generation_probability(filtered_segment, language)
            word_count = count_characters_text(segment, language)
            analysis.append({'text': segment, 'ai_prob': ai_prob, 'word_count': word_count})

        result = overall_ai_detection_result(analysis)

        return jsonify({'result': result, 'analysis': analysis})
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({'error': '请检查输入'}), 500

@flask_app.route('/config/upload', methods=['POST'])
@cross_origin()
def upload():
    file = request.files['file']
    language = request.form.get('language')  # 获取传递的语言参数
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
    print(language)  # 调试用，打印传递的语言参数

    file.save(file_path)

    if file_path.endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = extract_text_from_image(file_path)
        response = {'extracted_text': extracted_text}
        return jsonify(response), 200
    elif file_path.endswith('.pdf'):
        reader = PDFReader(file_path, language, OUTPUT_FOLDER)
        stats, segment_scores = reader.highlight_pdf()
        final_output_path = create_report(stats, segment_scores, filename, OUTPUT_FOLDER)
    elif file_path.endswith('.txt'):
        reader = PDFReader(file_path, language, OUTPUT_FOLDER)
        reader.process_txt(file_path, output_path)
        final_output_path = output_path
    elif file_path.endswith('.docx'):
        pdf_path = os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(filename)[0]}.pdf")
        convert_docx_to_pdf(file_path, pdf_path)
        reader = PDFReader(pdf_path, language, OUTPUT_FOLDER)
        stats, segment_scores = reader.highlight_pdf()
        final_output_path = create_report(stats, segment_scores, filename.replace('.docx', '.pdf'), OUTPUT_FOLDER)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    response = {'download_url': f'/download/{os.path.basename(final_output_path)}'}
    return jsonify(response), 200

@flask_app.route('/star.ico')
def star():
    return send_from_directory(os.path.join(BASE_DIR, 'config'), 'star.ico', mimetype='image/vnd.microsoft.icon')

@flask_app.route('/open_file', methods=['POST'])
@cross_origin()
def open_file():
    data = request.json
    filename = data.get('filename')
    if filename:
        possible_paths = [
            os.path.join(OUTPUT_FOLDER, f'final_processed_{filename}'),
            os.path.join(OUTPUT_FOLDER, f'processed_{filename}')
        ]

        if filename.lower().endswith('.docx'):
            docx_pdf_filename = filename.replace('.docx', '.pdf')
            possible_paths.insert(0, os.path.join(OUTPUT_FOLDER, f'final_processed_{docx_pdf_filename}'))
            possible_paths.insert(1, os.path.join(OUTPUT_FOLDER, f'processed_{docx_pdf_filename}'))

        file_path = next((path for path in possible_paths if os.path.exists(path)), None)

        if file_path:
            try:
                reader_exe_path = os.path.join(BASE_DIR, 'READER.exe')
                if file_path.lower().endswith('.pdf'):
                    subprocess.Popen([reader_exe_path, file_path], shell=True)
                elif file_path.lower().endswith('.txt'):
                    subprocess.Popen([reader_exe_path, file_path], shell=True)
                else:
                    return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400

                return jsonify({'status': 'success'}), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        else:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
    else:
        return jsonify({'status': 'error', 'message': 'Filename not provided'}), 400

@flask_app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'config', 'index.html'))

@flask_app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'config'), filename)

class Browser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("星鉴-机器生成文本检测系统")
        icon_path = os.path.join('config', 'star.png')
        self.setWindowIcon(QIcon(icon_path))
        screen_size = self.get_screen_size_3_4()
        self.setGeometry(100, 100, *screen_size)

        self.browser = QWebEngineView()
        self.browser.setUrl("http://localhost:5000")

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(self.browser)

        self.setCentralWidget(central_widget)
        self.adjust_zoom()
        self.center_window()

        self.showMaximized()

    def get_screen_size_3_4(self):
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_width = user32.GetSystemMetrics(78)
        screen_height = user32.GetSystemMetrics(79)
        adjusted_width = int(screen_width * 3 / 4)
        adjusted_height = int(screen_height * 3 / 4)
        return (adjusted_width, adjusted_height)
    
    def center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def on_file_click(self, filename):
        url = 'http://localhost:5000/open_file'
        try:
            response = requests.post(url, json={'filename': filename})
            if response.status_code == 200:
                print("File opened successfully.")
            else:
                print(f"Failed to open file: {response.json().get('message')}")
        except requests.RequestException as e:
            print(f"Error during request: {e}")

    def adjust_zoom(self):
        self.browser.setZoomFactor(1.0)

def start_flask():
    flask_app.run(port=5000)

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def convert_docx_to_pdf(docx_path, pdf_path=None):
    if pdf_path is None:
        convert(docx_path)
    else:
        convert(docx_path, pdf_path)

if __name__ == "__main__":
    start_time = time.time()
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()
    app = QApplication(sys.argv)
    window = Browser()
    window.show()
    end_time = time.time()
    startup_time = end_time - start_time
    print(f"Application startup time: {startup_time:.2f} seconds")
    clear_folder(os.path.join(BASE_DIR, 'config', 'uploads'))
    clear_folder(os.path.join(BASE_DIR, 'config', 'outputs'))
    sys.exit(app.exec())
#pyinstaller --noconsole --hidden-import=matplotlib.backends.backend_svg --hidden-import=model_inference --hidden-import=pdf_processor app.py   