import os
import fitz
import re
from decimal import Decimal
from PyPDF2 import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.units import inch  # 添加这一行来修复错误
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from model_inference import calculate_generation_probability
from datetime import datetime
import sys

# 读取PDF文件中的文本内容
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# 根据语言类型分割文本
def split_text(text, language):
    if language == 'zh':
        return split_text_chinese(text)
    elif language == 'en':
        return split_text_english(text)

# 将中文文本按句子分割
def split_text_chinese(text):
    segments = []
    current_segment = ""
    for char in text:
        current_segment += char
        if len(current_segment) >= 256 and char in "。！？":
            segments.append(current_segment)
            current_segment = ""
    if current_segment:
        segments.append(current_segment)
    return segments

def split_text_english(text):
    segments = []
    current_segment = ""
    word_count = 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= 128:
            current_segment += sentence + " "
            word_count += len(words)
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence + " "
            word_count = len(words)
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

# 过滤文本中的特殊字符
def filter_text(segment):
    return re.sub(r'[\n\r\t]', ' ', segment)


def count_characters_text(text, language):
    if language == 'zh':
        # 计算中文字符的数量
        return len(re.findall(r'[\u4e00-\u9fff]', text))
    elif language == 'en':
        # 计算英文单词的数量
        return len(re.findall(r'\b\w+\b', text))
    else:
        return 0


# 评估文本的AI生成概率
def evaluate_text(text, language):
    # 计算中文字符的数量
    zh_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    
    # 计算非中文字符的数量
    non_zh_chars = sum(1 for char in text if not ('\u4e00' <= char <= '\u9fff'))
    
    # 计算英文字符的数量
    en_chars = sum(1 for char in text if 'A' <= char <= 'Z' or 'a' <= 'z')
    
    # 计算非英文字符的数量
    non_en_chars = sum(1 for char in text if not ('A' <= char <= 'Z' or 'a' <= 'z'))
    
    # 检查非中文字符是否是中文字符的10倍以上
    if language == 'zh' and non_zh_chars > 10 * zh_chars:
        return 0
    
    # 检查非英文字符是否是英文字符的10倍以上
    if language != 'zh' and non_en_chars > 10 * en_chars:
        return 0
    
    if language == 'zh':
        if not re.search(r'[\u4e00-\u9fff]', text):
            return 0
    else:
        if not re.search(r'\b\w+\b', text):
            return 0

    return calculate_generation_probability(text, language)

# 添加 overall_ai_detection_result 函数
def overall_ai_detection_result(analysis):
    total_word_count = sum(item['word_count'] for item in analysis)
    if total_word_count == 0:
        return "请检查输入"
        
    weighted_ai_prob = sum(item['ai_prob'] * item['word_count'] for item in analysis) / total_word_count

    if 0.1 < weighted_ai_prob < 0.6:
        return predict_text(0, weighted_ai_prob)
    elif weighted_ai_prob >= 0.6:
        return predict_text(1, weighted_ai_prob)
    else:
        return predict_text(2, 1 - weighted_ai_prob)

# 根据AI生成概率预测文本类型
def predict_text(flag, probability):
    probability = format(probability * 100, ".2f")
    if flag == 0:
        return f"中度疑似大模型生成文本，概率为{probability}%"
    elif flag == 1:
        return f"高度疑似大模型生成文本，概率为{probability}%"
    else:
        return f"人类文本，概率为{probability}%"

# PDF阅读器类
class PDFReader:
    def __init__(self, file_path, language, output_folder):
        self.file_path = file_path
        self.language = language
        self.output_folder = output_folder  # 将 output_folder 作为类的一个属性
        self.output_path = os.path.join(output_folder, f"processed_{os.path.basename(file_path)}")
        self.doc = fitz.open(file_path)
        self.stats = {
            "total_ai_percent": 0,
            "high_ai_percent": 0,
            "prob_ai_percent": 0,
            "humanpercent": 0,
            "total_word_count": 0,
            "high_ai_count": 0,
            "prob_ai_count": 0,
            "humani_count": 0
        }

    def highlight_pdf(self):
        temp_output_path = os.path.join(self.output_folder, f"temp_{os.path.basename(self.file_path)}")
        original_width = self.increase_pdf_width(self.file_path, temp_output_path, 3.2 * inch)

        self.doc = fitz.open(temp_output_path)
        processed_segments = set()
        results = []

        total_pages = len(self.doc)
        total_segments = sum(len(split_text(self.doc.load_page(page_num).get_text(), self.language)) for page_num in range(total_pages))

        try:
            with tqdm(total=total_segments, desc='处理段落', dynamic_ncols=True, file=sys.stdout) as pbar:
                for page_num in range(total_pages):
                    page = self.doc.load_page(page_num)
                    text = page.get_text()
                    segments = split_text(text, self.language)

                    for segment in segments:
                        filtered_segment = filter_text(segment)
                        if filtered_segment in processed_segments:
                            pbar.update(1)
                            continue

                        ai_prob = evaluate_text(filtered_segment, self.language)
                        word_count = count_characters_text(filtered_segment, self.language)
                        self.stats["total_word_count"] += word_count

                        result = {
                            'text': segment,
                            'ai_prob': ai_prob,
                            'word_count': word_count
                        }
                        results.append(result)

                        # 输出每个段落的结果
                        print(f"段落: {segment}\nAI 概率: {ai_prob}\n字数: {word_count}\n")

                        self.stats["total_ai_percent"] += ai_prob * word_count

                        if 0.10 < ai_prob < 0.6:
                            color = (1, 1, 0)
                            self.stats["prob_ai_count"] += word_count
                            self.stats["prob_ai_percent"] += word_count
                        elif ai_prob >= 0.6:
                            color = (1, 0, 0)
                            self.stats["high_ai_count"] += word_count
                            self.stats["high_ai_percent"] += word_count
                        else:
                            self.stats["humani_count"] += word_count

                        if ai_prob > 0.1:
                            highlight = page.search_for(segment)
                            if not highlight:
                                pbar.update(1)
                                continue
                            for inst in highlight:
                                bbox = inst
                                if not self.is_in_text_box(page, bbox):
                                    continue
                                highlight_annot = page.add_highlight_annot(inst)
                                highlight_annot.set_colors(stroke=color)
                                highlight_annot.update()
                                if filtered_segment not in processed_segments:
                                    self.add_annotation(page, bbox, ai_prob, original_width)
                                    processed_segments.add(filtered_segment)
                        pbar.update(1)

                self.add_divider_and_title(page, original_width, 3.2 * inch, "LLM Detection Results")
        finally:
            self.doc.save(self.output_path)

        if self.stats["total_word_count"] > 0:
            self.stats["total_ai_percent"] = self.stats["total_ai_percent"] / self.stats["total_word_count"]
            self.stats["high_ai_percent"] = self.stats["high_ai_percent"] / self.stats["total_word_count"] * 100
            self.stats["prob_ai_percent"] = self.stats["prob_ai_percent"] / self.stats["total_word_count"] * 100
            self.stats["humanpercent"] = self.stats["humani_count"] / self.stats["total_word_count"] * 100

        # 输出最终的总结
        print("最终统计结果：")
        print(f"总字数: {self.stats['total_word_count']}")
        print(f"AI 文字总占比: {self.stats['total_ai_percent'] * 100:.2f}%")
        print(f"高AI概率文字占比: {self.stats['high_ai_percent']:.2f}%")
        print(f"中等AI概率文字占比: {self.stats['prob_ai_percent']:.2f}%")
        print(f"人类文字占比: {self.stats['humanpercent']:.2f}%")

        return self.stats, results
    
    def increase_pdf_width(self, input_pdf, output_pdf, added_width):
        added_width = Decimal(added_width)
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            original_width = page.mediabox.width
            original_height = page.mediabox.height

            new_page = PageObject.create_blank_page(width=original_width + added_width, height=original_height)
            new_page._content = None
            new_page.merge_page(page)
            writer.add_page(new_page)

        with open(output_pdf, 'wb') as f:
            writer.write(f)
        return Decimal(original_width)

    def add_annotation(self, page, bbox, ai_prob, original_width):
        _, y0, _, y1 = bbox
        annotation_x = float(original_width) + 10
        annotation_y = (y0 + y1) / 2 + 10
        fontsize = 12
        font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
        text = f"LLM_Generated_Probability: {100*ai_prob:.2f}%"
        page.insert_text((annotation_x, annotation_y), text, fontsize=fontsize, fontfile=font_path, color=(0, 0, 0))

    def add_divider_and_title(self, page, original_width, added_width, title_text):
        line_x = float(original_width)
        line_y0 = 0
        line_y1 = page.rect.height
        page.draw_line((line_x, line_y0), (line_x, line_y1), color=(0, 0, 0), width=1)

        title_x_center = float(original_width) + added_width / 2
        title_y = 20
        font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
        fontsize = 14

        title_width = fitz.get_text_length(title_text, fontsize=fontsize)
        title_x = title_x_center - title_width / 2

        page.insert_text((title_x, title_y), title_text, fontsize=fontsize, fontfile=font_path, color=(0, 0, 0))

    def is_in_text_box(self, page, bbox):
        text_boxes = page.get_text("blocks")
        for box in text_boxes:
            if box[0] <= bbox.x0 and box[1] <= bbox.y0 and box[2] >= bbox.x1 and box[3] >= bbox.y1:
                return True
        return False

    def process_txt(self, input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        segments = split_text(text, self.language)
        with open(output_path, 'w', encoding='utf-8') as file:
            for segment in segments:
                filtered_segment = filter_text(segment)
                ai_prob = evaluate_text(filtered_segment, self.language)
                if ai_prob > 0.10:
                    file.write(f"[大模型生成文本概率：{ai_prob:.4f}]\n")
                file.write(filtered_segment + "\n")

# 创建统计图表
def create_chart(stats, output_folder='config/outputs/figures'):
    os.makedirs(output_folder, exist_ok=True)

    font_path = "C:/Windows/Fonts/simhei.ttf"
    prop = fm.FontProperties(fname=font_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['高度疑似', '中度疑似', '人类文本']
    sizes = [stats["high_ai_percent"], stats["prob_ai_percent"], stats["humanpercent"]]
    colors = ['lightcoral', 'yellow', 'lightgreen']
    explode = (0.1, 0, 0)

    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=autopct_format, startangle=140, textprops={'fontproperties': prop})
    ax.axis('equal')

    for text in texts + autotexts:
        text.set_fontproperties(prop)

    figure_filename = os.path.join(output_folder, 'ai_detection_chart.svg')
    plt.savefig(figure_filename, format='svg')
    plt.close(fig)

    return figure_filename

# 创建段落评分图表
def create_segment_chart(segment_scores, output_folder='config/outputs/figures'):
    os.makedirs(output_folder, exist_ok=True)

    font_path = "C:/Windows/Fonts/simhei.ttf"
    prop = fm.FontProperties(fname=font_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    segments = range(1, len(segment_scores) + 1)
    scores = [score['ai_prob'] for score in segment_scores]
    colors = ['lightcoral' if score >= 0.6 else 'yellow' if score > 0.1 else 'lightgreen' for score in scores]

    bars = ax.bar(segments, scores, color=colors)
    ax.set_xlabel('段落序号', fontproperties=prop)
    ax.set_ylabel('大模型生成概率', fontproperties=prop)
    ax.set_title('大模型生成文本分布图', fontproperties=prop)

    ax.axhline(y=0.1, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0.6, color='gray', linestyle='--', linewidth=1)

    tick_frequency = max(len(segments) // 20, 1)
    ax.set_xticks([i for i in segments if (i-1) % tick_frequency == 0])
    ax.set_xticklabels([i for i in segments if (i-1) % tick_frequency == 0])

    legend_labels = [Patch(color='lightcoral', label='高度疑似大模型生成文本 (>= 0.6)'),
                     Patch(color='yellow', label='中度疑似大模型生成文本 (0.1-0.6)'),
                     Patch(color='lightgreen', label='人类文本 (<= 0.1)')]
    ax.legend(handles=legend_labels, prop=prop)

    figure_filename = os.path.join(output_folder, 'segment_scores_chart.svg')
    plt.savefig(figure_filename, format='svg')
    plt.close(fig)

    return figure_filename

# 创建报告
def create_report(stats, segment_scores, filename, output_folder):
    figures_folder = os.path.join(output_folder, 'figures')
    os.makedirs(figures_folder, exist_ok=True)

    chart_path = create_chart(stats, figures_folder)
    segment_chart_path = create_segment_chart(segment_scores, figures_folder)

    original_pdf_path = os.path.join(output_folder, f"processed_{filename}")
    reader = PdfReader(original_pdf_path)
    original_width = reader.pages[0].mediabox.width

    report_filename = os.path.join(figures_folder, 'detection_report.pdf')
    c = canvas.Canvas(report_filename, pagesize=(original_width, letter[1]))

    font_path = "C:/Windows/Fonts/simsun.ttc"
    pdfmetrics.registerFont(TTFont('SimSun', font_path))

    c.setFont("SimSun", 16)

    title = "星鉴-机器生成文本检测报告"
    title_width = pdfmetrics.stringWidth(title, "SimSun", 24)
    title_x = float(original_width) / 2 - title_width / 3
    c.drawString(title_x, letter[1] - 30, title)

    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = [
        ["文件名", filename],
        ["检测时间", detection_time],
        ["综合大模型生成指标", f"{100*stats['total_ai_percent']:.2f}%"],
        ["高度疑似部分占比", f"{stats['high_ai_percent']:.2f}%"],
        ["中度疑似部分占比", f"{stats['prob_ai_percent']:.2f}%"],
        ["人类文本部分占比", f"{stats['humanpercent']:.2f}%"]
    ]

    styleSheet = getSampleStyleSheet()
    styleSheet.add(ParagraphStyle(name='SimSun', fontName='SimSun', fontSize=14))
    table_data = [[Paragraph(cell, styleSheet['SimSun']) for cell in row] for row in data]
    table = Table(table_data, colWidths=[150, 190])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    for row in range(1, len(data)):
        bg_color = colors.white if row % 2 == 0 else colors.lightblue
        table.setStyle(TableStyle([('BACKGROUND', (0, row), (-1, row), bg_color)]))

    table.wrapOn(c, float(original_width) / 2, 200)
    table.drawOn(c, 30, letter[1] - 250)

    drawing1 = svg2rlg(chart_path)
    drawing1.scale(0.7, 0.7)
    renderPDF.draw(drawing1, c, float(original_width) / 2, letter[1] - 300)

    drawing2 = svg2rlg(segment_chart_path)
    drawing2.scale(0.9, 0.9)
    renderPDF.draw(drawing2, c, 0, letter[1] - 780)

    c.save()

    writer = PdfWriter()

    report_reader = PdfReader(report_filename)
    writer.add_page(report_reader.pages[0])

    for page_num in range(len(reader.pages)):
        writer.add_page(reader.pages[page_num])

    final_output_path = os.path.join(output_folder, f"final_processed_{filename}")
    with open(final_output_path, 'wb') as output_file:
        writer.write(output_file)

    return final_output_path
