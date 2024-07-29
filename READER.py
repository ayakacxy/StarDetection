import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import os
import io
import ctypes
import argparse
import threading
import queue
import chardet
import shutil

# 设置应用程序 DPI 感知
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # for per-monitor DPI awareness
except AttributeError:
    ctypes.windll.user32.SetProcessDPIAware()  # for system DPI awareness (Windows 8 and earlier)

class READER(tk.Tk):
    def __init__(self, file_path=None):
        super().__init__()
        self.title("检测结果查看")
        self.geometry("1500x1500")  # 放大窗口尺寸
        self.center_window()  # 居中窗口

        self.file_path = file_path
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.is_txt = False
        self.txt_queue = queue.Queue()
        self.txt_lines = []
        self.loading = False
        self.dragging = False  # 新增拖动状态

        self.create_widgets()

        if file_path:
            self.load_file(file_path)

    def create_widgets(self):
        # 创建顶部框架以容纳按钮
        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x")

        # 添加打开文件按钮
        self.open_button = tk.Button(top_frame, text="打开文件", command=self.open_file)
        self.open_button.pack(side="left", padx=5, pady=5)

        # 添加下载PDF按钮
        self.download_button = tk.Button(top_frame, text="下载检测报告", command=self.download)
        self.download_button.pack(side="left", padx=5, pady=5)

        # 添加退出按钮
        self.exit_button = tk.Button(top_frame, text="退出", command=self.quit)
        self.exit_button.pack(side="left", padx=5, pady=5)

        # 计算中间内容区的高度
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(side="top", expand=True, fill="both")

        self.canvas = tk.Canvas(self.content_frame)
        self.scroll_y = tk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.scroll_y.pack(side="right", fill="y")
        self.canvas.pack(side="left", expand=True, fill="both")

        self.page_frame = tk.Frame(self)
        self.page_frame.pack(side="bottom", fill="x")

        self.prev_button = tk.Button(self.page_frame, text="上一页", command=self.prev_page)
        self.prev_button.pack(side="left", padx=5)

        self.page_label = tk.Entry(self.page_frame, justify="center", width=5)
        self.page_label.pack(side="left", padx=5)

        self.total_pages_label = tk.Label(self.page_frame, text="", justify="center")
        self.total_pages_label.pack(side="left")

        self.jump_button = tk.Button(self.page_frame, text="跳转", command=self.go_to_page)
        self.jump_button.pack(side="left", padx=5)

        self.next_button = tk.Button(self.page_frame, text="下一页", command=self.next_page)
        self.next_button.pack(side="left", padx=5)

        # 进度条及其百分比显示
        style = ttk.Style()
        style.configure("TScale",
                        sliderlength=30,  # 使滑块更大
                        troughcolor='#D3D3D3',
                        background='#4CAF50')

        self.progress = ttk.Scale(self.page_frame, style="TScale", orient="horizontal", command=self.on_progress_changed)
        self.progress.bind("<ButtonRelease-1>", self.on_progress_released)  # 绑定鼠标释放事件
        self.progress.pack(side="left", padx=5, fill="x", expand=True)

        self.progress_label = tk.Label(self.page_frame, text="0%")
        self.progress_label.pack(side="left")

        # 正在加载标签
        self.loading_label = tk.Label(self, text="正在加载...", font=("SIMSUN", 24))
        self.loading_label.pack(side="top", pady=10)
        self.loading_label.pack_forget()  # 默认不显示

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def load_file(self, file_path):
        self.file_path = file_path
        self.is_txt = file_path.lower().endswith('.txt')
        self.clear_content()  # 清空当前显示内容

        if self.is_txt:
            thread = threading.Thread(target=self.load_txt, args=(file_path,))
        else:
            thread = threading.Thread(target=self.load_pdf, args=(file_path,))
        thread.start()

    def load_pdf(self, pdf_path):
        self.loading = True
        self.show_loading()  # 显示加载信息
        self.pdf_document = fitz.open(pdf_path)
        self.total_pages = len(self.pdf_document)
        self.current_page = 0
        self.loading = False
        self.hide_loading()  # 隐藏加载信息
        self.update_ui_after_load()
        self.show_page()

    def load_txt(self, txt_path):
        self.loading = True
        self.show_loading()  # 显示加载信息
        with open(txt_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(txt_path, 'r', encoding=encoding) as file:
            lines = file.readlines()
            self.txt_lines = [line for line in lines if line.strip() != '']  # 过滤掉多余的空行

        # 打印一些调试信息
        print(f"Total lines after filtering: {len(self.txt_lines)}")
        if len(self.txt_lines) > 0:
            print(f"First line: {self.txt_lines[0]}")
            print(f"Last line: {self.txt_lines[-1]}")

        self.current_page = 0
        self.loading = False
        self.hide_loading()  # 隐藏加载信息
        self.update_ui_after_load()
        self.show_page()

    def show_txt_page(self):
        if self.txt_lines:
            self.canvas.delete("all")
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.content_frame.winfo_height()

            # 计算每页显示的行数
            font_size = 14
            line_height = font_size + 2
            lines_per_page = max(canvas_height // line_height, 1)

            # 获取当前页内容
            start_line = self.current_page * lines_per_page
            end_line = min(start_line + lines_per_page, len(self.txt_lines))  # 确保不超出行数范围
            page_lines = self.txt_lines[start_line:end_line]

            # 打印当前页内容的调试信息
            print(f"Showing lines from {start_line} to {end_line}")
            if len(page_lines) > 0:
                print(f"First line on page: {page_lines[0]}")
                print(f"Last line on page: {page_lines[-1]}")

            # 显示当前页内容
            text = "".join(page_lines)
            self.canvas.create_text(10, 10, anchor=tk.NW, text=text, font=("Simsun", font_size), fill="black", width=canvas_width - 20)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))  # 更新滚动区域

            self.update_total_pages(lines_per_page)
            self.page_label.delete(0, tk.END)
            self.page_label.insert(0, str(self.current_page + 1))
            self.update_progress_label()

    def update_total_pages(self, lines_per_page):
        self.total_pages = (len(self.txt_lines) + lines_per_page - 1) // lines_per_page  # 计算总页数
        self.total_pages_label.config(text=f"/ {self.total_pages}")
        self.progress.config(to=self.total_pages)

    def show_loading(self):
        if self.loading:
            self.loading_label.pack(side="top", pady=10)

    def hide_loading(self):
        self.loading_label.pack_forget()

    def update_ui_after_load(self):
        self.page_label.delete(0, tk.END)
        self.page_label.insert(0, str(self.current_page + 1))
        self.total_pages_label.config(text=f"/ {self.total_pages}")
        self.progress.config(to=self.total_pages)
        self.show_page()

    def show_page(self):
        if self.is_txt:
            self.show_txt_page()
        else:
            self.show_pdf_page()

    def show_pdf_page(self):
        if self.pdf_document:
            thread = threading.Thread(target=self._show_pdf_page)
            thread.start()

    def _show_pdf_page(self):
        page = self.pdf_document[self.current_page]
        zoom = 3  # 缩放比例，值越大图像越清晰
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # 调整图像大小以适应窗口尺寸，同时保持纵横比
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 0 and canvas_height > 0:
            img = Image.open(io.BytesIO(pix.tobytes()))
            img_width, img_height = img.size

            # 计算适当的尺寸以保持纵横比
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            if new_width > 0 and new_height > 0:
                img = img.resize((new_width, new_height), Image.LANCZOS)
                self.img = ImageTk.PhotoImage(img)

                # 计算图像居中位置
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2

                self.canvas.delete("all")  # 清空画布内容
                self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.img)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))  # 更新滚动区域

                self.page_label.delete(0, tk.END)
                self.page_label.insert(0, str(self.current_page + 1))
                self.progress.set(self.current_page + 1)
                self.update_progress_label()

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page()

    def next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.show_page()

    def go_to_page(self, event=None):
        try:
            page_number = int(self.page_label.get().strip()) - 1
            if 0 <= page_number < self.total_pages:
                self.current_page = page_number
                self.show_page()
            else:
                messagebox.showwarning("警告", "页码超出范围")
        except ValueError:
            messagebox.showwarning("警告", "无效的页码")

    def on_resize(self, event):
        self.show_page()  # 调整窗口大小时重新绘制图像

    def on_progress_changed(self, value):
        if not self.loading:
            self.dragging = True  # 正在拖动
            page_number = int(float(value)) - 1
            if 0 <= page_number < self.total_pages:
                self.current_page = page_number

    def on_progress_released(self, event):
        self.dragging = False  # 拖动结束
        self.show_page()

    def update_progress_label(self):
        progress_percentage = (self.current_page + 1) / self.total_pages * 100 if self.total_pages else 0
        self.progress_label.config(text=f"{progress_percentage:.2f}%")

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF and Text files", "*.pdf *.txt"), ("PDF files", "*.pdf"), ("Text files", "*.txt")],
            defaultextension=".pdf"
        )
        if file_path:
            self.clear_content()  # 打开新文件前清空当前内容
            self.load_file(file_path)

    def download(self):
        # 判断文件类型是否为TXT
        if self.is_txt:
            # 获取当前文件名并处理文件名
            filename = os.path.basename(self.file_path)
            if filename.startswith("processed_"):
                filename = filename.replace("processed_", "")
            filename = f"{filename[:-4]}_检测报告.txt"  # 去掉扩展名并添加 "_检测报告.txt"

            # 设置默认保存路径
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=filename, filetypes=[("Text files", "*.txt")])
            if file_path:
                # 复制文件到指定位置
                shutil.copyfile(self.file_path, file_path)
                messagebox.showinfo("保存成功", f"TXT已保存至: {file_path}")
        else:
            # 获取当前文件名并处理文件名
            filename = os.path.basename(self.file_path)
            if filename.startswith("final_processed_"):
                filename = filename.replace("final_processed_", "")
            filename = f"{filename[:-4]}_检测报告.pdf"  # 去掉扩展名并添加 "_检测报告.pdf"

            # 设置默认保存路径
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=filename, filetypes=[("PDF files", "*.pdf")])
            if file_path:
                with open(self.file_path, 'rb') as f:
                    pdf_data = f.read()
                with open(file_path, 'wb') as f:
                    f.write(pdf_data)
                messagebox.showinfo("保存成功", f"PDF已保存至: {file_path}")

    def clear_content(self):
        self.canvas.delete("all")
        self.txt_lines = []
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.page_label.delete(0, tk.END)
        self.total_pages_label.config(text="")
        self.progress.set(0)
        self.progress_label.config(text="0%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文件查看器")
    parser.add_argument('file_path', nargs='?', help="文件路径（PDF或TXT）")
    args = parser.parse_args()

    file_path = args.file_path

    app = READER(file_path)
    app.mainloop()
