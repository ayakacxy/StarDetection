# 星鉴—多域多模型的机器生成文本检测系统

[TOC]

## 系统介绍

星鉴项目旨在实现对机器生成文本的检测。通过Zero-Shot和监督学习结合的方法，星鉴可以准确高效地区分多领域多模型的人类文本和机器生成文本。本项目适用于需要识别AI生成内容的各类应用场景。

### 系统架构

本项目基于`Flask`架构的前后端交互设计，用户通过前端页面交互，后端处理前端返回的文本或者文件，并将结果返回前端展示。

![星鉴系统架构图](figures\system_architecture_diagram.jpg)

### 系统功能

系统支持中英文两种语言检测，同时支持用户上传文件进行检测，文件格式支持包括`.txt`,`.docx`,`.pdf`

| 功能列表     | 功能描述                                                     |
| ------------ | ------------------------------------------------------------ |
| 文本检测功能 | 用户在客户端输入一段文本，检测完成后会返回一个这段文本由机器生成的概率 |
| 文件检测功能 | 长文本用户可以上传 PDF,DOCX 和 TXT 格式的文件，检测完成后会返回标注过的文件 |
| 深度文本分析 | 文本检测完成会告知用户各个段落由机器生成的概率，并标识各个成分的占比 |
| 文件标注功能 | 根据不同段落评估概率，标注高亮，并在侧面标注生成概率（PDF,DOCX）或直接标注概率（TXT） |

## 系统部署

本项目支持源代码部署以及`Windows`平台可执行程序`.exe`以及`Android`平台安装包`.apk`的安装。

### 源代码部署

- 先从`GitHub`上克隆我们的源代码

```bash
git clone https://github.com/ayakacxy/star_detection.git
cd  star_detection
```

- 本地已经安装`Anaconda`,

  - 那么创建一个新的环境并激活

  ```bash
  conda create --name star python=3.10
  conda activate star
  ```

  - 安装对应的依赖

  ```bash
  #从文件读取安装
  pip install -r requirements
  #直接安装对应的库
  pip install numpy<2.0 docx2pdf flask flask-cors pymupdf transformers torch pypdf2 reportlab matplotlib svglib requests tqdm PySide6
  ```

- 本地未安装`Anaconda`

  - 建议安装一个`python`的3.10版本。
  - 安装对应的依赖

  ```bash
  #从文件读取安装
  pip install -r requirements
  #直接安装对应的库
  pip install numpy<2.0 docx2pdf flask flask-cors pymupdf transformers torch pypdf2 reportlab matplotlib svglib requests tqdm PySide6
  ```

- 运行项目

  ```bash
  python app.py
  ```

最后的运行结果如下图所示

![运行结果](figures\execution_result.png)

#### 注意事项

1. 项目运行的日志输出存储在`config/log.jsonl`文件中，对应的报错信息可以检查日志文件进行调试。
2. 部分功能需要使用到`Windows`自带的字体文件，如果报错可以修改成本机自带的字体。
3. `DOCX`文件的检测需要本地存在`microsoft office`，如果未安装，健议自行把`DOCX`文件转换为`PDF`文件或者直接复制对应的文本进行检测。

###  `Windows`平台可执行程序安装

- 先从该链接下载我们的可执行程序[星鉴](https://rec.ustc.edu.cn/share/45b5f090-4bfa-11ef-81e7-0fc80e73d6c5)

- 下载之后双击`StarDetection.exe`进行安装

  ![安装](figures/installation.png)

- 选择安装语言

  ![选择语言](figures\select_language.png)

- 用户协议以及安装位置选择，默认安装在`D`盘

  ![选择语言](figures\select_language.png)

  ![选择安装路径](figures\select_installation_path.png)

- 创建快捷方式，并运行星鉴

  ![选择语言](figures\create_shortcut.png)

  ![运行星鉴](figures\run_star_detection.png)

安装成功运行结果如下所示

![成功运行](figures\successful_execution.png)

### 网页端访问

直接输入网址即可访问

### `Android`端安装

在该链接下载，安装即可。

## 使用指南

所有平台都是基于同一套操作以及交互逻辑，所以这里仅以本地为示范
