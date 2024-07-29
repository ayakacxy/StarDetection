import torch
from transformers import AutoTokenizer, RobertaModel, BertTokenizer, BertModel
import torch.nn as nn
import re
import os

# 定义映射器类
class Mapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

# 自定义RoBERTa模型
class CustomRoBERTaModel(nn.Module):
    def __init__(self, roberta, num_labels=2, dropout_rate=0.3):
        super(CustomRoBERTaModel, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 定义解耦BERT分类器
class DisentangledBERTClassifier(nn.Module):
    def __init__(self, num_classes=2, version="bert-base-uncased", device="cpu"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(version)
        self.bert = BertModel.from_pretrained(version)
        self.semantic_mapper = Mapper(dim=768)
        self.style_mapper = Mapper(dim=768)
        self.classifier = nn.Linear(768, num_classes)
        self.device = device
        self.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 应用语义和风格映射器
        semantic_features = self.semantic_mapper(cls_token)
        style_features = self.style_mapper(cls_token)

        # 组合特征（可以是拼接、求和等）
        combined_features = semantic_features + style_features  # [batch_size, hidden_size]

        # 分类
        logits = self.classifier(combined_features)  # [batch_size, num_classes]

        return logits

# 文本预处理函数
def preprocess_text(text):
    if text is None:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 检测AI生成的文本（英文）
def detect_ai_generated(text, model, tokenizer):
    text = preprocess_text(text)
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs
        probs = torch.softmax(logits, dim=-1)
        ai_prob = probs[0][1].item()
    return ai_prob

# 检测AI生成的文本（中文）
def detect_ai_generated_zh(text, model, tokenizer):
    text = preprocess_text(text)
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs
        probs = torch.softmax(logits, dim=-1)
        ai_prob = probs[0][1].item()
    return ai_prob

# 加载模型和分词器
def load_model_and_tokenizer(language):
    if language == 'zh':
        model_path = os.path.join('config', 'Model', 'modelDetection_Chinese_BERT_j_3.pth')
        tokenizer_name = os.path.join('config', 'Model', 'RoBERTa_ZH')
        model = DisentangledBERTClassifier(num_classes=2, version=tokenizer_name, device='cpu')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    else:
        model_path = os.path.join('config', 'Model', 'Detection_English_RoBERTa_try_1.pth')
        tokenizer_name = os.path.join('config', 'Model', 'RoBERTa_EN')
        roberta_model = RobertaModel.from_pretrained(tokenizer_name)
        model = CustomRoBERTaModel(roberta_model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

# 全局加载模型和分词器
model_zh, tokenizer_zh = load_model_and_tokenizer('zh')
model_en, tokenizer_en = load_model_and_tokenizer('en')

# 计算文本的生成概率
def calculate_generation_probability(text, language):
    if language == 'zh':
        return detect_ai_generated_zh(text, model_zh, tokenizer_zh)
    else:
        return detect_ai_generated(text, model_en, tokenizer_en)
