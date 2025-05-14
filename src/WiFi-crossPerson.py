import os
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import Dataset
from lib.gesture_mapping import number_to_gesture
from lib.gesture_mapping import gesture_descriptions


# 数据加载与预处理
def normalize_data(data_1):
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return data_1_norm

def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)

def load_data(path_to_data, motion_sel, test_orientation = 1):
    global T_MAX
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        if "Room1" in data_root or "Room2" in data_root or "Room3" in data_root:
            continue
        for data_file_name in data_files:
            file_path = os.path.join(data_root, data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(os.path.basename(data_root))  # 父文件夹名称
                torso_location = int(data_file_name.split('-')[2])  # 解析 torso location
                face_orientation = int(data_file_name.split('-')[3])  # 解析 face_orientation
                repetition_number = int(data_file_name.split('-')[4]) # 解析 repetition_number
                if label_1 not in motion_sel:
                    continue

                data_normed_1 = normalize_data(data_1)
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]

                    # 根据文件名是否包含 "user3" 决定训练/测试数据集
                if "user3" in data_file_name.lower():  # 转换为小写，避免大小写问题
                    test_data.append(data_normed_1.tolist())
                    test_labels.append(label_1)
                else:
                    train_data.append(data_normed_1.tolist())
                    train_labels.append(label_1)

            except Exception:
                continue

    # 处理形状
    train_data = zero_padding(train_data, T_MAX)
    test_data = zero_padding(test_data, T_MAX)

    train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 2, 3)
    test_data = np.swapaxes(np.swapaxes(test_data, 1, 3), 2, 3)

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, test_data, train_labels, test_labels


# ================== 1. 自定义 Dataset ==================
class WiFiTextDataset(Dataset):
    def __init__(self, wifi_data, labels, tokenizer, max_len=128):
        self.wifi_data = torch.tensor(wifi_data, dtype=torch.float32).permute(0, 4, 1, 2, 3)  # 调整维度顺序
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.wifi_data)

    def __getitem__(self, idx):
        wifi_sample = self.wifi_data[idx]
        label = self.labels[idx]
        gesture_name_1 = number_to_gesture.get(label, "unknown")
        gesture_name = gesture_descriptions.get(label, "This is an unknown hand motion.")
        gesture_name_2 = "This is a motion of" + gesture_name_1 + "," + gesture_name

        encoded_text = self.tokenizer(gesture_name_2, padding="max_length", max_length=self.max_len, truncation=True,
                                      return_tensors="pt")
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        return wifi_sample, input_ids, attention_mask, label

# ================== 2. WiFi 编码器 ==================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CNN_RNN_Transformer_Encoder(nn.Module):
    def __init__(self, input_shape, feature_dim=768, cnn_filters=128, rnn_units=512, transformer_dim=512,
                 num_heads=8, num_layers=3, dropout_rate=0.3, use_mean_pooling=True):
        super(CNN_RNN_Transformer_Encoder, self).__init__()

        C, T_MAX, H, W = input_shape  # (C=1, T_MAX=时间步长, H=20, W=20)

        self.use_mean_pooling = use_mean_pooling
        self.feature_dim = feature_dim

        # CNN部分：提取空间特征 (两层卷积)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_filters)
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_filters)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 计算 RNN 输入维度
        self.rnn_input_dim = cnn_filters * (20 // 2) * (20 // 2)

        # 双向GRU
        self.gru = nn.GRU(self.rnn_input_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Transformer 部分
        self.pos_encoder = PositionalEncoding(rnn_units * 2, max_len=T_MAX)  # 注意双向GRU，输出翻倍
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim * 2, nhead=num_heads,
                                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # 线性变换到 feature_dim
        self.fc_out = nn.Linear(transformer_dim * 2, feature_dim)

    def forward(self, x):
        batch_size, C, T_MAX, H, W = x.shape
        x = x.view(batch_size * T_MAX, 1, H, W)

        # CNN部分（两层卷积）
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 卷积1+BN+ReLU+池化
        x = torch.relu(self.bn2(self.conv2(x)))  # 卷积2+BN+ReLU
        x = self.dropout1(x)

        x = x.view(batch_size, T_MAX, -1)  # [batch_size, T_MAX, cnn_features]

        # 双向GRU部分
        rnn_out, _ = self.gru(x)  # [batch_size, T_MAX, rnn_units * 2]
        rnn_out = self.dropout2(rnn_out)

        # Transformer部分
        rnn_out = self.pos_encoder(rnn_out)
        transformer_out = self.transformer(rnn_out)  # [batch_size, T_MAX, transformer_dim]

        # 提取特征
        if self.use_mean_pooling:
            cls_output = torch.mean(transformer_out, dim=1)
        else:
            cls_output = transformer_out[:, -1, :]

        output = self.fc_out(cls_output)  # [batch_size, feature_dim]
        return output


# ================== 3. 文本编码器 ==================
class TextEncoder(nn.Module):
    def __init__(self, feature_dim=768):
        super(TextEncoder, self).__init__()

        bert_path = os.path.join(project_root, "bert_base_uncased")
        self.bert = BertModel.from_pretrained(bert_path)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(768, feature_dim)  # 降维

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量
        x = self.fc(cls_embedding)
        return x


class CLIPWiFiTextModel(torch.nn.Module):
    def __init__(self, wifi_encoder, text_encoder):
        super().__init__()
        self.wifi_encoder = wifi_encoder
        self.text_encoder = text_encoder
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))  # 可学习参数

    def forward(self, wifi_data, text_input_ids, text_attention_mask):
        wifi_features = self.wifi_encoder(wifi_data)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # L2 归一化
        wifi_features = F.normalize(wifi_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 计算可学习的 logit_scale
        logit_scale = self.logit_scale.exp()

        # 计算相似度 logits
        logits_per_image = logit_scale * (wifi_features @ text_features.t())  # WiFi → 文本
        logits_per_text = logits_per_image.t()  # 文本 → WiFi

        return logits_per_image, logits_per_text

# ================== 4. 对比学习损失 ==================
def info_nce_loss(logits_per_wifi, logits_per_text):
    batch_size = logits_per_wifi.shape[0]
    labels = torch.arange(batch_size, device=logits_per_wifi.device).long()

    loss_wifi_to_text = F.cross_entropy(logits_per_wifi, labels)
    loss_text_to_wifi = F.cross_entropy(logits_per_text, labels)

    return (loss_wifi_to_text + loss_text_to_wifi) / 2

def test(clip_model, dataloader, device, class_names, tokenizer, test_label=6):
    clip_model.eval()

    # 生成所有类别的文本特征
    class_labels = sorted(ALL_MOTION)
    prompts = [f"This is a motion of {number_to_gesture[label]}." for label in class_labels]

    # 使用 tokenizer 处理类别文本
    encoded_inputs = tokenizer(prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            wifi_data, _, _, labels = batch  # 忽略文本输入
            wifi_data = wifi_data.to(device)
            labels = labels.to(device)

            # 获取 WiFi 特征
            logits_per_wifi, _ = clip_model(wifi_data, input_ids, attention_mask)  # 只计算 WiFi 特征

            # 对相似度矩阵进行 softmax 归一化
            probs = F.softmax(logits_per_wifi, dim=1)  # 按行进行 softmax

            # 预测类别
            # preds = logits_per_wifi.argmax(dim=1)  # 预测类别索引
            preds = probs.argmax(dim=1)
            pred_labels = torch.tensor([class_labels[p] for p in preds.cpu().numpy()]).to(device)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=class_labels)

    target_label = test_label  # 例如，你要查看第六种手势的混淆情况
    target_index = class_labels.index(target_label)  # 找到对应的索引
    confusion_vector = conf_matrix[target_index].reshape(1, -1)  # 变为 1 行 n 列

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, labels=class_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", class_report)
    print("conf_matrix:\n", conf_matrix)

    return accuracy

def train_test(model, dataloader, optimizer, device, tokenizer, class_names, test_dataloader, test_fn=None, epochs=100,
               scheduler=None, test_interval=1):
    model.train()
    best_top1_acc = 0.0  # 新增：记录最高Top-1准确率

    print("eopch = 0")

    top1_acc = test_fn(model, test_dataloader, device, class_names, tokenizer)

    # 检查是否为最好准确率
    if top1_acc > best_top1_acc:
        best_top1_acc = top1_acc
        torch.save(wifi_encoder.state_dict(), "clip_wifi_encoder.pth")
        torch.save(text_encoder.state_dict(), "clip_text_encoder.pth")
        # print(f"✅ New best model saved at Epoch {epoch + 1}, Top-1 Accuracy: {top1_acc:.4f}")
        print(f"✅ New best model saved at Epoch 0, Top-1 Accuracy: {top1_acc:.4f}")

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            wifi_data, text_input_ids, text_attention_mask, _ = batch
            wifi_data, text_input_ids, text_attention_mask = wifi_data.to(device), text_input_ids.to(
                device), text_attention_mask.to(device)

            optimizer.zero_grad()

            logits_per_image, logits_per_text = model(wifi_data, text_input_ids, text_attention_mask)

            loss = info_nce_loss(logits_per_image, logits_per_text)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        if scheduler:
            scheduler.step()

        # 每 test_interval 个 epoch 调用一次测试函数
        if test_fn is not None and (epoch + 1) % test_interval == 0:
            print(f"\n>>> Epoch {epoch + 1}: Running test_1...")
            top1_acc = test_fn(model, test_dataloader, device, class_names, tokenizer)

            # 检查是否为最好准确率
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                torch.save(wifi_encoder.state_dict(), "clip_wifi_encoder.pth")
                torch.save(text_encoder.state_dict(), "clip_text_encoder.pth")
                print(f"✅ New best model saved at Epoch {epoch + 1}, Top-1 Accuracy: {top1_acc:.4f}")

            model.train()


# ================== 7. 训练初始化 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定是否使用已有模型
use_pretrained_model = False  # True 代表加载已有模型，False 代表训练新模型
# 加载 WiFi 数据
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "Data", "gesture")  # 数据存储的目录。
ALL_MOTION = [1, 2, 3, 4, 5, 6]  # 所有动作的类别列表。
fraction_for_test = 0.1  # 测试集所占比例。
number_for_test = 0
num_classes = len(ALL_MOTION)  # 假设有6个类别
# 示例输入数据
batch_size = 128
T_MAX = 0  # 时间步数
C = 1  # WiFi 数据的通道数
H, W = 20, 20  # 每个时间步的空间维度
input_dim = 1  # 每个时间步的输入通道数
dropout_rate = 0.5  # Dropout 层的比例。

# label:1-6
train_data, test_data, train_labels, test_labels = load_data(data_dir, ALL_MOTION, fraction_for_test)
print("data finish")

# 设定 tokenizer
bert_path = os.path.join(project_root, "bert_base_uncased")
tokenizer = BertTokenizer.from_pretrained(bert_path)
print("tokenizer finish")

# 创建 Dataset 和 DataLoader
train_dataset = WiFiTextDataset(train_data, train_labels, tokenizer)
test_dataset = WiFiTextDataset(test_data, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("data_loader finish")

# 初始化编码器和优化器
wifi_encoder = CNN_RNN_Transformer_Encoder(input_shape=(C, T_MAX, H, W)).to(device)
text_encoder = TextEncoder().to(device)
print("encoder finish")

clip_model = CLIPWiFiTextModel(wifi_encoder, text_encoder)

optimizer = torch.optim.Adam(list(wifi_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)
print("optimizer finish")

# 加载或训练模型
if use_pretrained_model:
    print("加载已有模型...")
    wifi_encoder.load_state_dict(torch.load("clip_wifi_encoder.pth"))
    text_encoder.load_state_dict(torch.load("clip_text_encoder.pth"))
else:
    print("训练新模型...")

class_names = [f"Motion type {i}" for i in ALL_MOTION]  # 生成类别名称

# 训练
train_test(clip_model, train_loader, optimizer, device, tokenizer, class_names, test_loader, test_fn=test)
print("train finish")
print("test finish")

