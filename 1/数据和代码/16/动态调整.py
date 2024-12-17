import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from transformers import BertModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
import warnings

# 忽略与gamma和beta重命名有关的警告
warnings.simplefilter("ignore", UserWarning)



def load_processed_data(file_path):
    # 使用 weights_only=True 加载权重
    data = torch.load(file_path, weights_only=True)
    input_ids = data['input_ids']
    attention_masks = data['attention_masks']
    aspect_embeddings = data['aspect_embeddings']
    labels = data['labels']

    # 统计类别数量
    unique_labels, counts = torch.unique(labels, return_counts=True)
    class_counts = torch.zeros(max(unique_labels) + 1, dtype=torch.float)
    class_counts[unique_labels] = counts.float()

    return input_ids, attention_masks, aspect_embeddings, labels, class_counts



# 根据类别数量计算类别权重
def calculate_class_weights(class_counts):
    class_weights = 1.0 / class_counts  # 类别权重为样本数量的倒数
    class_weights /= class_weights.sum()  # 归一化
    return class_weights.cuda()  # 转移到 GPU


import re


def load_original_texts(file_path):
    original_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # 行号方便调试
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 使用正则表达式匹配文本和方面词部分
            match = re.match(r'^"(.*)",\s*\[(.*)\]$', line)
            if match:
                text = match.group(1)  # 获取文本部分
                aspects = match.group(2)  # 获取方面词和情感极性部分

                # 提取方面词和情感极性对
                aspects_list = re.findall(r'"(.*?)","(.*?)"', aspects)

                # 将解析后的文本和方面词列表添加到原始数据中
                original_data.append((text, aspects_list))
            else:
                # 仅在格式不正确时打印行信息
                print(f"Line {line_num} does not match the expected format: {line}")

    return original_data


def forward(self, input_ids, attention_mask, aspect_emb, labels):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = bert_output.last_hidden_state

    # 方面词嵌入投影
    aspect_emb_proj = self.aspect_projection(aspect_emb).unsqueeze(1)

    # 使用权重多头注意力机制处理 BERT 输出
    attn_output = self.weighted_multihead_attention(sequence_output, labels)

    # 打印调试信息
    print("attn_output shape:", attn_output.shape)  # (batch_size, seq_len, hidden_dim)
    print("aspect_emb_proj shape:", aspect_emb_proj.shape)  # (batch_size, 1, hidden_dim)

    # 确保两者形状匹配
    combined_vector = attn_output[:, 0, :] + aspect_emb_proj.squeeze(1)

    # 通过分类层输出
    combined_vector = self.dropout(combined_vector)
    logits = self.classifier(combined_vector)

    return logits


class WeightedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, class_weights):
        super(WeightedMultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 静态类别权重
        self.class_weights = nn.Parameter(torch.stack([torch.full((hidden_dim,), w) for w in class_weights]))

        # 动态权重生成层，用于根据输入生成动态权重
        self.dynamic_weight_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, labels):
        # 使用多头注意力机制
        attn_output, _ = self.multihead_attn(x, x, x)

        # 检查输出形状
        # print("Original attn_output shape:", attn_output.shape)  # (batch_size, seq_len, hidden_dim)

        # 根据类别选择静态权重
        static_class_weights = self.class_weights[labels].unsqueeze(1)

        # 生成动态权重
        dynamic_weights = self.dynamic_weight_layer(x)  # (batch_size, seq_len, hidden_dim)

        # 最终权重结合静态和动态权重
        combined_weights = static_class_weights + dynamic_weights

        # 将结合后的权重应用到注意力输出
        attn_output = attn_output * combined_weights

        # 残差连接 + LayerNorm
        attn_output = self.layer_norm(x + attn_output)

        # 确保 attn_output 的最终形状为 (batch_size, seq_len, hidden_dim)
        # print("Final attn_output shape:", attn_output.shape)
        return attn_output


class EnhancedBertAttentionClassifier(nn.Module):
    def __init__(self, hidden_dim=300, num_classes=3, freeze_bert_layers=True, class_weights=None, num_heads=8):
        super(EnhancedBertAttentionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            output_attentions=False,
            output_hidden_states=False
        )

        if freeze_bert_layers:
            for name, param in self.bert.named_parameters():
                if 'layer' in name and int(name.split('.')[2]) < 6:
                    param.requires_grad = False

        # 将 hidden_dim 设置为输入维度
        self.aspect_projection = nn.Linear(hidden_dim, self.bert.config.hidden_size)

        # 权重多头注意力机制
        self.weighted_multihead_attention = WeightedMultiHeadAttention(self.bert.config.hidden_size, num_heads,
                                                                       class_weights)
        # 全连接层
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask, aspect_emb, labels):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # 方面词嵌入投影
        aspect_emb_proj = self.aspect_projection(aspect_emb).unsqueeze(1)

        # 使用权重多头注意力机制处理 BERT 输出
        attn_output = self.weighted_multihead_attention(sequence_output, labels)

        # 仅使用 attn_output 的第一个 token 表示
        sentence_representation = attn_output[:, 0, :]  # (batch_size, hidden_dim)

        # 打印调试信息
        # print("sentence_representation shape:", sentence_representation.shape)
        # print("aspect_emb_proj shape after squeeze:", aspect_emb_proj.squeeze(1).shape)

        # 与方面词嵌入相加
        combined_vector = sentence_representation + aspect_emb_proj.squeeze(1)

        # 通过分类层输出
        combined_vector = self.dropout(combined_vector)
        logits = self.classifier(combined_vector)

        return logits


# 训练函数
def train_model(model, train_loader, criterion, optimizer, scaler, accumulation_steps=4, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, (input_ids, attention_masks, aspect_embeddings, labels) in enumerate(train_loader):
            input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
            aspect_embeddings, labels = aspect_embeddings.cuda(), labels.cuda()

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_masks, aspect_embeddings, labels)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


# 测试函数
def evaluate_model(model, test_loader, original_data, output_limit=5):
    model.eval()
    all_preds = []
    all_labels = []
    current_text_idx = 0

    with torch.no_grad():
        for idx, (input_ids, attention_masks, aspect_embeddings, labels) in enumerate(test_loader):
            input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
            aspect_embeddings, labels = aspect_embeddings.cuda(), labels.cuda()

            outputs = model(input_ids, attention_masks, aspect_embeddings, labels)
            _, predicted = torch.max(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if current_text_idx < output_limit and current_text_idx < len(original_data):
                text, aspects = original_data[current_text_idx]
                print(f"\n原始文本：{text}")

                for i, (aspect, true_sentiment) in enumerate(aspects):
                    true_label = 'negative' if true_sentiment == 'negative' else 'neutral' if true_sentiment == 'neutral' else 'positive'
                    predicted_label = 'negative' if predicted[i].item() == 0 else 'neutral' if predicted[
                                                                                                   i].item() == 1 else 'positive'

                    print(
                        f"    Aspect {i + 1}: {aspect} | True Sentiment: {true_label} | Predicted Sentiment: {predicted_label}")

                current_text_idx += 1

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1


# 交叉验证函数
def cross_validate_model(model_class, dataset, test_loader, original_test_data, num_folds=5, batch_size=32,
                         num_epochs=5, class_weights=None, num_heads=8):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{num_folds}")

        # 创建训练和验证集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = model_class(class_weights=class_weights, num_heads=num_heads).cuda()

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        scaler = GradScaler()

        # 训练模型
        train_model(model, train_loader, criterion, optimizer, scaler, accumulation_steps=4, num_epochs=num_epochs)

        # 在验证集上评估模型
        val_accuracy, val_f1 = evaluate_model(model, val_loader, original_test_data, output_limit=5)
        print(f"\nFold {fold + 1} - Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {val_f1:.4f}")

        # 在独立测试集上评估模型
        test_accuracy, test_f1 = evaluate_model(model, test_loader, original_test_data, output_limit=5)
        print(
            f"Independent Test Accuracy after Fold {fold + 1}: {test_accuracy:.2f}%, Independent Test F1 Score: {test_f1:.4f}")

        # 记录每一折独立测试集结果
        fold_results.append((test_accuracy, test_f1))

    # 输出每一折的独立测试集结果
    print("\nAll Folds - Independent Test Results:")
    for i, (acc, f1) in enumerate(fold_results, start=1):
        print(f"Fold {i}: Independent Test Accuracy: {acc:.2f}%, Independent Test F1 Score: {f1:.4f}")

    # 计算平均结果
    avg_acc = sum([result[0] for result in fold_results]) / len(fold_results)
    avg_f1 = sum([result[1] for result in fold_results]) / len(fold_results)
    print(f"\nAverage Independent Test Accuracy: {avg_acc:.2f}%, Average Independent Test F1 Score: {avg_f1:.4f}")


# 主函数
if __name__ == "__main__":
    # 加载训练和测试数据，并统计类别样本数量
    train_file = 'bert_train_data.pt'
    test_file = 'bert_test_data.pt'
    original_test_file = '2016_Restaurants_test.txt'

    train_input_ids, train_attention_masks, train_aspect_embeddings, train_labels, class_counts = load_processed_data(
        train_file)
    test_input_ids, test_attention_masks, test_aspect_embeddings, test_labels, _ = load_processed_data(test_file)
    original_test_data = load_original_texts(original_test_file)

    # 计算类别权重
    class_weights = calculate_class_weights(class_counts)

    # 创建训练数据集
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_aspect_embeddings, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_aspect_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 执行交叉验证
    cross_validate_model(EnhancedBertAttentionClassifier, train_dataset, test_loader, original_test_data, num_folds=5,
                         batch_size=32, num_epochs=5, class_weights=class_weights, num_heads=8)
