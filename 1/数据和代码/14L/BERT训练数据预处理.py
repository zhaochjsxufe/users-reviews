import nltk
import re
from gensim.models import KeyedVectors
import numpy as np
import torch
from transformers import BertTokenizer

# 加载停用词表
with open('stopwords_English.txt', 'r') as f:
    stopwords = set(f.read().splitlines())

# 加载 Word2Vec 预训练模型
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 从文件加载数据
def load_processed_data(file_path):
    data = torch.load(file_path, weights_only=True)  # 增加 weights_only=True 参数
    return data['input_ids'], data['attention_masks'], data['aspect_embeddings'], data['labels']

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 读取数据文件
def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            text, aspects = line.strip().split('",[')
            text = text[1:]  # 去除文本的开头引号
            aspects = aspects[:-1]  # 去除方面词和情感极性的末尾方括号
            aspects_list = re.findall(r'\"(.*?)\",\"(.*?)\"', aspects)
            data.append((text, aspects_list))
    return data

# 分词（不对方面词应用停用词过滤）
def tokenize(text, is_aspect=False):
    tokens = nltk.word_tokenize(text.lower())
    if is_aspect:
        return tokens  # 对方面词不进行停用词过滤
    return [token for token in tokens if token not in stopwords]

# 生成词嵌入
def get_word_embedding(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return np.zeros(word2vec_model.vector_size)

# 生成方面词的平均词嵌入
def get_average_embedding(tokens):
    embeddings = [get_word_embedding(token) for token in tokens]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# 处理数据并生成 BERT 所需的输入
def process_data_for_bert(data, max_length=128):
    input_ids_list = []
    attention_mask_list = []
    aspect_embeddings_list = []
    labels_list = []

    # 遍历每个文本及其方面词
    for text, aspects in data:
        # BERT 分词并生成 input_ids 和 attention_mask
        tokens = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        # 打印当前文本和方面词总数
        print(f"\nProcessing text: {text}")
        print(f"Total aspects: {len(aspects)}")

        # 针对每个方面词生成独立的样本
        for idx, (aspect, sentiment) in enumerate(aspects):
            # 方面词分词并生成平均词嵌入
            aspect_tokens = tokenize(aspect, is_aspect=True)
            aspect_embedding = get_average_embedding(aspect_tokens)

            # 生成标签
            label = 0 if sentiment == 'negative' else 1 if sentiment == 'neutral' else 2

            # 将相同的 input_ids 和 attention_mask 与每个方面词关联
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            aspect_embeddings_list.append(aspect_embedding)
            labels_list.append(label)

            # 打印调试信息：每个方面词的处理结果
            print(f"Aspect {idx + 1}: '{aspect}', Sentiment: {sentiment}")
            print(f"Aspect Embedding (first 5 values): {aspect_embedding[:5]}")
            print(f"Label: {label}")

    # 转换为张量
    input_ids_list = torch.stack(input_ids_list)
    attention_mask_list = torch.stack(attention_mask_list)
    aspect_embeddings_list = torch.tensor(np.array(aspect_embeddings_list), dtype=torch.float32)
    labels_list = torch.tensor(labels_list, dtype=torch.long)

    return input_ids_list, attention_mask_list, aspect_embeddings_list, labels_list

# 保存数据到文件
def save_processed_data(file_path, input_ids, attention_masks, aspect_embeddings, labels):
    torch.save({
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'aspect_embeddings': aspect_embeddings,
        'labels': labels
    }, file_path)

# 从文件加载数据
def load_processed_data(file_path):
    data = torch.load(file_path)
    return data['input_ids'], data['attention_masks'], data['aspect_embeddings'], data['labels']

# 主函数
file_path = '2014_Laptop_train.txt'
data = read_data(file_path)

# 处理数据并生成 BERT 输入
input_ids, attention_masks, aspect_embeddings, labels = process_data_for_bert(data)

# 保存处理后的数据
save_file_path = 'bert_train_data.pt'
save_processed_data(save_file_path, input_ids, attention_masks, aspect_embeddings, labels)

# 打印样例
print("Data saved to:", save_file_path)
print("Sample input_ids:", input_ids[0])
print("Sample attention_mask:", attention_masks[0])
print("Sample aspect_embedding:", aspect_embeddings[0][:5], "...")
print("Sample label:", labels[0])

# 加载处理后的数据
input_ids, attention_masks, aspect_embeddings, labels = load_processed_data(save_file_path)

# 打印加载的数据样例
print("\nLoaded Data:")
print("Sample input_ids:", input_ids[0])
print("Sample attention_mask:", attention_masks[0])
print("Sample aspect_embedding:", aspect_embeddings[0][:5], "...")
print("Sample label:", labels[0])
