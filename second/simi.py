import csv
import pickle
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import ast
import multiprocessing
import jieba


def get_embedding(attr):
    # 确保模型和分词器每个进程中都被加载一次
    tokenizer = AutoTokenizer.from_pretrained(r'B:/bert-base-uncased')
    model = AutoModel.from_pretrained(r'B:/bert-base-uncased')
    # 将模型移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(attr, return_tensors='pt', truncation=True, max_length=model.config.max_position_embeddings, padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()


def get_embeddings_cn(attrs, stop_words):
    # 确保模型和分词器每个进程中都被加载一次
    # 加载中文预处理模型
    tokenizer = AutoTokenizer.from_pretrained(r'B:/chinese-roberta-wwm-ext')
    model = AutoModel.from_pretrained(r'B:/chinese-roberta-wwm-ext')
    # 将模型移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    for attr in attrs:
        processed_text = preprocess_text(attr, stop_words)
        inputs = tokenizer(processed_text, return_tensors='pt', truncation=True,
                           max_length=model.config.max_position_embeddings, padding=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    embeddings_matrix = torch.cat(embeddings, dim=0)
    return embeddings_matrix


def get_embeddings(attr):
    with multiprocessing.Pool(processes=4) as pool:
        embeddings = pool.map(get_embedding, attr)

    embeddings_matrix = torch.cat(embeddings, dim=0)
    return embeddings_matrix


def preprocess_text(text, stop_words):
    # 使用 jieba 进行分词
    words = jieba.cut(text)
    # 去除停用词
    filtered_words = [word for word in words if word not in stop_words]
    # 将处理后的词重新组合为字符串
    return " ".join(filtered_words)


if __name__ == '__main__':
    anchor_pairs = []
    with open(r'C:/Users/11792/Desktop/data_similarity/wd/anchors.txt', 'r') as file:
        data = file.read()
        anchor_pairs = ast.literal_eval(data)
    print(anchor_pairs)
    g1, g2 = pickle.load(open(r'C:/Users/11792/Desktop/data_similarity/wd/networks', 'rb'))
    attrs = pickle.load(open(r'C:/Users/11792/Desktop/data_similarity/wd/attrs', 'rb'))
    print(len(g1))
    print(len(g2))
    print(attrs[0])
    topic = []
    for i in range(len(attrs)):
        v = attrs[i]
        #dblp_1和wd是v[2], dblp_1是v[1]
        topic.append(v[2])

    #中文停用词
    with open(r'C:/Users/11792/Desktop/data_similarity/wd/stop_words_cn.pkl', 'rb') as f:
        stop_words = pickle.load(f)
    #dblp_1: 9086 9325
    #dblp_2 10000 10000
    #wd 9714 9526
    #先组成两个网络的文本的embedding矩阵
    #计算两个矩阵的相似性
    #得到相似度大节点的结果

    #已知锚点和锚点的邻居
    #看相似度大的节点是否在锚点上，如果在锚点上，这些节点的邻居是否也在锚点中，占比多少

    attrs_g1 = topic[:9000]  # g1 的属性
    attrs_g2 = topic[9714:9714+9000]  # g2 的属性
    #
    # 构造embedding矩阵cn版本
    embedding_matrix_g1 = get_embeddings_cn(attrs_g1, stop_words)
    embedding_matrix_g2 = get_embeddings_cn(attrs_g2, stop_words)

    # # 构造embedding矩阵
    # embedding_matrix_g1 = get_embeddings(attrs_g1)
    # embedding_matrix_g2 = get_embeddings(attrs_g2)

    #
    # 非anchor
    wd_node_similarity = []
    for i in range(9000):
        embedding_g1 = embedding_matrix_g1[i]
        embedding_g2 = embedding_matrix_g2[i]
        similarity = cosine_similarity([embedding_g1], [embedding_g2])[0][0]
        wd_node_similarity.append(similarity)

    #
    #anchor
    # anchor_similarities = []
    # for g1_node, g2_node in anchor_pairs:
    #     # 提取对应的embedding
    #     embedding_g1 = embedding_matrix_g1[g1_node]
    #     embedding_g2 = embedding_matrix_g2[g2_node - 9714]
    #
    #     # 计算并存储余弦相似度
    #     similarity = cosine_similarity([embedding_g1], [embedding_g2])[0][0]
    #     anchor_similarities.append(similarity)
    #
    # with open('wd_anchor_similarities.csv', 'w') as f:
    #     for pair, similarity in zip(anchor_pairs, anchor_similarities):
    #         f.write(f"{pair[0]},{pair[1]},{similarity}\n")

    with open('wd_node_similarities.csv', 'w') as f:
        for i, similarity in enumerate(wd_node_similarity):
            f.write(f"{i},{similarity}\n")