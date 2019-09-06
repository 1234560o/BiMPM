# @Author:zwj

import json
import jieba
import re
from collections import Counter
from operator import itemgetter
import numpy as np


class DataLoader(object):
    """数据加载父类"""


class BQLoader(DataLoader):

    def tokenizer(self, text, skip_space=False):
        if skip_space:
            text = re.sub(r" ", "", text)
        return jieba.lcut(text, cut_all=False)

    def read_data(self, filename, use_dict=None):
        """
        如果数据格式不一样，可修改应用于不同格式的数据集
        :param filename:
        :param use_dict:
        :return:
        """
        text_a = []
        text_b = []
        labels = []
        if use_dict:
            jieba.load_userdict(use_dict)
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                if line.strip() == "":
                    continue
                content = json.loads(line.strip())
                text_a.append(self.tokenizer(content["sentence1"]))
                text_b.append(self.tokenizer(content["sentence2"]))
                labels.append(int(content["gold_label"]))
        return text_a, text_b, labels


class SNLILoader(DataLoader):

    def read_data(self, filename):
        text_a = []
        text_b = []
        labels = []
        with open(filename, 'r', encoding='utf-8') as fin:
            labels2id = {"neutral": 0, "contradiction": 1, "entailment": 2}
            for line in fin.readlines():
                content = line.strip().split("\t")
                assert len(content) == 3
                if content[0] == "-":
                    continue
                text_a.append(content[1].split(" "))
                text_b.append(content[2].split(" "))
                labels.append(labels2id[content[0]])
        return text_a, text_b, labels


def write_token_data(text_a, text_b, labels, outfile):
    """
    将分词后的数据结果写入文件，以便观察分词效果以及添加分词词典
    """
    with open(outfile, 'w', encoding='utf-8') as fout:
        fout.write("label" + "\t" + "text_a" + "\t" + "text_b" + "\n")
        for i in range(len(labels)):
            fout.write(
                str(labels[i]) + "\t" + ' '.join(text_a[i]) + "\t" + ' '.join(text_b[i]) + "\n")


def visual_length(all_data):
    """
    查看数据的序列长度分布，以便确定句子最大长度大小
    """
    all_length = []
    for text in all_data:
        all_length.append(len(text))
    print(sorted(Counter(all_length).items(), key=itemgetter(0), reverse=True))


# def load_word2vec(vec_file):
#     word2index = {}
#     index2vector = []
#     word2vec = list(open(vec_file, 'r', encoding='utf-8').readlines())
#     word2vec = [s.split() for s in word2vec]
#     embedding_dim = int(word2vec[0][-1])    # 第一行第二个数是向量维度
#     word2index["<PAD>"] = 0
#     word2index["<UNK>"] = 1
#     index2vector.append(np.zeros(embedding_dim))
#     index2vector.append(np.random.normal(loc=0, scale=1, size=embedding_dim))
#     index = 2
#     for vec in word2vec[1:]:
#         a = np.zeros(embedding_dim)
#         for i in range(embedding_dim):
#             a[i] = float(vec[i + 1])
#         word2index[vec[0]] = index
#         index2vector.append(a)
#         index += 1
#
#     print("  WordTotal=\t", len(index2vector))
#     print("  Word dimension=\t", embedding_dim)
#
#     index2vector = np.array(index2vector).astype(np.float32)   # 必须
#     return word2index, index2vector


# def build_vocab(word2index, vocab_file):
#     """
#     为了避免在预测的时候需要加载预训练词向量文件，这里建立词典来获取word的id信息
#     :param word2index:
#     :param vocab_file:
#     :return:
#     """
#     with open(vocab_file, 'w', encoding='utf-8') as fin:
#         for word, index in word2index.items():
#             fin.write(word + "\n")

def build_vocab(all_data, vocab_file, vocab_size):
    counter = Counter()
    for i in range(len(all_data)):
        for word in all_data[i]:
            counter[word] += 1
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [k[0] for k in sorted_word_to_cnt]
    sorted_words = ["<PAD>", "<UNK>"] + sorted_words
    if len(sorted_words) > vocab_size:
        sorted_words = sorted_words[:vocab_size]
    with open(vocab_file, 'w', encoding='utf-8') as fout:
        for word in sorted_words:
            fout.write(word + "\n")


def read_vocab(vocab_file):
    word2index = {}
    words = []
    with open(vocab_file, "r", encoding='utf-8') as fout:
        for line in fout.readlines():
            content = line.strip()
            words.append(content)
    for i, word in enumerate(words):
        word2index[word] = i
    return word2index, words


def load_word2vec(words, vec_path, skip_first_line=True):
    word2vec = {}
    with open(vec_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if skip_first_line:
                skip_first_line = False
                continue
            tmp = line.strip().split(" ", 1)
            word2vec[tmp[0]] = [float(vec) for vec in tmp[1].split()]
            embedding_dim = len(tmp[1].split())
    embedding = []
    count = 0
    for word in words:
        if word in word2vec.keys():
            embedding.append(word2vec[word])
            count += 1
        else:
            embedding.append(list(np.random.normal(size=embedding_dim)))
    print(" 共有{}/{}个词使用了预训练词向量. ".format(count, len(words)))
    print(" 预训练的embedding table维度大小是[{} , {}] ".format(len(embedding), embedding_dim))
    return embedding


def convert_text2id(token_text, word2index, seq_length):
    if len(token_text) >= seq_length:
        # 截断,这里会改变变量token_text
        token_text = token_text[-seq_length:]
    else:
        token_text.extend(["<PAD>"] * (seq_length - len(token_text)))
    assert len(token_text) == seq_length
    token_id = []
    for word in token_text:
        if word in word2index:
            token_id.append(word2index[word])
        else:
            token_id.append(word2index["<UNK>"])
    return token_id


def process_data(text_a, text_b, label_id, word2index, seq_length):
    data_len = len(label_id)
    data_a_id = np.zeros([data_len, seq_length], dtype=np.int32)
    data_b_id = np.zeros([data_len, seq_length], dtype=np.int32)
    for i in range(data_len):
        data_a_id[i] = convert_text2id(text_a[i], word2index, seq_length)
        data_b_id[i] = convert_text2id(text_b[i], word2index, seq_length)

    for number in range(3):
        print("Example {}: ".format(number + 1))
        print("tokens_a: ", text_a[number])
        print("tokens_a_id: ", data_a_id[number])
        print("tokens_b: ", text_b[number])
        print("tokens_b_id: ", data_b_id[number])
        print("label_id: ", label_id[number])
    return data_a_id, data_b_id, np.array(label_id)


# def gen_batch(data_a_id, data_a_mask, data_b_id, data_b_mask, label_id, batch_size, is_training=True):
#     dataset = tf.data.Dataset.from_tensor_slices((data_a_id, data_a_mask, data_b_id, data_b_mask, label_id))  # [L, S]
#     if is_training:
#         dataset = dataset.shuffle(100000).repeat().batch(batch_size, drop_remainder=True)
#     else:
#         dataset = dataset.repeat().batch(batch_size, drop_remainder=False)
#     return dataset


def batch_iter(data_a_id, data_b_id, label_id, batch_size, is_training=True):
    """生成批次数据"""
    data_len = len(label_id)
    num_batch = int((data_len - 1) / batch_size) + 1

    if is_training:
        num_batch = int(data_len // batch_size)  # 训练模型使得batch_size保存一致
        indices = np.random.permutation(np.arange(data_len))
        data_a_id = data_a_id[indices]
        data_b_id = data_b_id[indices]
        label_id = label_id[indices]

    for i in range(num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, data_len)
        yield data_a_id[start:end], data_b_id[start:end], label_id[start:end]


if __name__ == "__main__":
    data_dir = "data/BQ_corpus/"
    Data = BQLoader()
    a, b, y = Data.read_data(data_dir + "BQ_train.json")
    visual_length(a + b)
    build_vocab(a + b, "hehe.txt", 2000)
    w2id, _ = read_vocab("hehe.txt")
    a1, b1, y1 = process_data(a, b, y, w2id, 64)
    dataset = batch_iter(a1, b1, y1, 32)
    j = 0
    for a11, b11, y11 in dataset:
        j += 1
        print(a11, b11, y11)
        if j > 2:
            break
