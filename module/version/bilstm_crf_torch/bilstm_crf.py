# -*- encoding: utf-8 -*-
"""
@file: bilstm_crf.py
@time: 2020/5/7 下午3:46
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: BiLSTM + CRF 命名实体识别
# 1. fit 拟合
# 2. valid 验证
# 3. test 测试
# 4. predict 预测
# 5. eval_score 实体评分
# 6. dump 保存
# 7. load 加载
"""
import torch
import os
import sys
import numpy as np
from torch import nn
from module.core.exception import exception_handling
from module.core.exception import ParameterError
from module.core.exception import UnknownError
from module.core.exception import FileNotFoundException
from module.core.color import Color
from module.core.utils import Writer


def _shuffle(*args):
    """
    洗牌操作。重新排序数据集。
    :param args: (list or tuple, mandatory) 需要洗牌的数据集。
    :return: (list) 重新排序的数据集
    """
    # 检查输入的参数是不是 list 或 tuple 类型
    for arg in args:
        if not isinstance(arg, (list, tuple)):
            raise ParameterError("args must be is list or tuple, but actually get {}".format(type(arg)))

    # 检查输入的每个参数大小是不是相等
    if len(args) > 1:
        for args_a, args_b in zip(args[:-1], args[1:]):
            if len(args_a) != len(args_b):
                raise ParameterError(
                    "args size must be equal. expect size:{}, actually size{}".format(len(args_a), len(args_b)))

    # 随机排序索引
    random_index = np.random.permutation(len(args[0]))

    # 洗牌
    result = list()
    for arg in args:
        arg_result = list()
        for index in random_index:
            arg_result.append(arg[index])
        result.append(arg_result)

    return result


def _split_valid_dataset(*args, ratio=0.2):
    """
    划分验证数据集。
    :param args: (list, mandatory) 数据集。列表格式。
    :param ratio: (int, optional, default=0.2) 验证数据集比例。0～1范围内。
    :return: (list) 训练数据集和验证数据集
    """
    if ratio > 1 or ratio < 0:
        raise ParameterError("dataset ratio must be is 0 ~ 1 range. actually get: {}".format(ratio))

    dataset = _shuffle(*args)

    sample_num = int(len(dataset[0]) * ratio)
    if sample_num == 0:
        sample_num = 1

    train_dataset, valid_dataset = list(), list()
    for data in dataset:
        valid_dataset.append(data[:sample_num])
        train_dataset.append(data[sample_num:])

    return train_dataset, valid_dataset


class BiLSTM_CRF(nn.Module):
    # 极小值。用于对转移概率矩阵的 <START> 和 <END> 进行初始化。
    MINIMUM_VALUE = -10000

    def __init__(self, word_dict, embedding_dim=200, hidden_size=256):
        super(BiLSTM_CRF, self).__init__()
        self.word_dict = word_dict
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 保存词汇量的大小
        self.vocab_size = len(self.word_dict.word_token)
        # 保存标识符的大小
        self.identifier_size = len(self.word_dict.identifier_token)
        self.start_tag = self.word_dict.identifier_to_token(self.word_dict.START_TAG)
        self.end_tag = self.word_dict.identifier_to_token(self.word_dict.END_TAG)
        self.num_layers = 1

        # 如果为 True 则为双边循环神经网络
        self.bidirectional = True
        self.num_directions = 2

        # 保存训练和验证的最小loss
        self.train_minimum_loss = None
        self.valid_minimum_loss = None

        self.__init_layer__()

    def __init_layer__(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 初始化 BiLSTM 。
        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional)

        # 初始化LSTM输出到线性输出层。
        self.hidden_to_identifier = nn.Linear(self.num_directions * self.hidden_size,
                                              self.identifier_size)

        # 初始化CRF转移矩阵转移参数。例如从 A 转移到 B
        # transition 横轴是目标节点 B。
        # transition 纵纵是发射节点 A。
        # transition[B] 表示到目标节点 B 的所有转移概率。
        self.transition = nn.Parameter(torch.randn(self.identifier_size, self.identifier_size))
        self.transition.data[self.start_tag, :] = BiLSTM_CRF.MINIMUM_VALUE
        self.transition.data[:, self.end_tag] = BiLSTM_CRF.MINIMUM_VALUE

    def init_hidden(self):
        """
        初始化LSTM模型第一个时间步输入 h 和 C
        :return:
        """
        return (torch.randn(self.num_layers * self.num_directions, 1, self.hidden_size),
                torch.randn(self.num_layers * self.num_directions, 1, self.hidden_size))

    def _lstm_forward(self, sent):
        """
        lstm 前向计算。
        :param sent: (tensor, mandatory) 句子张量。shape:[batch_size=1, seq_len]。
        :return: (tensor) 张量。shape:[seq_len, self.__identifier_size]
        """
        embed_output = self.embedding(sent)

        # shape：[seq_len, batch_size, embedding_dim]
        embed_output = embed_output.view(sent.shape[1], 1, -1)

        # lstm 输出, 如果 batch_first = False。则 shape：[seq_len, batch_size, self.hidden_size * self.num_direction]
        lstm_output, _ = self.lstm(embed_output, self.init_hidden())

        # linear 输出。shape：[seq_len, self.identifier_size]
        output = self.hidden_to_identifier(lstm_output.view(sent.shape[1], -1))
        return output

    @exception_handling
    def forward(self, features):
        """
        前向反馈。
        :param features: (tensor, mandatory) 句子特征。shape: [seq_len, identifier_size]
        :return: (list) 标识符列表
        """
        # 将字符串句子转换成tensor。shape: [seq_len]
        # sent_tensor = torch.tensor(self.encoding_sent(sent), dtype=torch.long).view(1, -1)

        # lstm 计算输出。 shape: [seq_len, self.identifier_size]
        # features = self._lstm_forward(sent_tensor)

        # 初始化viterbi算法的开始变量。shape： [1, identifier_size]. 例如：[[0, -1000, -1000, -1000]] 。
        # 其中 0 表示 <START> 标识符。
        init_var = torch.full((1, self.identifier_size), BiLSTM_CRF.MINIMUM_VALUE)
        # init_var[0][self.word_dict.identifier_to_token(self.word_dict.START_TAG)] = 0.0
        init_var[0][self.start_tag] = 0.0

        # 保存最佳路径的ID值。
        best_path_id = list()

        forward_var = init_var
        # Viterbi 算法计算路径
        for feat in features:
            identifier_list = list()
            viterbi_var = list()
            for _, next_identifier_id in self.word_dict.identifier_token.items():
                # 发射评分。shape: [1, identifier_size]。例如：[[1,1,1,1,1]], identifier_size = 5
                # emit_score = feat[next_identifier_id].view(1, -1).expand(1, self.__identifier_size)

                # 转移评分。 shape：[1, identifier_size]
                trans_score = self.transition[next_identifier_id].view(1, -1)

                # 当前标识符节点评分。例如 A 节点。有 A -> A、B -> A、C -> A .
                # emit_score 为目标 A 的发射评分
                # trans 为转移评分
                # next_identifier_score = forward_var + emit_score + trans_score
                next_identifier_score = forward_var + trans_score

                # # 转移。例如从 <START> 到下一个 标识符的概率评分。输出 shape：[1, identifier_size]
                # next_identifier_var = forward_var + self.transition[next_identifier_id].view(1, -1)

                # 保存当前最大的评分。例如 A -> A 和 B -> A 。B -> A 的评分大。则保存 B 的id
                best_identifier_id = next_identifier_score.argmax(1).item()
                identifier_list.append(best_identifier_id)

                # 保存当前转移中最大的一个评分值。例如: [tensor([0.1]), tensor([0.2])]
                viterbi_var.append(next_identifier_score[0][best_identifier_id].view(1))

            # 转移评分 + 发射评分。shape: [1, identifier_size]
            forward_var = (torch.cat(viterbi_var) + feat).view(1, -1)

            # 保存最佳路径id
            best_path_id.append(identifier_list)

        # 计算由最后一个字符到 <END> 转移概率。
        end_var = forward_var + self.transition[self.end_tag]
        # 保存概率最高的一条。选择概率最大的一条进行回溯
        best_identifier_id = end_var.argmax(1).item()

        reverse_path_id = [best_identifier_id]
        # 回朔。由后往前依次获取最好的路径。
        for b_path_id in reversed(best_path_id):
            reverse_path_id.append(b_path_id[best_identifier_id])

        # 调整逆向位置。例如将 "C、B、A" 调整为 "A、B、C"
        reverse_path_id.reverse()
        # 删除<START> 标识符
        best_path_identifier = reverse_path_id[1:]

        if features.shape[0] != len(best_path_identifier):
            raise UnknownError("Unknown Error! Please check code")

        return best_path_identifier

    def _log_likelihood_loss(self, features, labels):
        """
        计算损失。
        :param features: (tensor, mandatory) 特征。shape: [seq_len, identifier_size]
        :param labels: (tensor, mandatory) 标签。[seq_len]
        :return: (tensor) 损失
        """
        forward_score = self._forward_score(features)
        sentence_score = self._sentence_score(features, labels)
        return forward_score - sentence_score

    @exception_handling
    def _sentence_score(self, features, labels):
        """
        目标评分。根据提供的句子特征和标签进行评分。
        :param features: (tensor, mandatory) 句子张量。shape:[seq_len, identifier_size]
        :param labels: (tensor, mandatory) 句子标签。shape:[seq_len]
        :return: (tensor) 张量值。shape:[1]
        """
        if features.shape[0] != labels.shape[0]:
            raise ParameterError(
                "feature.shape[1] size: {} and labels length: {} must be equal".format(features.shape[0],
                                                                                       labels.shape[0]))
        score = torch.zeros(1)

        # 加上 <START> 标识符token
        tags = torch.cat([torch.tensor([self.start_tag], dtype=torch.long), labels])
        for i, feat in enumerate(features):
            # 发射评分
            emit_score = feat[tags[i + 1]]
            # 转移评分。发射节点到目标节点的评分。
            transition_score = self.transition[tags[i + 1]][tags[i]]
            # 叠加评分
            score = score + emit_score + transition_score

        score = score + self.transition[self.end_tag][tags[-1]]
        return score

    def _forward_score(self, features):
        """
        前向反馈评分。
        :param features: (tensor, mandatory) 文本特征。shape:[seq_len, identifier_size]
        :return: (tensor) 张量值。shape:[1]
        """

        def log_sum_exp(score_tensor):
            """
            对评分进行log、sum、exp 计算。将所有评分简化成一个评分。
            :param score_tensor: (tensor, mandatory) 评分张量。shape:[1, identifier_size]
            :return: (tensor) 评分值
            """
            max_score = score_tensor[0][score_tensor.argmax(1)]
            max_score_broadcast = max_score.view(1, -1).expand(1, score_tensor.shape[1])
            return max_score + torch.log(torch.sum(torch.exp(score_tensor - max_score_broadcast)))

        # 初始化
        init_var = torch.full([1, self.identifier_size], BiLSTM_CRF.MINIMUM_VALUE)
        init_var[0][self.start_tag] = 0.0

        forward_var = init_var
        for feat in features:
            score_list = list()
            for _, identifier_id in self.word_dict.identifier_token.items():
                # 发射节点的发射评分。shape:[1, identifier_size]
                emit_score = feat[identifier_id].view(1, -1).expand(1, self.identifier_size)

                # 发射节点到目标节点的转移评分。shape:[1, identifier_size]
                trans_score = self.transition[identifier_id].view(1, -1)

                # 叠加指向当前目标节点的评分。例如 B 为目标节点，B = identifier_id。其中A、B、C发射节点指向 B。
                # 当前目标节点B叠加该发射评分和转移评分。forward_var 为之前的评分
                # shape:[1, identifier_size]
                score = forward_var + emit_score + trans_score

                # 保存评分
                score_list.append(log_sum_exp(score))

            # 使用保存的评分，更新forward_var
            forward_var = torch.cat(score_list).view(1, -1)

        # 最终评分
        score = forward_var + self.transition[self.end_tag].view(1, -1)

        terminal_score = log_sum_exp(score)
        return terminal_score

    def save_pth(self, train_loss, valid_loss, pth="BiLSTM_CRF.pth"):
        """
        保存模型策略。如果 train_loss 和 valid_loss 都小于 self.train_minimum_loss 和 self.valid_minimum_loss 则保存模型
        :param train_loss: (float, mandatory) 每轮次训练的平均损失
        :param valid_loss: (float, mandatory) 每轮次验证的平均损失
        :param pth: (str, optional, default='BiLSTM_CRF.pth') 保存模型文件名
        :return:
        """
        is_save = False
        if self.train_minimum_loss is None and self.valid_minimum_loss is None:
            self.train_minimum_loss = train_loss
            self.valid_minimum_loss = valid_loss
            is_save = True

        elif self.train_minimum_loss > train_loss and self.valid_minimum_loss > valid_loss:
            self.train_minimum_loss = train_loss
            self.valid_minimum_loss = valid_loss
            is_save = True

        if is_save:
            self.dump(pth)

    def fit(self, x, y,
            epochs=30,
            lr=0.01,
            weight_decay=1e-4,
            ratio=0.2,
            pth="BiLSTM_CRF_NER.pth"):
        """
        拟合/训练模型。
        :param x: (list, mandatory) 训练数据。如：[[1,2,3...]...]
        :param y: (list, mandatory) 标签数据。如：[[1,2,3...]...]
        :param epochs: (int, optional, default=30) 训练轮次
        :param lr: (float, optional, default=0.01) 学习率
        :param weight_decay: (float, optional, default=1e-4) 权重衰减率。正则化作用。
        :param ratio: (int, optional, default=0.2) 采取 k 折交叉验证。验证数据集比例
        k=10，则有 len(train_data) // k * (k-1) 做训练集。len(train_data) // k * 1 做验证集。
        :param pth: (str, optional， default="BiLSTM_CRF_NER.pth") 保存模型的文件名
        :return:
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)

        print("start fit ...")
        for e in range(epochs):
            train_loss = 0

            # 划分数据集
            [train_x, train_y], [valid_x, valid_y] = _split_valid_dataset(x, y, ratio=ratio)

            for step, (sent, identifier) in enumerate(zip(train_x, train_y)):
                self.zero_grad()
                # 将句子编码并转换成张量。shape:[1, seq_len]
                sent_tensor = torch.tensor(sent, dtype=torch.long).view(1, -1)
                # 将标签编码并转换成张量。shape:[seq_len]
                identifier_tensor = torch.tensor(identifier, dtype=torch.long)

                # 句子特征
                features = self._lstm_forward(sent_tensor)

                # 计算损失
                loss = self._log_likelihood_loss(features, identifier_tensor)

                # 反向传播更新
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item()
                output_str = "Train Epochs: {}/{}, ".format(e + 1, epochs) \
                             + "Step: {}/{}, ".format(step + 1, len(train_x)) \
                             + "Loss: {:.4f}".format(train_loss / (step + 1))
                sys.stdout.write("\r" + output_str)
                sys.stdout.flush()
            print()

            # 使用验证数据集进行验证评分
            self.train(False)
            valid_loss = self.valid(valid_x, valid_y)
            self.train(True)

            self.save_pth(train_loss / len(train_x), valid_loss, pth=pth)
            print()

    def valid(self, x, y):
        """
        验证
        :param x: (list, mandatory) 验证数据
        :param y: (list, mandatory) 验证标签
        :return:
        """
        assert len(x) == len(y), ParameterError("Please check data size. x:{} and y:{} must be equal".format(len(x),
                                                                                                             len(y)))
        valid_loss = 0
        for step, (valid_x, valid_y) in enumerate(zip(x, y)):
            sent_tensor = torch.tensor(valid_x, dtype=torch.long).view(1, -1)
            identifier_tensor = torch.tensor(valid_y, dtype=torch.long)

            features = self._lstm_forward(sent_tensor)

            loss = self._log_likelihood_loss(features, identifier_tensor)

            valid_loss = valid_loss + loss.item()

            output_str = "Valid: Step: {}/{}, ".format(step + 1, len(x)) \
                         + "Loss: {:.4f}".format(valid_loss / (step + 1))
            sys.stdout.write("\r" + output_str)
            sys.stdout.flush()

        print()
        return valid_loss / len(x)

    def test(self, x, y):
        """
        测试
        :param x: (list, mandatory) 验证数据, 如：['桑普多vs卡塔尼推荐：3'...]
        :param y: (list, mandatory) 验证标签, 如：[[B_book, I_book, I_book, E_book, O, O, O, O, O]...]
        :return:
        """
        assert len(x) == len(y), ParameterError("Please check data size. x:{} and y:{} must be equal".format(len(x),
                                                                                                             len(y)))
        test_loss = 0
        for step, (test_x, test_y) in enumerate(zip(x, y)):
            sent_tensor = torch.tensor(test_x, dtype=torch.long).view(1, -1)
            identifier_tensor = torch.tensor(test_y, dtype=torch.long)

            features = self._lstm_forward(sent_tensor)

            loss = self._log_likelihood_loss(features, identifier_tensor)

            test_loss = test_loss + loss.item()

            output_str = "Test: Step: {}/{}, ".format(step + 1, len(x)) \
                         + "Loss: {:.4f}".format(test_loss / (step + 1))
            sys.stdout.write("\r" + output_str)
            sys.stdout.flush()

    def predict(self, sent):
        """
        预测
        :param sent: (list, mandatory) 句子列表。如：[1,2,3,4,]
        :return: (list) 预测标识符token
        """
        sent_tensor = torch.tensor(sent, dtype=torch.long).view(1, -1)
        features = self._lstm_forward(sent_tensor)
        identifier_token = self.forward(features)
        return identifier_token

    def dump(self, file="BiLSTM_CRF_NER.pth"):
        """
        保存模型。
        :param file: (str, optional, default="ner_bilstm_crf.pth") 模型文件名
        :return:
        """
        Writer.check_path(file)
        print("remove file: {} {}".format(file, Writer.remove_file(file)))
        torch.save(self, file)

        print("model save over! File: {}".format(file))

    @staticmethod
    @exception_handling
    def load(file):
        """
        加载模型.
        :param file: (str, mandatory) 模型文件名
        :return: (BiLSTM_CRF_NER) 模型对象
        """
        if not os.path.exists(file):
            raise FileNotFoundException("File: {}".format(file))

        model = torch.load(file)
        return model


def virtual_dataset_generator(size, x_size, y_size, sequence_length):
    import random
    x, y = list(), list()
    for i in range(size):
        random_seq = random.randint(sequence_length - 2, sequence_length)
        x.append(np.random.randint(low=0, high=x_size, size=[random_seq]))
        y.append(np.random.randint(low=2, high=y_size, size=[random_seq]))

    return x, y


def test_module_func(operation):
    if operation == virtual_dataset_generator.__name__:
        x, y = virtual_dataset_generator(10, 64, 32, 10)
        for i in range(10):
            print("{}, length:{}, x = {}".format(i, len(x[i]), x[i]))
            print("{}, length:{}, y = {}".format(i, len(y[i]), y[i]))

    elif operation == BiLSTM_CRF.fit.__name__:
        test_data = ['另外意大利的PlayGeneration杂志也刚刚给出了92%的高分。',
                     '生生不息CSOL生化狂潮让你填弹狂扫',
                     '突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，',
                     '吴三桂演义》小说的想像，说是为牛金星所毒杀。……在小说中加插一些历史背景，',
                     '你们是最棒的!#英雄联盟d学sanchez创作的原声王'
                     ]
        test_identifiers = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book',
             'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'E_book', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B_game', 'I_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['B_game', 'I_game', 'I_game', 'I_game', 'I_game', 'I_game', 'I_game', 'E_game', 'O', 'B_name', 'I_name',
             'I_name', 'I_name', 'I_name', 'I_name', 'E_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'B_name', 'I_name', 'I_name', 'I_name', 'E_name', 'O'],
            ['B_book', 'I_book', 'I_book', 'I_book', 'I_book', 'E_book', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'B_name', 'I_name', 'E_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_game', 'I_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ]
        from module.version.bilstm_crf_torch.word_dict import WordDict
        word_dict = WordDict()
        word_dict.fit(test_data, test_identifiers)

        embedding_dim = 100
        hidden_size = 64
        model = BiLSTM_CRF(word_dict, embedding_dim=embedding_dim, hidden_size=hidden_size)

        x = [word_dict.encoding_sentence(sent) for sent in test_data]
        y = [word_dict.encoding_identifier(identifier) for identifier in test_identifiers]

        model.fit(x, y, epochs=10)
        model.test(x, y)
        result = model.predict(x[0])
        assert len(result) == len(x[0]), \
            "forward identifiers size:{}, data size:{} unequal".format(len(result), len(x[0]))
        print("predict result:", result)
    elif operation == BiLSTM_CRF.load.__name__:
        test_data = ['另外意大利的PlayGeneration杂志也刚刚给出了92%的高分。',
                     '生生不息CSOL生化狂潮让你填弹狂扫',
                     '突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，',
                     '吴三桂演义》小说的想像，说是为牛金星所毒杀。……在小说中加插一些历史背景，',
                     '你们是最棒的!#英雄联盟d学sanchez创作的原声王'
                     ]
        test_identifiers = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book',
             'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'I_book', 'E_book', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B_game', 'I_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['B_game', 'I_game', 'I_game', 'I_game', 'I_game', 'I_game', 'I_game', 'E_game', 'O', 'B_name', 'I_name',
             'I_name', 'I_name', 'I_name', 'I_name', 'E_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O',
             'B_name', 'I_name', 'I_name', 'I_name', 'E_name', 'O'],
            ['B_book', 'I_book', 'I_book', 'I_book', 'I_book', 'E_book', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'B_name', 'I_name', 'E_name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O',
             'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_game', 'I_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O',
             'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ]
        from module.version.bilstm_crf_torch.word_dict import WordDict
        word_dict = WordDict()
        word_dict.fit(test_data, test_identifiers)

        embedding_dim = 100
        hidden_size = 64
        model = BiLSTM_CRF(word_dict, embedding_dim=embedding_dim, hidden_size=hidden_size)

        x = [word_dict.encoding_sentence(sent) for sent in test_data]
        y = [word_dict.encoding_identifier(identifier) for identifier in test_identifiers]

        pth = './model/pth/BiLSTM_CRF.pth'
        model.fit(x, y, epochs=10, pth=pth)

        new_model = BiLSTM_CRF.load(pth)
        new_model.fit(x, y, epochs=5, pth=pth)

        result = new_model.predict(x[0])
        print("predict result:", result)


if __name__ == "__main__":
    # test_module_func(virtual_dataset_generator.__name__)
    # test_module_func(BiLSTM_CRF.fit.__name__)
    test_module_func(BiLSTM_CRF.load.__name__)
