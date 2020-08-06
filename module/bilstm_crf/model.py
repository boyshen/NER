# -*- encoding: utf-8 -*-
"""
@file: model.py
@time: 2020/7/29 下午4:02
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""
import sys
import os
import torch
import json
from torch import nn
from module.core.utils import split_valid_dataset
from module.core.utils import batch
from module.core.utils import template
from module.core.utils import Writer
from module.core.exception import FileNotFoundException


def batch_log_sum_exp(vec):
    max_vec, _ = torch.max(vec, dim=1)
    broadcast_max_vec = max_vec.view(-1, 1).repeat(1, vec.shape[1])
    return max_vec + torch.log(torch.sum(torch.exp(vec - broadcast_max_vec), dim=1))


class DataLoader(object):
    def __init__(self, x, y, input_length, embedding_dim, batch_size, validation_rate=0.2, shuffle=True):
        """
        初始化
        :param x: (list, mandatory) 训练数据集
        :param y: (list, mandatory) 标签数据集
        :param input_length: (list, mandatory) 句子实际长度
        :param embedding_dim: (int, mandatory) 嵌入维度，用于创建掩码
        :param batch_size: (int, mandatory) 批量样本
        :param validation_rate: (float, optional, default=0.2) 划分验证数据集比例
        :param shuffle: (bool, optional, default=True) 是否洗牌
        """
        self.x = x
        self.y = y
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.shuffle = shuffle

        self.__create_mask__()

    def __create_mask__(self):
        """ 创建序列掩码 """
        self.mask = []
        for i in range(len(self.input_length)):
            sequence_mask = torch.zeros((1, len(self.x[i]), self.embedding_dim))
            sequence_mask[:, :self.input_length[i], :] = 1
            self.mask.append(sequence_mask)

    def take(self):
        """ 获取训练和验证数据集 """
        train_dataset, valid_dataset = split_valid_dataset(self.x, self.y, self.input_length, self.mask,
                                                           ratio=self.validation_rate,
                                                           is_shuffle=self.shuffle)
        train_dataset, train_size = batch(*train_dataset, batch_size=self.batch_size)
        valid_dataset, valid_size = batch(*valid_dataset, batch_size=self.batch_size)

        return zip(*train_dataset), zip(*valid_dataset), train_size, valid_size


class Mean(object):
    def __init__(self):
        self.counter = []

    def reset(self):
        self.counter = []

    def update(self, items):
        self.counter.append(items)

    def result(self):
        return sum(self.counter) / len(self.counter)


class CRF(nn.Module):
    def __init__(self, target_size, start_index, end_index, device='cpu'):
        """
        初始化。
        :param target_size: (int, mandatory) 目标转移矩阵大小
        :param start_index: (int, mandatory) 词典中 <START> 索引
        :param end_index: (int, mandatory) 词典中 <END> 索引
        """
        super(CRF, self).__init__()
        self.target_size = target_size
        self.START_TAG = start_index
        self.END_TAG = end_index
        self.device = device

        self.minimum = -1.0e4

        self.__init_parameter__()

    def __init_parameter__(self):
        self.transitions = torch.randn(self.target_size, self.target_size)
        self.transitions.detach()[self.START_TAG, :] = self.minimum
        self.transitions.detach()[:, self.END_TAG] = self.minimum
        self.transitions.to(self.device)
        self.transitions = nn.Parameter(self.transitions)

    def forward(self, features, seq_len, labels=None):
        # return self._forward_alg(features, seq_len)
        # return self._score_sentence(features, labels, seq_len)
        pass

    def _forward_alg(self, features, seq_len):
        """
        前向序列评分
        :param features: (FloatTensor, mandatory) 样本序列特征。shape: [batch_size, sequence, feature_dim]
        :param seq_len: (list, mandatory) 样本序列实际长度。如 batch_size=3。 seq_len=[12, 13, 14]。元素是每个样本的序列大小
        :return: (FloatTensor) 样本评分。shape:[batch_size, 1]
        """
        # 根据输入的 features 。获取 batch_size, sequence_size .
        batch_size = features.shape[0]
        sequence_size = features.shape[1]

        # 初始化 START. shape: [batch_size, target_size].
        # example: batch_size=2, target_size = 2, start_tag = 0
        #   [[0.0, -10000],
        #    [0.0, -10000]]
        init_alphas = torch.full((batch_size, self.target_size), self.minimum)
        init_alphas[:, self.START_TAG] = 0.0

        # 初始化前向评分变量. shape: [batch_size, sequence, target_size]. 序列 + 1 并初始化 START
        forward_var = torch.zeros((batch_size, sequence_size + 1, features.shape[2]), dtype=torch.float,
                                  device=self.device)
        forward_var[:, 0, :] = init_alphas

        # 初始化转移矩阵。 使用 self.transitions 改变维度为: [batch_size, target_size, target_size]
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(batch_size, 1, 1)

        for i in range(features.shape[1]):
            emit_score = features[:, i, :]
            emit_score = emit_score[:, :, None].repeat(1, 1, transitions.shape[2])

            forward_v = forward_var[:, i, :]
            forward_v = forward_v[:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1)

            # 发射评分 + 转移评分 + 前向评分
            tag_var = emit_score + transitions + forward_v
            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])

            # log_sum_exp 计算。获取到当前序列 i 的所有状态评分。shape:[batch_size, target_size].
            score = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            # 将当前序列结果叠加给 forward_var 前向状态评分
            forward_cloned = forward_var.clone()
            forward_cloned[:, i + 1, :] = max_tag_var + score
            forward_var = forward_cloned

        # 获取每个样本的最后一个序列。shape:[batch_size, target_size]
        forward_var = forward_var[range(forward_var.shape[0]), seq_len, :]

        # 添加序列 END_TAG 结束评分
        end_var = self.transitions[self.END_TAG]
        end_var = end_var[None, :].repeat(batch_size, 1)
        terminal_var = forward_var + end_var

        # 批量进行 log_sum_exp 计算
        alpha = batch_log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, features, labels, seq_len):
        """
        句子评分
        :param features: (FloatTensor, mandatory) 样本序列特征。shape: [batch_size, sequence, feature_dim]
        :param labels: (list, mandatory) 样本序列标签。shape: [batch_size, sequence]
        :param seq_len: 样本序列实际长度。如 batch_size=3。 seq_len=[12, 13, 14]。元素是每个样本的序列大小
        :return: (list) 样本评分。shape:[batch_size, 1]
        """
        batch_size = features.shape[0]

        # <START> 和 <END> 转换成样本张量。shape:[batch_size, 1]
        start = torch.tensor(self.START_TAG, dtype=torch.long).view(1, -1).repeat(batch_size, 1).to(self.device)
        end = torch.tensor(self.END_TAG, dtype=torch.long).view(1, -1).repeat(batch_size, 1).to(self.device)

        # 将标签序列填充 <START> 和 <END>
        pad_start_label = torch.cat((start, labels), dim=1)
        pad_end_label = torch.cat((labels, end), dim=1)

        # 将样本序列多余的填充字符都替换成 <END> 结束
        for i in range(batch_size):
            pad_end_label[i, seq_len[i]:] = self.END_TAG

        score = torch.zeros(batch_size)
        for i in range(batch_size):
            # 获取发射评分。 i 为样本. seq_len[i] 为当前样本的实际序列大小。使用 range 迭代循环。
            # 例如：当 seq_len[i]=3 时, range(3) = [0, 1, 2]
            # labels[i, :seq_len[i]] 为当前样本序列的实际标签。
            emit_score = features[i, range(seq_len[i]), labels[i, :seq_len[i]]]

            # 获取序列转移评分。
            # 获取样本序列到 <END> 和 <END> 的索引.
            # 例如:
            # pad_end_label=[[2,3,4,1,1]]. pad_end_label[0][:3+1] = [2,3,4,1]
            # pad_start_label=[[0,2,3,4,1]]. pad_end_label[0][:3+1] = [0,2,3,4]
            # self.transitions[[2,3,4,1], [0,2,3,4]] =
            #  self.transitions[2][0], self.transitions[3][2], self.transitions[4][1], self.transitions[1][4]
            end_label = pad_end_label[i][:seq_len[i] + 1]
            start_label = pad_start_label[i][:seq_len[i] + 1]
            transitions_score = self.transitions[end_label, start_label]

            # 保存评分
            score[i] = torch.sum(emit_score) + torch.sum(transitions_score)

        return score

    def viterbi_decode(self, features):
        """
        解码
        :param features: (FloatTensor, mandatory) 样本序列特征。shape: [sequence, feature_dim]
        :return: (list) 解码序列
        """
        sequence = features.shape[0]

        init_var = torch.full((1, self.target_size), self.minimum)
        init_var[0][self.START_TAG] = 0.0

        best_path = []
        forward_var = init_var
        for i in range(sequence):
            # 获取下一个转移状态
            next_state = self.transitions + forward_var.view(1, -1).repeat(self.target_size, 1).to(self.device)
            # 选择转移状态中最大的值和索引
            viterbi_ver, b_path = torch.max(next_state, dim=1)
            forward_var = viterbi_ver + features[i]
            best_path.append(b_path)

        # 添加 <END>
        terminal_var = forward_var.to(self.device) + self.transitions[self.END_TAG].to(self.device)

        # 获取评分最高的 id
        _, best_id = torch.max(terminal_var, dim=0)
        best_path_id = [best_id.item()]
        # 回朔。由后往前依次获取最好的路径。
        for b_path in reversed(best_path):
            best_path_id.append(b_path[best_id.item()].item())

        # 调整逆向位置。例如将 "C、B、A" 调整为 "A、B、C"
        best_path_id.reverse()
        assert best_path_id[0] == self.START_TAG

        # 删除 <START> 索引
        best_path_id = best_path_id[1:]
        return best_path_id

    def computer_loss(self, features, labels, seq_len):
        """
        计算损失
        :param features: (FloatTensor, mandatory) 样本序列特征。shape: [batch_size, sequence, feature_dim]
        :param labels: (list, mandatory) 样本序列标签。shape: [batch_size, sequence]
        :param seq_len: (list, mandatory)样本序列实际长度。如 batch_size=3。 seq_len=[12, 13, 14]。元素是每个样本的序列大小
        :return: (Tensor) 评分
        """
        forward_score = self._forward_alg(features, seq_len).to(self.device)
        sentence_score = self._score_sentence(features, labels, seq_len).to(self.device)
        score = forward_score - sentence_score
        return score.mean()


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, target_size, hidden_size, start_index, end_index, device='cpu',
                 dropout=0.5):
        """
        初始化
        :param vocab_size: (int, mandatory) 词汇量大小
        :param embedding_dim: (int, mandatory) 嵌入维度大小
        :param target_size: (int, mandatory) 目标转移矩阵大小
        :param hidden_size: (int, mandatory) 目标隐藏层大小
        :param start_index: (int, mandatory) (int, mandatory) 词典中 <START> 索引
        :param end_index: (int, mandatory) (int, mandatory) 词典中 <END> 索引
        :param device: (str, optional, default='cpu') 训练设备。cpu 或 cuda
        :param dropout: (float, optional, default=0.5)
        """
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.device = device
        self.dropout = dropout

        self.num_layers = 2
        self.bidirectional = True
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.__init_layer__()

    def __init_layer__(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.bilstm = nn.LSTM(self.embedding_dim,
                              self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)
        input_size = self.num_directions * self.hidden_size
        self.layer_normal = nn.LayerNorm(input_size)
        self.dense = nn.Linear(input_size, self.target_size)

        self.crf = CRF(self.target_size, self.start_index, self.end_index, device=self.device)

    def forward(self, inputs, input_mask=None):
        """
        前向反馈
        :param inputs: (tensor, mandatory) 输入序列。shape:[batch_size, sequence]
        :param input_mask: (tensor, mandatory) 掩码。shape:[batch_size, sequence, embedding_dim]
        :return: (tensor) shape: [batch_size, sequence, target_size]
        """
        output = self.embedding(inputs)
        if input_mask is not None:
            output = output * input_mask
        output, _ = self.bilstm(output)
        output = self.layer_normal(output)
        output = self.dense(output)
        return output

    def forward_loss(self, inputs, input_mask, seq_len, labels):
        """
        前向损失
        :param inputs: (tensor, mandatory) 输入序列。shape:[batch_size, sequence]
        :param input_mask: (tensor, mandatory) 掩码。shape:[batch_size, sequence, embedding_dim]
        :param seq_len: (list, mandatory)样本序列实际长度。如 batch_size=3。 seq_len=[12, 13, 14]。元素是每个样本的序列大小
        :param labels: (list, mandatory) 样本序列标签。shape: [batch_size, sequence]
        :return:
        """
        features = self.forward(inputs, input_mask)
        loss = self.crf.computer_loss(features, labels, seq_len)
        return loss


class NameEntityRecognitionModel(object):

    def __init__(self, vocab_size, embedding_dim, target_size, hidden_size, start_index, end_index, dropout=0.5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.dropout = dropout

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 保存训练和验证的最小loss
        self.train_minimum_loss = None
        self.valid_minimum_loss = None

        self.learning_rate = 0

        self.__init_model__()
        self.__init_metrics__()

    def __init_model__(self):
        # 初始化模型
        self.model = BiLSTM_CRF(vocab_size=self.vocab_size,
                                embedding_dim=self.embedding_dim,
                                target_size=self.target_size,
                                hidden_size=self.hidden_size,
                                start_index=self.start_index,
                                end_index=self.end_index,
                                device=self.device,
                                dropout=self.dropout)
        self.model.to(self.device)

    def __init_metrics__(self):
        self.train_loss = Mean()
        self.valid_loss = Mean()

    def fit(self, x, y, input_length,
            epochs=10,
            lr=0.0001,
            weight_decay=1.0e-4,
            batch_size=4,
            shuffle=True,
            validation_rate=0.2,
            grad_norm=5,
            factor=0.1,
            patience=2,
            eps=1.0e-8,
            checkpoint='NER_BiLSTM_CRF.pth'):
        """
        拟合
        :param x: (list, mandatory) 训练数据集
        :param y: (list, mandatory) 标签数据集
        :param input_length: (list, mandatory) 数据集实际长度
        :param epochs: (int, optional, default=10) 训练轮次
        :param lr: (float, optional, default=0.001) 学习率大小
        :param weight_decay: (float, optional, default=1.0e-4) 权重衰减。l2惩罚
        :param batch_size: (int, optional, default=4) 批量样本大小
        :param shuffle: (bool, optional, default=True) 是否在每轮次训练时候洗牌
        :param validation_rate: (float, optional, default=0.2) 验证数据集比例
        :param grad_norm: (int, optional, default=5) 梯度阀值。防止梯度爆炸系数。
        :param factor: (float, optional, default=0.1) 学习率衰减因子。new_lr = lr * factor
        :param patience: (int, optional, default=2) 在指定的评估指标2次没有变化之后，修改学习率
        :param eps: (float, optional, default=1.0e-8) 应用于lr的最小衰减。如果新旧lr之间的差异小于eps，则忽略该更新
        :param checkpoint: (str, optional, default='NER_BiLSTM_CRF.pth') 保存检查点的文件
        :return:
        """

        dataloader = DataLoader(x, y, input_length, self.embedding_dim, batch_size, validation_rate=validation_rate,
                                shuffle=shuffle)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, verbose=True,
                                                                  patience=patience, eps=eps)

        print("start fit ...")
        for e in range(epochs):
            train_dataset, valid_dataset, train_size, valid_size = dataloader.take()
            self.train_loss.reset()
            self.valid_loss.reset()
            self.model.train()
            for step, (train_x, train_y, sequence, mask) in enumerate(train_dataset):
                train_x = torch.tensor(train_x, dtype=torch.long).to(self.device)
                train_y = torch.tensor(train_y, dtype=torch.long).to(self.device)
                sequence = torch.tensor(sequence, dtype=torch.long).to(self.device)
                mask = torch.cat(mask, dim=0).to(self.device)
                optimizer.zero_grad()
                loss = self.model.forward_loss(train_x, mask, sequence, train_y)
                self.train_loss.update(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)
                optimizer.step()

                output = template(e + 1, epochs, step + 1, train_size, self.train_loss.result(), head='Train')
                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            print()

            self.model.eval()
            with torch.no_grad():
                for step, (valid_x, valid_y, sequence, mask) in enumerate(valid_dataset):
                    valid_x = torch.tensor(valid_x, dtype=torch.long, device=self.device)
                    valid_y = torch.tensor(valid_y, dtype=torch.long, device=self.device)
                    sequence = torch.tensor(sequence, dtype=torch.long, device=self.device)
                    mask = torch.cat(mask, dim=0).to(self.device)
                    loss = self.model.forward_loss(valid_x, mask, sequence, valid_y)
                    self.valid_loss.update(loss.item())

                    output = template(e + 1, epochs, step + 1, valid_size, self.valid_loss.result(), head='Valid')
                    sys.stdout.write('\r' + output)
                    sys.stdout.flush()
                print()

            self.save_pth(checkpoint)
            lr_scheduler.step(self.valid_loss.result())
            print()

    def predict(self, x):
        """
        预测
        :param x: (list, mandatory) 句子列表。如：[1,2,3,4]
        :return: (list) 预测标识符token
        """
        with torch.no_grad():
            inputs_x = torch.tensor(x, dtype=torch.long).view(1, -1).to(self.device)
            features = self.model.forward(inputs_x)
            features = features.squeeze(0)
            output = self.model.crf.viterbi_decode(features)
        return output

    def save_pth(self, pth="BiLSTM_CRF.pth"):
        """
        保存模型策略。如果 train_loss 和 valid_loss 都小于 self.train_minimum_loss 和 self.valid_minimum_loss 则保存模型
        :param pth: (str, optional, default='BiLSTM_CRF.pth') 保存模型文件名
        :return:
        """
        is_save = False
        if self.train_minimum_loss is None and self.valid_minimum_loss is None:
            self.train_minimum_loss = self.train_loss.result()
            self.valid_minimum_loss = self.valid_loss.result()
            is_save = True

        else:
            if self.train_loss.result() < self.train_minimum_loss and \
                    self.valid_loss.result() < self.valid_minimum_loss:
                self.train_minimum_loss = self.train_loss.result()
                self.valid_minimum_loss = self.valid_loss.result()
                is_save = True

        if is_save:
            self.save_checkpoint(pth)

    def save_checkpoint(self, file="BiLSTM_CRF_NER.pth"):
        """
        保存模型。
        :param file: (str, optional, default="ner_bilstm_crf.pth") 模型文件名
        :return:
        """
        Writer.check_path(file)
        print("remove file: {} {}".format(file, Writer.remove_file(file)))

        torch.save(self.model.state_dict(), file)
        print("model save over! File: {}".format(file))

    def dump_config(self, file='BiLSTM_CRF_NER.json'):
        """
        保存配置信息
        :param file: (str, mandatory) 配置信息文件名
        :return:
        """
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'target_size': self.target_size,
            'hidden_size': self.hidden_size,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'dropout': self.dropout,
            'train_minimum_loss': self.train_minimum_loss,
            'valid_minimum_loss': self.valid_minimum_loss
        }

        Writer.check_path(file)
        with open(file, 'w') as f:
            json.dump(config, f, indent=4)

        print("Dump config over! File:{}".format(file))

    @staticmethod
    def restore(config, checkpoint):
        """
        还原模型
        :param config: (str, mandatory) 配置信息文件
        :param checkpoint: （str, mandatory) 检查点文件
        :return:
        """
        assert os.path.isfile(config), FileNotFoundException(config)
        assert os.path.isfile(checkpoint), FileNotFoundException(checkpoint)

        with open(config, 'r') as f:
            conf = json.load(f)

        ner = NameEntityRecognitionModel(vocab_size=conf['vocab_size'],
                                         embedding_dim=conf['embedding_dim'],
                                         target_size=conf['target_size'],
                                         hidden_size=conf['hidden_size'],
                                         start_index=conf['start_index'],
                                         end_index=conf['end_index'],
                                         dropout=conf['dropout'])
        ner.train_minimum_loss = conf['train_minimum_loss']
        ner.valid_minimum_loss = conf['valid_minimum_loss']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint, map_location=device)
        ner.model.load_state_dict(state_dict)
        return ner


def test_module_func(operation):
    # if operation == 'forward_alg':
    #     target_size = 5
    #     start_index = 0
    #     end_index = 1
    #     crf = CRF(target_size, start_index, end_index)
    #     features = torch.randn(3, 4, 5)
    #     sequence = [2, 3, 3]
    #     output = crf.forward(features, sequence)
    #     print("Forward score: ", output)
    # elif operation == 'score_sentence':
    #     target_size = 5
    #     start_index = 0
    #     end_index = 1
    #     crf = CRF(target_size, start_index, end_index)
    #     features = torch.randn(3, 4, 5)
    #     sequence = [2, 3, 3]
    #     labels = torch.randint(2, 5, size=(3, 4))
    #     output = crf.forward(features, sequence, labels=labels)
    #     print("Score sentence: ", output)

    if operation == CRF.viterbi_decode.__name__:
        features = torch.randn(5, 5)
        crf = CRF(5, 0, 1)
        output = crf.viterbi_decode(features)
        print("viterbi decode: ", output)

    elif operation == BiLSTM_CRF.forward.__name__:
        vocab_size = 4
        embedding_dim = 6
        target_size = 5
        hidden_size = 8

        # 输入数据
        inputs = torch.randint(0, vocab_size, size=(2, 5))
        mask = torch.zeros((2, 5, embedding_dim))
        mask[0][:4] = 1
        mask[1][:3] = 1

        bilstm_crf = BiLSTM_CRF(vocab_size, embedding_dim, target_size, hidden_size, start_index=0, end_index=1)
        # optimizer = torch.optim.Adam(bilstm_crf.parameters(), lr=0.001)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        output = bilstm_crf.forward(inputs, mask)
        print("BiLSTM output: ", output)
        print("BiLSTM output shape: ", output.shape)

    elif operation == DataLoader.__name__:
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6]]
        y = [[11, 22, 33], [44, 55, 66], [77, 88, 99], [11, 33, 55], [22, 44, 66]]
        input_length = [2, 3, 2, 3, 2]
        embedding_dim = 4
        dataloader = DataLoader(x, y, input_length, embedding_dim, batch_size=2)
        train_dataset, valid_dataset, train_sample_size, valid_sample_size = dataloader.take()

        print("train sample size: ", train_sample_size)
        print("valid sample size: ", valid_sample_size)

        for train_x, train_y, seq_len, mask in train_dataset:
            print("x: ", train_x)
            print("y: ", train_y)
            print("seq_len: ", seq_len)
            print("mask: ", mask)
            print("mask[0]: ", mask[0])
            print()

    elif operation == NameEntityRecognitionModel.fit.__name__:
        import numpy as np
        vocab_size = 10
        embedding_dim = 8
        target_size = 6

        x = np.random.randint(0, vocab_size, size=(100, 10)).tolist()
        y = np.random.randint(2, target_size, size=(100, 10)).tolist()
        input_length = np.random.randint(7, 10, size=100).tolist()

        ner = NameEntityRecognitionModel(vocab_size, embedding_dim, target_size, hidden_size=8, start_index=0,
                                         end_index=1)
        ner.fit(x, y, input_length, checkpoint='./model/NER_BiLSTM_CRF.pth')

    elif operation == NameEntityRecognitionModel.predict.__name__:
        import numpy as np
        vocab_size = 10
        embedding_dim = 8
        target_size = 6

        x = np.random.randint(0, vocab_size, size=(1, 10)).tolist()
        ner = NameEntityRecognitionModel(vocab_size, embedding_dim, target_size, hidden_size=8, start_index=0,
                                         end_index=1)
        output = ner.predict(x)
        print("predict: ", output)

    elif operation == NameEntityRecognitionModel.restore.__name__:
        import numpy as np
        vocab_size = 10
        embedding_dim = 8
        target_size = 6
        pth = './model/NER_BiLSTM_CRF.pth'
        config = './model/NER_BiLSTM_CRF.json'

        x = np.random.randint(0, vocab_size, size=(100, 10)).tolist()
        y = np.random.randint(2, target_size, size=(100, 10)).tolist()
        input_length = np.random.randint(7, 10, size=100).tolist()

        ner = NameEntityRecognitionModel(vocab_size, embedding_dim, target_size, hidden_size=8, start_index=0,
                                         end_index=1)
        ner.fit(x, y, input_length, checkpoint=pth, epochs=30)
        ner.dump_config(config)
        print()

        new_ner = NameEntityRecognitionModel.restore(config, pth)
        new_ner.fit(x, y, input_length, checkpoint=pth, epochs=2)

        x = np.random.randint(0, vocab_size, size=(1, 10)).tolist()
        predict = new_ner.predict(x)
        print("predict: ", predict)


if __name__ == '__main__':
    # test_module_func('forward_alg')
    # test_module_func('score_sentence')
    # test_module_func(CRF.viterbi_decode.__name__)
    # test_module_func(BiLSTM_CRF.forward.__name__)
    # test_module_func(DataLoader.__name__)
    # test_module_func(NameEntityRecognitionModel.fit.__name__)
    # test_module_func(NameEntityRecognitionModel.predict.__name__)
    test_module_func(NameEntityRecognitionModel.restore.__name__)
