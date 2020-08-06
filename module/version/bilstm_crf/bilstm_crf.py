# -*- encoding: utf-8 -*-
"""
@file: bilstm_crf.py
@time: 2020/7/15 下午2:59
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from module.core.utils import Writer
from module.core.color import Color
from module.core.exception import ParameterError
from module.core.exception import FileNotFoundException


def template(e, epochs, step, steps, loss, head='Train'):
    """ 训练输出模版 """
    output = "{}: Epochs: {}/{}, Step: {}/{}, Loss:{:.4f}, ".format(head, e, epochs, step, steps, loss)
    return output


def packaging_dataset(x, y, input_length, batch_size=2, shuffle=True):
    """
    包装数据集。将 x 和 y 包装成数据集
    :param x: (list, mandatory) 训练数据集
    :param y: (list, mandatory) 验证数据集
    :param input_length: (int, mandatory) 序列长度
    :param batch_size: (int, optional, default=1) 批次样本大小
    :param shuffle: (bool, optional, default=True) 是否进行洗牌
    :return: (tf.data.Dataset) 数据集
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y, input_length))
    if shuffle:
        dataset = dataset.shuffle(len(x) // 2).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    return dataset


def get_sample_size(dataset, validation_split=0.1):
    """
    获取训练数据集和验证数据集样本比例大小
    :param dataset: (tf.data.Dataset, mandatory) 数据集
    :param validation_split: (float, optional, default=0.1) 验证数据集比例
    :return: (int)  训练样本大小和验证样本大小
    """
    if validation_split > 1 or validation_split < 0:
        raise ParameterError("Validation rate range: 0~1, actually get:{}".format(validation_split))

    total_size = len(list(dataset))
    valid_sample_size = int(total_size * validation_split)
    valid_sample_size = valid_sample_size if valid_sample_size > 0 else 1

    train_sample_size = total_size - valid_sample_size

    return train_sample_size, valid_sample_size


def log_sum_exp(score_tensor):
    """
    log + sum + exp 运算
    :param score_tensor: (Tensor, mandatory) 二维张量
    :return: (Tensor) 张量
    """
    # 归一化. 主要是使用每行的值向量 减去 每行的对应的最大值。
    # 这样得到的元素将小于等于 0。 使用 exp 进行运算的范围则在 [0 ~ 1] 之间
    max_score = tf.reshape(tf.reduce_max(score_tensor, axis=1), shape=[1, -1])
    score = score_tensor - tf.repeat(max_score, repeats=score_tensor.shape[0], axis=0)

    return max_score + tf.math.log(tf.reduce_sum(tf.math.exp(score), axis=1))


def validation_sample_size(dataset_size, validation_rate):
    """
    根据数据比例，获取验证样本大小
    :param dataset_size:
    :param validation_rate:
    :return:
    """
    sample_size = int(dataset_size * validation_rate)
    if sample_size == 0:
        sample_size = 1
    return sample_size


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


def _split_valid_dataset(*args, shuffle=True, validation_rate=0.2):
    """
    划分验证数据集。
    :param args: (list, mandatory) 数据集。列表格式。
    :param validation_rate: (int, optional, default=0.2) 验证数据集比例。0～1范围内。
    :return: (list) 训练数据集和验证数据集
    """
    if validation_rate > 1 or validation_rate < 0:
        raise ParameterError("dataset ratio must be is 0 ~ 1 range. actually get: {}".format(validation_rate))

    dataset = args
    if shuffle:
        dataset = _shuffle(*args)

    sample_num = validation_sample_size(len(dataset[0]), validation_rate)

    train_dataset, valid_dataset = list(), list()
    for data in dataset:
        valid_dataset.append(data[:sample_num])
        train_dataset.append(data[sample_num:])

    return train_dataset, valid_dataset


class Dataset(object):
    """ 预处理输入数据集 """

    def __init__(self, input_x, input_y, shuffle=True, validation_rate=None):
        """
        初始化。
        :param input_x: (list, mandatory) 训练数据
        :param input_y: (list, mandatory) 标签数据
        :param shuffle: (bool, optional, default=True) 是否进行洗牌
        :param validation_rate: (float, optional, default=0.1) 划分验证数据集比例
        """
        assert len(input_x) == len(input_y), ParameterError(
            "Different data size sample. x:{}, y:{}".format(len(input_x), len(input_y)))

        self.input_x = input_x
        self.input_y = input_y
        self.shuffle = shuffle
        self.validation_rate = validation_rate

        self.__init_size__()
        self.__convert_to_tensor__()

    def __init_size__(self):
        self.size = len(self.input_x)
        if self.validation_rate is None:
            self.valid_size = 0
            self.train_size = self.size
        else:
            self.valid_size = validation_sample_size(self.size, self.validation_rate)
            self.train_size = self.size - self.valid_size

    def __convert_to_tensor__(self):
        self.x, self.y = [], []
        for i_x, i_y in zip(self.input_x, self.input_y):
            self.x.append(tf.reshape(tf.convert_to_tensor(i_x, dtype=tf.int32), shape=(1, -1)))
            self.y.append(tf.reshape(tf.convert_to_tensor(i_y, dtype=tf.int32), shape=(1, -1)))

    def take(self):
        """ 获取训练数据和验证数据 """
        if self.validation_rate is None:
            if self.shuffle:
                train_dataset = _shuffle(self.x, self.y)
                train_dataset = zip(train_dataset[0], train_dataset[1])
            else:
                train_dataset = zip(self.x, self.y)
            valid_dataset = None
        else:
            train_dataset, valid_dataset = _split_valid_dataset(self.x, self.y,
                                                                shuffle=self.shuffle,
                                                                validation_rate=self.validation_rate)
            train_dataset = zip(train_dataset[0], train_dataset[1])
            valid_dataset = zip(valid_dataset[0], valid_dataset[1])
        return train_dataset, valid_dataset


class CRF(tf.keras.layers.Layer):
    """ CRF 模型 """

    def __init__(self, target_size, start_index, end_index, name='CRF'):
        """
        初始化
        :param target_size: (int, mandatory) 转移矩阵目标大小
        :param start_index: (int, mandatory) 序列开始索引
        :param end_index: (int, mandatory) 序列结束索引
        :param name: (str, optional, default='CRF')
        """
        super(CRF, self).__init__(name=name)
        self.target_size = target_size
        self.start = start_index
        self.end = end_index

        self.minimum = 1.0e-4

        self.__init_transition__()

    def __init_transition__(self):
        """
        初始化转移矩阵
        1. 定义随机均匀分布矩阵 transition_matrix。shape: (target_size, target_size)
        2. 定义 start 的列: shape:(self.target_size, self.target_size).
            example: start = 0
                [[0, 1, 1],
                 [0, 1, 1],
                 [0, 1, 1]]
            transition_matrix * start 则对应的 start 列为 0
        3. 定义 end 的行：shape: (self.target_size, self.target_size)
            example: end = 1
                [[1, 1, 1],
                 [0, 0, 0],
                 [1, 1, 1]]
            transition_matrix * end 则对应的 end 行为 0

        4. 定义 start 和 end 的值为最小值。
            使用 start * end , 同时转换成 bool 类型，再使用 logical_not 运算。之后再转换成 float 类型，则可得到如下矩阵 value ：
                [[1, 0, 0],
                [1, 1, 1],
                [1, 0, 0]]
            再将该矩阵乘最小值 value = value * minimum
        5. 转换矩阵
            transition = transition_matrix * start * end + value
        :return:
        """
        # 使用均匀分布随机初始化
        transition_matrix = tf.random.uniform((self.target_size, self.target_size))

        # 定义 start 的列: shape:(self.target_size, self.target_size).
        start = tf.one_hot([self.start] * self.target_size, self.target_size)
        start = tf.ones((self.target_size, self.target_size)) - start

        # 定义 end 的行：shape: (self.target_size, self.target_size)
        end = tf.transpose(tf.one_hot([self.end] * self.target_size, self.target_size), perm=(1, 0))
        end = tf.ones((self.target_size, self.target_size)) - end

        # 定义 start 和 end 的值为最小值。
        value = start * end
        value = tf.cast(tf.logical_not(tf.cast(value, dtype=tf.bool)), dtype=tf.float32) * self.minimum

        # transition * start * end 之后则对应的行和列为 0 。同时在加上 value，则将最 start 和 end 初始化为最小值
        transition_matrix = transition_matrix * start * end + value

        self.transition = tf.Variable(transition_matrix)
        # print(self.transition)

    def call(self, inputs, training=None, mask=None):
        if training:
            return self.forward_score(inputs)
        else:
            return self.viterbi_decode(inputs)

    def sentence_score(self, features, labels):
        """
        标签评分
        :param features: (Tensor, mandatory) 特征向量 shape: [seq_len, target_size]
        :param labels: (Tensor, mandatory) 标签向量 shape: [seq_len]
        :return:(Tensor) 评分
        """
        # 加上 <START>
        labels = tf.squeeze(labels)
        start = tf.constant([self.start])
        tags = tf.concat([start, labels], axis=0)

        score = tf.zeros(1)
        i = 0
        for feat in features:
            # 发射评分
            emit_score = feat[tags[i + 1]]

            # 转移评分。发射节点到目标节点的评分。
            transition_score = self.transition[tags[i + 1]][tags[i]]

            # 发射评分 + 转移评分
            score = score + emit_score + transition_score

            i += 1

        # 加上 <END>
        score = score + self.transition[self.end][tags[-1]]
        return score

    def forward_score(self, features):
        """
        前向评分
        :param features: (Tensor, mandatory) 特征。shape: [seq_len, self.target_size]
        :return:(Tensor) 评分张量。shape: [1, 1]
        """
        # 定义初始化变量。shape: [1, self.target_size], 将 start 所在的值初始化为 0
        init_var = tf.fill((1, self.target_size), self.minimum)
        start = tf.ones([1, self.target_size]) - tf.one_hot([self.start], self.target_size)
        init_var = init_var * start

        forward_var = init_var
        for feat in features:
            # 转换shape 为 [1, self.target_size].
            emit_score = tf.reshape(feat, shape=[1, -1])
            # 使用 repeat 方式扩展元素。shape:[self.target_size, self.target_size].
            # 同时进行矩阵转换。
            # 例如如下格式：
            # [[1, 1, 1],
            #  [2, 2, 2],
            #  [3, 3, 3]]
            emit_score = tf.transpose(tf.repeat(emit_score, repeats=self.target_size, axis=0), perm=[1, 0])

            # 前向评分 + 转移评分 + 发射评分
            score = forward_var + self.transition + emit_score
            forward_var = log_sum_exp(score)

        score = forward_var + tf.reshape(self.transition[self.end], shape=[1, -1])
        score = tf.reshape(log_sum_exp(score), shape=[-1])
        return score

    def viterbi_decode(self, features):
        """
        维特比解码路径
        :param features: (Tensor, mandatory) 特征向量
        :return: (list) 解码路径
        """
        # 定义初始化变量。shape: [1, self.target_size], 将 start 所在的值初始化为 0
        init_var = tf.fill((1, self.target_size), self.minimum)
        start = tf.ones([1, self.target_size]) - tf.one_hot([self.start], self.target_size)
        init_var = init_var * start

        best_index = []

        forward_var = init_var
        for feature in features:
            # 转移评分
            score = self.transition + forward_var

            # 保存路径
            best_index.append(tf.argmax(score, axis=1))

            # 转移评分 + 发射评分
            max_score = tf.reshape(tf.reduce_max(score, axis=1), shape=(1, -1))
            forward_var = max_score + tf.reshape(feature, shape=(1, -1))

        # 计算由最后一个字符到 <END> 转移概率。
        score = forward_var + tf.reshape(self.transition[self.end], shape=(1, -1))
        # 保存概率最高的一条。选择概率最大的一条进行回溯
        best_path_id = tf.argmax(score, axis=1)
        best_path_id = best_path_id.numpy()[0]

        # 回朔。由后往前依次获取最好的路径。
        best_path = [best_path_id]
        for b_index in reversed(best_index):
            best_path.append(b_index[best_path_id].numpy())

        # 调整逆向位置。例如将 "C、B、A" 调整为 "A、B、C"
        best_path.reverse()
        # 删除<START> 标识符
        best_path = best_path[1:]

        return best_path

    def log_likelihood_loss(self, features, label, forward_score):
        """
        对数似然损失
        :param features: (Tensor, mandatory) 特征向量。shape：[seq_len, target_size]
        :param label: (Tensor, mandatory) 标签向量。shape: [seq_len]
        :param forward_score: (Tensor, mandatory) 前向评分。shape: [1]
        :return: (Tensor) loss
        """
        # forward_score = self.forward_score(features)
        target_score = self.sentence_score(features, label)
        return tf.squeeze(forward_score - target_score)


class BiLSTM(tf.keras.layers.Layer):
    """ BiLSTM 模型 """

    def __init__(self, embed_size, vocab_size, lstm_units, target_size, l1=1.0e-4, l2=1.0e-4, merge_mode='sum',
                 name='BiLSTM'):
        """
        初始化
        :param embed_size: (int, mandatory) 输入维度
        :param vocab_size: (int, mandatory) 词汇量大小
        :param lstm_units: (int, mandatory) LSTM 隐藏单元数量
        :param target_size: (int, mandatory) 目标输出大小
        :param l1: (float, optional, default=1.0e-4) l1 正则化系数
        :param l2: (float, optional, default=1.0e-4) l2 正则化系数
        :param merge_mode: (str, optional, default='sum') BiLSTM 输出序列操作模式。sum 则将两个序列相加，concat 则拼接序列
        :param name: (str, optional, default='BiLSTM_CRF') 名称
        """
        super(BiLSTM, self).__init__(name=name)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.target_size = target_size
        self.l1 = l1
        self.l2 = l2
        self.merge_mode = merge_mode

        self.__init_layer__()

    def __init_layer__(self):
        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.embed_size, name='Embedding')
        lstm = tf.keras.layers.LSTM(self.lstm_units,
                                    kernel_regularizer=tf.keras.regularizers.l1(self.l1),
                                    activity_regularizer=tf.keras.regularizers.l2(self.l2),
                                    return_sequences=True, name='LSTM')
        self.bi_lstm = tf.keras.layers.Bidirectional(lstm, merge_mode=self.merge_mode, name='Bi_LSTM')
        self.dn = tf.keras.layers.Dense(self.target_size, use_bias=False, name='Dense')

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        x = self.bi_lstm(x)
        x = tf.squeeze(x, axis=0)
        x = self.dn(x)

        return x


class BiLSTM_CRF(tf.keras.Model):
    """ BiLSTM_CRF 模型 """

    def __init__(self, embed_size, vocab_size, target_size, lstm_units, start_index, end_index,
                 l1=1.0e-4,
                 l2=1.0e-4,
                 merge_mode='sum',
                 name='BiLSTM_CRF'):
        """
        初始化
        :param embed_size: (int, mandatory) 输入维度
        :param vocab_size: (int, mandatory) 词汇量大小
        :param lstm_units: (int, mandatory) LSTM 隐藏单元数量
        :param start_index: (int, mandatory) 序列开始索引
        :param end_index: (int, mandatory) 序列结束索引
        :param target_size: (int, mandatory) 目标输出大小
        :param l1: (float, optional, default=1.0e-4) l1 正则化系数
        :param l2: (float, optional, default=1.0e-4) l2 正则化系数
        :param merge_mode: (str, optional, default='sum') BiLSTM 输出序列操作模式。sum 则将两个序列相加，concat 则拼接序列
        :param name: (str, optional, default='BiLSTM_CRF') 名称
        """
        super(BiLSTM_CRF, self).__init__(name=name)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.lstm_units = lstm_units
        self.start_index = start_index
        self.end_index = end_index
        self.l1 = l1
        self.l2 = l2
        self.merge_mode = merge_mode

        self.__init_layer__()

    def __init_layer__(self):
        self.bi_lstm = BiLSTM(embed_size=self.embed_size,
                              vocab_size=self.vocab_size,
                              lstm_units=self.lstm_units,
                              target_size=self.target_size,
                              l1=self.l1,
                              l2=self.l2,
                              merge_mode=self.merge_mode)
        self.crf = CRF(target_size=self.target_size, start_index=self.start_index, end_index=self.end_index)

    def call(self, inputs, training=None, mask=None):
        feature = self.bi_lstm(inputs)
        forward_score = self.crf(feature, training=training)

        return forward_score, feature


class NER_BiLSTM_CRF(object):

    def __init__(self, embed_size, vocab_size, target_size, lstm_units, start_index, end_index,
                 l1=1.0e-4,
                 l2=1.0e-4,
                 merge_mode='sum',
                 learning_rate=0.01,
                 momentum=0.0,
                 decay=1.0e-5,
                 use_polynomial_decay=False,
                 polynomial_decay_steps=9000,
                 polynomial_end_learning_rate=0.0001,
                 polynomial_power=0.5,
                 checkpoint_dir='./NER_BiLSTM_CRF/ckpt',
                 max_to_keep=3):
        """
        初始化
        :param embed_size: (int, mandatory) 词嵌入大小
        :param vocab_size: (int, mandatory) 输入词汇量的大小
        :param target_size: (int, mandatory) 目标输出词汇量的大小
        :param lstm_units: (int, mandatory) LSTM 隐藏层大小
        :param start_index: (int, mandatory) 序列开始<START>索引
        :param end_index: (int, mandatory) 序列结束 <END> 索引
        :param l1: (float, optional, default=1.0e-4) L1 正则化系数
        :param l2: (float, optional, default=1.0e-4) L2 正则化系数
        :param merge_mode: (str, optional, default='sum') BiLSTM 输出序列操作模式。sum 则将两个序列相加，concat 则拼接序列
        :param learning_rate: (float, optional, default=0.01) 学习率
        :param momentum: (float, optional, default=0.0) 动量
        :param decay: (float, optional, default=1.0e-5) 衰减率
        :param use_polynomial_decay: (bool, optional, default=False) 是否使用 polynomial_decay 衰减
        :param polynomial_decay_steps: (int, optional, default=9000) PolynomialDecay 的衰减 steps。
            参考：tf.keras.optimizers.schedules.PolynomialDecay
        :param polynomial_end_learning_rate: (int, optional, default=0.0001) 衰减率。
            参考：tf.keras.optimizers.schedules.PolynomialDecay
        :param polynomial_power: (float, optional, default=0.5) 多项式的幂。
            参考：tf.keras.optimizers.schedules.PolynomialDecay
        :param checkpoint_dir: (str, optional, default='./NER_BiLSTM_CRF/ckpt') 检查点保存目录
        :param max_to_keep: (int, optional, default=3) 保存检查点的数量
        """
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.lstm_units = lstm_units
        self.start_index = start_index
        self.end_index = end_index
        self.l1 = l1
        self.l2 = l2
        self.merge_mode = merge_mode
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.use_polynomial_decay = use_polynomial_decay
        self.polynomial_decay_steps = polynomial_decay_steps
        self.polynomial_end_learning_rate = polynomial_end_learning_rate
        self.polynomial_power = polynomial_power
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

        self.__init_model__()
        self.__init_optimizer__()
        self.__init_checkpoint__()
        self.__init_metrics__()

        # 保存每轮次训练的训练最小值和验证最小值
        self.train_minimum_loss = None
        self.valid_minimum_loss = None

    def __init_model__(self):
        """ 初始化模型 """
        self.bi_lstm_crf = BiLSTM_CRF(embed_size=self.embed_size,
                                      vocab_size=self.vocab_size,
                                      target_size=self.target_size,
                                      lstm_units=self.lstm_units,
                                      start_index=self.start_index,
                                      end_index=self.end_index,
                                      l1=self.l1,
                                      l2=self.l2,
                                      merge_mode=self.merge_mode)

        data = tf.random.uniform((1, 16), minval=0, maxval=self.vocab_size, dtype=tf.int32)
        self.bi_lstm_crf(data, training=True)

    def __init_optimizer__(self):
        """ 初始化Adam """
        if self.use_polynomial_decay:
            lr = tf.keras.optimizers.schedules.PolynomialDecay(self.learning_rate, self.polynomial_decay_steps,
                                                               end_learning_rate=self.polynomial_end_learning_rate,
                                                               power=self.polynomial_decay_steps)
        else:
            lr = self.learning_rate
        self.optimizer = tf.keras.optimizers.SGD(lr, momentum=self.momentum, decay=self.decay)

    def __init_checkpoint__(self):
        """ 初始化检查点 """
        self.checkpoint = tf.train.Checkpoint(model=self.bi_lstm_crf, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,
                                                             max_to_keep=self.max_to_keep)

    def __init_metrics__(self):
        """ 初始化指标 """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def summary(self):
        """ 返回模型简介信息 """
        return self.bi_lstm_crf.summary()

    def save_checkpoint(self):
        """ 保存检查点 """
        self.checkpoint_manager.save()
        print("save checkpoint: {}".format(self.checkpoint_manager.latest_checkpoint))

    def train_save_checkpoint(self):
        """
        在训练过程中保存检查点。
        当 train_loss.result < train_minimum_loss and valid_loss.result < valid_minimum_loss 保存，
        同时更新 train_minimum_loss 和 valid_minimum_loss
        """
        is_save = False
        if self.train_minimum_loss is None and self.valid_minimum_loss is None:
            self.train_minimum_loss = self.train_loss.result()
            self.valid_minimum_loss = self.valid_loss.result()
            is_save = True

        else:
            if self.train_minimum_loss > self.train_loss.result() \
                    and self.valid_minimum_loss > self.valid_loss.result():
                self.train_minimum_loss = self.train_loss.result()
                self.valid_minimum_loss = self.valid_loss.result()
                is_save = True

        if is_save:
            self.save_checkpoint()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        """ 训练 """
        with tf.GradientTape() as tape:
            forward_score, features = self.bi_lstm_crf(x, training=True)
            likelihood_loss = self.bi_lstm_crf.crf.log_likelihood_loss(features, y, forward_score)
            regularizer_loss = tf.add_n(self.bi_lstm_crf.losses)
            loss = likelihood_loss + regularizer_loss

        gradient = tape.gradient(loss, self.bi_lstm_crf.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.bi_lstm_crf.trainable_variables))

        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def valid_step(self, x, y):
        """ 验证 """
        forward_score, feature = self.bi_lstm_crf(x, training=True)
        likelihood_loss = self.bi_lstm_crf.crf.log_likelihood_loss(feature, y, forward_score)

        self.valid_loss(likelihood_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, x, y):
        """ 测试 """
        forward_score, feature = self.bi_lstm_crf(x, training=True)
        likelihood_loss = self.bi_lstm_crf.crf.log_likelihood_loss(feature, y, forward_score)

        self.test_loss(likelihood_loss)

    def fit(self, x, y, epochs=10, shuffle=True, validation_split=0.1):
        """
        拟合
        :param x: (tuple, mandatory) 训练数据。
        :param y: (tuple, mandatory) 标签数据。
        :param epochs: (int, optional, default=10) 训练轮次
        :param shuffle: (bool, optional, default=True) 是否在没轮次结束之后进行洗牌
        :param validation_split: (float, optional, default=0.1) 划分验证数据集比例
        :return:
        """
        # dataset = packaging_dataset(x, y, batch_size=1, shuffle=shuffle)
        # train_sample_size, valid_sample_size = get_sample_size(dataset, validation_split)

        dataset = Dataset(x, y, shuffle=shuffle, validation_rate=validation_split)

        for e in range(epochs):
            train_dataset, valid_dataset = dataset.take()

            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            for step, (train_x, train_y) in enumerate(train_dataset):
                self.train_step(train_x, train_y)

                output = template(e + 1, epochs, step + 1, dataset.train_size, self.train_loss.result(), head='Train')
                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            print()

            for step, (valid_x, valid_y) in enumerate(valid_dataset):
                self.valid_step(valid_x, valid_y)

                output = template(e + 1, epochs, step + 1, dataset.valid_size, self.valid_loss.result(), head='Valid')
                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            print()

            self.train_save_checkpoint()
            print()

    def eval(self, x, y):
        """
        评估
        :param x: (tuple, mandatory) 训练数据。
        :param y: (tuple, mandatory) 标签数据。
        :return:
        """
        self.test_loss.reset_states()

        # dataset = packaging_dataset(x, y, input_length, batch_size=1, shuffle=False)
        dataset = Dataset(x, y, shuffle=False, validation_rate=None)
        eval_dataset, _ = dataset.take()
        for step, (test_x, test_y) in enumerate(eval_dataset):
            self.test_step(test_x, test_y)

            output = template(1, 1, step + 1, dataset.size, self.test_loss.result(), head='Test')
            sys.stdout.write('\r' + output)
            sys.stdout.flush()
        print()

    def predict(self, x):
        """
        预测
        :param x: (list, mandatory) 预测数据
        :return: (list) 预测结果
        """
        x = tf.reshape(tf.constant(x, dtype=tf.int32), shape=[1, -1])
        output, _ = self.bi_lstm_crf(x, training=False)

        return output

    def dump_config(self, file='BiLSTM_CRF.json'):
        """
        备份模型配置信息
        :param file: (str, optional, default='BiLSTM_CRF.json') 配置文件名
        :return:
        """
        config = {
            'class': NER_BiLSTM_CRF.__name__,
            'embed_size': self.embed_size,
            'vocab_size': self.vocab_size,
            'target_size': self.target_size,
            'lstm_units': self.lstm_units,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'l1': self.l1,
            'l2': self.l2,
            'merge_mode': self.merge_mode,
            'learning_rate': self.learning_rate,
            'polynomial_decay_steps': self.polynomial_decay_steps,
            'polynomial_end_learning_rate': self.polynomial_end_learning_rate,
            'polynomial_power': self.polynomial_power,
            'momentum': self.momentum,
            'decay': self.decay,
            'use_polynomial_decay': self.use_polynomial_decay,
            'checkpoint_dir': self.checkpoint_dir,
            'max_to_keep': self.max_to_keep
        }

        Writer.check_path(file)

        with open(file, 'w') as f:
            json.dump(config, f, indent=4)

        print("Dump config file: {}".format(file))

    @staticmethod
    def from_config(file):
        """
        通过配置文件还原模型
        :param file: (str, mandatory) 配置文件名
        :return: (NER_BiLSTM_CRF) 模型对象
        """
        assert os.path.isfile(file), FileNotFoundException(file)

        with open(file, 'r') as f:
            data = json.load(f)

        assert data['class'] == NER_BiLSTM_CRF.__name__, ParameterError(
            "config:{} class:{} expect get:{}".format(file, data['class'], NER_BiLSTM_CRF.__name__))

        model = NER_BiLSTM_CRF(embed_size=data['embed_size'],
                               vocab_size=data['vocab_size'],
                               target_size=data['target_size'],
                               lstm_units=data['lstm_units'],
                               start_index=data['start_index'],
                               end_index=data['end_index'],
                               l1=data['l1'],
                               l2=data['l2'],
                               merge_mode=data['merge_mode'],
                               learning_rate=data['learning_rate'],
                               polynomial_decay_steps=data['polynomial_decay_steps'],
                               polynomial_end_learning_rate=data['polynomial_end_learning_rate'],
                               polynomial_power=data['polynomial_power'],
                               momentum=data['momentum'],
                               decay=data['decay'],
                               use_polynomial_decay=data['use_polynomial_decay'],
                               checkpoint_dir=data['checkpoint_dir'],
                               max_to_keep=data['max_to_keep'])
        print(Color.green("restore NER_BiLSTM_CRF from: {}".format(file)))
        return model

    @staticmethod
    def restore(config, checkpoint):
        """
        通过配置文件和检查点还原网络模型
        :param config: (str, mandatory) 模型配置文件
        :param checkpoint: (str, mandatory) 检查点目录
        :return:
        """
        model = NER_BiLSTM_CRF.from_config(config)
        if tf.train.latest_checkpoint(checkpoint):
            model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
            output = "restore checkpoint from: {}".format(tf.train.latest_checkpoint(checkpoint))
        else:
            output = "Not found checkpoint:{}, Initializing from scratch".format(checkpoint)

        print(Color.green(output))
        return model


def virtual_data_generator(size=100, seq_len=16, x_maxval=200, y_maxval=32):
    """ 虚拟数据集生成 """
    x = tf.random.uniform((size, seq_len), minval=0, maxval=x_maxval, dtype=tf.int32)
    y = tf.random.uniform((size, seq_len), minval=2, maxval=y_maxval, dtype=tf.int32)
    # input_length = tf.random.uniform((size, 1), minval=seq_len - 4, maxval=seq_len, dtype=tf.int32)

    return x, y


def test_module_func(operation):
    if operation == Dataset.__name__:
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 4, 7]]
        y = [[11, 22, 33], [44, 55, 66], [77, 88, 99], [22, 44, 77]]

        data = _shuffle(x, y)
        print("shuffle x: ", data[0])
        print("shuffle y: ", data[1])

        train_dataset, valid_dataset = _split_valid_dataset(x, y, shuffle=True)
        print("_split_valid_dataset, train: ", train_dataset)
        print("_split_valid_dataset, valid: ", valid_dataset)

        dataset = Dataset(x, y, shuffle=True, validation_rate=0.1)
        train_dataset, valid_dataset = dataset.take()
        for i, (train_x, train_y) in enumerate(train_dataset):
            print("Input tensor x: ", train_x)
            print("Input tensor y: ", train_y)
        print("valid size: ", dataset.valid_size)
        print("train size: ", dataset.train_size)

        dataset = Dataset(x, y, shuffle=True, validation_rate=None)
        train_dataset, valid_dataset = dataset.take()
        for i, (train_x, train_y) in enumerate(train_dataset):
            print("validation_rate is None Input tensor x: ", train_x)
            print("validation_rate is None Input tensor y: ", train_y)

        print("valid size: ", dataset.valid_size)
        print("train size: ", dataset.train_size)

    elif operation == CRF.__name__:
        target_size = 16
        crf = CRF(target_size, start_index=0, end_index=1)
        features = tf.random.uniform((10, target_size))

        forward_score = crf.forward_score(features)
        print("forward score: ", forward_score)

        label = tf.random.uniform([10], minval=2, maxval=target_size, dtype=tf.int32)
        sentence_score = crf.sentence_score(features, label)
        print("sentence score: ", sentence_score)

        loss = crf.log_likelihood_loss(features, label, forward_score)
        print("loss: ", loss)

        path = crf.viterbi_decode(features)
        print("length:{} path:{} ".format(len(path), path))

    elif operation == BiLSTM.__name__:
        data = tf.random.uniform((1, 16), minval=0, maxval=100, dtype=tf.int32)

        target_size = 16
        bi_lstm = BiLSTM(embed_size=64, vocab_size=100, lstm_units=64, target_size=target_size)
        output = bi_lstm(data)
        print("BiLSTM output: ", output.shape)

    elif operation == BiLSTM_CRF.__name__:
        data = tf.random.uniform((1, 16), minval=0, maxval=200, dtype=tf.int32)
        model = BiLSTM_CRF(embed_size=128, vocab_size=200, target_size=32, lstm_units=64, start_index=0, end_index=1)
        output = model(data, training=False)
        print("BiLSTM_CRF output: ", output)

    elif operation == NER_BiLSTM_CRF.__name__:
        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=200, target_size=32, lstm_units=64, start_index=0,
                               end_index=1)
        print(model.summary())
        print(model.bi_lstm_crf.losses)
        print(tf.add_n(model.bi_lstm_crf.losses))

    elif operation == NER_BiLSTM_CRF.fit.__name__:
        x_vocab_size = 200
        y_vocab_size = 32

        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=x_vocab_size, target_size=y_vocab_size,
                               lstm_units=64, start_index=0, end_index=1)
        print(model.summary())

        x, y = virtual_data_generator(100, 16, x_maxval=x_vocab_size, y_maxval=y_vocab_size)
        model.fit(x, y)

    elif operation == NER_BiLSTM_CRF.predict.__name__:
        x_vocab_size = 200
        y_vocab_size = 32
        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=x_vocab_size, target_size=y_vocab_size,
                               lstm_units=64, start_index=0, end_index=1)
        print(model.summary())

        data = tf.random.uniform((1, 16), minval=0, maxval=x_vocab_size, dtype=tf.int32)
        output = model.predict(data)
        print("Predict path: ", output)

    elif operation == NER_BiLSTM_CRF.eval.__name__:
        x_vocab_size = 200
        y_vocab_size = 32

        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=x_vocab_size, target_size=y_vocab_size,
                               lstm_units=64, start_index=0, end_index=1)
        print(model.summary())

        x, y = virtual_data_generator(100, 16, x_maxval=x_vocab_size, y_maxval=y_vocab_size)
        model.eval(x, y)

    elif operation == NER_BiLSTM_CRF.dump_config.__name__:
        x_vocab_size = 200
        y_vocab_size = 32
        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=x_vocab_size, target_size=y_vocab_size,
                               lstm_units=64, start_index=0, end_index=1)

        model.dump_config('./BiLSTM_CRF.json')

    elif operation == NER_BiLSTM_CRF.restore.__name__:
        x_vocab_size = 200
        y_vocab_size = 32
        config = './model/BiLSTM_CRF.json'
        checkpoint_dir = './model/ckpt'

        x, y = virtual_data_generator(100, 16, x_maxval=x_vocab_size, y_maxval=y_vocab_size)
        model = NER_BiLSTM_CRF(embed_size=128, vocab_size=x_vocab_size, target_size=y_vocab_size,
                               lstm_units=64, start_index=0, end_index=1, checkpoint_dir=checkpoint_dir)
        model.fit(x, y)
        model.dump_config(config)

        NER_BiLSTM_CRF.restore(config, checkpoint_dir)


if __name__ == '__main__':
    # test_module_func(Dataset.__name__)
    # test_module_func(BiLSTM.__name__)
    # test_module_func(CRF.__name__)
    # test_module_func(BiLSTM.__name__)
    # test_module_func(BiLSTM_CRF.__name__)
    # test_module_func(NER_BiLSTM_CRF.__name__)
    # test_module_func(NER_BiLSTM_CRF.fit.__name__)
    # test_module_func(NER_BiLSTM_CRF.predict.__name__)
    # test_module_func(NER_BiLSTM_CRF.eval.__name__)
    test_module_func(NER_BiLSTM_CRF.dump_config.__name__)
    test_module_func(NER_BiLSTM_CRF.restore.__name__)
