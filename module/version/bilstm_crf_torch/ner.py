# -*- encoding: utf-8 -*-
"""
@file: version.py
@time: 2020/7/15 下午3:06
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 训练：使用训练数据集进行训练
    example: python version.py train

# 评估：将使用 dev 数据集进行评估。输出损失
    example: python version.py eval

# 评分: 使用 dev 数据集进行预测评分
    example: python version.py score

# 测试：将使用测试数据集，将测试数据集识别结果写入文件
    example：python version.py test --file "predict_result.json"
            --file          指定写入的文件名
# 预测：
    example: python version.py prediction --sentence "《加勒比海盗3：世界尽头》的去年同期成绩死死甩在身后，后者则即将赶超《变形金刚》，"
            --sentence      指定预测的句子
"""
import re
import sys
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

try:
    from module.core.utils import DataUtils
    from module.core.utils import Writer
    from module.version.bilstm_crf_torch.word_dict import WordDict
    from module.core.color import Color
    from module.version.bilstm_crf_torch.bilstm_crf import BiLSTM_CRF
    from module.core.utils import Config
    from module.core.exception import UnknownError
except ModuleNotFoundError:
    sys.path.append("../../../")
    from module.core.utils import DataUtils
    from module.core.utils import Writer
    from module.version.bilstm_crf_torch.word_dict import WordDict
    from module.core.color import Color
    from module.version.bilstm_crf_torch.bilstm_crf import BiLSTM_CRF
    from module.core.utils import Config
    from module.core.exception import UnknownError

# 定义语料标识符
identifier_b, identifier_i, identifier_o, identifier_e, identifier_s = "B", "I", "O", "E", "S"

# 训练模式
M_TRAIN = 'train'

# 测试模式。将使用测试数据集，将测试数据集识别结果写入文件
M_TEST = 'test'

# 评估模式。将使用 dev 数据集进行评估。输出损失
M_EVAL = 'eval'

# 预测模式。输入句子，返回对应的实体识别结果
M_PREDICTION = 'prediction'

M_SCORE = 'score'

MODE = [M_TRAIN, M_TEST, M_EVAL, M_PREDICTION, M_SCORE]

CONFIG = './config.json'


def input_args():
    parser = argparse.ArgumentParser(description="1.train 2.test 3.eval 4.prediction 5.score ")
    parser.add_argument("mode", metavar="operation mode", choices=MODE, help="Choices mode: {}".format(MODE))
    parser.add_argument("--sentence", dest="sentence", action='store', type=str, default='',
                        help="Please Enter the predicted sentence")
    parser.add_argument("--file", dest='file', action='store', type=str, default='./predict_result.json',
                        help="Write the prediction results to a file")
    args = parser.parse_args()
    return args


def regular(sentence):
    """ 正则化表达式，剔除语料中的空格 """
    sentence = re.sub('[ ]+', '', sentence)
    return sentence


def identifier_format(i, s):
    """
    定义语料标识符的格式
    :param i: (str, mandatory) 标识符
    :param s: (str, mandatory) 名称
    :return:
    """
    return "{}_{}".format(i, s)


def handle_train(line):
    """ 提取训练文本、标识符、标签数据 """
    json_data = json.loads(line)

    # 获取文本数据和标签数据
    text = json_data['text']
    label = json_data['label']

    identifier = [identifier_o] * len(text)

    for ner_name, ner_value in label.items():
        for ner_str, ner_index in ner_value.items():
            for n_index in ner_index:
                if text[n_index[0]:n_index[1] + 1] != ner_str:
                    print("Data Error: no specific character found . text: {}, label: {}".format(text, label))
                    exit()
                # 单个字符的实体。在中文语料中可能不存在。
                if len(ner_str) == 1:
                    identifier[n_index[0]] = identifier_format(identifier_s, ner_name)

                # 两个字符的实体
                elif len(ner_str) == 2:
                    identifier[n_index[0]] = identifier_format(identifier_b, ner_name)
                    identifier[n_index[1]] = identifier_format(identifier_e, ner_name)

                # 两个字符以上的实体
                elif len(ner_str) > 2:
                    identifier[n_index[0]] = identifier_format(identifier_b, ner_name)
                    for i in range(1, len(ner_str) - 2 + 1):
                        identifier[n_index[0] + i] = identifier_format(identifier_i, ner_name)
                    identifier[n_index[1]] = identifier_format(identifier_e, ner_name)

    return [text, identifier, label]


def handle_test(line):
    """ 提取测试文本、标识符、标签数据 """
    data = json.loads(line)

    # 获取 text 和 id
    text_id = data['id']
    text = data['text']

    return [text_id, text]


def load_train_or_dev_dataset(file):
    """
    加载训练数据集
    :param file: (str, mandatory) 数据文件
    :return: (list) 文本数据，标识符数据，标签数据，实体类型
    """
    dataset = DataUtils.read_text_data(file, handle_func=handle_train)
    text_data, identifier_data, label_data = [], [], []
    for (text, identifier, label) in dataset:
        text = list(text)
        assert len(text) == len(identifier), \
            UnknownError("Please input data, text:{}, identifier:{}".format(text, identifier))

        text_data.append(text)
        identifier_data.append(identifier)
        label_data.append(label)

    entity_counter = Counter()
    for label in label_data:
        for entity_name, _, in label.items():
            entity_counter.update([entity_name])

    entity_type = [entity_name for entity_name, _ in entity_counter.items()]

    return text_data, identifier_data, label_data, entity_type


def load_test_dataset(file):
    """
    加载测试数据集
    :param file: (str, mandatory) 测试数据集文件
    :return: (list) 测试数据ID和文本
    """
    dataset = DataUtils.read_text_data(file, handle_func=handle_test)
    text_id, texts = [], []
    for (t_id, text) in dataset:
        text_id.append(t_id)
        texts.append(list(text))

    return text_id, texts


def init_or_restore_word_dict(file, text, identifier):
    """
    初始化或还原 WordDict 对象。首先检查 file， 如果 file存在，则尝试使用file进行还原，否则重新初始化。
    :param file: (str, mandatory) 保存字典文件
    :param text: (list, mandatory) 文本数据
    :param identifier: (list, mandatory) 标识符数据
    :return: (WordDict) WordDict 对象
    """
    if os.path.isfile(file):
        word_dict = WordDict.load(file)
        output = "Restore word dict from: {}".format(file)
    else:
        word_dict = WordDict()
        word_dict.fit(text, identifier)
        word_dict.save(file)

        output = "Not found: {}, Initializing from scratch".format(file)

    print(Color.green(output))
    return word_dict


def init_bilstm_crf(word_dict, conf):
    """
    初始化 BiLSTM_CRF 模型
    :param word_dict: (int, mandatory) 词典对象
    :param conf: (obj, mandatory) 字典对象
    :return: (NER_BiLSTM_CRF) 模型对象
    """
    model = BiLSTM_CRF(word_dict,
                       embedding_dim=conf.MODEL.BILSTM_CRF.embedding_dim,
                       hidden_size=conf.MODEL.BILSTM_CRF.hidden_size)
    return model


def init_or_restore_bilstm_crf(file, word_dict, conf):
    """
    初始化或还原 BiLSTM_CRF 模型。当发现配置文件和检查点目录时候，则尝试从配置文件和检查点恢复模型
    :param file: (str, mandatory) 配置文件
    :param word_dict: (int, mandatory) 词典对象
    :param conf: (obj, mandatory) 字典对象
    :return: (NER_BiLSTM_CRF) 模型对象
    """
    if os.path.isfile(file):
        model = BiLSTM_CRF.load(file)
        output = "Restore model from:{}".format(file)
    else:
        model = init_bilstm_crf(word_dict, conf)
        output = "Not found pth file:{}, Initializing from scratch".format(file)

    print(Color.green(output))
    return model


def find_entity(texts, text_identifier):
    """
    查找实体。 根据预测文本和文本标识符找到对应的实体
    :param texts: (str, mandatory) 文本
    :param text_identifier: (list, mandatory) 序列标识符。
    :return:
    """
    assert len(texts) == len(text_identifier), \
        UnknownError("Please check input data size. text:{}, identifier:{}".format(texts, text_identifier))

    def check_entity_type(entity):
        """
        检查标识符序列后缀类型是否都一致
        :param entity: (list) 序列标识符。如: [B_game, I_game, E_game]
        :return: (bool) True or False
        """
        # 检查标识符的后缀。如：B_game, I_game, E_game 的 game 是不是都相同。相同则返回 True， 否则返回 False 。
        for a_entity, b_entity in zip(entity[:-1], entity[1:]):
            if a_entity == identifier_o or b_entity == identifier_o:
                return False

            if a_entity[1:] != b_entity[1:]:
                return False

        # 检查序列内是否包括两个 B 或 E。如果有则返回 False。例如：'B_company', 'E_company', 'B_company', 'E_company'
        for a_entity in entity[1:-1]:
            if a_entity[:1] == identifier_b or a_entity[:1] == identifier_e:
                return False

        return True

    entity_index = {}
    # 正向匹配方法
    for i in range(len(text_identifier)):
        if text_identifier[i] == identifier_o:
            continue
        if text_identifier[i][:1] in [identifier_i, identifier_e]:
            continue

        new_text_identifier = text_identifier[i:]
        for j in range(len(new_text_identifier)):
            sample_identifier = new_text_identifier[:len(new_text_identifier) - j]

            if len(sample_identifier) == 0:
                continue

            if sample_identifier[-1] == identifier_o:
                continue

            # 检查获取的样本标识符第一个和最后一个是不是 B 和 E。标记开始和结尾
            if sample_identifier[0][:1] == identifier_b and sample_identifier[-1][:1] == identifier_e:
                if check_entity_type(sample_identifier):
                    index = [i, i + len(sample_identifier) - 1]
                    entity_str = ''.join(texts[index[0]:index[1] + 1])
                    entity_name = sample_identifier[0][2:]

                    # 更新到 entity_index 字典
                    if entity_name in entity_index.keys():
                        if entity_str in entity_index[entity_name].keys():
                            entity_index[entity_name][entity_str].append(index)
                        else:
                            entity_index[entity_name][entity_str] = [index]
                    else:
                        entity_dict = dict()
                        entity_dict[entity_str] = [index]
                        entity_index[entity_name] = entity_dict
    return entity_index


class Score(object):
    """ 评分 """

    def __init__(self):
        # 二维矩阵. shape:[2, 2]。
        # shape[0] 为计算 precise 的 TP 和 FN。
        # shape[1] 为计算 recall 的 TP he FP。
        # example:
        #   「[TP, FN],
        #     [TP, FP]]
        self.confusion_matrix = np.zeros((2, 2))

    def update(self, pred_identifier, true_identifier, y_pred, y_true):
        """
        更新数据矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_pred: (dict) 预测实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :param y_true: (dict) 正确实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        self.update_recall(pred_identifier, true_identifier, y_true)
        self.update_precise(pred_identifier, true_identifier, y_pred)

    def update_recall(self, pred_identifier, true_identifier, y_true):
        """
        更新Recall矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_true: (dict) 正确实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        for entity_name, entity_value in y_true.items():
            for _, entity_index in entity_value.items():
                for index in entity_index:
                    if pred_identifier[index[0]:index[1] + 1] == true_identifier[index[0]:index[1] + 1]:
                        self.confusion_matrix[1][0] += 1
                    else:
                        self.confusion_matrix[1][1] += 1

    def update_precise(self, pred_identifier, true_identifier, y_pred):
        """
        更新Precise矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_pred: (dict) 预测实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        for entity_name, entity_value in y_pred.items():
            for _, entity_index in entity_value.items():
                for index in entity_index:
                    if pred_identifier[index[0]:index[1] + 1] == true_identifier[index[0]:index[1] + 1]:
                        self.confusion_matrix[0][0] += 1
                    else:
                        self.confusion_matrix[0][1] += 1

    def get_recall(self):
        recall = self.confusion_matrix[1][0] / self.confusion_matrix[1].sum()
        return recall

    def get_precise(self):
        precise = self.confusion_matrix[0][0] / self.confusion_matrix[0].sum()
        return precise

    def get_f1_score(self):
        recall = self.get_recall()
        precise = self.get_precise()

        f1_score = 2 * (recall * precise) / (recall + precise)
        return f1_score


class MultiEntityScore(object):
    """ 多实体评分 """

    def __init__(self, entity):
        self.entity = entity
        self.__init_confusion_matrix__()

    def __init_confusion_matrix__(self):
        self.entity_dict = {}
        for entity_name in self.entity:
            self.entity_dict[entity_name] = len(self.entity_dict)

        # 三维矩阵. shape:[实体数量, 2, 2]。其中 shape[0] 为实体类型。
        # shape[实体类型][0] 为计算 precise 的 TP 和 FN。
        # shape[实体类型][1] 为计算 recall 的 TP he FP。
        # example:
        #   [[[TP, FN]],
        #    [[TP, FP]],
        #
        #    [[TP, FN]],
        #    [[TP, FP]]]
        self.confusion_matrix = np.zeros((len(self.entity_dict), 2, 2))

    def update(self, pred_identifier, true_identifier, y_pred, y_true):
        """
        更新数据矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_pred: (dict) 预测实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :param y_true: (dict) 正确实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        assert len(pred_identifier) == len(true_identifier), \
            UnknownError("Please check input data size. y_pred:{}, y_true:{}".format(pred_identifier, true_identifier))

        self.update_recall(pred_identifier, true_identifier, y_true)
        self.update_precise(pred_identifier, true_identifier, y_pred)

    def update_recall(self, pred_identifier, true_identifier, y_true):
        """
        更新Recall矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_true: (dict) 正确实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        for entity_name, entity_value in y_true.items():
            for _, entity_index in entity_value.items():
                for index in entity_index:
                    if pred_identifier[index[0]: index[1] + 1] == true_identifier[index[0]: index[1] + 1]:
                        self.confusion_matrix[self.entity_dict[entity_name]][1][0] += 1
                    else:
                        self.confusion_matrix[self.entity_dict[entity_name]][1][1] += 1

    def update_precise(self, pred_identifier, true_identifier, y_pred):
        """
        更新Precise矩阵
        :param pred_identifier: (list) 预测序列标识符。例如：[B_game, I_game, E_game]
        :param true_identifier: (list) 正确序列标识符。例如: [B_game, I_game, E_game]
        :param y_pred: (dict) 预测实体索引标记。如： {'position': {'老师': [[18, 19]]}}
        :return:
        """
        for entity_name, entity_value in y_pred.items():
            for _, entity_index in entity_value.items():
                for index in entity_index:
                    if pred_identifier[index[0]: index[1] + 1] == true_identifier[index[0]: index[1] + 1]:
                        self.confusion_matrix[self.entity_dict[entity_name]][0][0] += 1
                    else:
                        self.confusion_matrix[self.entity_dict[entity_name]][0][1] += 1

    def get_recall(self):
        """ 获取 Recall """
        result = {}
        for entity_name, entity_id in self.entity_dict.items():
            recall = self.confusion_matrix[entity_id][1][0] / self.confusion_matrix[entity_id][1].sum()
            result[entity_name] = recall

        return result

    def get_precise(self):
        """ 获取 Precise """
        result = {}
        for entity_name, entity_id in self.entity_dict.items():
            precise = self.confusion_matrix[entity_id][0][0] / self.confusion_matrix[entity_id][0].sum()
            result[entity_name] = precise
        return result

    def get_f1_score(self):
        """ 获取 F1 score """
        recall_result = self.get_recall()
        precise_result = self.get_precise()

        result = {}
        entity_list = list(recall_result.keys())
        for entity_name in entity_list:
            recall = recall_result[entity_name]
            precise = precise_result[entity_name]

            f1_score = 2 * (recall * precise) / (recall + precise)
            result[entity_name] = f1_score

        return result


class NameEntityRecognition(object):

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        self.__init_config__()
        self.__load_dataset__()
        self.__init_word_dict__()
        self.__init_dataset__()
        self.__init_model__()

    def __init_config__(self):
        """" 初始化配置文件 """
        self.conf = Config(self.config).get_data()

    def __load_dataset__(self):
        """ 初始化数据集 """
        self.dataset = None
        # 训练使用 train 数据集。如果 USING_DEV_AS_VALIDATION_DATASET = True 。则将 dev 数据集加入训练
        if self.mode == M_TRAIN:
            train_text, train_identifier, train_label, train_entity = load_train_or_dev_dataset(
                self.conf.PATH_DATA_TRAIN)
            if self.conf.USING_DEV_AS_VALIDATION_DATASET:
                valid_text, valid_identifier, valid_label, valid_entity = load_train_or_dev_dataset(
                    self.conf.PATH_DATA_DEV)

                text_data = train_text + valid_text
                identifier_data = train_identifier + valid_identifier
                label_data = train_label + valid_label
                entity_type = list(set(train_entity + valid_entity))
            else:
                text_data = train_text
                identifier_data = train_identifier
                label_data = train_label
                entity_type = train_entity

            self.dataset = (text_data, identifier_data, label_data, entity_type)

        # 评分 或 评估。使用 dev 数据集
        elif self.mode == M_SCORE or self.mode == M_EVAL:
            score_text, score_identifier, score_label, entity_type = load_train_or_dev_dataset(self.conf.PATH_DATA_DEV)
            self.dataset = (score_text, score_identifier, score_label, entity_type)

        # 测试。使用 test 数据集
        elif self.mode == M_TEST:
            text_id, text = load_test_dataset(self.conf.PATH_DATA_TEST)
            self.dataset = (text_id, text, [], [])

    def __init_word_dict__(self):
        """ 初始化 WordDict """
        text, identifier = None, None
        if self.mode == M_TRAIN:
            text, identifier, _, _ = self.dataset

        self.word_dict = init_or_restore_word_dict(self.conf.PATH_WORD_DICT, text, identifier)

    def __init_dataset__(self):
        """" 初始化数据集 """
        self.input_dataset = None
        if self.mode == M_TRAIN or self.mode == M_SCORE or self.mode == M_EVAL:
            text_data, identifier_data, _, _ = self.dataset

            input_x, input_y = list(), list()
            for text, identifier in zip(text_data, identifier_data):
                x = self.word_dict.encoding_sentence(text)
                y = self.word_dict.encoding_identifier(identifier)

                assert len(x) == len(y), \
                    UnknownError("Please check input data. text:{}, identifier:{]".format(text, identifier))

                input_x.append(x)
                input_y.append(y)

            self.input_dataset = (input_x, input_y)

    def __init_model__(self):
        """ 初始化模型 """
        self.model = init_or_restore_bilstm_crf(file=self.conf.MODEL.BILSTM_CRF.pth,
                                                word_dict=self.word_dict,
                                                conf=self.conf)

    def train(self):
        """ 训练 """
        x, y = self.input_dataset
        self.model.fit(x, y,
                       epochs=self.conf.MODEL.BILSTM_CRF.EPOCHS,
                       lr=self.conf.MODEL.BILSTM_CRF.learning_rate,
                       weight_decay=self.conf.MODEL.BILSTM_CRF.weight_decay,
                       ratio=self.conf.MODEL.BILSTM_CRF.VALIDATION_SPLIT,
                       pth=self.conf.MODEL.BILSTM_CRF.pth)

    def test(self, file='predict_result.json'):
        """
        测试。将测试数据集识别结果写入文件
        :param file: (str, optional, default='predict_result.json') 文件名
        :return:
        """
        result = []
        text_id, texts, _, _ = self.dataset
        for t_id, text in tqdm(zip(text_id, texts)):
            x = self.word_dict.encoding_sentence(text)
            prediction = self.model.predict(x)
            pred_identifier = self.word_dict.decoding_identifier(prediction)
            entity_index = find_entity(text, pred_identifier)

            json_str = json.dumps({'id': t_id, 'label': entity_index}, ensure_ascii=False)
            result.append(json_str)

        Writer.check_path(file)
        with open(file, 'a+') as f:
            for line in result:
                f.write(line + '\n')

        print(Color.green("Write computer! File: {}".format(file)))

    def eval(self):
        """ 评估 """
        x, y = self.input_dataset
        self.model.test(x, y)

    def score(self):
        """ 评分 """
        texts, true_identifier, y_true, entity_type = self.dataset

        score = Score()
        multi_entity_score = MultiEntityScore(entity_type)

        for i in range(len(texts)):
            x = self.word_dict.encoding_sentence(texts[i])
            prediction = self.model.predict(x)
            pred_identifier = self.word_dict.decoding_identifier(prediction)
            y_pred = find_entity(''.join(texts[i]), pred_identifier)

            score.update(pred_identifier, true_identifier[i], y_pred, y_true[i])
            multi_entity_score.update(pred_identifier, true_identifier[i], y_pred, y_true[i])

            template = "Step:{}/{}, Recall:{:.4f}, Precise: {:.4f}, F1_score:{:.4f} ".format(i + 1,
                                                                                             len(texts),
                                                                                             score.get_recall(),
                                                                                             score.get_precise(),
                                                                                             score.get_f1_score())
            sys.stdout.write('\r' + template)
            sys.stdout.flush()
        print()

        # 获取每个实体的 recall、precise、f1_score
        recall = multi_entity_score.get_recall()
        precise = multi_entity_score.get_precise()
        f1_score = multi_entity_score.get_f1_score()

        # 计算出每个实体的大小，方便规范化输出
        size = 0
        for entity_name in entity_type:
            if len(entity_name) > size:
                size = len(entity_name)
        size += 1

        for entity_name in entity_type:
            output_entity_name = entity_name + (size - len(entity_name)) * " "
            print("Entity:{}, Recall:{:.4f}, Precise:{:.4f}, F1_score:{:.4f}".format(output_entity_name,
                                                                                     recall[entity_name],
                                                                                     precise[entity_name],
                                                                                     f1_score[entity_name]))

    def predict(self, sentence):
        """
        预测
        :param sentence: (str, mandatory) 字符型文本
        :return: (list and dict) 标识符序列和实体索引
        """
        sentence = regular(sentence)
        if sentence == ' ' or len(sentence) == 0:
            print("Input sentence cannot be empty")
            return False

        x = self.word_dict.encoding_sentence(list(sentence))
        prediction = self.model.predict(x)
        pred_identifier = self.word_dict.decoding_identifier(prediction)
        entity = find_entity(sentence, pred_identifier)

        return pred_identifier, entity

    def run(self, sentence="", file='predict_result.json'):
        """ 运行 """
        if self.mode == M_TRAIN:
            return self.train()
        elif self.mode == M_TEST:
            return self.test(file)
        elif self.mode == M_EVAL:
            return self.eval()
        elif self.mode == M_SCORE:
            return self.score()
        elif self.mode == M_PREDICTION:
            return self.predict(sentence)


def test_module_func(operation):
    if operation == load_train_or_dev_dataset.__name__:
        dataset = load_train_or_dev_dataset('../../../data/cluener_public/dev.json')
        text_data, identifier_data, label_data, entity_type = dataset
        for i, (text, identifier, label) in enumerate(zip(text_data, identifier_data, label_data)):
            print("text: ", text)
            print("identifier: ", identifier)
            print("label: ", label)
            print()
            if i == 10:
                break

        print("entity type: ", entity_type)

    elif operation == load_test_dataset.__name__:
        dataset = load_test_dataset('../../../data/cluener_public/test.json')
        text_id, texts = dataset
        for i, (t_id, text) in enumerate(zip(text_id, texts)):
            print("text id: ", t_id)
            print("text: ", text)
            print()
            if i == 10:
                break

    elif operation == init_or_restore_word_dict.__name__:
        file = './model/word_dict.pickle'
        text_data, identifier_data, _, _ = load_train_or_dev_dataset('../../../data/cluener_public/dev.json')
        word_dict = init_or_restore_word_dict(file, text_data, identifier_data)
        print("sequence max length : {}".format(word_dict.max_length))
        print("words: {}".format(len(word_dict.word_token)))
        print("identifier: {}".format(len(word_dict.identifier_token)))

        new_word_dict = init_or_restore_word_dict(file, text_data, identifier_data)
        print("sequence max length : {}".format(new_word_dict.max_length))
        print("words: {}".format(len(new_word_dict.word_token)))
        print("identifier: {}".format(len(new_word_dict.identifier_token)))

    elif operation == find_entity.__name__:
        texts = "腾讯新闻昨天金庸逝世江湖再无金大侠订阅号消息昨天【15条】王者荣耀福利抢鲜看!队友的京东京东jd."
        identifier = ['B_company', 'I_company', 'I_company', 'E_company', 'O', 'O', 'B_name', 'E_name', 'O', 'O', 'O',
                      'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                      'B_game', 'I_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B_company',
                      'E_company', 'B_company', 'E_company', 'O', 'O', 'O']
        target_entity = {'game': {'王者荣耀': [[29, 32]]}, 'name': {'金庸': [[6, 7]]},
                         'company': {'腾讯新闻': [[0, 3]], '京东': [[42, 43], [44, 45]]}}

        entity = find_entity(texts, identifier)
        print("find entity: ", entity)
        print("target entity: ", target_entity)
        print(entity == target_entity)

        # text_data, identifier_data, label_data = load_dataset('../../../data/cluener_public/dev.json')
        # for i, (text, identifier, label) in enumerate(zip(text_data, identifier_data, label_data)):
        #     entity = find_entity(''.join(text), identifier)
        #
        #     if entity == label:
        #         print("find entity: ", entity)
        #         print("target entity: ", label)
        #         print(True)
        #     else:
        #         print(Color.red("find entity: {}".format(entity)))
        #         print(Color.red("target entity: {}".format(label)))
        #         print(Color.red(False))
        #     print()

    elif operation == Score.__name__:
        true_identifier = ['O', 'B_government', 'I_government', 'E_government', 'O', 'O', 'O']
        pred_identifier = ['O', 'B_government', 'I_government', 'E_government', 'B_government', 'I_government',
                           'E_government']
        y_true = {'government': {'北京市': [[1, 3]]}}
        y_pred = {'government': {'北京市': [[1, 3], [4, 6]]}}
        score = Score()
        score.update(pred_identifier, true_identifier, y_pred, y_true)
        print("recall: ", score.get_recall())
        print("precise: ", score.get_precise())
        print("F1 score: ", score.get_f1_score())

    elif operation == MultiEntityScore.__name__:
        true_identifier = ['O', 'B_government', 'I_government', 'E_government', 'O', 'O', 'O']
        pred_identifier = ['O', 'B_government', 'I_government', 'E_government', 'B_government', 'I_government',
                           'E_government']
        y_true = {'government': {'北京市': [[1, 3]]}}
        y_pred = {'government': {'北京市': [[1, 3], [4, 6]]}}
        score = MultiEntityScore(['government'])
        score.update(pred_identifier, true_identifier, y_pred, y_true)

        print("recall: ", score.get_recall())
        print("precise: ", score.get_precise())
        print("F1 score: ", score.get_f1_score())

    elif operation == 'train':
        ner = NameEntityRecognition(mode=M_TRAIN, config=CONFIG)
        ner.run()

    elif operation == 'test':
        ner = NameEntityRecognition(mode=M_TEST, config=CONFIG)
        ner.run(file='./model/predict_result.json')

    elif operation == 'eval':
        ner = NameEntityRecognition(mode=M_EVAL, config=CONFIG)
        ner.run()

    elif operation == 'score':
        ner = NameEntityRecognition(mode=M_SCORE, config=CONFIG)
        ner.run()

    elif operation == 'predict':
        sentence = "《加勒比海盗3：世界尽头》的去年同期成绩死死甩在身后，后者则即将赶超《变形金刚》，"
        ner = NameEntityRecognition(mode=M_PREDICTION, config=CONFIG)
        identifier, entity_index = ner.run(sentence)
        print("identifier: ", identifier)
        print("entity index: ", entity_index)


def main():
    args = input_args()

    if args.mode == M_PREDICTION:
        sentence = regular(args.sentence)
        if sentence == ' ' or len(sentence) == 0:
            print("Input sentence cannot be empty")
            return False
        ner = NameEntityRecognition(mode=M_PREDICTION, config=CONFIG)
        identifier, entity_index = ner.run(args.sentence)
        print("sentence: ", sentence)
        print("identifier: ", identifier)
        print("entity index: ", entity_index)

    else:
        ner = NameEntityRecognition(mode=args.mode, config=CONFIG)
        ner.run(file=args.file)


if __name__ == '__main__':
    # test_module_func(load_train_or_dev_dataset.__name__)
    # test_module_func(load_test_dataset.__name__)
    # test_module_func(init_or_restore_word_dict.__name__)
    test_module_func(find_entity.__name__)
    # test_module_func(Score.__name__)
    # test_module_func(MultiEntityScore.__name__)
    # test_module_func('train')
    # test_module_func('test')
    # test_module_func('eval')
    # test_module_func('score')
    # test_module_func('predict')
    # main()
