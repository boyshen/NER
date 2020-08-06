# -*- encoding: utf-8 -*-
"""
@file: run.py
@time: 2020/7/30 下午6:34
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 训练：使用训练数据集进行训练
    example: python run.py train

# 评估：将使用 dev 数据集进行评估。输出Recall、Precise、F1-score
    example: python run.py eval

# 测试：将使用测试数据集，将测试数据集识别结果写入文件
    example：python run.py test --file "predict_result.json"
            --file          指定写入的文件名
# 预测：
    example: python run.py prediction --sentence "《加勒比海盗3：世界尽头》的去年同期成绩死死甩在身后，后者则即将赶超《变形金刚》，"
            --sentence      指定预测的句子
"""
import re
import sys
import os
import json
import argparse
from tqdm import tqdm
from collections import Counter

try:
    from module.core.utils import DataUtils
    from module.core.utils import Config
    from module.core.utils import Writer
    from module.core.exception import UnknownError
    from module.bilstm_crf.word_dict import WordDict
    from module.core.color import Color
    from module.bilstm_crf.model import NameEntityRecognitionModel
except ModuleNotFoundError:
    sys.path.append('../../')
    from module.core.utils import DataUtils
    from module.core.utils import Config
    from module.core.utils import Writer
    from module.core.exception import UnknownError
    from module.bilstm_crf.word_dict import WordDict
    from module.core.color import Color
    from module.bilstm_crf.model import NameEntityRecognitionModel

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

MODE = [M_TRAIN, M_TEST, M_EVAL, M_PREDICTION]

CONFIG = './config.json'


def input_args():
    parser = argparse.ArgumentParser(description="1.train 2.test 3.eval 4.prediction ")
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

        output = "Not found:{}, Initializing from scratch".format(file)

    print(Color.green(output))
    return word_dict


def init_model(word_dict, conf):
    """
    初始化 BiLSTM_CRF 模型
    :param word_dict: (int, mandatory) 词典对象
    :param conf: (obj, mandatory) 字典对象
    :return: (NER_BiLSTM_CRF) 模型对象
    """
    vocab_size = len(word_dict.word_token)
    target_size = len(word_dict.identifier_token)
    start_index = word_dict.identifier_token[word_dict.START_TAG]
    end_index = word_dict.identifier_token[word_dict.END_TAG]
    model = NameEntityRecognitionModel(vocab_size=vocab_size,
                                       embedding_dim=conf.MODEL.embedding_dim,
                                       target_size=target_size,
                                       hidden_size=conf.MODEL.hidden_size,
                                       start_index=start_index,
                                       end_index=end_index,
                                       dropout=0.5)
    return model


def init_or_restore_model(word_dict, conf):
    """
    初始化或还原模型。当发现配置文件和检查点目录时候，则尝试从配置文件和检查点恢复模型
    :param word_dict: (int, mandatory) 词典对象
    :param conf: (obj, mandatory) 字典对象
    :return: (NER_BiLSTM_CRF) 模型对象
    """
    if os.path.isfile(conf.MODEL.checkpoint) and os.path.isfile(conf.MODEL.config):
        model = NameEntityRecognitionModel.restore(conf.MODEL.config, conf.MODEL.checkpoint)
        output = "Restore model from:{}, Checkpoint:{} ".format(conf.MODEL.config, conf.MODEL.checkpoint)
    else:
        model = init_model(word_dict, conf)
        output = "Not found checkpoint and config file:{}, Initializing from scratch".format(conf.MODEL.config)

    print(Color.green(output))
    return model


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


def find_entity(texts, text_identifier):
    """
    查找实体。 根据预测文本和文本标识符找到对应的实体
    :param texts: (str, mandatory) 文本
    :param text_identifier: (list, mandatory) 序列标识符。
    :return:
    """
    assert len(texts) == len(text_identifier), \
        UnknownError("Please check input data size. text:{}, identifier:{}".format(texts, text_identifier))

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


def find_entity_index(text_identifier):
    entity_index = []
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
                    entity_name = sample_identifier[0][2:]
                    entity_index.append({entity_name: index})

    return entity_index


class NameEntityRecognitionScore(object):
    """
    命名实体评分。提供 recall、precise、F1_score 评估指标
    """

    def __init__(self):
        self.y_pred = []
        self.y_true = []
        self.y_positive = []

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.y_positive = []

    def update(self, prediction, label):
        """
        更新。
        :param prediction: (list, mandatory) 预测序列。 如: ['B_book', 'I_book', 'I_book', 'O']
        :param label: (list, mandatory) 标签序列。如：['B_book', 'I_book', 'I_book', 'O']
        :return:
        """
        prediction_entity = find_entity_index(prediction)
        label_entity = find_entity_index(label)

        self.y_pred.extend(prediction_entity)
        self.y_true.extend(label_entity)
        self.y_positive.extend([p_entity for p_entity in prediction_entity if p_entity in label_entity])

    def f1_score(self):
        """ F1 score """
        recall = len(self.y_positive) / len(self.y_true) if len(self.y_true) != 0 else 0.0
        precise = len(self.y_positive) / len(self.y_pred) if len(self.y_pred) != 0 else 0.0
        f1_score = 2 * (recall * precise) / (recall + precise) if (recall + precise) != 0 else 0.0
        return f1_score

    def recall(self):
        """ recall """
        recall = len(self.y_positive) / len(self.y_true) if len(self.y_true) != 0 else 0.0
        return recall

    def precise(self):
        """ precise """
        precise = len(self.y_positive) / len(self.y_pred) if len(self.y_pred) != 0 else 0.0
        return precise

    def result(self):
        """
        :return: (dict) 每个实体的 recall、precise、f1_score
        """
        y_pred_counter = Counter()
        for entity in self.y_pred:
            y_pred_counter.update(list(entity.keys()))

        y_true_counter = Counter()
        for entity in self.y_true:
            y_true_counter.update(list(entity.keys()))

        y_positive_counter = Counter()
        for entity in self.y_positive:
            y_positive_counter.update(list(entity.keys()))

        recall, precise, f1_score = {}, {}, {}
        for name, counter in y_positive_counter.items():
            recall_score = counter / y_true_counter[name]
            precise_score = counter / y_pred_counter[name]
            recall[name] = recall_score
            precise[name] = precise_score
            f1_score[name] = 2 * (recall_score * precise_score) / (recall_score + precise_score)

        return recall, precise, f1_score


class NER(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        self.__init_config__()
        self.__load_dataset__()
        self.__init_word_dict__()
        self.__init_dataset__()
        self.__init_model__()

    def __init_config__(self):
        """ 初始化配置 """
        self.conf = Config(self.config).get_data()

    def __load_dataset__(self):
        """ 加载数据集 """
        self.dataset = None
        if self.mode == M_TRAIN:
            text, identifier, _, _ = load_train_or_dev_dataset(self.conf.PATH_DATA_TRAIN)
            self.dataset = (text, identifier)
        elif self.mode == M_EVAL:
            text, identifier, _, _ = load_train_or_dev_dataset(self.conf.PATH_DATA_DEV)
            self.dataset = (text, identifier)
        elif self.mode == M_TEST:
            text_id, text = load_test_dataset(self.conf.PATH_DATA_TEST)
            self.dataset = (text_id, text)

    def __init_word_dict__(self):
        """ 初始化字典 """
        text, identifier = None, None
        if self.mode == M_TRAIN:
            text, identifier = self.dataset
        self.word_dict = init_or_restore_word_dict(self.conf.PATH_WORD_DICT, text, identifier)

    def __init_dataset__(self):
        """ 初始化训练数据集 """
        self.input_dataset = None
        if self.mode == M_TRAIN:
            texts, identifiers = self.dataset
            x, y, input_length = [], [], []
            for text, identifier in zip(texts, identifiers):
                input_x, seq_len = self.word_dict.encoding_sentence(text, alignment=True)
                input_y, _ = self.word_dict.encoding_identifier(identifier, alignment=True)

                assert len(input_x) == len(input_y), \
                    UnknownError("Please check input data size. text:{}, identifier:{}".format(text, identifier))

                x.append(input_x)
                y.append(input_y)
                input_length.append(seq_len)
            self.input_dataset = (x, y, input_length)

    def __init_model__(self):
        """ 初始化模型 """
        self.model = init_or_restore_model(self.word_dict, self.conf)

    def train(self):
        """ 训练 """
        x, y, input_length = self.input_dataset
        self.model.fit(x, y, input_length,
                       epochs=self.conf.MODEL.epochs,
                       lr=self.conf.MODEL.learning_rate,
                       weight_decay=self.conf.MODEL.weight_decay,
                       batch_size=self.conf.MODEL.batch_size,
                       shuffle=self.conf.MODEL.shuffle,
                       validation_rate=self.conf.MODEL.validation_rate,
                       grad_norm=self.conf.MODEL.gradient_normal,
                       factor=self.conf.MODEL.factor,
                       patience=self.conf.MODEL.patience,
                       eps=self.conf.MODEL.eps,
                       checkpoint=self.conf.MODEL.checkpoint)
        self.model.dump_config(self.conf.MODEL.config)

    def eval(self):
        """ 评分 """
        score = NameEntityRecognitionScore()

        texts, identifiers = self.dataset
        for step, (text, identifier) in enumerate(zip(texts, identifiers)):
            input_x, _ = self.word_dict.encoding_sentence(text, alignment=False)
            y_pred = self.model.predict(input_x)
            y_pred = self.word_dict.decoding_identifier(y_pred)
            score.update(y_pred, identifier)

            outputs = "Step:{}/{}, Recall:{:.4f}, Precise:{:.4f}, F1_score:{:.4f}".format(step + 1, len(texts),
                                                                                          score.recall(),
                                                                                          score.precise(),
                                                                                          score.f1_score())
            sys.stdout.write('\r' + outputs)
            sys.stdout.flush()
        print()

        recall, precise, f1_score = score.result()
        max_length = 0
        for entity, _ in recall.items():
            if len(entity) > max_length:
                max_length = len(entity)
        max_length += 1

        for entity, _ in recall.items():
            str_entity = entity + ' ' * (max_length - len(entity))
            print("Entity: {}, Recall:{:.4f}, Precise:{:.4f}, F1_score:{:.4f}".format(str_entity, recall[entity],
                                                                                      precise[entity],
                                                                                      f1_score[entity]))

    def test(self, file='predict_result.json'):
        """
        测试。将测试数据集识别结果写入文件
        :param file: (str, optional, default='predict_result.json') 文件名
        :return:
        """
        result = []
        text_id, texts = self.dataset
        for t_id, text in tqdm(zip(text_id, texts)):
            input_x, _ = self.word_dict.encoding_sentence(text, alignment=False)
            prediction = self.model.predict(input_x)
            pred_identifier = self.word_dict.decoding_identifier(prediction)
            entity_index = find_entity(text, pred_identifier)

            json_str = json.dumps({'id': t_id, 'label': entity_index}, ensure_ascii=False)
            result.append(json_str)

        Writer.check_path(file)
        with open(file, 'a+') as f:
            for line in result:
                f.write(line + '\n')

        print(Color.green("Write computer! File: {}".format(file)))

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

        input_x, _ = self.word_dict.encoding_sentence(list(sentence), alignment=False)
        prediction = self.model.predict(input_x)
        pred_identifier = self.word_dict.decoding_identifier(prediction)
        entity = find_entity(sentence, pred_identifier)

        return pred_identifier, entity

    def run(self, file='predict_result.json', sentence=''):
        if self.mode == M_TRAIN:
            return self.train()
        elif self.mode == M_EVAL:
            return self.eval()
        elif self.mode == M_TEST:
            return self.test(file)
        elif self.mode == M_PREDICTION:
            return self.predict(sentence)


def main():
    args = input_args()

    if args.mode == M_PREDICTION:
        sentence = regular(args.sentence)
        if sentence == ' ' or len(sentence) == 0:
            print("Input sentence cannot be empty")
            return False
        ner = NER(mode=M_PREDICTION, config=CONFIG)
        identifier, entity_index = ner.run(sentence=args.sentence)
        print("sentence: ", sentence)
        print("identifier: ", identifier)
        print("entity index: ", entity_index)

    else:
        ner = NER(mode=args.mode, config=CONFIG)
        ner.run(file=args.file)


def test_module_func(operation):
    if operation == find_entity_index.__name__:
        text_identifier = ['B_book', 'I_book', 'I_book', 'O', 'B_name', 'I_name', 'E_name', 'B_name', 'I_name',
                           'E_name']
        entity = find_entity_index(text_identifier)
        print("entity: ", entity)

    elif operation == NameEntityRecognitionScore.__name__:
        y_true = ['B_book', 'I_book', 'E_book', 'O', 'B_name', 'I_name', 'E_name', 'B_name', 'I_name',
                  'E_name']
        y_pred = ['B_book', 'I_book', 'E_book', 'O', 'B_name', 'I_name', 'I_name', 'B_name', 'I_name',
                  'E_name']

        score = NameEntityRecognitionScore()
        score.update(y_pred, y_true)

        print("recall: {:.4f}, precise: {:.4f}, F1 score:{:.4f}".format(score.recall(),
                                                                        score.precise(),
                                                                        score.f1_score()))
        recall, precise, f1_score = score.result()
        for name, score in recall.items():
            print("Entity: {}, recall: {:.4f}, precise: {:.4f}, f1_score: {:.4f}".format(name,
                                                                                         recall[name],
                                                                                         precise[name],
                                                                                         f1_score[name]))

    elif operation == 'train':
        ner = NER(mode=M_TRAIN, config=CONFIG)
        ner.run()

    elif operation == 'eval':
        ner = NER(mode=M_EVAL, config=CONFIG)
        ner.run()

    elif operation == 'test':
        ner = NER(mode=M_TEST, config=CONFIG)
        ner.run(file='./model/result/predict_result.json')

    elif operation == 'predict':
        sentence = "加勒比海盗3：世界尽头》的去年同期成绩死死甩在身后，后者则即将赶超《变形金刚》"
        ner = NER(mode=M_PREDICTION, config=CONFIG)
        identifier, entity = ner.run(sentence=sentence)
        print("identifier: ", identifier)
        print("entity: ", entity)


if __name__ == '__main__':
    # test_module_func(find_entity_index.__name__)
    # test_module_func(NameEntityRecognitionScore.__name__)
    # test_module_func('train')
    # test_module_func('eval')
    # test_module_func('test')
    # test_module_func('predict')
    main()
