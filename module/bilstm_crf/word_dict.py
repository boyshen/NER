# -*- encoding: utf-8 -*-
"""
@file: word_dict.py
@time: 2020/7/15 下午2:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""

from tqdm import tqdm
from collections import Counter
from module.core.exception import exception_handling, ParameterError, NotFitException
from module.core.utils import Dictionary


class WordDict(Dictionary):

    @exception_handling
    def __init__(self):
        super(WordDict, self).__init__()

        self.word_token = {
            self.PAD_TAG: 0
        }

        self.identifier_token = {
            self.PAD_TAG: 0
        }
        self.token_identifier = dict()

        # 保存fit语料中最大的序列size
        self.max_length = 0
        self.__is_fit = False

    @exception_handling
    def fit(self, data, identifiers):
        """
        拟合。根据数据和标示符号拟合词典。初始化 word_token、token_word、identifier_token
        :param data: (list or tuple, mandatory) 数据集。字符列表或字符元祖格式。
            例如：[['大', '众', '软', '件', '精', '彩', '分', '享', ...]]
        :param identifiers: (list or tuple, mandatory) 标识符。列表列表格式
            例如：[[B_book, I_book, I_book, E_book, O, O, O, O, O], ...]
        :return:
        """
        if not isinstance(data, (list, tuple)):
            raise ParameterError("Data type must be is list or tuple, but actually get {}".format(type(data)))
        if not isinstance(identifiers, (list, tuple)):
            raise ParameterError(
                "Identifier type must be is list or tuple, but actually get {}".format(type(identifiers)))
        if len(data) != len(identifiers):
            raise ParameterError(
                "Data length:{} and identifier length:{} must be equal".format(len(data), len(identifiers)))

        word_counter = Counter()
        identifier_counter = Counter()
        for sent, identifier in tqdm(zip(data, identifiers)):
            if len(sent) != len(identifier):
                raise ParameterError("Data element length:{} and identifier element length:{} must be equal! "
                                     "Data:{}, identifier:{}".format(len(sent), len(identifier), sent, identifier))
            word_counter.update(sent)
            identifier_counter.update(identifier)

            if len(sent) > self.max_length:
                self.max_length = len(sent)

        for word, _ in word_counter.items():
            self.word_token[word] = len(self.word_token)

        for identifier, _ in identifier_counter.items():
            self.identifier_token[identifier] = len(self.identifier_token)
        self.identifier_token[self.START_TAG] = len(self.identifier_token)
        self.identifier_token[self.END_TAG] = len(self.identifier_token)

        self.token_identifier = {token: identifier for identifier, token in self.identifier_token.items()}

        self.__is_fit = True

        print("word num: {}, identifier num: {}".format(len(self.word_token), len(self.identifier_token)))

    @exception_handling
    def word_to_token(self, word):
        """
        将word转换成token
        :param word: (str, mandatory) 字或词
        :return: (int) 返回 token 或 PAD
        """
        if not self.__is_fit:
            raise NotFitException("NERDictionary not fit")
        try:
            token = self.word_token[word]
        except KeyError:
            token = self.word_token[self.PAD_TAG]

        return token

    @exception_handling
    def identifier_to_token(self, identifier):
        """
        将标识符转换成token
        :param identifier: (str, mandatory) 标识符
        :return: (int) token
        """
        if not self.__is_fit:
            raise NotFitException("NERDictionary not fit")

        if identifier not in list(self.identifier_token.keys()):
            raise ParameterError("identifier={} not in {} ".format(identifier, list(self.identifier_token.keys())))
        return self.identifier_token[identifier]

    @exception_handling
    def token_to_identifier(self, token):
        """
        将token转换成标识符
        :param token: (int, mandatory) token
        :return: (str) 标识符
        """
        if not self.__is_fit:
            raise NotFitException("NERDictionary not fit")

        if token not in list(self.token_identifier.keys()):
            raise ParameterError("token={} not in {}".format(token, list(self.token_identifier.keys())))
        return self.token_identifier[token]

    def alignment_sentence(self, sentence, alignment_size=None):
        """
        对齐句子。对不满足指定大小的序列进行填充和截断
        :param sentence: (list, mandatory) 句子序列
        :param alignment_size: (int, optional, default=None) 对齐大小
        :return: (list) 句子序列
        """

        def alignment(sequence, size, padding):
            # 如果句子序列大于 max_length 则进行截断
            sequence_length = len(sequence)
            if sequence_length > size:
                seq = sequence[:size]
                sequence_length = len(seq)

            # 如果句子序列小于 max_length 则进行填充
            elif sequence_length < size:
                seq = sequence + [padding] * (size - len(sequence))
            else:
                seq = sequence
            return seq, sequence_length

        sent = [self.word_to_token(word) for word in sentence]
        if alignment_size is None:
            return alignment(sent, self.max_length, self.word_to_token(self.PAD_TAG))
        else:
            return alignment(sent, alignment_size, self.word_to_token(self.PAD_TAG))

    def alignment_identifier(self, sentence, alignment_size=None):
        """
        对齐标识符。对不满足指定大小的序列进行填充和截断
        :param sentence: (list, mandatory) 句子序列
        :param alignment_size: (int, optional, default=None) 对齐大小
        :return: (list) 句子序列
        """

        def alignment(sequence, size, padding):
            # 如果句子序列大于 max_length 则进行截断
            sequence_length = len(sequence)
            if sequence_length > size:
                seq = sequence[:size]
                sequence_length = len(seq)
            # 如果句子序列小于 max_length 则进行填充
            elif sequence_length < size:
                seq = sequence + [padding] * (size - len(sequence))
            else:
                seq = sequence
            return seq, sequence_length

        sent = [self.identifier_to_token(word) for word in sentence]
        if alignment_size is None:
            return alignment(sent, self.max_length, self.identifier_to_token(self.PAD_TAG))
        else:
            return alignment(sent, alignment_size, self.identifier_to_token(self.PAD_TAG))

    def encoding_sentence(self, sentence, alignment=False, alignment_size=None):
        """
        编码句子。将字符句子转换成 token
        :param sentence: (list, mandatory) 字符句子。例如：['大', '众', '软', '件', '精', '彩', '分', '享']
        :param alignment: (bool, optional, default=False) 是否对齐序列。如果为 True。则在将输出指定序列的大小
        :param alignment_size: (int, optional, default=None) 指定对齐序列大小。如果为 None, 则使用默认对齐序列
        :return: (list) token 列表
        """
        if alignment:
            return self.alignment_sentence(sentence, alignment_size=alignment_size)
        else:
            seq = [self.word_to_token(word) for word in sentence]
            seq_length = len(seq)
            return seq, seq_length

    def encoding_identifier(self, identifiers, alignment=False, alignment_size=None):
        """
        编码标识符列表。将标识符转换成 token 列表
        :param identifiers: (list, mandatory) 标识符列表。例如：[B_book, I_book, I_book, E_book, O, O, O, O, O]
        :param alignment: (bool, optional, default=False) 是否对齐序列。如果为 True。则在将输出指定序列的大小
        :param alignment_size: (int, optional, default=None) 指定对齐序列大小。如果为 None, 则使用默认对齐序列
        :return: (list) token 列表
        """
        if alignment:
            return self.alignment_identifier(identifiers, alignment_size=alignment_size)
        else:
            seq = [self.identifier_to_token(identifier) for identifier in identifiers]
            seq_length = len(seq)
            return seq, seq_length

    def decoding_identifier(self, tokens):
        """
        解码标识符。将token列表的标识符转换成字符列表
        :param tokens: (list, mandatory) 标识符 token 列表
        :return: (list) 标识符列表
        """
        sequence = list()
        for token in tokens:
            # if token == self.identifier_to_token(self.PAD_TAG):
            #     continue
            sequence.append(self.token_to_identifier(token))

        return sequence


def test_module_func():
    texts = [list("无论魔兽3日后的发展会如何，我永远都是一名魔兽争霸3选手。"),
             list("桑普多vs卡塔尼推荐：3")]
    identifiers = [
        ['O', 'O', 'B_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'B_game', 'I_game', 'I_game', 'I_game', 'E_game', 'B_position', 'E_position', 'O'],
        ['B_organization', 'I_organization', 'E_organization', 'O', 'O', 'B_organization', 'I_organization',
         'E_organization', 'O', 'O', 'O', 'O']
    ]
    word_dict = WordDict()
    word_dict.fit(texts, identifiers)

    print("word_token: ", word_dict.word_token)
    print("identifier_token: ", word_dict.identifier_token)
    print("token_identifier: ", word_dict.token_identifier)
    print("max length: ", word_dict.max_length)

    seq, _ = word_dict.encoding_sentence(list("魔兽3"))
    assert seq == [3, 4, 5]
    seq, _ = word_dict.encoding_identifier(['B_game', 'I_game', 'E_game'])
    assert seq == [2, 3, 4]
    assert word_dict.decoding_identifier([2, 3, 4]) == ['B_game', 'I_game', 'E_game']

    output, seq_len = word_dict.encoding_sentence(list("魔兽3"), alignment=True)
    print("size:{}, sequence length:{}, alignment output:{}".format(len(output), seq_len, output))

    output, seq_len = word_dict.encoding_sentence(list("魔兽3") + ['3'] * 30, alignment=True)
    print("size:{}, sequence length:{}, alignment output:{}".format(len(output), seq_len, output))

    output, seq_len = word_dict.encoding_identifier(['B_game', 'I_game', 'E_game'], alignment=True)
    print("size:{}, sequence length:{}, alignment output:{}".format(len(output), seq_len, output))

    output, seq_len = word_dict.encoding_identifier(['B_game'] * 30, alignment=True)
    print("size:{}, sequence length:{}, alignment output:{}".format(len(output), seq_len, output))

    output = word_dict.decoding_identifier([3, 4, 5, 11, 11, 11])
    print("size: {} decoding identifier: {}".format(len(output), output))


if __name__ == '__main__':
    test_module_func()
