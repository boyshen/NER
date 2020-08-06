# -*- encoding: utf-8 -*-
"""
@file: word_dict.py
@time: 2020/5/6 下午5:08
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 1. fit 拟合。创建词典和标识符词典
# 2. _word_to_token 将中文字转换成token。
# 3. _identifier_to_token 将文本转换成token的形式。
# 4. _token_to_identifier 将token格式的标识符转换成字符形式的标示符。
"""
from tqdm import tqdm
from collections import Counter
from module.core.exception import exception_handling, ParameterError, NotFitException
from module.core.utils import Dictionary


class WordDict(Dictionary):

    @exception_handling
    def __init__(self):
        super(WordDict, self).__init__()

        self.word_token = dict()
        # self.token_word = {}
        self.UNK = 0

        self.identifier_token = {
            self.START_TAG: 0,
            self.END_TAG: 1
        }
        self.token_identifier = dict()

        self.__is_fit = False

    @exception_handling
    def fit(self, data, identifiers):
        """
        拟合。根据数据和标示符号拟合词典。初始化 word_token、token_word、identifier_token
        :param data: (list or tuple, mandatory) 数据集。字符列表或字符元祖格式。
            例如：['大众软件, 精彩分享', ...]
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
            word_counter.update(list(sent))
            identifier_counter.update(identifier)

        for word, _ in word_counter.items():
            self.word_token[word] = len(self.word_token)
        self.word_token[self.UNK_TAG] = len(self.word_token)
        self.UNK = self.word_token[self.UNK_TAG]
        # self.token_word = {token: word for word, token in self.word_token.items()}

        for identifier, _ in identifier_counter.items():
            self.identifier_token[identifier] = len(self.identifier_token)
        self.token_identifier = {token: identifier for identifier, token in self.identifier_token.items()}

        self.__is_fit = True

        print("word num: {}, identifier num: {}".format(len(self.word_token), len(self.identifier_token)))

    @exception_handling
    def word_to_token(self, word):
        """
        将word转换成token
        :param word: (str, mandatory) 字或词
        :return: (int) 返回 token 或 UNK=0
        """
        if not self.__is_fit:
            raise NotFitException("NERDictionary not fit")

        try:
            token = self.word_token[word]
        except KeyError:
            token = self.UNK

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

    @exception_handling
    def encoding_sentence(self, sent):
        """
        编码文本。将字符串文本转换成token形式
        :param sent: (str, mandatory) 中文句子或文本。例如：'大众软件, 精彩分享'
        :return: (int list) 列表token
        """
        sent_token = [self.word_to_token(word) for word in sent]
        return sent_token

    @exception_handling
    def encoding_identifier(self, identifiers):
        """
        编码标识符。将字符列表形式的标识符转换成token形式
        :param identifiers: (str of list, mandatory) 标识符。例如：[B_book, I_book, I_book, E_book, O, O, O, O, O]
        :return: (int list) 列表token
        """
        identifier_token = [self.identifier_to_token(identifier) for identifier in identifiers]
        return identifier_token

    @exception_handling
    def decoding_identifier(self, identifiers_token):
        """
        解码标识符。将token格式的标识符转换成字符形式的标示符。
        :param identifiers_token: (list or tuple, mandatory) token格式的标识符列表。例如：[0, 1, 3, 4, 0]
        :return: (list) 字符列表。
        """
        identifiers = [self.token_to_identifier(token) for token in identifiers_token]
        return identifiers


def test():
    texts = ["无论魔兽3日后的发展会如何，我永远都是一名魔兽争霸3选手。",
             "桑普多vs卡塔尼推荐：3"]
    identifiers = [
        ['O', 'O', 'B_game', 'I_game', 'E_game', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'B_game', 'I_game', 'I_game', 'I_game', 'E_game', 'B_position', 'E_position', 'O'],
        ['B_organization', 'I_organization', 'E_organization', 'O', 'O', 'B_organization', 'I_organization',
         'E_organization', 'O', 'O', 'O', 'O']
    ]
    ner_dict = WordDict()
    ner_dict.fit(texts, identifiers)

    print("source text: ", texts[0])
    print("source identifier: ", identifiers[0])
    print("word_token: ", ner_dict.word_token)
    print("identifier_token: ", ner_dict.identifier_token)
    print("token_identifier: ", ner_dict.token_identifier)

    print("encoding sentence: ", ner_dict.encoding_sentence(list(texts[0])))

    identifier = ['O', 'O', 'B_game', 'I_game', 'E_game', 'O', 'O', 'O']
    print("encoding identifier: ", ner_dict.encoding_identifier(identifier))

    identifier_token = [0, 1, 2, 3, 4, 1]
    print("decoding sentence: ", ner_dict.decoding_identifier(identifier_token))


if __name__ == "__main__":
    test()
