# -*- encoding: utf-8 -*-
"""
@file: utils.py
@time: 2020/5/23 下午4:58
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 数据处理工具
"""

import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    from module.core.exception import exception_handling
    from module.core.exception import FileNotFoundException
    from module.core.exception import ParameterError
except ModuleNotFoundError:
    import sys

    sys.path.append('../../')
    from module.core.exception import exception_handling
    from module.core.exception import FileNotFoundException
    from module.core.exception import ParameterError


def template(e, epochs, step, steps, loss, accuracy=None, head='Train'):
    """
    网络训练输出模版
    """
    output = '{}: Epochs: {}/{}, Steps: {}/{}, Loss: {:.4f}, '.format(head, e, epochs, step, steps, loss)
    if accuracy is not None:
        output += 'Accuracy: {:.4f}, '.format(accuracy)

    return output


def batch(*args, batch_size):
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

    total_size = len(args[0]) // batch_size
    dataset = [[] for _ in range(len(args))]
    for i in range(total_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        for j in range(len(args)):
            dataset[j].append(args[j][start:end])

    return dataset, total_size


def shuffle(*args):
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


def split_valid_dataset(*args, ratio=0.2, is_shuffle=True):
    """
    划分验证数据集。
    :param args: (list, mandatory) 数据集。列表格式。
    :param ratio: (int, optional, default=0.2) 验证数据集比例。0～1范围内。
    :param is_shuffle: (bool, optional, default=True) 是否进行洗牌
    :return: (list) 训练数据集和验证数据集
    """
    if ratio > 1 or ratio < 0:
        raise ParameterError("dataset ratio must be is 0 ~ 1 range. actually get: {}".format(ratio))

    if is_shuffle:
        dataset = shuffle(*args)
    else:
        dataset = args

    sample_num = int(len(dataset[0]) * ratio)
    if sample_num == 0:
        sample_num = 1

    train_dataset, valid_dataset = list(), list()
    for data in dataset:
        valid_dataset.append(data[:sample_num])
        train_dataset.append(data[sample_num:])

    return train_dataset, valid_dataset


def computer_conv_size(input_size, kernel_size, strides, padding):
    """
    计算卷积输出尺寸
    :param input_size: (int, mandatory)  输入尺寸大小。可以是 height or width
    :param kernel_size: (int, mandatory) 卷积 kernel 大小。
    :param strides: (int, mandatory) 卷积 strides
    :param padding: (int, mandatory) 卷积填充大小
    :return: (int) 尺寸大小
    """
    if kernel_size is None:
        return None

    i = input_size
    f = kernel_size
    s = strides
    p = padding

    # if padding.lower() == 'same':
    #     new_input_size = int(i / s)
    #     p = (new_input_size - 1) * s + f - i
    # elif padding.lower() == 'valid':
    #     p = (i - f + 1) / s
    # else:
    #     p = 0

    return (i - f + 2 * p) // s + 1


class ComputerConvSize(tf.keras.layers.Layer):

    def __init__(self, kernel_size, strides, padding):
        super(ComputerConvSize, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs, **kwargs):
        pass


class DataUtils(object):

    def __init__(self):
        pass

    @staticmethod
    @exception_handling
    def read_text_data(text_file, handle_func=None, show_progress_bar=True):
        """
        读取文本数据
        :param text_file: (str, mandatory) 文本数据
        :param handle_func: (func, optional, default="None") 文本处理函数。需要对每行数据进行的处理。
        :param show_progress_bar: (bool, optional, default=True) 是否显示进度条
        :return: (list) 文本数据列表。
        """
        assert os.path.isfile(text_file), FileNotFoundException(text_file)

        data = list()
        with open(text_file, 'rb') as lines:
            data_lines = tqdm(lines) if show_progress_bar else lines

            for line in data_lines:
                line = str(line, encoding="utf-8").strip('\n').strip()

                if handle_func is not None:
                    line = handle_func(line)

                data.append(line)

        return data


class DictToObj(dict):
    """
    字典转对象。将字典 key 转换成对象
    """

    def __init__(self, *args, **kwargs):
        super(DictToObj, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            value = DictToObj(value)
        return value


class Writer(object):
    DIR_SEPARATOR = '/'

    def __init__(self):
        pass

    @staticmethod
    def check_path(file):
        """
        检查文件路径是否存在。不存在则创建
        :param file: (str, mandatory) 文件路径. 例如: '/data/file.pth'
        :return:
        """
        parameter = file.split(Writer.DIR_SEPARATOR)
        if len(parameter) > 1:
            path = file[:len(file) - len(parameter[-1])]
            if not os.path.exists(path):
                os.makedirs(path)

    @staticmethod
    def remove_file(file):
        """
        删除文件
        :param file: (str, mandatory) 删除文件
        :return: (bool)
        """
        if os.path.isfile(file):
            os.remove(file)
            return True
        return False


class Dictionary(object):
    START_TAG = "<START>"
    END_TAG = "<END>"
    UNK_TAG = "<UNK>"
    PAD_TAG = "<PAD>"

    def __init__(self):
        pass

    def save(self, file):
        """
        保存字典
        :param file: (str, mandatory) 保存文件名
        :return:
        """
        file = file.strip()

        Writer.check_path(file)

        pickle.dump(self, open(file, 'wb'))

        print("save dictionary success! File: ", file)

    @staticmethod
    @exception_handling
    def load(file):
        """
        加载字典
        :param file: (str, mandatory) 字典文件
        :return: (dictionary) 字典对象
        """
        assert os.path.isfile(file), FileNotFoundException(file)

        with open(file, 'rb') as f_read:
            dictionary = pickle.loads(f_read.read())

        return dictionary


class Features(object):
    """
    特征提取
    """

    def __init__(self):
        pass

    def save(self, file):
        """
        保存
        :param file: (str, mandatory) 保存文件名
        :return:
        """
        file = file.strip()

        Writer.check_path(file)

        pickle.dump(self, open(file, 'wb'))

        print("save dictionary success! File: ", file)

    @staticmethod
    @exception_handling
    def load(file):
        """
        加载
        :param file: (str, mandatory) 文件
        :return: (Features) 对象
        """
        assert os.path.isfile(file), FileNotFoundException(file)

        with open(file, 'rb') as f_read:
            dictionary = pickle.loads(f_read.read())

        return dictionary


class Generator(object):

    def __init__(self):
        pass

    @staticmethod
    @exception_handling
    def split_valid_dataset(*args, ratio=0.2):
        """
        划分验证数据集。
        :param args: (list, mandatory) 数据集。列表格式。
        :param ratio: (int, optional, default=0.2) 验证数据集比例。0～1范围内。
        :return: (list) 训练数据集和验证数据集
        """
        if ratio > 1 or ratio < 0:
            raise ParameterError("dataset ratio must be is 0 ~ 1 range. actually get: {}".format(ratio))

        dataset = Generator.shuffle(*args)

        sample_num = int(len(dataset[0]) * ratio)
        if sample_num == 0:
            sample_num = 1

        train_dataset, valid_dataset = list(), list()
        for data in dataset:
            train_dataset.append(data[sample_num:])
            valid_dataset.append(data[:sample_num])

        return train_dataset, valid_dataset

    @staticmethod
    @exception_handling
    def shuffle(*args):
        """
        洗牌
        :param args:(list, tuple) 需要洗牌的数据集
        :return:
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


class Config(object):
    """
    读取json配置文件信息，并转换成对象格式
    """

    def __init__(self, file):
        self.file = file
        self.__read_json_info__()

    def __read_json_info__(self):
        assert os.path.isfile(self.file), FileNotFoundException(self.file)
        with open(self.file, 'r') as f:
            data = json.load(f)

        self.data = DictToObj(data)

    def get_data(self):
        return self.data


def test_module_func(module_name):
    if module_name == DataUtils.read_text_data.__name__:
        def handle(sent):
            line_data = list()
            for word in sent.split(' '):
                if word == '' or word == " ":
                    continue
                line_data.append(word)
            return line_data

        file = '../../data/data/A2_0.wav.trn'
        data = DataUtils.read_text_data(file, handle, show_progress_bar=False)

        with open(file, 'rb') as lines:
            for i, line in enumerate(lines):
                line = str(line, encoding='utf8').strip('\n').strip()
                assert ' '.join(data[i]) == line

        # print("test module: {} is ok!".format(DataUtils.read_text_data.__name__))

    elif module_name == template.__name__:
        output = template(1, 10, 1, 2, 0.45678, accuracy=0.9888)
        print(output)
        # print("test module:{} is ok!".format(template.__name__))

    elif module_name == DictToObj.__name__:
        test_dict = {'a': 1, 'b': 2, 'c': {'c_key_1': 'c_value', 'c_key_2': 2}, 'd': 'd_value'}
        test_obj = DictToObj(test_dict)
        assert test_obj.a == 1
        assert test_obj.b == 2
        assert test_obj.c.c_key_1 == 'c_value'
        assert test_obj.c.c_key_2 == 2
        assert test_obj.d == 'd_value'

        print('a: ', test_obj.a)
        print('b: ', test_obj.b)
        print('c_key_1: ', test_obj.c.c_key_1)
        print('c_key_2: ', test_obj.c.c_key_2)
        print('d:', test_obj.d)

        # print("test module: {} is ok!".format(DictToObj.__name__))

    elif module_name == Generator.shuffle.__name__:
        def _print(s_data, s_label):
            for d, l in zip(s_data, s_label):
                print("data: ", d)
                print("label: ", l)
                print()

        data = ["a", "b", "c", "d", "e"]
        label = [1, 2, 3, 4, 5]
        dataset = (data, label)

        (shuffle_data, shuffle_label) = Generator.shuffle(*dataset)
        print("shuffle data")
        _print(shuffle_data, shuffle_label)

    elif module_name == Generator.split_valid_dataset.__name__:
        def _print(s_data, s_label):
            for d, l in zip(s_data, s_label):
                print("data: ", d)
                print("label: ", l)
                print()

        data = ["a", "b", "c", "d", "e"]
        label = [1, 2, 3, 4, 5]
        dataset = (data, label)

        (train_data, train_label), (valid_data, valid_label) = Generator.split_valid_dataset(*dataset, ratio=0.5)
        print("split train dataset: ")
        _print(train_data, train_label)

        print("split valid dataset:")
        _print(valid_data, valid_label)

    elif module_name == Config.__name__:
        data = {'test': 1, 'test1': 'hello'}
        file = './test.json'
        with open(file, 'w') as f:
            json.dump(data, f)

        config = Config(file).get_data()
        assert config.test == 1
        assert config.test1 == 'hello'

        print("config test: ", config.test)
        print("config test1: ", config.test1)
        os.remove(file)

    elif module_name == computer_conv_size.__name__:
        # conv = tf.keras.Sequential([
        #     tf.keras.layers.Conv1D(32, 3, strides=1, padding='same'),
        #     tf.keras.layers.MaxPool1D(pool_size=3, strides=2)
        # ])
        # images = tf.random.uniform((1, 128, 256))
        # output = conv(images)
        #
        # conv_size = computer_conv_size(128, 3, 2, 0)
        # assert output.shape[1] == conv_size
        # print("1D conv size: {}, output shape: {}".format(conv_size, output.shape))

        # images = tf.random.uniform((1, 126, 126, 3))
        # conv2d = tf.keras.layers.Conv2D(32, 3, strides=2, padding='valid')
        # output = conv2d(images)
        # conv_size = computer_conv_size(126, 3, 2, 0)
        # print("2D conv size: {}, output shape: {}".format(conv_size, output.shape))

        conv_length = tf.keras.layers.Lambda(lambda x: computer_conv_size(x, kernel_size=3, strides=2, padding=0))
        output_length = tf.keras.Sequential([conv_length, conv_length, conv_length])
        images = tf.random.uniform((4, 1), minval=256, maxval=257, dtype=tf.int32)
        output = output_length(images)
        print(output)

    elif module_name == 'split_valid_dataset_func':
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = [[11, 22, 33], [44, 55, 66], [77, 88, 99]]
        z = ['a', 'b', 'c']

        train_dataset, valid_dataset = split_valid_dataset(x, y, z, ratio=0.1, is_shuffle=True)
        for train_x, train_y, train_z in zip(train_dataset[0], train_dataset[1], train_dataset[2]):
            print("train x: ", train_x)
            print("train y: ", train_y)
            print("train z: ", train_z)
            print()

        train_dataset, valid_dataset = split_valid_dataset(x, y, z, ratio=0.1, is_shuffle=False)
        for train_x, train_y, train_z in zip(train_dataset[0], train_dataset[1], train_dataset[2]):
            print("No shuffle train x: ", train_x)
            print("No shuffle train y: ", train_y)
            print("No shuffle train z: ", train_z)
            print()

    elif module_name == batch.__name__:
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6]]
        y = [[11, 22, 33], [44, 55, 66], [77, 88, 99], [11, 33, 55], [22, 44, 66]]
        z = ['a', 'b', 'c', 'd', 'e']
        batch_dataset, sample_size = batch(x, y, z, batch_size=3)
        print("sample size: ", sample_size)
        for train_x, train_y, train_z in zip(batch_dataset[0], batch_dataset[1], batch_dataset[2]):
            print("train x: ", train_x)
            print("train y: ", train_y)
            print("train z: ", train_z)
            print()


def main():
    # test_module_func(DataUtils.read_text_data.__name__)
    # test_module_func(template.__name__)
    # test_module_func(DictToObj.__name__)
    # test_module_func(Generator.shuffle.__name__)
    # test_module_func(Generator.split_valid_dataset.__name__)
    # test_module_func(Config.__name__)
    # test_module_func(computer_conv_size.__name__)
    # test_module_func('split_valid_dataset_func')
    test_module_func(batch.__name__)


if __name__ == "__main__":
    main()
