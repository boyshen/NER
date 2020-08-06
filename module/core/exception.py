# -*- encoding: utf-8 -*-
"""
@file: exception.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 模块主要包括：自定义异常、异常处理、异常捕获
# 自定义异常：
#   创建 AppException 异常类，继承python默认的异常Exception类。在此基础上自定义异常类型。
#   自定义异常定义异常等级、异常消息、异常模块、异常输出颜色。
#
# 捕获异常：
#   使用 python 装饰器机制，定义异常 exception_handling 异常处理装饰器，对可能抛出异常的模块使用该装饰器
#
# 异常处理：
#   对捕获的异常进行处理。可分为自定义异常和未知异常。
#   自定义异常则根据等级进行处理。warning 以及 error 等级的异常中断执行退出程序
#   未知异常则捕获异常消息，回溯出错代码块
"""

import functools
import traceback

try:
    from module.core.color import Color
except ModuleNotFoundError:
    from .color import Color


class AppException(Exception):
    """
    异常模块的父类
    """
    # 定义异常的级别，分别为 debug、warning、error 三种异常级别
    debug = "debug"
    warning = "warning"
    error = "error"

    # 定义每种异常级别的输出颜色。Color.green、Color.yellow、Color.red 为Color模块中的函数。
    # 这里的字典值value是一个函数，利用python的函数回调机制。
    color = {debug: Color.green,
             warning: Color.yellow,
             error: Color.red}

    def __init__(self):
        # 用于保存异常的模块名
        self.name = None

        # 用于定义异常模块的id
        # self.id = None

        # 用于保存异常的等级
        self.level = None

        # 用于保存异常的输出消息
        self.msg = None

    @staticmethod
    def print_format(name, level, msg, obj, tips=None):
        """
        异常信息的格式化输出
        :param name: (str, mandatory) 异常的名称
        :param level: (str, mandatory) 异常的级别
        :param msg: (str, mandatory) 异常信息
        :param obj: (str, mandatory) 其他需要输出的信息
        :param tips: (str, optional, default=None) 异常详细提醒信息
        :return: (str, mandatory) 格式化字符串
        """
        name = AppException.color[level](name)
        level = AppException.color[level](level)
        tips = tips if tips is not None else ""
        return "{}: [level:{}] {}:{} {}".format(name, level, msg, tips, obj)


class FileNotFoundException(AppException):
    """
    文件异常模块，继承 AppException ，用于当某个文件没有发现时候抛出该异常
    """

    def __init__(self, filename, level=AppException.error):
        """
        初始化操作
        :param filename: (str, mandatory) 文件名，报错的文件名
        :param level: (int, optional, default=error) 级别，报错类型的级别
        """
        super(FileNotFoundException, self).__init__()
        self.filename = filename
        self.level = level
        self.name = __class__.__name__
        self.msg = "No Such file or directory"

    def __str__(self):
        """
        __str__ 是 python 中返回对 Class 的叙述的字符串操作。用于返回当前异常信息
        其中AppException.color 为父类中定义的字典，字典key 为异常等级，value 为颜色函数回调。
        :return: (str) 异常信息
        """
        return AppException.print_format(self.name, self.level, self.msg, self.filename)


class NotFitException(AppException):
    """
    没有fit异常，用于对模型、词典没有进行fit时候抛出异常
    """

    def __init__(self, obj, level=AppException.error):
        """
        初始化，保存抛出异常的类名和等级
        :param obj: (str, mandatory) 异常的类名
        :param level: (str, optional, default=error) 异常的级别
        """
        self.obj = obj
        self.level = level
        self.name = __class__.__name__
        self.msg = "Object Not Fit"

    def __str__(self):
        return AppException.print_format(self.name, self.level, self.msg, self.obj)


class NullCharacterException(AppException):
    """
    空字符异常，用于定义输入文本时候，发现输入的为空字符则抛出该异常
    """

    def __init__(self, obj, tips=None, level=AppException.error):
        self.obj = obj
        self.tips = tips
        self.level = level
        self.name = __class__.__name__
        self.msg = "Null Character Exception"

    def __str__(self):
        return AppException.print_format(self.name, self.level, self.msg, self.obj, self.tips)


class ParameterError(AppException):
    """
    参数错误，用于检查输入的参数，如果参数错误则抛出该异常
    """

    def __init__(self, obj, tips=None, level=AppException.error):
        self.obj = obj
        self.tips = tips
        self.level = level
        self.name = __class__.__name__
        self.msg = "Parameter Error"

    def __str__(self):
        return AppException.print_format(self.name, self.level, self.msg, self.obj, self.tips)


class UnknownError(AppException):

    def __init__(self, obj, tips=None, level=AppException.error):
        self.obj = obj
        self.tips = tips
        self.level = level
        self.name = __class__.__name__
        self.msg = "Unknown Error"

    def __str__(self):
        return AppException.print_format(self.name, self.level, self.msg, self.obj, self.tips)


def handler(e, is_custom_exception):
    """
    异常处理的函数，如果为自定义的异常，则根据level进行判断是否退出程序，
    当 level 为 error 或 warning 时候则退出程序
    非自定义异常则进行回溯，打印保存的程序和提示信息。
    :param e: (Exception, mandatory) 捕获的异常对象
    :param is_custom_exception: (bool, mandatory) 是否为自定义的异常类型。
    :return:
    """
    print(e)
    if is_custom_exception:
        if e.level == AppException.error or e.level == AppException.warning:
            print(traceback.format_exc())
            exit()
    else:
        # 回溯异常的代码块
        print(traceback.format_exc())
        exit()


def exception_handling(func):
    """
    处理异常信息函数的装饰器。用于捕获异异常信息，并对异常信息进行处理。
    处理异常信息根据异常的等级而定，如果异常的等级为 error 或 warning，则退出程序。
    :param func: (func, mandatory) 异常函数
    :return:
    """

    @functools.wraps(func)
    def handling(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result

        except FileNotFoundException as e:
            handler(e, is_custom_exception=True)

        except NotFitException as e:
            handler(e, is_custom_exception=True)

        except NullCharacterException as e:
            handler(e, is_custom_exception=True)

        except ParameterError as e:
            handler(e, is_custom_exception=True)

        except UnknownError as e:
            handler(e, is_custom_exception=True)

        except Exception as e:
            handler(e, is_custom_exception=False)

    return handling
