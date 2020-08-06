# -*- encoding: utf-8 -*-
"""
@file: color.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 颜色代码
"""


class Color(object):
    # 前景色
    class Fore(object):
        black = 30  # 黑色
        red = 31  # 红色
        green = 32  # 绿色
        yellow = 33  # 黄色
        blue = 34  # 蓝色
        purple = 35  # 紫色
        cyan = 36  # 青色
        white = 37  # 白色

    # 背景色
    class Back(object):
        black = 40  # 黑色
        red = 41  # 红色
        green = 42  # 绿色
        yellow = 43  # 黄色
        blue = 44  # 蓝色
        purple = 45  # 紫色
        cyan = 46  # 青色
        white = 47  # 白色
        default = 48

    # 模式
    class Mode(object):
        default = 0  # 终端默认设置
        bold = 1  # 高亮显示
        underline = 2  # 使用下划线
        blink = 3  # 闪烁

    end = 0

    # c_start = "\033["
    # c_end = "m"

    def __init__(self):
        pass

    @staticmethod
    def color_format(string, fore, back, mode):
        return "\033[{};{};{}m{}\033[{}m".format(mode, fore, back, string, Color.end)

    @staticmethod
    def red(string):
        return Color.color_format(string,
                                  Color.Mode.default,
                                  Color.Fore.red,
                                  Color.Back.default)

    @staticmethod
    def yellow(string):
        return Color.color_format(string,
                                  Color.Mode.default,
                                  Color.Fore.yellow,
                                  Color.Back.default)

    @staticmethod
    def green(string):
        return Color.color_format(string,
                                  Color.Mode.default,
                                  Color.Fore.green,
                                  Color.Back.default)
