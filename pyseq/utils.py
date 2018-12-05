#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午4:35
# @Author  : 林利芳
# @File    : utils.py
import os
import re
from scipy.misc import logsumexp
import numpy as np
import pickle


def read_template(filename):
    """
    读取模板特征
    :param filename: 模板文件  U08:%x[-1,0]/%x[0,0]
    :return: tp_list [[U00,[0,0],[1,0]]]
    """
    if not os.path.isfile(filename):
        print("模板文件[{}]不存在!".format(filename))
        exit()
    tp_list = []
    pattern = re.compile(r'\[-?\d+,-?\d+\]')  # -?[0-9]*
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            fl = line.find("#")
            if fl != -1:  # 移除注释
                line = line[0:fl]
            if valid_template_line(line) is False:
                continue
            fl = line.find(":")
            if fl != -1:  # just a symbol 模板符号 -> U00:%x[0,0]
                each_list = [line[0:fl]]
            else:
                each_list = [line[0]]

            for a in list(pattern.finditer(line)):
                loc_str = line[a.start() + 1:a.end() - 1]
                loc = loc_str.split(",")
                each_list.append(loc)
            tp_list.append(each_list)
    print("有效模板数量:", len(tp_list))
    return tp_list


def valid_template_line(line):
    if_valid = True
    if line.count("[") != line.count("]"):
        if_valid = False
    if "UuBb".find(line[0]) == -1:
        if_valid = False
    if if_valid is False:
        print("模板错误:", line)
    return if_valid


def read_data(filename):
    """
    读取训练数据
    :param filename:
    :return:
    """
    if not os.path.isfile(filename):
        print("文件[{}]不存在".format(filename))
        exit()
    texts = []
    labels = []
    text = []
    label = []
    oby_dic = dict()
    oby_id = 0
    space_cnt = 0
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            # 一行结尾符 空行 \n
            if len(line) == 0:
                if len(text) > 0:
                    texts.append(text)
                    labels.append(label)
                text = []
                label = []
            else:
                chunk = line.split()
                if space_cnt == 0:
                    space_cnt = len(chunk)
                else:
                    if len(chunk) != space_cnt:
                        print("输入错误:\t", line)
                text.append([chunk[0]])
                y_label = chunk[-1]
                if oby_dic.get(y_label) is None:
                    oby_dic[y_label] = oby_id
                    label.append(oby_id)
                    oby_id += 1
                else:
                    label.append(oby_dic[y_label])
    # 最后一行
    if len(text) > 0:
        texts.append(text)
        labels.append(label)

    oys = labels
    seq_num = len(oys)
    seq_lens = [len(x) for x in texts]
    num_k = len(oby_dic)
    y2label = dict([(oby_dic[key], key) for key in oby_dic.keys()])
    print("标签数量:", num_k)
    return texts, seq_lens, oys, seq_num, num_k, oby_dic, y2label


def load_data(filename):
    """
    读取训练数据
    :param filename:
    :return:
    """
    if not os.path.isfile(filename):
        print("文件[{}]不存在".format(filename))
        exit()
    texts = []
    labels = []
    text = []
    label = []
    space_cnt = 0
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            # 一行结尾符 空行 \n
            if len(line) == 0:
                if len(text) > 0:
                    texts.append(text)
                    labels.append(label)
                text = []
                label = []
            else:
                chunk = line.split()
                if space_cnt == 0:
                    space_cnt = len(chunk)
                else:
                    if len(chunk) != space_cnt:
                        print("输入错误:\t", line)
                        continue
                text.append([chunk[0]])
                y_label = chunk[-1]
                label.append(y_label)
    # 最后一行
    if len(text) > 0:
        texts.append(text)
        labels.append(label)
    return texts, labels




def save_model(model, model_file):
    with open(model_file, 'wb') as fp:
        pickle.dump(model, fp)


def load_model(model_file):
    """
    加载模型
    :param model_file:
    :return:
    """
    if not os.path.isfile(model_file):
        print("Error: 模型文件不存在!")
        return -1
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def output_file(x_test, y_test, max_ys, res_file):
    """
    输出文件
    :param x_test:
    :param y_test:
    :param max_ys:
    :param res_file:
    :return:
    """
    if res_file is None:
        return 0
    result = []
    for seq_id, text in enumerate(x_test):
        sentence = []
        for loc_id in range(len(text)):
            line = ""
            for x in text[loc_id]:
                line += x + '\t'
            if y_test:
                line += y_test[seq_id][loc_id] + "\t"
            line += max_ys[seq_id][loc_id] + '\n'
            sentence.append(line)
        result.append(''.join(sentence))
    with open(res_file, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(result))
    return 0


def random_param(uf_num, bf_num):
    theta = np.ones(uf_num + bf_num)
    return theta
