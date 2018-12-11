#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午10:15
# @Author  : 林利芳
# @File    : crf_train.py
from pyseq.linear_crf import CRF
from pyseq.utils import load_data


def main():
    x_train, y_train = load_data(filename='model_301_JZ_NAME')
    x_test, y_test = load_data(filename='model_301_JZ_NAME')
    model = CRF()
    model.fit(x_train, y_train, template_file='templates.txt', model_file="model")
    a = model.predict(x_test, y_test, model_file="model", res_file='res.txt')


if __name__ == "__main__":
    main()
