#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午10:15
# @Author  : 林利芳
# @File    : crf_train.py
from pyseq.linear_crf import CRF


def main():
	model = CRF(mp=1)
	model.fit(data_file='train.txt', template_file='templates.txt', model_file="model")
	model.predict(data_file='test.txt', result_file='res.txt')


if __name__ == "__main__":
	main()
