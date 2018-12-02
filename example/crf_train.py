#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午10:15
# @Author  : 林利芳
# @File    : crf_train.py
import multiprocessing
from pyseq.linear_crf import CRF


def main():
	model = CRF(mp=1)
	model.fit(data_file='ned.train', template_file='templatesimple.txt', model_file="model")


if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
