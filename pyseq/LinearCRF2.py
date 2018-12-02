# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:31:31 2014

@author: Huang,Zheng

Linear CRF in Python

License (BSD)
==============
Copyright (c) 2013, Huang,Zheng.  huang-zheng@sjtu.edu.cn
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import time
import pickle
import numpy
from scipy.misc import logsumexp
import os
import codecs
import re
import datetime
import ctypes
import multiprocessing
from multiprocessing import Process, Queue
import sys

_gradient = None  # global variable used to store the gradient calculated in liklihood function.


def log_sum_exp_vec_mat(log_a, log_m):
	return logsumexp(log_a + log_m, axis=1)


def log_sum_exp_mat_vec(log_m, logb):
	return logsumexp(log_m + logb[:, numpy.newaxis], axis=0)


def valid_template_line(line):
	if_valid = True
	if line.count("[") != line.count("]"):
		if_valid = False
	if "UuBb".find(line[0]) == -1:
		if_valid = False
	if if_valid is False:
		print("error in template file:", line)
	return if_valid


def read_data(data_file):
	if not os.path.isfile(data_file):
		print("数据不存在")
		exit()
	texts = []
	labels = []
	text = []
	label = []
	oby_dic = dict()
	oby_id = 0
	space_cnt = 0
	with open(data_file, 'r', encoding='utf-8') as fp:
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

	texts = texts
	oys = labels
	seq_num = len(oys)
	seq_lens = [len(x) for x in texts]
	num_classify = len(oby_dic)
	y2label = dict([(oby_dic[key], key) for key in oby_dic.keys()])
	print("标签数量:", num_classify)
	return texts, seq_lens, oys, seq_num, num_classify, oby_dic, y2label


def read_template(tmp_file):
	"""
	读取模板
	:param tmp_file: 模板文件
	:return:
	"""
	if not os.path.isfile(tmp_file):
		print("无法找到模板!")
		exit()
	tp_list = []
	pattern = re.compile(r'\[-?\d+,-?\d+\]')  # -?[0-9]*

	with open(tmp_file, 'r', encoding='utf-8') as fp:
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


def expand_observation(sentence, loc_id, tp):
	"""
	expend the observation at loc_id for sequence
	:param sentence: 
	:param loc_id: 
	:param tp: tp = ['U01',[0,0],[1,0]]
	:return: 
	"""
	line = tp[0]
	for li in tp[1::]:
		row = loc_id + int(li[0])
		col = int(li[1])
		if len(sentence) > row >= 0:
			if len(sentence[row][col]) > col >= 0:
				line += ":" + sentence[row][col]
	return line


def process_features(tp_list, texts, num_classify, fd=1):
	"""
	特征提取
	:param tp_list:模板
	:param texts:
	:param num_classify:标签数量 int
	:param fd:特征频次
	:return:
	"""
	uf_obs = dict()
	bf_obs = dict()
	for ti, tp in enumerate(tp_list):  # for each template line
		for text in texts:
			for lid in range(text):
				obx = expand_observation(text, lid, tp)

				if obx[0] == "B":
					if bf_obs.get(obx) is None:
						bf_obs[obx] = 1
					else:
						t_val = bf_obs[obx]
						bf_obs[obx] = t_val + 1

				if obx[0] == "U":
					if uf_obs.get(obx) is None:
						uf_obs[obx] = 1
					else:
						t_val = uf_obs[obx]
						uf_obs[obx] = t_val + 1

	if fd >= 2:  # 移除频次小于2的特征
		uf_obs = {k: v for k, v in uf_obs.items() if v >= fd}
		bf_obs = {k: v for k, v in bf_obs.items() if v >= fd}

	uf_num, bf_num = 0, 0
	for obx in bf_obs.keys():
		bf_obs[obx] = bf_num
		bf_num += num_classify * num_classify
	for obx in uf_obs.keys():
		uf_obs[obx] = uf_num
		uf_num += num_classify
	return uf_obs, bf_obs, uf_num, bf_num


def cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num):
	"""
	speed up the feature calculation
	calculate the on feature functions 
	"""
	uon = []
	bon = []
	for sid in range(seq_num):  # for each training sequence.
		seq_uon = []
		seq_bon = []
		for lid in range(len(texts[sid])):
			l_uon = []
			l_bon = []
			for ti, tp in enumerate(tp_list):  # for each template line
				obx = expand_observation(texts, sid, lid, tp)
				if tp[0][0] == "B":
					fid = bf_obs.get(obx)
					# print fid
					if fid is not None:
						l_bon.append(fid)
				if tp[0][0] == "U":
					fid = uf_obs.get(obx)
					if fid is not None:
						l_uon.append(fid)
			seq_uon.append(l_uon)
			seq_bon.append(l_bon)
		uon.append(seq_uon)
		bon.append(seq_bon)
	return uon, bon


def cal_observe_on_loc(uon, bon, seq_num, mp):
	"""
	speed up the feature calculation (muliprocessing)
	calculate the on feature list and location 
	"""
	u_len = 0
	loc_len = 0
	for a in uon:
		loc_len += len(a)
		for b in a:
			u_len += len(b)
	if sys.platform == "win32" and mp == 1:  # windows system need shared memory to do multiprocessing
		uon_arr = multiprocessing.Array('i', u_len)
		uon_seq_sta = multiprocessing.Array('i', seq_num)
		uon_loc_sta = multiprocessing.Array('i', loc_len)
		uon_loc_end = multiprocessing.Array('i', loc_len)
	else:
		uon_arr = numpy.zeros((u_len,), dtype=numpy.int)
		uon_seq_sta = numpy.zeros((seq_num,), dtype=numpy.int)
		uon_loc_sta = numpy.zeros((loc_len,), dtype=numpy.int)
		uon_loc_end = numpy.zeros((loc_len,), dtype=numpy.int)

	uid = 0
	seq_i = 0
	loci = 0
	for seq in uon:  # for each training sequence.
		uon_seq_sta[seq_i] = loci
		for loco in seq:
			uon_loc_sta[loci] = uid
			for aon in loco:
				uon_arr[uid] = aon
				uid += 1
			uon_loc_end[loci] = uid
			loci += 1
		seq_i += 1

	b_len = 0
	loc_len = 0
	for a in bon:
		loc_len += len(a)
		for b in a:
			b_len += len(b)
	# u_len = sum[(len(b)) for b in a for a in uon]

	if sys.platform == "win32" and mp == 1:  # windows system need shared memory to do multiprocessing
		bon_arr = multiprocessing.Array('i', u_len)
		bon_seq_sta = multiprocessing.Array('i', seq_num)
		bon_loc_sta = multiprocessing.Array('i', loc_len)
		bon_loc_end = multiprocessing.Array('i', loc_len)
	else:
		bon_arr = numpy.zeros((u_len,), dtype=numpy.int)
		bon_seq_sta = numpy.zeros((seq_num,), dtype=numpy.int)
		bon_loc_sta = numpy.zeros((loc_len,), dtype=numpy.int)
		bon_loc_end = numpy.zeros((loc_len,), dtype=numpy.int)

	bid = 0
	seq_i = 0
	loci = 0
	for seq in bon:  # for each training sequence.
		bon_seq_sta[seq_i] = loci
		for loco in seq:
			bon_loc_sta[loci] = bid
			for aon in loco:
				bon_arr[bid] = aon
				bid += 1
			bon_loc_end[loci] = bid
			loci += 1
		seq_i += 1
	return uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta, bon_loc_end


def cal_FSS(texts, oys, uon, bon, uf_num, bf_num, seq_num, num_classify, y0):
	fss = numpy.zeros((uf_num + bf_num))
	fssb = fss[0:bf_num]
	fssu = fss[bf_num:]
	for i in range(seq_num):
		for li in range(len(texts[i])):
			for ao in uon[i][li]:
				fssu[ao + oys[i][li]] += 1.0
			for ao in bon[i][li]:
				if li == 0:  # the first , yt-1=y0
					fssb[ao + oys[i][li] * num_classify + y0] += 1.0
				else:
					fssb[ao + oys[i][li] * num_classify + oys[i][li - 1]] += 1.0
	return fss


def random_param(uf_num, bf_num):
	theta = numpy.ones(uf_num + bf_num)
	return theta


def regularity(theta, regtype=0, sigma=1.0):
	if regtype == 0:
		regular = 0
	elif regtype == 1:
		regular = numpy.sum(numpy.abs(theta)) / sigma
	else:
		v = sigma ** 2
		v2 = v * 2
		regular = numpy.sum(numpy.dot(theta, theta)) / v2
	return regular


def regularity_deriv(theta, regtype=0, sigma=1.0):
	if regtype == 0:
		regular_deriv = 0
	elif regtype == 1:
		regular_deriv = numpy.sign(theta) / sigma
	else:
		v = sigma ** 2
		regular_deriv = theta / v
	return regular_deriv


def log_m_array(seq_len, auon, abon, num_classify, theta_u, theta_b):
	# log_m_list (n, num_classify, num_classify ) --> (sequence length, Yt, Yt-1)
	m_list = []
	for li in range(seq_len):
		fv = numpy.zeros((num_classify, num_classify))
		for ao in auon[li]:
			fv += theta_u[ao:ao + num_classify][:, numpy.newaxis]
		for ao in abon[li]:
			fv += theta_b[ao:ao + num_classify * num_classify].reshape((num_classify, num_classify))
		m_list.append(fv)

	for i in range(0, num_classify):  # set the emerge function for ~y(0) to be -inf.
		m_list[0][i][1:] = - float("inf")
	return m_list


def logM_sa(seq_len, seq_id, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta,
			bon_loc_end, num_classify, theta_u, theta_b):
	# log_m_list (n, num_classify, num_classify ) --> (sequence length, Yt, Yt-1)
	m_list = []
	i_loc = uon_seq_sta[seq_id]
	i_locb = bon_seq_sta[seq_id]
	for li in range(seq_len):
		fv = numpy.zeros((num_classify, num_classify))
		for i in range(uon_loc_sta[i_loc + li], uon_loc_end[i_loc + li]):
			ao = uon_arr[i]
			fv += theta_u[ao:ao + num_classify][:, numpy.newaxis]
		for i in range(bon_loc_sta[i_locb + li], bon_loc_end[i_locb + li]):
			ao = bon_arr[i]
			fv += theta_b[ao:ao + num_classify * num_classify].reshape((num_classify, num_classify))
		m_list.append(fv)

	for i in range(0, num_classify):
		m_list[0][i][1:] = - float("inf")
	return m_list


def cal_log_alphas(m_list):
	log_alpha = m_list[0][:, 0]  # alpha(1)
	log_alphas = [log_alpha]
	for logM in m_list[1:]:
		log_alpha = log_sum_exp_vec_mat(log_alpha, logM)
		log_alphas.append(log_alpha)
	return log_alphas


def cal_log_betas(m_list):
	log_beta = numpy.zeros_like(m_list[-1][:, 0])
	log_betas = [log_beta]
	for logM in m_list[-1:0:-1]:
		log_beta = log_sum_exp_mat_vec(logM, log_beta)
		log_betas.append(log_beta)
	return log_betas[::-1]


def likelihood_standalone(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma):
	"""conditional log likelihood log p(Y|X)"""
	likelihood = numpy.dot(fss, theta)
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	# likelihood=0.0
	for seq_id in range(seq_num):
		log_m_list = log_m_array(seq_lens[seq_id], uon[seq_id], bon[seq_id], num_classify, theta_u, theta_b)
		log_z = logsumexp(cal_log_alphas(log_m_list)[-1])
		likelihood -= log_z
	return likelihood - regularity(theta, regtype, sigma)


def likelihood_thread_o(seq_lens, uon, bon, theta_u, theta_b, seq_num, num_classify, uf_num, bf_num, start, end, que):
	likelihood = 0.0
	for seq_id in range(start, end):
		log_m_list = log_m_array(seq_lens[seq_id], uon[seq_id], bon[seq_id], num_classify, theta_u, theta_b)
		log_z = logsumexp(cal_log_alphas(log_m_list)[-1])
		likelihood -= log_z
	que.put(likelihood)


def likelihood_thread(seq_lens, theta, seq_num, num_classify, uf_num, bf_num, start, end, que1, que2, ns):
	uon = ns.uon
	bon = ns.bon
	grad = numpy.zeros(uf_num + bf_num)
	likelihood = 0
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	for si in range(start, end):
		log_m_list = log_m_array(seq_lens[si], uon[si], bon[si], num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		likelihood -= log_z
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			for ao in uon[si][i]:
				grad_u[ao:ao + num_classify] -= p_yi
			for ao in bon[si][i]:
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))
	que1.put(likelihood)
	que2.put(grad)


def likelihood_thread_simple(seq_lens, uon, bon, theta, num_classify, uf_num, bf_num, que1, que2):
	grad = numpy.zeros(uf_num + bf_num)
	likelihood = 0
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	for si in range(len(seq_lens)):
		log_m_list = log_m_array(seq_lens[si], uon[si], bon[si], num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		likelihood -= log_z
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			for ao in uon[si][i]:
				grad_u[ao:ao + num_classify] -= p_yi
			for ao in bon[si][i]:
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))
	que1.put(likelihood)
	que2.put(grad)


def likelihood_thread_sa(seq_lens, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta,
						 bon_loc_end, theta, start, end, seq_num, num_classify, uf_num, bf_num, que1, que2):
	grad = numpy.zeros(uf_num + bf_num)
	likelihood = 0
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	for si in range(start, end):
		log_m_list = logM_sa(seq_lens[si], si, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta,
							 bon_loc_sta, bon_loc_end, num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		likelihood -= log_z
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			iloc = uon_seq_sta[si]
			for it in range(uon_loc_sta[iloc + i], uon_loc_end[iloc + i]):
				ao = uon_arr[it]
				grad_u[ao:ao + num_classify] -= p_yi

			iloc = bon_seq_sta[si]
			for it in range(bon_loc_sta[iloc + i], bon_loc_end[iloc + i]):
				ao = bon_arr[it]
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))

	que1.put(likelihood)
	que2.put(grad)


def likelihood_multithread_o(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num):
	# multithread version of likelihood
	"""conditional log likelihood log p(Y|X)"""
	likelihood = numpy.dot(fss, theta)
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	que = Queue()
	np = 0
	sub_processes = []
	core_num = multiprocessing.cpu_count()
	# core_num=1
	if core_num > 1:
		chunk = seq_num / core_num + 1
	else:
		chunk = seq_num
	start = 0
	while start < seq_num:
		end = start + chunk
		if end > seq_num:
			end = seq_num
		args = (seq_lens, uon, bon, theta_u, theta_b, seq_num, num_classify, uf_num, bf_num, start, end, que)
		p = Process(target=likelihood_thread, args=args)
		p.start()
		np += 1
		sub_processes.append(p)
		start += chunk
	for i in range(np):
		likelihood += que.get()
	while sub_processes:
		sub_processes.pop().join()
	return likelihood - regularity(theta)


def likelihood_mp(seq_lens, fss, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma, ns):
	global _gradient
	grad = numpy.array(fss, copy=True)  # data distribution
	likelihood = numpy.dot(fss, theta)
	que1 = Queue()  # for the likelihood output
	que2 = Queue()  # for the gradient output
	np = 0
	sub_processes = []
	core_num = multiprocessing.cpu_count()
	if core_num > 1:
		chunk = seq_num / core_num + 1
	else:
		chunk = seq_num
	start = 0
	while start < seq_num:
		end = start + chunk
		if end > seq_num:
			end = seq_num
		args = (seq_lens, theta, seq_num, num_classify, uf_num, bf_num, start, end, que1, que2, ns)
		p = Process(target=likelihood_thread, args=args)
		p.start()
		np += 1
		sub_processes.append(p)
		start += chunk
	for i in range(np):
		likelihood += que1.get()
	for i in range(np):
		grad += que2.get()
	while sub_processes:
		sub_processes.pop().join()
	grad -= regularity_deriv(theta, regtype, sigma)
	_gradient = grad
	return likelihood - regularity(theta, regtype, sigma)


# """simply use windows multiprocess, pickle everything"""
def likelihood_mp_simple(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma):
	global _gradient
	grad = numpy.array(fss, copy=True)  # data distribution
	likelihood = numpy.dot(fss, theta)
	que1 = Queue()  # for the likelihood output
	que2 = Queue()  # for the gradient output
	np = 0
	sub_processes = []
	core_num = multiprocessing.cpu_count()
	# core_num=1
	if core_num > 1:
		chunk = seq_num / core_num + 1
	else:
		chunk = seq_num
	start = 0
	while start < seq_num:
		end = start + chunk
		if end > seq_num:
			end = seq_num
		args = (seq_lens[start:end], uon[start:end], bon[start:end], theta, num_classify, uf_num, bf_num, que1, que2)
		p = Process(target=likelihood_thread_simple, args=args)
		p.start()
		np += 1
		sub_processes.append(p)
		start += chunk
	for i in range(np):
		likelihood += que1.get()
	for i in range(np):
		grad += que2.get()
	while sub_processes:
		sub_processes.pop().join()
	grad -= regularity_deriv(theta, regtype, sigma)
	_gradient = grad
	return likelihood - regularity(theta, regtype, sigma)


# """using shared array to do the multiprocessing in windows"""
def likelihood_mp_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta,
					 bon_loc_end, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma):
	global _gradient
	grad = numpy.array(fss, copy=True)  # data distribution
	likelihood = numpy.dot(fss, theta)
	que1 = Queue()  # for the likelihood output
	que2 = Queue()  # for the gradient output
	np = 0
	sub_processes = []
	core_num = multiprocessing.cpu_count()
	# core_num=1
	if core_num > 1:
		chunk = int(seq_num / core_num) + 1
	else:
		chunk = seq_num
	start = 0
	while start < seq_num:
		end = start + chunk
		if end > seq_num:
			end = seq_num
		args = (seq_lens, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta,
				bon_loc_end, theta, start, end, seq_num, num_classify, uf_num, bf_num, que1, que2)
		p = Process(target=likelihood_thread_sa, args=args)
		p.start()
		np += 1
		# print 'delegated %s:%s to subprocess %s' % (start, end, np)
		sub_processes.append(p)
		start += chunk
	for i in range(np):
		likelihood += que1.get()
	for i in range(np):
		grad += que2.get()
	while sub_processes:
		sub_processes.pop().join()
	grad -= regularity_deriv(theta, regtype, sigma)
	_gradient = grad
	return likelihood - regularity(theta, regtype, sigma)


def likelihood_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta,
				  bon_loc_end, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma):
	global _gradient
	grad = numpy.array(fss, copy=True)  # data distribution
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	likelihood = numpy.dot(fss, theta)
	for si in range(seq_num):
		log_m_list = logM_sa(seq_lens[si], si, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta,
							 bon_loc_sta, bon_loc_end, num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		likelihood -= log_z
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			iloc = uon_seq_sta[si]
			for it in range(uon_loc_sta[iloc + i], uon_loc_end[iloc + i]):
				ao = uon_arr[it]
				grad_u[ao:ao + num_classify] -= p_yi

			iloc = bon_seq_sta[si]
			for it in range(bon_loc_sta[iloc + i], bon_loc_end[iloc + i]):
				ao = bon_arr[it]
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))
	grad -= regularity_deriv(theta, regtype, sigma)
	_gradient = grad
	return likelihood - regularity(theta, regtype, sigma)


def cal_likelihood(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma):
	global _gradient
	grad = numpy.array(fss, copy=True)  # data distribution
	likelihood = numpy.dot(fss, theta)
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	# likelihood = numpy.dot(fss,theta)
	for si in range(seq_num):
		log_m_list = log_m_array(seq_lens[si], uon[si], bon[si], num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		likelihood -= log_z
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			for ao in uon[si][i]:
				grad_u[ao:ao + num_classify] -= p_yi
			for ao in bon[si][i]:
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))
	grad -= regularity_deriv(theta, regtype, sigma)
	_gradient = grad
	return likelihood - regularity(theta, regtype, sigma)


def gradient_likelihood_standalone(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype,
								   sigma):
	grad = numpy.array(fss, copy=True)  # data distribution
	grad_b = grad[0:bf_num]
	grad_u = grad[bf_num:]
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	# likelihood = numpy.dot(fss,theta)
	for si in range(seq_num):
		log_m_list = log_m_array(seq_lens[si], uon[si], bon[si], num_classify, theta_u, theta_b)
		log_alphas = cal_log_alphas(log_m_list)
		log_betas = cal_log_betas(log_m_list)
		log_z = logsumexp(log_alphas[-1])
		expect = numpy.zeros((num_classify, num_classify))
		for i in range(len(log_m_list)):
			if i == 0:
				expect = numpy.exp(log_m_list[0] + log_betas[i][:, numpy.newaxis] - log_z)
			elif i < len(log_m_list):
				expect = numpy.exp(
					log_m_list[i] + log_alphas[i - 1][numpy.newaxis, :] + log_betas[i][:, numpy.newaxis] - log_z)
			p_yi = numpy.sum(expect, axis=1)
			# minus the parameter distribution
			for ao in uon[si][i]:
				grad_u[ao:ao + num_classify] -= p_yi
			for ao in bon[si][i]:
				grad_b[ao:ao + num_classify * num_classify] -= expect.reshape((num_classify * num_classify))
	return grad - regularity_deriv(theta, regtype, sigma)


def gradient_likelihood(theta):
	# this is a dummy function
	global _gradient
	return _gradient


def check_crf_dev(data_file, template_file):
	"""Check if the Derivative calculation is correct.
	Don't call this function if your model has millions of features.
	Otherwise it will run forever...      """
	if not os.path.isfile(template_file):
		print("Can't find the template file!")
		return -1
	tp_list = read_template(template_file)

	if not os.path.isfile(data_file):
		print("Data file doesn't exist!")
		return -1
	texts, seq_lens, oys, seq_num, num_classify, obydic, y2label = read_data(data_file)

	uf_obs, bf_obs, uf_num, bf_num = process_features(tp_list, texts, num_classify)
	f_num = uf_num + bf_num

	uon, bon = cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num)

	y0 = 0
	regtype = 2
	sigma = 1.0
	fss = cal_FSS(texts, oys, uon, bon, uf_num, bf_num, seq_num, num_classify, y0)
	print("Linear CRF in Python.. ver 0.1 ")
	print("B features:", bf_num, "U features:", uf_num, "total num:", f_num)
	print("training sequence number:", seq_num)

	theta = random_param(uf_num, bf_num)
	delta = 0.0001
	for i in range(f_num):
		ta = likelihood_mp_simple(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma)
		dev = gradient_likelihood(theta)
		theta[i] = theta[i] + delta
		tb = likelihood_mp_simple(seq_lens, fss, uon, bon, theta, seq_num, num_classify, uf_num, bf_num, regtype, sigma)
		devest = (tb - ta) / delta
		print("dev:", dev[i], "dev numeric~:", devest, str(datetime.datetime.now())[10:19])
		theta[i] = theta[i] - delta  # reverse to original


def check_crf_dev_sa(data_file, template_file, mp=0):
	"""
	Check if the Derivative calculation is correct.
	Don't call this function if your model has millions of features.
	Otherwise it will run forever...      
	"""
	if not os.path.isfile(template_file):
		print("Can't find the template file!")
		return -1
	tp_list = read_template(template_file)

	if not os.path.isfile(data_file):
		print("Data file doesn't exist!")
		return -1
	texts, seq_lens, oys, seq_num, num_classify, obydic, y2label = read_data(data_file)

	uf_obs, bf_obs, uf_num, bf_num = process_features(tp_list, texts, num_classify)
	f_num = uf_num + bf_num

	uon, bon = cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num)
	uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta, bon_loc_end = cal_observe_on_loc(
		uon, bon,
		seq_num, mp)

	y0 = 0
	regtype = 2
	sigma = 1.0
	fss = cal_FSS(texts, oys, uon, bon, uf_num, bf_num, seq_num, num_classify, y0)
	print("Linear CRF in Python.. ver 0.1 ")
	print("B features:", bf_num, "U features:", uf_num, "total num:", f_num)
	print("training sequence number:", seq_num)

	if sys.platform == "win32" and mp == 1:
		theta = multiprocessing.Array(ctypes.c_double, uf_num + bf_num)
		theta = numpy.ctypeslib.as_array(theta.get_obj())
		theta = theta.reshape(uf_num + bf_num)

	else:
		theta = random_param(uf_num, bf_num)
	delta = 0.0001
	for i in range(f_num):
		ta = likelihood_mp_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end,
							  bon_arr, bon_seq_sta, bon_loc_sta, bon_loc_end, theta, seq_num, num_classify, uf_num,
							  bf_num,
							  regtype, sigma)
		dev = gradient_likelihood(theta)
		theta[i] = theta[i] + delta
		tb = likelihood_mp_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end,
							  bon_arr, bon_seq_sta, bon_loc_sta, bon_loc_end, theta, seq_num, num_classify, uf_num,
							  bf_num,
							  regtype, sigma)
		devest = (tb - ta) / delta
		print("dev:", dev[i], "dev numeric~:", devest, str(datetime.datetime.now())[10:19])
		theta[i] = theta[i] - delta  # reverse to original


def save_model(bf_num, uf_num, tlist, obydic, uf_obs, bf_obs, theta, modelfile):
	with open(modelfile, 'wb') as f:
		pickle.dump([bf_num, uf_num, tlist, obydic, uf_obs, bf_obs, theta], f)


def output_file(texts, oys, max_ys, y2label, res_file):
	"""
	输出文件
	:param texts:
	:param oys:
	:param max_ys:
	:param y2label:
	:param res_file:
	:return:
	"""
	if res_file == "":
		return 0
	result = []
	for si in range(len(oys)):
		sentence = []
		for li in range(len(oys[si])):
			line = ""
			for x in texts[si][li]:
				line += x
			line += ' '
			line += y2label[oys[si][li]] + " "
			line += y2label[max_ys[si][li]]
			line += "\n"
			sentence.append(line)
		result.append(''.join(sentence))
	with open(res_file, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(result))
	return 0


def load_model(model_file):
	"""
	加载模型
	:param model_file: 
	:return: 
	"""
	if not os.path.isfile(model_file):
		print("Error: model file does not Exist!")
		return -1
	with open(model_file, 'rb') as f:
		bf_num, uf_num, tp_list, oby_dic, uf_obs, bf_obs, theta = pickle.load(f)
	num_classify = len(oby_dic)
	y2label = dict([(oby_dic[key], key) for key in oby_dic.keys()])
	return bf_num, uf_num, tp_list, oby_dic, uf_obs, bf_obs, theta, num_classify, y2label


def tagging(seq_lens, uon, bon, theta, seq_num, num_classify, uf_num, bf_num):
	theta_b = theta[0:bf_num]
	theta_u = theta[bf_num:]
	max_ys = []
	for si in range(seq_num):
		log_m_list = log_m_array(seq_lens[si], uon[si], bon[si], num_classify, theta_u, theta_b)
		max_alpha = numpy.zeros((len(log_m_list), num_classify))
		my = []
		max_ilist = []
		seq_len = len(log_m_list)
		for i in range(seq_len):
			if i == 0:
				max_alpha[i] = log_m_list[0][:, 0]
			elif i < seq_len:
				at = log_m_list[i] + max_alpha[i - 1]
				max_alpha[i] = at.max(axis=1)
				max_ilist.append(at.argmax(axis=1))
		ty = max_alpha[-1].argmax()
		my.append(ty)
		for a in (reversed(max_ilist)):
			my.append(a[ty])
			ty = a[ty]
		max_ys.append(my[::-1])
	return max_ys


def check_tagging(max_ys, oys):
	tc = 0
	te = 0
	for si in range(len(oys)):
		for li in range(len(oys[si])):
			if oys[si][li] == max_ys[si][li]:
				tc += 1
			else:
				te += 1
	print("Note: If Y is useless, correct rate is also useless.")
	print("correct:", tc, "error:", te, " correct rate:", float(tc) / (tc + te))


def crf_predict(data_file, model_file, result_file=""):
	start_time = time.time()
	"""read all the data"""
	bf_num, uf_num, tp_list, oby_dic, uf_obs, bf_obs, theta, num_classify, y2label = load_model(model_file)
	f_num = uf_num + bf_num
	if f_num == 0:
		print("ERROR: Load the model file failed!")
		return -1
	texts, seq_lens, oys, seq_num, t1, oby_dic_tmp, y2ltmp = read_data(data_file)
	if seq_num == 0 or len(oby_dic) == 0:
		print("ERROR: Read data file failed!")
		return -1
	for i in range(len(oys)):
		for j in range(len(oys[i])):
			s_label = y2ltmp[oys[i][j]]
			if oby_dic.get(s_label):  # some
				oys[i][j] = oby_dic[y2ltmp[oys[i][j]]]
			else:
				oys[i][j] = 0

	print("Linear CRF in Python.. ver 0.1 ")
	print("B 特征:", bf_num, "U 特征:", uf_num, "total num:", f_num)
	print("Prediction sequence number:", seq_num)
	uon, bon = cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num)
	max_ys = tagging(seq_lens, uon, bon, theta, seq_num, num_classify, uf_num, bf_num)
	check_tagging(max_ys, oys)
	print("写入预测结果:", result_file)
	output_file(texts, oys, max_ys, y2label, result_file)
	print("Test finished in ", time.time() - start_time, "seconds. \n ")


def train(data_file, template_file, model_file, mp=1, regtype=2, sigma=1.0, fd=5):
	"""
	训练模型
	:param data_file: 训练集
	:param template_file: 模板
	:param model_file: 模型文件
	:param mp: 并行
	:param regtype:
	:param sigma:
	:param fd:
	:return:
	"""
	start_time = time.time()

	tp_list = read_template(template_file)
	texts, seq_lens, oys, seq_num, num_classify, oby_dic, y2label = read_data(data_file)

	uf_obs, bf_obs, uf_num, bf_num = process_features(tp_list, texts, num_classify, fd=fd)
	f_num = uf_num + bf_num
	print("线性CRF 版本 1.0.")
	print("B 特征:", bf_num, "U 特征:", uf_num, "B-U 特征:", f_num)
	print("训练集序列数量:", seq_num)
	print("开始计算特征:", round(time.time() - start_time, 2), "seconds. \n ")
	if f_num == 0:
		print("没有学习参数. ")
		return
	uon, bon = cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num)
	with open("ubobx", 'wb') as f:
		pickle.dump([uf_obs, bf_obs], f)
	del uf_obs
	del bf_obs

	y0 = 0
	fss = cal_FSS(texts, oys, uon, bon, uf_num, bf_num, seq_num, num_classify, y0)
	del texts
	del oys

	uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr, bon_seq_sta, bon_loc_sta, bon_loc_end = \
		cal_observe_on_loc(uon, bon, seq_num, mp)

	del uon
	del bon

	print("开始学习参数:\t", round(time.time() - start_time, 2), "seconds. \n ")

	from scipy import optimize
	if sys.platform == "win32" and mp == 1:  # using shared memory
		theta = multiprocessing.Array(ctypes.c_double, uf_num + bf_num)
		theta = numpy.ctypeslib.as_array(theta.get_obj())
		theta = theta.reshape(uf_num + bf_num)
	else:
		theta = random_param(uf_num, bf_num)

	if mp == 1:  # using multi processing
		likeli = lambda x: -likelihood_mp_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr,
											 bon_seq_sta, bon_loc_sta, bon_loc_end, x, seq_num, num_classify, uf_num,
											 bf_num, regtype, sigma)
	else:
		likeli = lambda x: -likelihood_sa(seq_lens, fss, uon_arr, uon_seq_sta, uon_loc_sta, uon_loc_end, bon_arr,
										  bon_seq_sta, bon_loc_sta, bon_loc_end, x, seq_num, num_classify, uf_num,
										  bf_num, regtype, sigma)
	likelihood_deriv = lambda x: -gradient_likelihood(x)
	theta, fobj, dtemp = optimize.fmin_l_bfgs_b(likeli, theta, fprime=likelihood_deriv, disp=1, factr=1e12, maxiter=20)

	with open("ubobx", 'rb') as f:
		uf_obs, bf_obs = pickle.load(f)
	save_model(bf_num, uf_num, tp_list, oby_dic, uf_obs, bf_obs, theta, model_file)
	print("Training finished in ", time.time() - start_time, "seconds. \n ")


def train_simple(data_file, template_file, model_file, mp=1, regtype=2, sigma=1.0):
	start_time = time.time()

	tp_list = read_template(template_file)
	texts, seq_lens, oys, seq_num, num_classify, oby_dic, y2label = read_data(data_file)

	uf_obs, bf_obs, uf_num, bf_num = process_features(tp_list, texts, num_classify)
	f_num = uf_num + bf_num
	print("Linear CRF in Python.. ver 0.1 ")
	print("B features:", bf_num, "U features:", uf_num, "total num:", f_num)
	print("training sequence number:", seq_num)
	print("start to calculate ON feature.  ", time.time() - start_time, "seconds. \n ")
	uon, bon = cal_observe_on(tp_list, texts, uf_obs, bf_obs, seq_num)

	print("start to calculate data distribution. ", time.time() - start_time, "seconds. \n ")

	y0 = 0
	fss = cal_FSS(texts, oys, uon, bon, uf_num, bf_num, seq_num, num_classify, y0)
	del texts
	del oys

	print("start to learn distribution. ", time.time() - start_time, "seconds. \n ")

	# return

	from scipy import optimize
	if mp == 1:  # using multi processing
		theta = random_param(uf_num, bf_num)
		likeli = lambda x: -likelihood_mp_simple(seq_lens, fss, uon, bon, x, seq_num, num_classify, uf_num, bf_num,
												 regtype, sigma)
	else:
		theta = random_param(uf_num, bf_num)
		likeli = lambda x: -cal_likelihood(seq_lens, fss, uon, bon, x, seq_num, num_classify, uf_num, bf_num, regtype,
										   sigma)
	likelihood_deriv = lambda x: -gradient_likelihood(x)
	theta, fobj, dtemp = optimize.fmin_l_bfgs_b(likeli, theta, fprime=likelihood_deriv, disp=1, factr=1e12)

	save_model(bf_num, uf_num, tp_list, oby_dic, uf_obs, bf_obs, theta, model_file)
	print("Training finished in ", time.time() - start_time, "seconds. \n ")


def main():
	# checkCrfDev("train.txt","template.txt")
	# checkCrfDev("train2.txt","template.txt")
	# checkCrfDev_sa("..\\train1.txt","..\\template.txt",mp=1)
	# checkCrfDev("trainexample2.txt","template.txt")
	# checkCrfDev_sa("trainsimple.data","templatesimple.txt",mp=1)
	# train("..\\train1.txt","..\\template.txt","model",mp=0,fd=1)
	# train("train1.txt","template.txt","model",mp=1)
	# train("datas\\4.msr_training.data","templatechunk","model")
	# train("..\\train2.txt","..\\template.txt","model",mp=1,fd=2)
	# train("datas\\train.txt","templatechunk","model")
	# train("ned.train", "templatesimple.txt", "model", mp=1)
	# train("tr1.utf8.txt","templatesimple.txt","model")

	crf_predict("ned.testa", "model", "res.txt")


# crf_predict("tr1.utf8.txt","model","res.txt")


if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
