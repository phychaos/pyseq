#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午4:33
# @Author  : 林利芳
# @File    : linear_crf.py
import ctypes
import multiprocessing
import sys
from multiprocessing import Process
from queue import Queue
from scipy import optimize
import time
from .utils import *


class CRF(object):
	def __init__(self, mp=1, regtype=2, sigma=1.0, fd=5):
		"""
		CRF 初始化
		:param mp: 并行
		:param regtype:
		:param sigma:
		:param fd: 特征频次
		"""
		self.seq_num = 0
		self.bf_num = 0
		self.uf_num = 0
		self.f_num = 0
		self.theta = None
		self.uf_obs = dict()
		self.bf_obs = dict()
		self.oby_dic = dict()
		self.tp_list = []
		self.num_classify = 0
		self.mp = mp
		self.fd = fd
		self.sigma = sigma
		self.regtype = regtype
		self.uon_arr = None
		self.uon_seq_sta = None
		self.uon_loc_sta = None
		self.uon_loc_end = None
		self.bon_arr = None
		self.bon_seq_sta = None
		self.bon_loc_sta = None
		self.bon_loc_end = None

	def print_parameter(self, seq_num):
		print("线性CRF 版本 1.0.")
		print("B 特征:{}\tU 特征:{}\tB+U 特征:{}.".format(self.bf_num, self.uf_num, self.f_num))
		print("序列数量:", seq_num)

	def fit(self, data_file, template_file, model_file, max_iter=10):
		"""
		训练模型
		:param data_file: 训练集
		:param template_file: 模板
		:param model_file: 模型文件
		:param max_iter: 迭代次数
		:return:
		"""
		self.tp_list = read_template(template_file)
		texts, seq_lens, oys, self.seq_num, self.num_classify, self.oby_dic, y2label = read_data(data_file)
		self.process_features(texts)
		self.f_num = self.uf_num + self.bf_num
		self.print_parameter(self.seq_num)
		if self.f_num == 0:
			ValueError("没有学习参数.")
		uon, bon = self.cal_observe_on(texts)

		y0 = 0
		fss = self.cal_fss(texts, oys, uon, bon, y0)
		self.cal_observe_on_loc(uon, bon)
		del texts
		del oys
		del uon
		del bon

		theta = random_param(self.uf_num, self.bf_num)

		if self.mp == 1:  # using multi processing
			likelihood = lambda x: -self.likelihood_mp_sa(seq_lens, fss, x)
		else:
			likelihood = lambda x: -self.likelihood_sa(seq_lens, fss, x)
		start_time = time.time()
		likelihood_der = lambda x: -self.gradient_likelihood(x)
		self.theta, f_obj, d_temp = optimize.fmin_l_bfgs_b(
			likelihood, theta, fprime=likelihood_der, disp=1, factr=1e12, maxiter=max_iter)

		model = [self.bf_num, self.uf_num, self.tp_list, self.oby_dic, self.uf_obs, self.bf_obs, self.theta]
		save_model(model, model_file)
		print("训练结束 ", time.time() - start_time, "seconds. \n ")

	def predict(self, data_file, result_file=""):
		start_time = time.time()
		texts, seq_lens, oys, seq_num, t1, oby_dic_tmp, y2label_temp = read_data(data_file)
		if seq_num == 0 or len(self.oby_dic) == 0:
			print("ERROR: Read data file failed!")
			return -1
		for i in range(len(oys)):
			for j in range(len(oys[i])):
				s_label = y2label_temp[oys[i][j]]
				if self.oby_dic.get(s_label):  # some
					oys[i][j] = self.oby_dic[y2label_temp[oys[i][j]]]
				else:
					oys[i][j] = 0

		self.print_parameter(seq_num)
		uon, bon = self.cal_observe_on(texts)
		max_ys = self.tagging(seq_lens, uon, bon, seq_num)
		self.check_tagging(max_ys, oys)
		print("写入预测结果:", result_file)

		y2label = dict([(self.oby_dic[key], key) for key in self.oby_dic.keys()])
		output_file(texts, oys, max_ys, y2label, result_file)
		print("Test finished in ", time.time() - start_time, "seconds. \n ")

	def tagging(self, seq_lens, uon, bon, seq_num):
		theta_b = self.theta[0:self.bf_num]
		theta_u = self.theta[self.bf_num:]
		max_ys = []
		for si in range(seq_num):
			log_m_list = self.log_m_array(seq_lens[si], uon[si], bon[si], theta_u, theta_b)
			max_alpha = np.zeros((len(log_m_list), self.num_classify))
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

	@staticmethod
	def check_tagging(max_ys, oys):
		tc = 0
		te = 0
		for si in range(len(oys)):
			for li in range(len(oys[si])):
				if oys[si][li] == max_ys[si][li]:
					tc += 1
				else:
					te += 1
		print("正确:", tc, "错误:", te, " 准确率:", float(tc) / (tc + te))

	def process_features(self, texts):
		"""
		特征提取
		:param texts:
		:return:
		"""
		uf_obs = dict()
		bf_obs = dict()
		for ti, tp in enumerate(self.tp_list):  # for each template line
			for text in texts:
				for loc_id in range(len(text)):
					obx = self.expand_observation(text, loc_id, tp)

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

		if self.fd >= 2:  # 移除频次小于fd的特征
			uf_obs = {k: v for k, v in uf_obs.items() if v >= self.fd}
			bf_obs = {k: v for k, v in bf_obs.items() if v >= self.fd}
		print(list(uf_obs.keys())[-5:])
		uf_num, bf_num = 0, 0
		for obx in bf_obs.keys():
			bf_obs[obx] = bf_num
			bf_num += self.num_classify * self.num_classify
		for obx in uf_obs.keys():
			uf_obs[obx] = uf_num
			uf_num += self.num_classify
		self.uf_num = uf_num
		self.bf_num = bf_num
		self.uf_obs = uf_obs
		self.bf_obs = bf_obs

	@staticmethod
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

	def cal_observe_on(self, texts):
		"""
		speed up the feature calculation
		calculate the on feature functions
		"""
		uon = []
		bon = []
		for text in texts:
			seq_uon = []
			seq_bon = []
			for loc_id in range(len(text)):
				loc_uon = []
				loc_bon = []
				for ti, tp in enumerate(self.tp_list):  # for each template line
					obx = self.expand_observation(text, loc_id, tp)
					if tp[0][0] == "B":
						fid = self.bf_obs.get(obx)
						if fid is not None:
							loc_bon.append(fid)
					if tp[0][0] == "U":
						fid = self.uf_obs.get(obx)
						if fid is not None:
							loc_uon.append(fid)
				seq_uon.append(loc_uon)
				seq_bon.append(loc_bon)
			uon.append(seq_uon)
			bon.append(seq_bon)
		return uon, bon

	def cal_observe_on_loc(self, uon, bon):
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
		if sys.platform == "win32" and self.mp == 1:  # windows system need shared memory to do multiprocessing
			self.uon_arr = multiprocessing.Array('i', u_len)
			self.uon_seq_sta = multiprocessing.Array('i', self.seq_num)
			self.uon_loc_sta = multiprocessing.Array('i', loc_len)
			self.uon_loc_end = multiprocessing.Array('i', loc_len)
		else:
			self.uon_arr = np.zeros((u_len,), dtype=np.int)
			self.uon_seq_sta = np.zeros((self.seq_num,), dtype=np.int)
			self.uon_loc_sta = np.zeros((loc_len,), dtype=np.int)
			self.uon_loc_end = np.zeros((loc_len,), dtype=np.int)

		uid = 0
		seq_i = 0
		loci = 0
		for seq in uon:  # for each training sequence.
			self.uon_seq_sta[seq_i] = loci
			for loco in seq:
				self.uon_loc_sta[loci] = uid
				for aon in loco:
					self.uon_arr[uid] = aon
					uid += 1
				self.uon_loc_end[loci] = uid
				loci += 1
			seq_i += 1
		# ------------------------------------------------------------------------------------------
		b_len = 0
		loc_len = 0
		for a in bon:
			loc_len += len(a)
			for b in a:
				b_len += len(b)

		if sys.platform == "win32" and self.mp == 1:  # windows system need shared memory to do multiprocessing
			self.bon_arr = multiprocessing.Array('i', u_len)
			self.bon_seq_sta = multiprocessing.Array('i', self.seq_num)
			self.bon_loc_sta = multiprocessing.Array('i', loc_len)
			self.bon_loc_end = multiprocessing.Array('i', loc_len)
		else:
			self.bon_arr = np.zeros((u_len,), dtype=np.int)
			self.bon_seq_sta = np.zeros((self.seq_num,), dtype=np.int)
			self.bon_loc_sta = np.zeros((loc_len,), dtype=np.int)
			self.bon_loc_end = np.zeros((loc_len,), dtype=np.int)

		bid = 0
		seq_i = 0
		loci = 0
		for seq in bon:  # for each training sequence.
			self.bon_seq_sta[seq_i] = loci
			for loco in seq:
				self.bon_loc_sta[loci] = bid
				for aon in loco:
					self.bon_arr[bid] = aon
					bid += 1
				self.bon_loc_end[loci] = bid
				loci += 1
			seq_i += 1

	def cal_fss(self, texts, oys, uon, bon, y0):
		"""
		统计特征数量 每个特征对应 num_classify 个特征
		:param texts: 序列文本
		:param oys: 标签
		:param uon: u特征 序列 [[[],[],[]],[],]
		:param bon: b特征
		:param y0: 起始值
		:return:
		"""
		fss = np.zeros((self.f_num,))
		fss_b = fss[0:self.bf_num]
		fss_u = fss[self.bf_num:]
		for i in range(self.seq_num):
			for li in range(len(texts[i])):
				for ao in uon[i][li]:
					fss_u[ao + oys[i][li]] += 1.0
				for ao in bon[i][li]:
					if li == 0:  # the first , yt-1=y0
						fss_b[ao + oys[i][li] * self.num_classify + y0] += 1.0
					else:
						fss_b[ao + oys[i][li] * self.num_classify + oys[i][li - 1]] += 1.0
		return fss

	@staticmethod
	def gradient_likelihood(theta):
		# this is a dummy function
		global _gradient
		return _gradient

	def likelihood_mp_sa(self, seq_lens, fss, theta):
		global _gradient
		grad = np.array(fss, copy=True)  # data distribution
		likelihood = np.dot(fss, theta)
		que1 = Queue()  # for the likelihood output
		que2 = Queue()  # for the gradient output
		num_p = 0
		sub_processes = []
		core_num = multiprocessing.cpu_count()
		# core_num=1
		if core_num > 1:
			chunk = int(self.seq_num / core_num) + 1
		else:
			chunk = self.seq_num
		start = 0
		while start < self.seq_num:
			end = start + chunk
			if end > self.seq_num:
				end = self.seq_num
			p = Process(target=self.likelihood_thread_sa, args=(seq_lens, theta, start, end, que1, que2))
			p.start()
			num_p += 1
			sub_processes.append(p)
			start += chunk
		for i in range(num_p):
			likelihood += que1.get()
		for i in range(num_p):
			grad += que2.get()
		while sub_processes:
			sub_processes.pop().join()
		grad -= self.regularity_der(theta)
		_gradient = grad
		return likelihood - self.regularity(theta)

	def likelihood_sa(self, seq_lens, fss, theta):
		global _gradient
		grad = np.array(fss, copy=True)  # data distribution
		grad_b = grad[0:self.bf_num]
		grad_u = grad[self.bf_num:]
		theta_b = theta[0:self.bf_num]
		theta_u = theta[self.bf_num:]
		likelihood = np.dot(fss, theta)
		for si in range(self.seq_num):
			log_m_list = self.logM_sa(seq_lens[si], si, theta_u, theta_b)
			log_alphas = self.cal_log_alphas(log_m_list)
			log_betas = self.cal_log_betas(log_m_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((self.num_classify, self.num_classify))
			for i in range(len(log_m_list)):
				if i == 0:
					expect = np.exp(log_m_list[0] + log_betas[i][:, np.newaxis] - log_z)
				elif i < len(log_m_list):
					expect = np.exp(
						log_m_list[i] + log_alphas[i - 1][np.newaxis, :] + log_betas[i][:, np.newaxis] - log_z)
				p_yi = np.sum(expect, axis=1)
				# minus the parameter distribution
				iloc = self.uon_seq_sta[si]
				for it in range(self.uon_loc_sta[iloc + i], self.uon_loc_end[iloc + i]):
					ao = self.uon_arr[it]
					grad_u[ao:ao + self.num_classify] -= p_yi

				iloc = self.bon_seq_sta[si]
				for it in range(self.bon_loc_sta[iloc + i], self.bon_loc_end[iloc + i]):
					ao = self.bon_arr[it]
					grad_b[ao:ao + self.num_classify * self.num_classify] -= expect.reshape(
						(self.num_classify * self.num_classify))
		grad -= self.regularity_der(theta)
		_gradient = grad
		return likelihood - self.regularity(theta)

	def likelihood_thread_o(self, seq_lens, uon, bon, theta_u, theta_b, start, end, que):
		likelihood = 0.0
		for seq_id in range(start, end):
			log_m_list = self.log_m_array(seq_lens[seq_id], uon[seq_id], bon[seq_id], theta_u, theta_b)
			log_z = logsumexp(self.cal_log_alphas(log_m_list)[-1])
			likelihood -= log_z
		que.put(likelihood)

	def likelihood_thread_sa(self, seq_lens, theta, start, end, que1, que2):
		grad = np.zeros(self.f_num)
		likelihood = 0
		grad_b = grad[0:self.bf_num]
		grad_u = grad[self.bf_num:]
		theta_b = theta[0:self.bf_num]
		theta_u = theta[self.bf_num:]
		for si in range(start, end):
			log_m_list = self.logM_sa(seq_lens[si], si, theta_u, theta_b)
			log_alphas = self.cal_log_alphas(log_m_list)
			log_betas = self.cal_log_betas(log_m_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((self.num_classify, self.num_classify))
			for i in range(len(log_m_list)):
				if i == 0:
					expect = np.exp(log_m_list[0] + log_betas[i][:, np.newaxis] - log_z)
				elif i < len(log_m_list):
					expect = np.exp(
						log_m_list[i] + log_alphas[i - 1][np.newaxis, :] + log_betas[i][:, np.newaxis] - log_z)
				p_yi = np.sum(expect, axis=1)
				# minus the parameter distribution
				iloc = self.uon_seq_sta[si]
				for it in range(self.uon_loc_sta[iloc + i], self.uon_loc_end[iloc + i]):
					ao = self.uon_arr[it]
					grad_u[ao:ao + self.num_classify] -= p_yi

				iloc = self.bon_seq_sta[si]
				for it in range(self.bon_loc_sta[iloc + i], self.bon_loc_end[iloc + i]):
					ao = self.bon_arr[it]
					grad_b[ao:ao + self.num_classify * self.num_classify] -= expect.reshape(
						(self.num_classify * self.num_classify))

		que1.put(likelihood)
		que2.put(grad)

	def regularity(self, theta):
		if self.regtype == 0:
			regular = 0
		elif self.regtype == 1:
			regular = np.sum(np.abs(theta)) / self.sigma
		else:
			v = self.sigma ** 2
			v2 = v * 2
			regular = np.sum(np.dot(theta, theta)) / v2
		return regular

	def regularity_der(self, theta):
		if self.regtype == 0:
			regular_der = 0
		elif self.regtype == 1:
			regular_der = np.sign(theta) / self.sigma
		else:
			v = self.sigma ** 2
			regular_der = theta / v
		return regular_der

	def log_m_array(self, seq_len, auon, abon, theta_u, theta_b):
		# log_m_list (n, num_classify, num_classify ) --> (sequence length, Yt, Yt-1)
		m_list = []
		for li in range(seq_len):
			fv = np.zeros((self.num_classify, self.num_classify))
			for ao in auon[li]:
				fv += theta_u[ao:ao + self.num_classify][:, np.newaxis]
			for ao in abon[li]:
				fv += theta_b[ao:ao + self.num_classify * self.num_classify].reshape(
					(self.num_classify, self.num_classify))
			m_list.append(fv)

		for i in range(0, self.num_classify):  # set the emerge function for ~y(0) to be -inf.
			m_list[0][i][1:] = - float("inf")
		return m_list

	def logM_sa(self, seq_len, seq_id, theta_u, theta_b):
		m_list = []
		i_loc = self.uon_seq_sta[seq_id]
		i_locb = self.bon_seq_sta[seq_id]
		for li in range(seq_len):
			fv = np.zeros((self.num_classify, self.num_classify))
			for i in range(self.uon_loc_sta[i_loc + li], self.uon_loc_end[i_loc + li]):
				ao = self.uon_arr[i]
				fv += theta_u[ao:ao + self.num_classify][:, np.newaxis]
			for i in range(self.bon_loc_sta[i_locb + li], self.bon_loc_end[i_locb + li]):
				ao = self.bon_arr[i]
				fv += theta_b[ao:ao + self.num_classify * self.num_classify].reshape(
					(self.num_classify, self.num_classify))
			m_list.append(fv)

		for i in range(0, self.num_classify):
			m_list[0][i][1:] = - float("inf")
		return m_list

	@staticmethod
	def cal_log_alphas(m_list):
		log_alpha = m_list[0][:, 0]  # alpha(1)
		log_alphas = [log_alpha]
		for logM in m_list[1:]:
			log_alpha = log_sum_exp_vec_mat(log_alpha, logM)
			log_alphas.append(log_alpha)
		return log_alphas

	@staticmethod
	def cal_log_betas(m_list):
		log_beta = np.zeros_like(m_list[-1][:, 0])
		log_betas = [log_beta]
		for logM in m_list[-1:0:-1]:
			log_beta = log_sum_exp_mat_vec(logM, log_beta)
			log_betas.append(log_beta)
		return log_betas[::-1]
