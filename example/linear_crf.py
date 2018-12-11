#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午4:33
# @Author  : 林利芳
# @File    : linear_crf.py
from scipy import optimize
import time
from pyseq.utils import *
from concurrent import futures

_gradient = None  # global variable used to store the gradient calculated in liklihood function.


class SeqFeature(object):
	def __init__(self, fd=5):
		self.fd = fd
		self.seq_lens = []
		self.oby_dict = dict()
		self.fss = None
		self.num_k = 0
		self.feature_node = []
		self.feature_edge = []
		self.node_matrix = []
		self.edge_matrix = []
		self.node_obs = {}
		self.edge_obs = {}
		self.template = [
			['U00', ['-2', '0']],
			['U01', ['-1', '0']],
			['U02', ['0', '0']],
			['U03', ['1', '0']],
			['U04', ['2', '0']],
			['U05', ['-2', '0'], ['-1', '0'], ['0', '0']],
			['U06', ['-1', '0'], ['0', '0'], ['1', '0']],
			['U07', ['0', '0'], ['1', '0'], ['2', '0']],
			['U08', ['-1', '0'], ['0', '0']],
			['U09', ['0', '0'], ['1', '0']],
			['B'],
		]

	def extract_features(self, texts):
		"""
		特征函数提取
		:param texts: 序列文本 [[['你',],['好',]],[['你',],['好',]]]
		:return:
		"""
		print("特征提取...")
		uf_obs = dict()
		bf_obs = dict()
		for tp in self.template:  # for each template line tp = [['U00',[-1,0],[0,0]],[]]
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

		node_f = [key for key, v in sorted(uf_obs.items(), key=lambda x: x[1], reverse=True) if v >= self.fd]
		edge_f = [key for key, v in sorted(bf_obs.items(), key=lambda x: x[1], reverse=True) if v >= self.fd]
		self.node_obs = {key: kk * self.num_k for kk, key in enumerate(node_f)}
		self.edge_obs = {key: kk * self.num_k * self.num_k for kk, key in enumerate(edge_f)}

	# 特征函数
	# for state_p in range(self.num_k):
	# 	for state in range(self.num_k):
	# 		for edge in edge_f:
	# 			# self.add_feature_edge(lambda y_p, y, x, i: 1 if y_p == state_p and state == y and x == edge else 0)
	# 			self.add_feature_edge(1)
	# for state in range(self.num_k):
	# 	for node in node_f:
	# 		# self.add_feature_node(lambda y, x, i: 1 if y == state and node == x else 0)
	# 		self.add_feature_node(1)

	@staticmethod
	def expand_observation(sentence, loc_id, tp):
		"""
		expend the observation at loc_id for sequence
		:param sentence: 字符序列
		:param loc_id: 字符在sentence的位置序号
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
			else:
				line += ':B' + li[0]
		return line

	def cal_observe_on(self, texts):
		"""
		获取文本特征 [[['U:你','U:你:好'],['U:你','U:你:好'],[]],[],[]] =[[[145,456,566],[3455,]],[]]
		:param texts:
		:return:
		"""
		for text in texts:
			seq_uon = []
			seq_bon = []
			for loc_id in range(len(text)):
				uf = []
				bf = []
				for ti, tp in enumerate(self.template):
					obx = self.expand_observation(text, loc_id, tp)
					if tp[0][0] == "B":
						bf_id = self.edge_obs.get(obx)
						if bf_id is not None:
							bf.append(bf_id)
					if tp[0][0] == "U":
						uf_id = self.node_obs.get(obx)
						if uf_id is not None:
							uf.append(uf_id)
				seq_uon.append(uf)
				seq_bon.append(bf)
			self.node_matrix.append(seq_uon)
			self.edge_matrix.append(seq_bon)
		return self.node_matrix, self.edge_matrix

	def cal_fss(self, labels, y0):
		"""
		统计特征数量 每个特征对应 num_k 个特征
		:param labels: 标签
		:param y0: 起始值0
		:return:
		"""
		self.fss = np.zeros((self.size(),))
		fss_b = self.fss[0:self.bf_size()]
		fss_u = self.fss[self.bf_size():]
		for seq_id, label in enumerate(labels):
			y_p = y0
			for loc_id, y in enumerate(label):
				for uf_id in self.node_matrix[seq_id][loc_id]:
					fss_u[uf_id + y] += 1
				for bf_id in self.edge_matrix[seq_id][loc_id]:
					fss_b[bf_id + y_p * self.num_k + y] += 1
				# y_p = label[loc_id - 1] if loc_id > 0 else y0
				# fss_b += self.edge_matrix[seq_id][loc_id][y_p, y, :]
				y_p = y

	def size(self):
		"""特征函数个数"""
		return len(self.edge_obs) * self.num_k * self.num_k + len(self.node_obs) * self.num_k

	def uf_size(self):
		"""u特征函数个数"""
		return len(self.node_obs) * self.num_k

	def bf_size(self):
		"""B特征函数大小"""
		return len(self.edge_obs) * self.num_k * self.num_k

	def process_state(self, labels):
		"""
		状态预处理
		:param labels:
		:return:
		"""
		new_label = []
		oby_id = 0
		for sentence in labels:
			s_label = []
			for label in sentence:
				label_id = self.oby_dict.get(label)
				if label_id is None:
					label_id = oby_id
					self.oby_dict[label] = oby_id
					oby_id += 1
				s_label.append(label_id)
			new_label.append(s_label)
		self.num_k = len(self.oby_dict)
		return new_label

	def __call__(self, texts, labels, template_file, *args, **kwargs):
		if template_file:
			self.template = read_template(template_file)
		self.seq_lens = [len(x) for x in labels]
		labels = self.process_state(labels)
		self.extract_features(texts)
		self.cal_observe_on(texts)
		self.cal_fss(labels, y0=0)


# self.save_feature(template)


class CRF(object):
	def __init__(self, regtype=2, sigma=1.0, fd=5):
		"""
		CRF 初始化
		:param regtype: L1、L2正则化
		:param sigma: 正则化参数
		:param fd: 特征频次
		"""
		self.theta = None
		self.sigma = sigma
		self.regtype = regtype
		self.feature = SeqFeature(fd=fd)

	def fit(self, x_train, y_train, model_file=None, template_file=None, max_iter=20, n_jobs=None):
		"""
		训练模型
		:param x_train: x
		:param y_train: label
		:param template_file: 模板
		:param model_file: 模型文件
		:param max_iter: 迭代次数
		:param n_jobs: 进程数
		:return:
		"""
		self.feature(x_train, y_train, template_file)
		del x_train, y_train

		theta = random_param(self.feature.size())

		if n_jobs:
			n_jobs = min([os.cpu_count() - 1, n_jobs])
		else:
			n_jobs = os.cpu_count() - 1

		# self.likelihood_parallel(theta, n_jobs)
		# exit()
		likelihood = lambda x: -self.likelihood_parallel(x, n_jobs)
		likelihood_deriv = lambda x: -self.gradient_likelihood(x)
		start_time = time.time()
		theta, _, _ = optimize.fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_deriv, maxiter=max_iter)
		self.theta = theta
		if model_file:
			self.save_model(model_file)
		print("L-BFGS-B 训练耗时:\t{}s".format(int(time.time() - start_time)))

	def predict(self, x_test, y_test=None, model_file=None, res_file=None):
		"""
		预测结果
		:param x_test:
		:param y_test:
		:param model_file:
		:param res_file:
		:return:
		"""
		if model_file:
			self.load_model(model_file)
		seq_lens = [len(x) for x in x_test]
		y2label = dict([(self.feature.oby_dict[key], key) for key in self.feature.oby_dict.keys()])
		uon, bon = self.feature.cal_observe_on(x_test)
		max_ys = self.tagging_viterbi(seq_lens, uon, bon, y2label)
		if y_test:
			check_tagging(max_ys, y_test)
		if res_file:
			output_file(x_test, y_test, max_ys, res_file)
		return max_ys

	def tagging_viterbi(self, seq_lens, uon, bon, y2label):
		"""
		动态规划计算序列状态
		:param seq_lens: [10,8,3,10] 句子长度
		:param uon: u特征
		:param bon: b特征
		:param y2label: y2label id-label
		:return:
		"""
		bf_num = self.feature.bf_size()
		theta_b = self.theta[0:bf_num]
		theta_u = self.theta[bf_num:]
		max_ys = []
		for seq_id, seq_len in enumerate(seq_lens):
			matrix_list = self.log_matrix(uon[seq_id], bon[seq_id], theta_u, theta_b, self.feature.num_k)
			max_alpha = np.zeros((len(matrix_list), self.feature.num_k))

			max_index = []
			for i in range(seq_len):
				if i == 0:
					# 起始状态
					max_alpha[i] = matrix_list[0][0, :]
				elif i < seq_len:
					# 取状态概率最大的序列(num_k,num_k)
					at = matrix_list[i] + max_alpha[i - 1][:, np.newaxis]
					max_alpha[i] = at.max(axis=0)
					max_index.append(at.argmax(axis=0))  # 索引代表前一时刻和当前时刻求和的最大值
			# 最终状态 取概率最大一个最为最终序列结果
			max_state = []
			ty = max_alpha[-1].argmax()
			max_state.append(y2label.get(ty, 'O'))
			# 反向追踪路径
			for a in (reversed(max_index)):
				max_state.append(y2label.get(a[ty], 'O'))
				ty = a[ty]
			max_ys.append(max_state[::-1])
		return max_ys

	@staticmethod
	def gradient_likelihood(theta):
		"""
		梯度-全局变量 dummy function
		:param theta: 参数
		:return:
		"""
		global _gradient
		return _gradient

	def likelihood_parallel(self, theta, n_jobs):
		"""
		并行计算参数 损失函数likelihood 梯度grad
		:param theta: 参数
		:param n_jobs: 进程数
		:return:
		"""
		global _gradient
		grad = np.array(self.feature.fss, copy=True)  # data distribution
		likelihood = np.dot(self.feature.fss, theta)
		seq_lens = self.feature.seq_lens
		seq_num = len(seq_lens)

		node_matrix = self.feature.node_matrix
		edge_matrix = self.feature.edge_matrix
		n_thread = 2 * n_jobs
		chunk = seq_num / n_thread
		chunk_id = [int(kk * chunk) for kk in range(n_thread + 1)]
		jobs = []
		with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
			for ii, start in enumerate(chunk_id[:-1]):
				end = chunk_id[ii + 1]
				job = executor.submit(self.likelihood, theta, node_matrix[start:end], edge_matrix[start:end])
				jobs.append(job)
		for job in jobs:
			_likelihood, _grad = job.result()
			likelihood += _likelihood
			grad += _grad
		# 正则化
		grad -= self.regularity_deriv(theta, self.regtype, self.sigma)
		_gradient = grad
		return likelihood - self.regularity(theta, self.regtype, self.sigma)

	def likelihood(self, theta, node_matrix, edge_matrix):
		"""
		计算序列特征概率
		对数似然函数 L(theta) = theta * fss -sum(log Z)
		梯度 grad = fss - sum (exp(theta * f) * f)
		:param node_matrix: u特征 [[[10,25,30],[45,394],[]],[]]
		:param edge_matrix: b特征 [[[10,25,30],[45,394],[]],[]]
		:param theta: 参数 shape=(uf_num + bf_num,)
		:return:
		"""
		grad = np.zeros(self.feature.size())

		bf_size = self.feature.bf_size()
		num_k = self.feature.num_k
		likelihood = 0
		grad_b = grad[0:bf_size]
		grad_u = grad[bf_size:]
		theta_b = theta[0:bf_size]
		theta_u = theta[bf_size:]
		for seq_id, (node_f, edge_f) in enumerate(zip(node_matrix, edge_matrix)):

			matrix_list = self.log_matrix(node_f, edge_f, theta_u, theta_b, num_k)
			log_alphas = self.forward_alphas(matrix_list)
			log_betas = self.backward_betas(matrix_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((num_k, num_k))
			for i in range(len(matrix_list)):
				if i == 0:
					expect = np.exp(matrix_list[0] + log_betas[i] - log_z)
				elif i < len(matrix_list):
					expect = np.exp(
						matrix_list[i] + log_alphas[i - 1][:, np.newaxis] + log_betas[i] - log_z)

				for ao in node_f[i]:
					grad_u[ao:ao + num_k] -= np.sum(expect, axis=0)
				for ao in edge_f[i]:
					grad_b[ao:ao + num_k * num_k] -= expect.reshape((num_k * num_k))
		# node_grad = np.dot(np.exp(log_alphas[i] + log_betas[i]), node_f[i])
		# edge_grad = np.dot(expect.reshape((num_k * num_k)), edge_f.reshape((num_k * num_k)))

		# grad_u -= node_grad
		# grad_b -= edge_grad
		return likelihood, grad

	@staticmethod
	def log_matrix(node_f, edge_f, theta_u, theta_b, num_k):
		"""
		特征抽取 条件随机场矩阵形式 M_i = sum( theta * f )
		:param node_f: 序列u特征 shape =(seq_len, num_k, uf)
		:param edge_f: 序列u特征  shape =(seq_len, num_k, num_k, bf)
		:param theta_u: u特征参数
		:param theta_b: b特征参数
		:param num_k: 状态数
		:return: num_k 阶矩阵 shape = (seq_len,num_k,num_k)
		"""
		matrix_list = []
		for loc_id, (node, edge) in enumerate(zip(node_f, edge_f)):
			fv = np.zeros((num_k, num_k))
			for uf_id in node:
				fv += theta_u[uf_id:uf_id + num_k]
			for bf_id in edge:
				fv += theta_b[bf_id:bf_id + num_k * num_k].reshape((num_k, num_k))
			# fv += np.dot(node, theta_u)
			# for kk, value in enumerate(edge):
			# 	fv[kk] += np.dot(value, theta_b)
			matrix_list.append(fv)
		for state in range(1, num_k):
			matrix_list[0][state, :] = -float('inf')
		# 初始状态
		return matrix_list

	def forward_alphas(self, m_list):
		"""
		前向算法 alpha
		:param m_list: 条件随机场矩阵形式 M_i = sum( theta * fss )
		:return:
		"""
		log_alpha = m_list[0][0, :]  # alpha(1)
		log_alphas = [log_alpha]
		for logM in m_list[1:]:
			log_alpha = self.logsumexp_vec_mat(log_alpha, logM)
			log_alphas.append(log_alpha)
		return log_alphas

	def backward_betas(self, m_list):
		"""
		后向算法 beta
		:param m_list: 条件随机场矩阵形式 M_i = sum( theta * fss )
		:return:
		"""
		log_beta = np.zeros_like(m_list[-1][0, :])
		log_betas = [log_beta]
		for logM in m_list[-1:0:-1]:
			log_beta = self.logsumexp_mat_vec(logM, log_beta)
			log_betas.append(log_beta)
		return log_betas[::-1]

	@staticmethod
	def logsumexp_vec_mat(log_a, log_m):
		"""
		计算logsumexp log(e^x) = a-log(e^(-x+a))
		:param log_a:
		:param log_m:
		:return:
		"""
		return logsumexp(log_a[:, np.newaxis] + log_m, axis=0)

	@staticmethod
	def logsumexp_mat_vec(log_m, logb):
		return logsumexp(log_m + logb, axis=1)

	@staticmethod
	def regularity(theta, regtype, sigma):
		"""
		正则化 regtype=0,1,2  L1, L2 正则
		:param theta: 参数 shape = (f_num,) = (uf_num + bf_num,)
		:param regtype: 正则化类型0,1,2 L1正则(loss + |w|/sigma), L2正则(loss + |w|^2/(2*sigma^2))
		:param sigma:
		:return:
		"""
		if regtype == 0:
			regular = 0
		elif regtype == 1:
			regular = np.sum(np.abs(theta)) / sigma
		else:
			v = sigma ** 2
			v2 = v * 2
			regular = np.sum(np.dot(theta, theta)) / v2
		return regular

	@staticmethod
	def regularity_deriv(theta, regtype, sigma):
		"""
		正则化微分 regtype=0,1,2  L1, L2 正则
		:param theta: 参数 shape = (f_num,) = (uf_num + bf_num,)
		:param regtype: 正则化类型0,1,2 L1正则(loss' + sign(w)/sigma), L2正则(loss + |w|^2/(2*sigma^2))
		:param sigma:
		:return:
		"""
		if regtype == 0:
			regular_der = 0
		elif regtype == 1:
			regular_der = np.sign(theta) / sigma
		else:
			v = sigma ** 2
			regular_der = theta / v
		return regular_der

	def load_model(self, model_file):
		"""
		加载模型
		:param model_file:
		:return:
		"""
		self.feature, self.theta = load_model(model_file)

	def save_model(self, model_file):
		"""
		保存模型
		:param model_file:
		:return:
		"""
		model = [self.feature, self.theta]
		save_model(model, model_file)
