#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-2 下午4:33
# @Author  : 林利芳
# @File    : linear_crf.py
from scipy import optimize
import time
from .utils import *
from concurrent import futures

_gradient = None  # global variable used to store the gradient calculated in liklihood function.


class SeqFeature(object):
	def __init__(self, fd=5):
		self.fd = fd
		self.fss = None
		self.bf_num = 0
		self.uf_num = 0
		self.f_num = 0
		self.uf_obs = dict()
		self.bf_obs = dict()
		self.uon = None
		self.bon = None

	def process_features(self, texts, tp_list, num_k):
		"""
		特征提取
		:param texts: 序列文本 [[['你',],['好',]],[['你',],['好',]]]
		:param tp_list: 特征模板
		:param num_k: 状态数
		:return:
		"""
		print("特征提取...")
		uf_obs = dict()
		bf_obs = dict()
		for ti, tp in enumerate(tp_list):  # for each template line tp = [['U00',[-1,0],[0,0]],[]]
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
		# print("特征:\t", list(uf_obs.keys())[-10:])
		uf_num, bf_num = 0, 0
		for obx in bf_obs.keys():
			bf_obs[obx] = bf_num
			bf_num += num_k * num_k
		for obx in uf_obs.keys():
			uf_obs[obx] = uf_num
			uf_num += num_k
		self.uf_num = uf_num
		self.bf_num = bf_num
		self.f_num = uf_num + bf_num
		self.uf_obs = uf_obs
		self.bf_obs = bf_obs
		print("B 特征:\t{}\nU 特征:\t{}\n总特征:\t{}\n".format(self.bf_num, self.uf_num, self.f_num))

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

	def cal_observe_on(self, texts, tp_list):
		"""
		获取文本特征 [[['U:你','U:你:好'],['U:你','U:你:好'],[]],[],[]] =[[[145,456,566],[3455,]],[]]
		:param texts:
		:param tp_list:
		:return:
		"""
		self.uon = []
		self.bon = []
		for text in texts:
			seq_uon = []
			seq_bon = []
			for loc_id in range(len(text)):
				loc_uon = []
				loc_bon = []
				for ti, tp in enumerate(tp_list):  # for each template line
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
			self.uon.append(seq_uon)
			self.bon.append(seq_bon)
		return self.uon, self.bon

	def cal_fss(self, x_train, y_train, num_k, y0):
		"""
		统计特征数量 每个特征对应 num_k 个特征
		:param x_train: 序列文本
		:param y_train: 标签
		:param num_k: 状态数
		:param y0: 起始值0
		:return:
		"""
		self.fss = np.zeros((self.f_num,))
		fss_b = self.fss[0:self.bf_num]
		fss_u = self.fss[self.bf_num:]
		for seq_id, text in enumerate(x_train):
			for li in range(len(text)):
				for ao in self.uon[seq_id][li]:
					fss_u[ao + y_train[seq_id][li]] += 1.0
				for ao in self.bon[seq_id][li]:
					if li == 0:  # the first , yt-1=y0
						fss_b[ao + y_train[seq_id][li] * num_k + y0] += 1.0
					else:
						fss_b[ao + y_train[seq_id][li] * num_k + x_train[seq_id][li - 1]] += 1.0

	def save_feature(self, tp_list):
		result = ['#CRF Feature Templates.\n\n']
		for tp in tp_list:
			feature = tp[0] + ':'
			for start, end in tp[1:]:
				feature += '%x[' + start + ',' + end + ']'
			result.append(feature)
		result.append('\n\n#U')
		u_feature = list(sorted(self.uf_obs.keys(), key=lambda x: x))
		result.extend(u_feature)
		with open('feature.txt', 'w', encoding='utf-8') as fp:
			fp.write('\n'.join(result))

	def __call__(self, x_texts, y_label, tp_list, num_k, y0=0, *args, **kwargs):
		self.process_features(x_texts, tp_list, num_k)
		self.cal_observe_on(x_texts, tp_list)
		self.cal_fss(x_texts, y_label, num_k, y0)
		self.save_feature(tp_list)


class CRF(object):
	def __init__(self, regtype=2, sigma=1.0, fd=5):
		"""
		CRF 初始化
		:param regtype: L1、L2正则化
		:param sigma: 正则化参数
		:param fd: 特征频次
		"""
		self.theta = None
		self.oby_dict = dict()
		self.tp_list = [['U00', ['-2', '0']],
		                ['U01', ['-1', '0']],
		                ['U02', ['0', '0']],
		                ['U03', ['1', '0']],
		                ['U04', ['2', '0']],
		                ['U05', ['-2', '0'], ['-1', '0'], ['0', '0']],
		                ['U06', ['-1', '0'], ['0', '0'], ['1', '0']],
		                ['U07', ['0', '0'], ['1', '0'], ['2', '0']],
		                ['U08', ['-1', '0'], ['0', '0']],
		                ['U09', ['0', '0'], ['1', '0']]]
		self.num_k = 0
		self.sigma = sigma
		self.regtype = regtype
		self.feature = SeqFeature(fd=fd)

	def fit(self, x_train, y_train, model_file=None, mp=1, template_file=None, max_iter=20, n_jobs=None):
		"""
		训练模型
		:param x_train: x
		:param y_train: label
		:param mp: 并行
		:param template_file: 模板
		:param model_file: 模型文件
		:param max_iter: 迭代次数
		:param n_jobs: 进程数
		:return:
		"""
		if template_file:
			self.tp_list = read_template(template_file)
		seq_lens = [len(x) for x in y_train]
		y_train = self.process_state(y_train)
		self.feature(x_train, y_train, self.tp_list, self.num_k, y0=0)

		if self.feature.f_num == 0:
			return

		del x_train, y_train

		theta = random_param(self.feature.uf_num, self.feature.bf_num)

		if mp == 1:
			if n_jobs:
				n_jobs = min([os.cpu_count() - 1, n_jobs])
			else:
				n_jobs = os.cpu_count() - 1
			likelihood = lambda x: -self.likelihood_parallel(x, seq_lens, n_jobs)
		else:
			likelihood = lambda x: -self.likelihood(x, seq_lens)

		start_time = time.time()
		likelihood_der = lambda x: -self.gradient_likelihood(x)
		theta, _, _ = optimize.fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_der, disp=1, maxiter=max_iter)
		self.theta = theta
		if model_file:
			self.save_model(model_file)
		print("L-BFGS-B 训练耗时:\t{}s".format(int(time.time() - start_time)))

	def process_state(self, y_train):
		"""
		状态预处理
		:param y_train:
		:return:
		"""
		new_label = []
		oby_id = 0
		for sentence in y_train:
			s_label = []
			for label in sentence:
				label_id = self.oby_dict.get(label)
				if label_id:
					s_label.append(label_id)
				else:
					self.oby_dict[label] = oby_id
					s_label.append(oby_id)
					oby_id += 1
			new_label.append(s_label)
		self.num_k = len(self.oby_dict)
		return new_label

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
		y2label = dict([(self.oby_dict[key], key) for key in self.oby_dict.keys()])
		uon, bon = self.feature.cal_observe_on(x_test, self.tp_list)
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
		bf_num = self.feature.bf_num
		theta_b = self.theta[0:bf_num]
		theta_u = self.theta[bf_num:]
		max_ys = []
		for seq_id, seq_len in enumerate(seq_lens):
			matrix_list = self.log_matrix(seq_len, uon[seq_id], bon[seq_id], theta_u, theta_b, self.num_k)
			max_alpha = np.zeros((len(matrix_list), self.num_k))

			max_index = []
			for i in range(seq_len):
				if i == 0:
					# 起始状态
					max_alpha[i] = matrix_list[0][:, 0]
				elif i < seq_len:
					# 取状态概率最大的序列(num_k,num_k)
					at = matrix_list[i] + max_alpha[i - 1]
					max_alpha[i] = at.max(axis=1)
					max_index.append(at.argmax(axis=1))  # 索引代表前一时刻和当前时刻求和的最大值
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

	def likelihood_parallel(self, theta, seq_lens, n_jobs):
		"""
		并行计算参数 损失函数likelihood 梯度grad
		:param seq_lens: 序列长度 [5,9,6,...]
		:param theta: 参数
		:param n_jobs: 进程数
		:return:
		"""
		global _gradient
		grad = np.array(self.feature.fss, copy=True)  # data distribution
		likelihood = np.dot(self.feature.fss, theta)
		seq_num = len(seq_lens)
		uon = self.feature.uon
		bon = self.feature.bon

		n_thread = 2 * n_jobs
		chunk = seq_num / n_thread
		chunk_id = [int(kk * chunk) for kk in range(n_thread + 1)]
		jobs = []
		with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
			for ii, start in enumerate(chunk_id[:-1]):
				end = chunk_id[ii + 1]
				job = executor.submit(self.likelihood_thread, theta, seq_lens[start:end], uon[start:end], bon[start:end])
				jobs.append(job)
		for job in jobs:
			_likelihood, _grad = job.result()
			likelihood += _likelihood
			grad += _grad
		# 正则化
		grad -= self.regularity_deriv(theta, self.regtype, self.sigma)
		_gradient = grad
		return likelihood - self.regularity(theta, self.regtype, self.sigma)

	def likelihood(self, theta, seq_lens):
		"""
		损失函数 梯度
		对数似然函数 L(theta) = theta * fss -sum(log Z)
		梯度 grad = fss - sum (exp(theta * f) * f)
		:param theta: 参数 shape = (f_num,)
		:param seq_lens: 序列长度 shape=(N,)
		:return:
		"""
		global _gradient
		bf_num = self.feature.bf_num
		num_k = self.num_k
		grad = np.array(self.feature.fss, copy=True)  # data distribution
		grad_b = grad[0:bf_num]
		grad_u = grad[bf_num:]
		theta_b = theta[0:bf_num]
		theta_u = theta[bf_num:]
		likelihood = np.dot(self.feature.fss, theta)
		uon = self.feature.uon
		bon = self.feature.bon
		for seq_id, seq_len in enumerate(seq_lens):
			matrix_list = self.log_matrix(seq_len, uon[seq_id], bon[seq_id], theta_u, theta_b, num_k)
			log_alphas = self.forward_alphas(matrix_list)
			log_betas = self.backward_betas(matrix_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((num_k, num_k))
			for i in range(len(matrix_list)):
				if i == 0:
					expect = np.exp(matrix_list[0] + log_betas[i][:, np.newaxis] - log_z)
				elif i < len(matrix_list):
					expect = np.exp(
						matrix_list[i] + log_alphas[i - 1][np.newaxis, :] + log_betas[i][:, np.newaxis] - log_z)
				p_yi = np.sum(expect, axis=1)
				# minus the parameter distribution
				for ao in uon[seq_id][i]:
					grad_u[ao:ao + num_k] -= p_yi
				for ao in bon[seq_id][i]:
					grad_b[ao:ao + num_k * num_k] -= expect.reshape((num_k * num_k))
		# 正则化
		grad -= self.regularity_deriv(theta, self.regtype, self.sigma)
		_gradient = grad
		return likelihood - self.regularity(theta, self.regtype, self.sigma)

	def likelihood_thread(self, theta, seq_lens, uon, bon):
		"""
		计算序列特征概率
		:param seq_lens: 序列长度
		:param uon: u特征 [[[10,25,30],[45,394],[]],[]]
		:param bon: b特征
		:param theta: 参数 shape=(uf_num + bf_num,)
		:return:
		"""
		grad = np.zeros(self.feature.f_num)

		bf_num = self.feature.bf_num
		num_k = self.num_k
		likelihood = 0
		grad_b = grad[0:bf_num]
		grad_u = grad[bf_num:]
		theta_b = theta[0:bf_num]
		theta_u = theta[bf_num:]
		for seq_id, seq_len in enumerate(seq_lens):
			matrix_list = self.log_matrix(seq_len, uon[seq_id], bon[seq_id], theta_u, theta_b, num_k)
			log_alphas = self.forward_alphas(matrix_list)
			log_betas = self.backward_betas(matrix_list)
			log_z = logsumexp(log_alphas[-1])
			likelihood -= log_z
			expect = np.zeros((num_k, num_k))
			for i in range(len(matrix_list)):
				if i == 0:
					expect = np.exp(matrix_list[0] + log_betas[i][:, np.newaxis] - log_z)
				elif i < len(matrix_list):
					expect = np.exp(
						matrix_list[i] + log_alphas[i - 1][np.newaxis, :] + log_betas[i][:, np.newaxis] - log_z)
				p_yi = np.sum(expect, axis=1)
				# minus the parameter distribution
				for ao in uon[seq_id][i]:
					grad_u[ao:ao + num_k] -= p_yi
				for ao in bon[seq_id][i]:
					grad_b[ao:ao + num_k * num_k] -= expect.reshape((num_k * num_k))
		return likelihood, grad

	@staticmethod
	def log_matrix(seq_len, auon, abon, theta_u, theta_b, num_k):
		"""
		特征抽取 条件随机场矩阵形式 M_i = sum( theta * f )
		:param seq_len: 序列长度 int
		:param auon: 序列u特征 shape =(seq_len,) [[1245,4665],[2,33,455],...]
		:param abon: 序列u特征  shape =(seq_len,)
		:param theta_u: u特征参数
		:param theta_b: b特征参数
		:param num_k: 状态数
		:return: num_k 阶矩阵 shape = (seq_len,num_k,num_k)
		"""
		matrix_list = []
		for li in range(seq_len):
			fv = np.zeros((num_k, num_k))
			for ao in auon[li]:
				m = theta_u[ao:ao + num_k]
				fv += m[:, np.newaxis]

			for ao in abon[li]:
				m = theta_b[ao:ao + num_k * num_k]
				fv += m.reshape((num_k, num_k))
			matrix_list.append(fv)
		# 初始状态
		for i in range(0, num_k):  # set the emerge function for ~y(0) to be -inf.
			matrix_list[0][i][1:] = - float("inf")
		return matrix_list

	def forward_alphas(self, m_list):
		"""
		前向算法 alpha
		:param m_list: 条件随机场矩阵形式 M_i = sum( theta * fss )
		:return:
		"""
		log_alpha = m_list[0][:, 0]  # alpha(1)
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
		log_beta = np.zeros_like(m_list[-1][:, 0])
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
		return logsumexp(log_a + log_m, axis=1)

	@staticmethod
	def logsumexp_mat_vec(log_m, logb):
		return logsumexp(log_m + logb[:, np.newaxis], axis=0)

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
		bf_num, uf_num, f_num, self.tp_list, self.oby_dict, uf_obs, bf_obs, self.theta, self.num_k = load_model(
			model_file)
		self.feature.bf_num = bf_num
		self.feature.uf_num = uf_num
		self.feature.uf_obs = uf_obs
		self.feature.bf_obs = bf_obs

	def save_model(self, model_file):
		"""
		保存模型
		:param model_file:
		:return:
		"""
		bf_num = self.feature.bf_num
		uf_num = self.feature.uf_num
		f_num = bf_num + uf_num
		uf_obs = self.feature.uf_obs
		bf_obs = self.feature.bf_obs
		model = [bf_num, uf_num, f_num, self.tp_list, self.oby_dict, uf_obs, bf_obs, self.theta, self.num_k]
		save_model(model, model_file)
