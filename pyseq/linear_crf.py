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
        self.num_k = 0
        self.mp = mp
        self.fd = fd
        self.sigma = sigma
        self.regtype = regtype

    def print_parameter(self, seq_num):
        print("线性CRF 版本 1.0.")
        print("B 特征:\t{}\nU 特征:\t{}".format(self.bf_num, self.uf_num))
        print("序列长度:\t{}".format(seq_num))

    def fit(self, data_file, template_file, model_file, max_iter=20, n_jobs=None):
        """
        训练模型
        :param data_file: 训练集
        :param template_file: 模板
        :param model_file: 模型文件
        :param max_iter: 迭代次数
        :param n_jobs: 进程数
        :return:
        """
        self.tp_list = read_template(template_file)
        texts, seq_lens, oys, self.seq_num, self.num_k, self.oby_dic, y2label = read_data(data_file)
        self.process_features(texts)
        self.f_num = self.uf_num + self.bf_num
        self.print_parameter(self.seq_num)
        if self.f_num == 0:
            print("没有学习参数.")
            return
        uon, bon = self.cal_observe_on(texts)

        y0 = 0
        fss = self.cal_fss(texts, oys, uon, bon, y0)
        del texts
        del oys

        theta = random_param(self.uf_num, self.bf_num)

        if self.mp == 1:
            if n_jobs:
                n_jobs = min([os.cpu_count() - 1, n_jobs])
            else:
                n_jobs = os.cpu_count() - 1

            likelihood = lambda x: -self.likelihood_mp_sa(seq_lens, fss, x, uon, bon, self.regtype, self.sigma, n_jobs)
        else:
            likelihood = lambda x: -self.likelihood_sa(seq_lens, fss, x, uon, bon, self.seq_num, self.uf_num,
                                                       self.bf_num, self.num_k, self.regtype, self.sigma)
        start_time = time.time()
        likelihood_der = lambda x: -self.gradient_likelihood(x)
        theta, _, _ = optimize.fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_der, disp=1, maxiter=max_iter)
        self.theta = theta
        model = [self.bf_num, self.uf_num, self.tp_list, self.oby_dic, self.uf_obs, self.bf_obs, self.theta, self.num_k]
        save_model(model, model_file)
        print("L-BFGS-B 训练耗时:\t{}s".format(int(time.time() - start_time)))

    def predict(self, data_file, model_file='model', result_file=""):
        """
        预测结果
        :param data_file:
        :param model_file:
        :param result_file:
        :return:
        """
        self.bf_num, self.uf_num, self.tp_list, self.oby_dic, self.uf_obs, self.bf_obs, self.theta, self.num_k = load_model(
            model_file)
        texts, seq_lens, oys, seq_num, num_k, oby_dic_tmp, y2label_temp = read_data(data_file)
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

    def tagging(self, seq_lens, uon, bon, seq_num):
        """
        动态规划计算序列状态
        :param seq_lens: [10,8,3,10] 句子长度
        :param uon: u特征
        :param bon: b特征
        :param seq_num: 样本数量
        :return:
        """
        theta_b = self.theta[0:self.bf_num]
        theta_u = self.theta[self.bf_num:]
        max_ys = []
        for seq_id in range(seq_num):
            matrix_list = log_matrix(seq_lens[seq_id], uon[seq_id], bon[seq_id], theta_u, theta_b, self.num_k)
            max_alpha = np.zeros((len(matrix_list), self.num_k))
            my = []
            max_ilist = []
            seq_len = len(matrix_list)
            for i in range(seq_len):
                if i == 0:
                    # 起始状态
                    max_alpha[i] = matrix_list[0][:, 0]
                elif i < seq_len:
                    # 取状态概率最大的序列(num_k,num_k)
                    at = matrix_list[i] + max_alpha[i - 1]
                    max_alpha[i] = at.max(axis=1)
                    max_ilist.append(at.argmax(axis=1))
            # 最终状态 取概率最大一个最为最终序列结果
            ty = max_alpha[-1].argmax()
            my.append(ty)
            # 反向追踪路径
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
        :param texts: 序列文本 [[['你',],['好',]],[['你',],['好',]]]
        :return:
        """
        uf_obs = dict()
        bf_obs = dict()
        for ti, tp in enumerate(self.tp_list):  # for each template line tp = [['U00',[-1,0],[0,0]],[]]
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
        print("特征:\t", list(uf_obs.keys())[-10:])
        uf_num, bf_num = 0, 0
        for obx in bf_obs.keys():
            bf_obs[obx] = bf_num
            bf_num += self.num_k * self.num_k
        for obx in uf_obs.keys():
            uf_obs[obx] = uf_num
            uf_num += self.num_k
        self.uf_num = uf_num
        self.bf_num = bf_num
        self.uf_obs = uf_obs
        self.bf_obs = bf_obs

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
        return line

    def cal_observe_on(self, texts):
        """
        获取文本特征 [[['U:你','U:你:好'],['U:你','U:你:好'],[]],[],[]] =[[[145,456,566],[3455,]],[]]
        :param texts:
        :return:
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

    def cal_fss(self, texts, oys, uon, bon, y0):
        """
        统计特征数量 每个特征对应 num_k 个特征
        :param texts: 序列文本
        :param oys: 标签
        :param uon: u特征 序列 [[[18283,384,3930],[],[]],[],]
        :param bon: b特征
        :param y0: 起始值0
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
                        fss_b[ao + oys[i][li] * self.num_k + y0] += 1.0
                    else:
                        fss_b[ao + oys[i][li] * self.num_k + oys[i][li - 1]] += 1.0
        return fss

    @staticmethod
    def gradient_likelihood(theta):
        """
        梯度-全局变量
        :param theta: 参数
        :return:
        """
        # this is a dummy function
        global _gradient
        return _gradient

    def likelihood_mp_sa(self, seq_lens, fss, theta, uon, bon, regtype, sigma, n_jobs):
        """
        并行计算参数 损失函数likelihood 梯度grad
        :param seq_lens: 序列长度 [5,9,6,...]
        :param fss:
        :param theta: 参数
        :param uon:
        :param bon:
        :param regtype:正则化类型 0,1,2 L1 正则 L2正则
        :param sigma: 正则化参数
        :param n_jobs: 进程数
        :return:
        """
        global _gradient
        grad = np.array(fss, copy=True)  # data distribution
        likelihood = np.dot(fss, theta)
        seq_num = self.seq_num
        n_thread = 2 * n_jobs
        chunk = seq_num / n_thread
        chunk_id = [int(kk * chunk) for kk in range(n_thread + 1)]
        jobs = []
        with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for ii, start in enumerate(chunk_id[:-1]):
                end = chunk_id[ii + 1]
                job = executor.submit(self.likelihood_thread_sa, theta, seq_lens[start:end], uon[start:end],
                                      bon[start:end], self.num_k, self.uf_num, self.bf_num)
                jobs.append(job)
        for job in jobs:
            _likelihood, _grad = job.result()
            likelihood += _likelihood
            grad += _grad
        # 正则化
        grad -= regularity_der(theta, regtype, sigma)
        _gradient = grad
        return likelihood - regularity(theta, regtype, sigma)

    @staticmethod
    def likelihood_sa(fss, theta, seq_lens, uon, bon, seq_num, uf_num, bf_num, num_k, regtype, sigma):
        """
        损失函数 梯度
        :param fss: 特征分布
        :param theta: 参数 shape = (f_num,)
        :param seq_lens: 序列长度 shape=(N,)
        :param uon: u序列特征
        :param bon: b序列特征
        :param seq_num: 序列数量 int
        :param uf_num: u特征数量
        :param bf_num: b特征数量
        :param num_k: 状态数
        :param regtype: 正则化 L1 L2
        :param sigma: 正则化参数
        :return:
        """
        global _gradient
        grad = np.array(fss, copy=True)  # data distribution
        grad_b = grad[0:bf_num]
        grad_u = grad[bf_num:]
        theta_b = theta[0:bf_num]
        theta_u = theta[bf_num:]
        likelihood = np.dot(fss, theta)
        for seq_id in range(seq_num):
            matrix_list = log_matrix(seq_lens[seq_id], uon[seq_id], bon[seq_id], theta_u, theta_b, num_k)
            log_alphas = cal_log_alphas(matrix_list)
            log_betas = cal_log_betas(matrix_list)
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
        grad -= regularity_der(theta, regtype, sigma)
        _gradient = grad
        return likelihood - regularity(theta, regtype, sigma)

    @staticmethod
    def likelihood_thread_sa(theta, seq_lens, uon, bon, num_k, uf_num, bf_num):
        """
        计算序列特征概率
        :param seq_lens: 序列长度
        :param uon: u特征 [[[10,25,30],[45,394],[]],[]]
        :param bon: b特征
        :param theta: 参数 shape=(uf_num + bf_num,)
        :param num_k: 标签数量
        :param uf_num: u特征个数
        :param bf_num: b特征个数
        :return:
        """
        grad = np.zeros(uf_num + bf_num)
        likelihood = 0
        grad_b = grad[0:bf_num]
        grad_u = grad[bf_num:]
        theta_b = theta[0:bf_num]
        theta_u = theta[bf_num:]
        for seq_id in range(len(seq_lens)):
            matrix_list = log_matrix(seq_lens[seq_id], uon[seq_id], bon[seq_id], theta_u, theta_b, num_k)
            log_alphas = cal_log_alphas(matrix_list)
            log_betas = cal_log_betas(matrix_list)
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


def main():
    model = CRF(mp=1)
    model.fit(data_file='train.txt', template_file='templates.txt', model_file="model", max_iter=20)
    model.predict(data_file='test.txt', result_file='res.txt')


if __name__ == "__main__":
    main()
