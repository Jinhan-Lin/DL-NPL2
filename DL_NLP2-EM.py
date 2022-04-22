import numpy as np
import time
import random


class EM:
    def __init__(self, theta_t, theta_i, N, M):
        self.alpha_t = [theta_t[0], theta_t[1], 1 - theta_t[0] - theta_t[1]]  # 硬币分布概率
        self.beta_t = [theta_t[2], theta_t[3], theta_t[4]]  # 硬币正面概率
        self.alpha = [theta_i[0], theta_i[1], 1 - theta_i[0] - theta_i[1]]  # 硬币分布概率
        self.beta = [theta_i[2], theta_i[3], theta_i[4]]  # 硬币正面概率
        self.N = N
        self.M = M
        self.theta = theta_i

    def gen_data(self):
        data = []
        for alpha, beta in zip(self.alpha_t, self.beta_t):
            for _ in range(int(alpha * self.N)):
                data.append(np.random.binomial(1, beta, size=self.M))
        self.data = data

    # 判断EM算法是否收敛
    def cal_error(self, old, new):
        old = np.mat(old).T
        new = np.mat(new).T
        temp = old - new
        if temp.T * temp < 0.000001:
            return True
        return False

    def run_EM(self):
        self.gen_data()
        [s1, s2, s3] = self.alpha
        [p, q, r] = self.beta
        epoch = 0
        theta_old = np.array(self.theta)
        print('start')
        while epoch < 1000:
            #E步，计算隐藏参数
            u1 = ((p ** np.sum(self.data, axis=1, keepdims=True)) * (
                    (1 - p) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s1) / (
                         (p ** np.sum(self.data, axis=1, keepdims=True)) * (
                         (1 - p) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s1
                         + (q ** np.sum(self.data, axis=1, keepdims=True)) * (
                                 (1 - q) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s2
                         + (r ** np.sum(self.data, axis=1, keepdims=True)) * (
                                 (1 - r) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s3)
            u2 = ((q ** np.sum(self.data, axis=1, keepdims=True)) * (
                    (1 - q) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s2) / (
                         (p ** np.sum(self.data, axis=1, keepdims=True)) * (
                         (1 - p) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s1
                         + (q ** np.sum(self.data, axis=1, keepdims=True)) * (
                                 (1 - q) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * s2
                         + (r ** np.sum(self.data, axis=1, keepdims=True)) * (
                                 (1 - r) ** (self.M - np.sum(self.data, axis=1, keepdims=True))) * (1 - s1 - s2))
            u3 = 1 - u1 - u2
            #M步，计算模型参数
            s1 = np.sum(u1) / self.N
            s2 = np.sum(u2) / self.N
            p = np.sum(u1 * np.sum(self.data, axis=1, keepdims=True)) / np.sum(u1 * self.M)
            q = np.sum(u2 * np.sum(self.data, axis=1, keepdims=True)) / np.sum(u2 * self.M)
            r = np.sum(u3 * np.sum(self.data, axis=1, keepdims=True)) / np.sum(u3 * self.M)
            theta_new = np.array([s1, s2, p, q, r])
            print('epoch:', epoch, 'theta：', theta_new)
            if self.cal_error(theta_new, theta_old):
                print('end')
                break
            theta_old = theta_new
            epoch += 1
        return theta_new


if __name__ == "__main__":
    # 设定初始参数
    theta_t = [0.4, 0.3, 0.5, 0.6, 0.8]  # 真实参数
    theta_i = [0.2, 0.5, 0.3, 0.4, 0.5]  # 初始迭代参数
    N = 500  # 取硬币次数
    M = 100  # 单个硬币投掷次数
    solver = EM(theta_t, theta_i, N, M)
    theta = solver.run_EM()
    print('_______________________________________')
    print("true theta:", theta_t)
    print("initial theta:", theta_i)
    print("em_solve result:", theta)
