# 时间：2024/07/03
# 介绍：本代码用于将列车各工况的数据进行POD分解与CS重构 并评估误差

# 用到的库
import os
import csv
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def POD_SVD(UTX, k):
    """POD分解

    Args:
        UTX (array): 待分解的二维矩阵
        k (int): 表示截取前K阶模态

    Returns:
        UX_mean_2 (array): 工况平均值 (m, 1)
        U_reduced (array): 空间模态 (m, k)
    """
    m, n = UTX.shape  # m为空间长度（行数）, n为时间长度（列数）
    # 获得空间平均值
    UX_mean = np.mean(UTX, axis=1)
    UX_mean_2 = np.mean(UTX, axis=1).reshape(-1, 1)
    UTX = UTX - UX_mean_2
    U, s, V = np.linalg.svd(UTX)  # SVD就实现了POD分解
    # 截取U的前50列
    U_reduced = U[:, :k]
    return UX_mean, U_reduced


def CAL_ERROR(x, y, type_err):
    """计算两个同长度一维向量间的误差

    Args:
        x (array): 第一个一维向量
        y (array): 第二个一维向量
        type_err (int): 1-均方根误差

    Returns:
        error (float): 两个向量间的误差
    """
    if type_err == 1:
        error = np.sqrt(np.mean((x - y) ** 2))
    return error


def BP_Linprog(A, y):
    """L1 范数求最优解

    Args:
        A (array): 传感矩阵
        y (array): 观测值

    Returns:
        s (array): 重构出的稀疏系数
    """
    k = A.shape[1]  # 模态阶数
    c = np.zeros((2 * k))
    Aeq = np.hstack((A, -A))
    beq = y
    #  求解线性规划方程
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    # 用前k个减去后k个
    s = res.x[0:k] - res.x[k : 2 * k]
    return s


def CS(UTX, order_test, sensors, mean, U):
    """压缩感知重构

    Args:
        UTX (array): 测试集数据(二维矩阵)
        order_test (int): 随机挑选的测试集工况序号
        sensors (array): 传感器测点位置
        mean (array): 工况平均值
        U (array): 最重要的POD模态

    Returns:
        value_rec (array): 重构值(一维矩阵)
        error (float): 重构误差
    """
    # 提取真实值
    value_true = UTX[:, order_test]
    m = value_true.shape[0]
    k = U.shape[1]
    p = len(sensors)
    # 初始化测量矩阵
    Phi = np.zeros((p, m))
    UTX_observe = value_true[sensors] - mean[sensors]
    for i in range(p):
        Phi[i, sensors[i]] = 1
    s = BP_Linprog(Phi @ U, UTX_observe)
    value_rec = np.zeros_like(value_true)
    for i in range(k):
        value_rec = value_rec + s[i] * U[:, i]
    value_rec = value_rec + mean
    error = CAL_ERROR(value_true, value_rec, 1)
    return value_rec, error


def ERROR_RANGE(UTX, Mean):
    """寻找UTX矩阵中任意列矩阵与工况平均值Mean之间的最大误差

    Args:
        UTX (array): 测试集数据(二维矩阵)
        Mean (array): 工况平均值

    Returns:
        error_max (float): 最大误差
    """
    num_columns = UTX.shape[1]
    error_max = 0

    # 遍历所有可能的列对
    for i in range(num_columns):
        error = CAL_ERROR(UTX[:, i], Mean, 1)
        if error > error_max:
            error_max = error
    return error_max

def FITNESS(UTX, sensors, mean, U):
    """计算PSO中每个粒子的适应度 即对整个训练集数据求重构误差

    Args:
        UTX (array): 训练集数据(二维矩阵)
        sensors (array): 测量点位(一维矩阵)
        mean (array): 工况平均值
        U (array): 最重要的POD模态

    Returns:
        error (float): 整体重构误差
    """
    error = 0
    m = UTX.shape[1]
    for i in range(m):
        value_rec, error_rec = CS(UTX, i, sensors, mean, U)
        error = error + error_rec
    error = error / m
    return error


def PSO(UTX, mean, U, p, add_pmax, iter_max, partical_size, bound, error_limit, v_max, c1, c2, w):
    """进行粒子群优化遴选更优的测量点位

    Args:
        UTX (array): 训练集数据
        mean (array): 工况平均值
        U (array): 最重要的POD模态
        p (int): 粒子维度 即测量点数
        add_max (int): 最大可增加的测点数
        iter_max (int): 最大迭代次数
        partical_size (int): 粒子群规模
        bound (int): 空间界限
        error_limit (float): 判断收敛误差的界限
        v_max (float): 粒子的最大飞翔速度
        c1 (float): 个体学习因子
        c2 (float): 社会学习因子
        w (float): 惯性因子

    Returns:
        sensors (array): 最优传感器点位
        p_new (int): 实际采用的测点数
    """
    p_new = p
    while p_new <= p+add_pmax:
        # 初始化粒子的飞翔速度
        v = np.random.randint(0, v_max, size=(partical_size, p_new))
        # 初始化粒子的位置
        x = np.random.randint(bound[0], bound[1], size=(partical_size, p_new))
        # 初始化粒子适应度
        f = np.zeros((partical_size))
        bestf_personal = np.zeros((partical_size))
        for i in range(partical_size): f[i] = FITNESS(UTX, x[i, :], mean, U)
        # 初始化最优值
        bestx_personal = x.copy()
        bestf_personal = f.copy()
        bestf_global = min(bestf_personal)
        index_global = np.argmin(bestf_personal)
        bestx_global = bestx_personal[index_global, :]
        # 迭代开始
        k = 0
        bestf_k = np.zeros((iter_max))
        while k < iter_max:
            # 更新每个粒子的适应度
            for i in range(partical_size):
                f[i] = FITNESS(UTX, x[i, :], mean, U)
                # 比较更新局部最优解
                if f[i] < bestf_personal[i]:
                    bestf_personal[i] = f[i]
                    bestx_personal[i, :] = x[i, :]
            # 比较更新全局最优解
            bestf_global = min(bestf_personal)
            index_global = np.argmin(bestf_personal)
            bestx_global = bestx_personal[index_global, :]
            # 更新每个粒子的速度与位置
            for i in range(partical_size):
                # 更新速度
                v[i, :] = (
                    w * v[i, :]
                    + c1 * random.random() * (bestx_personal[i, :] - x[i, :])
                    + c2 * random.random() * (bestx_global - x[i, :])
                )
                v[i, :] = np.where(v[i, :] < v_max, v[i, :], v_max)
                v[i, :] = np.where(v[i, :] > -v_max, v[i, :], -v_max)
                # 更新位置
                x[i, :] = x[i, :] + np.rint(v[i, :])
                x[i, :] = np.where(x[i, :] < bound[1], x[i, :], bound[1])
                x[i, :] = np.where(x[i, :] > bound[0], x[i, :], bound[0])
            bestf_k[k] = bestf_global
            print("当前测点数：{}个|第{}轮迭代的最小误差为：{}".format(p_new, k, bestf_k[k]))
            if bestf_global < error_limit:
                sensors = bestx_global
                return sensors, p_new
            k = k + 1
        p_new = p_new + 5
    print("未达到迭代精度要求！")
    return sensors, p_new

# 主函数
if __name__ == "__main__":
    # 读取CSV文件
    start_time_1 = time.time()  # 记录耗时start
    data = pd.read_csv("Data_all.csv", header=None, skiprows=0)
    UTX_temp = data.to_numpy()
    m = data.shape[0]
    n = data.shape[1]
    # 按照列打乱不同工况的数据
    # order = list(range(0, n))
    # np.random.shuffle(order)
    order = [
        5,
        24,
        29,
        14,
        15,
        11,
        9,
        25,
        19,
        21,
        16,
        1,
        18,
        12,
        20,
        27,
        28,
        8,
        0,
        35,
        30,
        33,
        32,
        4,
        34,
        23,
        31,
        2,
        6,
        26,
        17,
        7,
        22,
        13,
        10,
        3,
    ]
    UTX = UTX_temp[:, order]
    # 释放内存
    del UTX_temp

    # 设置训练集
    start_train = 0
    end_train = int(0.6 * n)
    num_train = end_train - start_train
    UTX_train = UTX[:, start_train:end_train]
    # 设置验证集
    start_val = end_train
    end_val = start_val + int(0.2 * n)
    num_val = end_val - start_val
    UTX_val = UTX[:, start_val:end_val]
    # 设置测试集
    start_test = end_val
    num_test = n - start_test
    UTX_test = UTX[:, start_test:]
    # 释放内存
    del start_train, end_train, start_val, end_val, start_test
    end_time_1 = time.time()  # 记录耗时end

    # 分段求POD模态
    start_time_2 = time.time()  # 记录耗时start
    k = 50  # 保留前k阶模态
    for i in range(m // 1000 + 1):
        row_start = i * 1000
        row_end = min((i + 1) * 1000, m)
        if i == 0:
            UX_mean, U = POD_SVD(UTX_train[row_start:row_end, :], k)
        else:
            UX_mean_temp, U_temp = POD_SVD(UTX_train[row_start:row_end, :], k)
            UX_mean = np.concatenate((UX_mean, UX_mean_temp))
            U = np.vstack((U, U_temp))
    # 释放内存
    del UX_mean_temp, U_temp
    end_time_2 = time.time()  # 记录耗时end

    # 首先求取UTX矩阵中工况平均值与任意列工况间的最大误差 以辨认重构误差精度是否符合合理
    error_max = ERROR_RANGE(UTX, UX_mean)
    print("各工况与平均工况的最大误差为：{}".format(error_max))

    # PSO优化传感器的位置
    start_time_3 = time.time()  # 记录耗时start
    sensors, p_new = PSO(
        UTX_train,
        UX_mean,
        U,
        p=50,
        add_pmax=50,
        iter_max=20,
        partical_size=20,
        bound=[0, m - 1],
        error_limit=0.02,
        v_max=50000,
        c1=2,
        c2=2,
        w=0.6,
    )
    end_time_3 = time.time()  # 记录耗时end

    # 压缩感知重构测试
    start_time_4 = time.time()  # 记录耗时start
    # 随机挑选幸运测试工况
    order_test = 5  # random.randint(0, num_test - 1)
    # 给出预测值与重构误差
    value_rec, error_rec = CS(UTX_test, order_test, sensors, UX_mean, U)
    end_time_4 = time.time()  # 记录耗时end

    # 程序各部分耗时情况
    print(f"数据处理过程耗时: {end_time_1 - start_time_1} 秒")
    print(f"POD分解过程耗时: {end_time_2 - start_time_2} 秒")
    print(f"PSO优化过程耗时: {end_time_3 - start_time_3} 秒")
    print(f"CS重构过程耗时: {end_time_4 - start_time_4} 秒")
    #df1 = pd.DataFrame(U[:, 0:6])
    #df1.to_csv("pod_of_U.csv", index=False)
    #df2 = pd.DataFrame(V[0:6, :])
    #df2.to_csv("pod_of_V.csv", index=False)
    #df3 = pd.DataFrame(s[0:6])
    #df3.to_csv("pod_of_s.csv", index=False)
