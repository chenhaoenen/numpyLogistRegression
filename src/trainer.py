# -*- coding: utf-8 -*-
# --------------------------------------------
# Author: chen hao
# Date: 2022/4/28 10:59
# Description:
# --------------------------------------------
import time
import random
import numpy as np
from utils import stats_time
import matplotlib.pyplot as plt


def sigmoid(x):
    # sigmoid函数公式
    return 1 / (1 + np.exp(-x))

def compute_loss(X, y, param):
    data_size = len(y)
    h = sigmoid(X @ param) # w*x+b
    epsilon = 1e-5 # 平滑系数
    loss = (1/data_size)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))  #logits regression 损失函数
    return loss

def predict(X, params):
    return np.round(sigmoid(X @ params)) # 四舍五入求最终的Y值


def train(X, Y, params, learning_rate, iterations, x_devSet, y_devSet):
    start = int(time.time())
    train_data_size = len(Y)
    loss_history = np.zeros((iterations, 1)) # 保存每次迭代的损失,用于后续画图
    acc_history = np.zeros((iterations//500, 1))
    global_acc = 0.0

    for step in range(iterations):
        # 梯度下降更新参数 也就是更新 w*x + b中的 w 和 b
        params = params - (learning_rate/train_data_size) * (X.T @ (sigmoid(X @ params) - Y))  # X.T @ (sigmoid(X @ params) - Y) 损失函数对参数的求导
        cur_loss= compute_loss(X, Y, params)
        loss_history[step] = cur_loss

        # 下面的都是日志输出
        if step != 0 and step % 500 == 0: # 每500个step测试一次，并输出相应日志
            # 验证集
            y_pred = predict(x_devSet, params)
            accurate = float(sum(y_pred == y_devSet)) / float(len(y_devSet))
            acc_history[step//500] = accurate

            # 保存模型的最优参数
            if global_acc < accurate:
                model_path = f'../data/model/param_step-{step}_acc-{accurate}.npy'
                np.save(model_path, params)
                global_acc = accurate

            # 输出log日志
            end = int(time.time())
            print(f'step:{step} | loss:{"%.4f" % cur_loss} | cur_acc:{"%.4f" % accurate} | global_acc:{"%.4f" % global_acc} | eta:{stats_time(start, end, step, iterations)}h | time:{time.strftime("%H:%M:%S")}')

    return loss_history, acc_history


def read_feature(path):
    # 读取预处理得到的tf-idf值作为特征训练
    dataSet = []
    count = 0
    with open(path, 'r', encoding='utf-8') as reader:
        line = reader.readline()
        while line:
            count += 1
            y, *x = line.strip().split(' ')
            x = [float(f) for f in x]
            y = [float(y)]
            dataSet.append((x, y))
            line = reader.readline()
            if count % 10000 == 0:
                print(f'read the training data to line {count}')

    random.seed(12345) #设计随机数种子
    random.shuffle(dataSet) # 打乱数据集，打乱数据集的目的是为了后续对训练集和验证集的划分
    X, Y = zip(*dataSet)
    return np.array(X), np.array(Y)


def main():
    # 训练配置
    X, Y = read_feature('../data/preprocess/tf_idf_res')
    data_size, feat_dim = X.shape  # data_size：数据个数， feat_dim: 特征维度
    X = np.hstack((np.ones((data_size, 1)), X)) # 这里 相当于将w*x+b 中的 b和x融合到一起
    params = np.zeros((feat_dim+1, 1)) # 初始化参数，也就是w和b
    iterations = 100000  # 训练迭代次数
    learning_rate = 0.8 # 学习率

    rate = 5000 # 从数据集中取rate条数据作为验证集，其他作为训练集
    x_trainSet = X[:-rate]     # 训练集
    y_trainSet = Y[:-rate]

    x_devSet = X[-rate:]      # 验证集
    y_devSet = Y[-rate:]

    print('-' * 40)
    print(f'data size: {data_size}')
    print(f'train set size: {len(x_trainSet)}')
    print(f'dev set size: {len(x_devSet)}')
    print(f'feat dimension: {feat_dim}')
    print(f'iterations: {iterations}')
    print(f'learning rate: {learning_rate}')
    print('-' * 40)

    # 开始训练
    loss_history, acc_history = train(x_trainSet, y_trainSet, params, learning_rate, iterations, x_devSet, y_devSet)

    # loss 损失图
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(loss_history)), loss_history, 'green')
    plt.title("Convergence Graph of Loss Function")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # acc 准确率图
    plt.subplot(2, 1, 2)
    plt.plot(range(len(acc_history)), acc_history, 'deepskyblue')
    plt.title("Convergence Graph of Accurate")
    plt.xlabel("Step")
    plt.ylabel("Acc")
    plt.ylim(0.2, 1)


    fig.tight_layout()  # 调整整体空白
    plt.savefig('../doc/pic/loss-acc.jpg')
    plt.show()


if __name__ == '__main__':
    main()
