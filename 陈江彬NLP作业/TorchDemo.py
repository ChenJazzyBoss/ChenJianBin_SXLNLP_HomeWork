import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于PyTorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数 > 第5个数，则为正样本，反之为负样本
"""


# 定义神经网络模型类
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()  # 调用父类的初始化方法
        # 第一层
        self.linear1 = nn.Linear(input_size, 10)  # 定义线性层，输入大小为input_size，输出大小为10
        # 第二层
        self.linear2 = nn.Linear(10, 15)  # 定义线性层，输入大小为10，输出大小为15
        # 第三层
        self.linear3 = nn.Linear(15, 1)  # 定义线性层，输入大小为15，输出大小为1
        # self.linear = nn.Linear(input_size, 1)  # 定义线性层，输入大小为input_size，输出大小为1
        self.activation = torch.sigmoid  # 使用sigmoid激活函数将输出归一化到0和1之间
        self.loss = nn.functional.mse_loss  # 使用均方差损失函数作为loss计算

    # 定义前向传播方法
    # 输入数据x，真实标签y
    # 返回损失值或者预测值
    def forward(self, x, y=None):
        # x为输入数据，y为真实标签，如果y不为空，则计算损失值，否则返回预测值
        x1 = self.linear1(x)  # 通过第一层线性层计算
        x2 = self.linear2(x1)  # 通过第二层线性层计算
        x3 = self.linear3(x2)  # 通过第三层线性层计算
        y_pred = self.activation(x3)  # 通过sigmoid激活函数归一化
        if y is not None:
            return self.loss(y_pred, y)  # 如果有真实标签，返回损失值
        else:
            # 如果没有真实标签，返回预测值;说明这是在进行测试
            return y_pred  # 否则返回预测值


# 生成一个样本，代表我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)  # 随机生成一个5维向量
    # 开始对正负样本进行判断了
    # 判断规则：对应的权重矩阵相乘值大于矩阵的平均值则为正样本
    # 否则为负样本
    # 返回一个元组，第一个元素是5维向量，第二个元素是0或1
    weight = np.array([0.1, 0.35, 0.15, 0.1, 0.3])  # 权重矩阵
    matrix = np.dot(x, weight)
    if matrix >= 0.5:
        return x, 1  # 返回正样本
    else:
        return x, 0  # 否则为负样本


# 随机生成一批样本
# 正负样本均匀生成
# 返回一个包含所有样本的列表 total_sample_num：样本总数
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()  # 生成样本，解构赋值
        X.append(x)
        Y.append([y])
    # 转换为numpy数组
    # PyTorch要求输入数据为张量形式，所以需要转换为张量
    # 转换为张量的过程是将numpy数组转换为PyTorch的张量
    # 提前转化为numpy数组为后续转化为张量提高效率
    X_np = np.array(X)
    Y_np = np.array(Y)
    return torch.FloatTensor(X_np), torch.FloatTensor(Y_np)  # 将数据转换为PyTorch的张量


# 测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 设置模型为评估模式
    test_sample_num = 100  # 测试样本数量
    test_x, test_y = build_dataset(test_sample_num)  # 生成测试集
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(test_y), test_sample_num - sum(test_y)))  # 输出正负样本数量
    correct, wrong = 0, 0
    with torch.no_grad():  # 禁用梯度计算 禁用梯度计算可以减少计算量
        y_pred = model(test_x)  # 模型预测
        # 这段代码是与真实标签进行对比
        for y_p, y_t in zip(y_pred, test_y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))  # 输出正确预测个数和正确率
    return correct / (correct + wrong)  # 返回正确率


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    # 优化器的作用是更新模型参数，以降低损失函数的值
    # 常用的优化器有：SGD, Adam, RMSProp, AdaGrad等
    # 以下是手写的优化器
    """
    # 梯度计算
        grad_w1 = 2 * (y_pred - y_true) * x ** 2  # 计算w1的梯度
        grad_w2 = 2 * (y_pred - y_true) * x  # 计算w2的梯度
        grad_w3 = 2 * (y_pred - y_true)  # 计算w3的梯度
        # 权重更新
        w1 = w1 - lr * grad_w1  # 更新w1
        w2 = w2 - lr * grad_w2  # 更新w2
        w3 = w3 - lr * grad_w3  # 更新w3
    """
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器
    log = []  # 记录训练日志
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)  # 生成训练集
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 设置模型为训练模式
        watch_loss = []  # 记录每个batch的损失
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]  # 获取一个batch的输入 也就是获取20样本
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]  # 获取一个batch的标签
            """
            在 PyTorch 中，当你调用模型实例（例如 model(x, y)）时，实际上是在调用模型的 forward 方法。
            这是神经网络模型的一个核心特性，由 nn.Module 这个基类定义。nn.Module 是所有网络模块的基类，
            它提供了很多基础功能，包括 forward 方法。
            """
            loss = model(x, y)  # 计算loss
            loss.backward()  # 反向传播计算梯度
            # Adam优化器中的函数
            # 以下这两段代码不明白
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # 记录损失
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))  # 输出每轮的平均损失
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])  # 记录准确率和损失
    # 保存模型
    torch.save(model.state_dict(), "model.pt")  # 保存模型参数
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画损失曲线
    plt.legend()
    plt.show()  # 显示图像
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5  # 输入向量维度
    model = TorchModel(input_size)  # 创建模型实例
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())  # 打印模型参数

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()  # 运行主函数进行训练
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.79349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
                [0.14184231, 0.97703277, 0.92579291, 0.33814932, 0.1358894]
                ]
    predict("model.pt", test_vec)  # 使用训练好的模型进行预测
