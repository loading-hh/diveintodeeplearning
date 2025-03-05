'''
Author: loading-hh
Date: 2025-03-01 17:12:33
LastEditTime: 2025-03-05 17:09:51
LastEditors: loading-hh
Description: 
FilePath: \pytorch\第十章-注意力机制\vision transformer\train.py
可以输入预定的版权声明、个性签名、空行等
'''
import tqdm
import model
import transformer
import torch
import torchvision
from torch import nn
from IPython import display
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import datasets
import plot


def statistics_confusion(y_true, y_predict, class_num):
    """
        输出的混淆矩阵的每一行是预测类别的，列是真实类别的。

    """
    """for i in range(10):#代表类别
        for j in range(len(y_val_hat1)):#代表元素下表
            if y_val_hat1[j] == i and label_val[j] == i:
                confusion[i][i] += 1
    for j in range(len(y_val_hat1)):
        if y_val_hat1[j] != label_val[j]:
            confusion[y_val_hat1[j]][label_val[j]] += 1"""
    confusion = torch.zeros(class_num, class_num)  # 横着代表预测的类别，竖着代表真实的类别
    for i in range(len(y_true)):
        confusion[y_predict[i]][y_true[i]] += 1
    return confusion


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   model_load_path 已有模型路径，使用自己模型
    #                   当pretrain这个参数是为True时就不能用model_load_path这个参数
    # ---------------------------------------------------------------------------#
    train_data_path = "C:/Users/CCU6/Desktop/肺癌检测/cancer/trian"
    val_data_path = "C:/Users/CCU6/Desktop/肺癌检测/cancer/validation"
    # ---------------------------------#
    #   num_epochs是指定模型整体训练轮数
    # ---------------------------------#
    num_epochs = 10
    # ---------------------------------#
    #   alpha是学习率
    # ---------------------------------#
    alpha = 0.002
    # ---------------------------------#
    #   batch_size是批量大小
    # ---------------------------------#
    batch_size = 16
    # ----------------------------------------------------------------#
    #   optimizer_type  进行优化器的选择，可选的种类有Adam，SGD
    #                   当使用Adam优化器时建议设置  Init_lr=6e-4
    #                   当使用SGD优化器时建议设置   Init_lr=2e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ----------------------------------------------------------------#
    optimizer_type = "Adam"
    momentum = 0.937
    weight_decay = 0
    # ---------------------------------#
    #   class_num   是类别数
    # ---------------------------------#
    class_num = 3
    # ---------------------------------#
    #   device是使用cpu还是gpu
    # ---------------------------------#
    device = torch.device("cuda:0")
    # ---------------------------------#
    #   通过optimizer_type选择优化器
    # ---------------------------------#

    net = model.VisionTransformer(3, 224, 3, 16, 768, class_num, 12, 4, 0)
    optimizer = {
        'Adam': torch.optim.Adam(net.parameters(), lr = alpha, betas = (momentum, 0.999), weight_decay = weight_decay),
        'SGD': torch.optim.SGD(net.parameters(), lr = alpha, momentum = momentum, nesterov=True, weight_decay = weight_decay)
    }[optimizer_type]

    train_dataset = datasets.Load_my_datasets(train_data_path, val_data_path)

    # 定义的东西
    plotter = plot.DynamicPlotter()
    train_loss_his = []
    val_loss_his = []
    train_his_acc = []
    val_his_acc = []
    train_recall = torch.zeros(10)
    train_precision = torch.zeros(10)
    val_recall = torch.zeros(10)
    val_precision = torch.zeros(10)
    # 进行混淆矩阵的统计
    confusion_train = torch.zeros(class_num, class_num)
    confusion_val = torch.zeros(class_num, class_num)

    train, val = train_dataset.load_datasets()
    train_data_batch = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_data_batch = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0)
    # 正式代码
    print(f"train on:{device}")
    net.to(device)
    loss = nn.CrossEntropyLoss(reduction='none')
    for i in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        net.train()
        # 增加进度条
        loop_1 = tqdm.tqdm(enumerate(train_data_batch), total=len(train_data_batch), leave=True)
        loop_1.set_description(f"Epoch [{i + 1}/{num_epochs}], train")
        # 每次迭代用batch_size大小的数据集进行训练，一轮共every_num_epoch此迭代。
        for k, (data_train, label_train) in loop_1:
            data_train_k = data_train.to(device)
            label_train_k = label_train.to(device)
            optimizer.zero_grad()
            y_trian_hat = net(data_train_k)
            l_train = loss(y_trian_hat, label_train_k)
            l_train.mean().backward()
            optimizer.step()
            with torch.no_grad():
                train_max_hat = torch.argmax(y_trian_hat, dim=1)
                train_acc_k = sum(train_max_hat == label_train_k)
                train_acc += train_acc_k
                train_loss += l_train.sum()
                a = statistics_confusion(label_train, train_max_hat, class_num)
                confusion_train = confusion_train + a

            # 在进度条最后加精度与损失
            if (k == len(loop_1) - 1):
                loop_1.set_postfix(train_acc = ((train_acc * 1.0) / len(train)).cpu(),
                                   train_loss = ((train_loss * 1.0) / len(train)).cpu())
        for j in range(class_num):
            train_recall[j] = confusion_train[j][j] / torch.sum(confusion_train, dim=0)[j]
            train_precision[j] = confusion_train[j][j] / torch.sum(confusion_train, dim=1)[j]

        net.eval()
        with torch.no_grad():
            val_loss = 0.
            val_acc = 0.
            # 增加进度条
            loop_2 = tqdm.tqdm(enumerate(val_data_batch), total=len(val_data_batch), leave=True)
            loop_2.set_description(f"Epoch [{i + 1}/{num_epochs}], validation")
            # 验证集的预测
            for j, (data_val, label_val) in loop_2:
                data_val_k = data_val.to(device)
                label_val_k = label_val.to(device)
                y_val_hat = net(data_val_k)
                val_max_hat = torch.argmax(y_val_hat, dim=1)
                val_acc_k = sum(val_max_hat == label_val_k)
                l_val = loss(y_val_hat, label_val_k)
                val_acc += val_acc_k
                val_loss += l_val.sum()
                confusion_val = confusion_val + statistics_confusion(label_val, val_max_hat, class_num)

                # 在进度条最后加精度与损失
                if (k == len(loop_2) - 1):
                    loop_2.set_postfix(val_acc = ((val_acc * 1.0) / len(val)).cpu(),
                                       val_loss = ((val_loss * 1.0) / len(val)).cpu())

            train_loss_his.append(((train_loss * 1.0) / len(train)).cpu())
            val_loss_his.append(((val_loss * 1.0) / len(val)).cpu())
            # train_his_acc.append(((train_acc * 1.0) / len(train)).cpu())
            # val_his_acc.append(((val_acc * 1.0) / len(val)).cpu())
        for j in range(class_num):
            val_recall[j] = confusion_val[j][j] / torch.sum(confusion_val, dim=0)[j]
            val_precision[j] = confusion_val[j][j] / torch.sum(confusion_val, dim=1)[j]
        print(f"训练集召回率 lung_aca：{train_recall[0]}, lung_n：{train_recall[1]}, lung_scc：{train_recall[2]}")
        print(f"训练集精确率 lung_aca：{train_precision[0]}, lung_n：{train_precision[1]}, lung_scc：{train_precision[2]}")
        print(f"测试集召回率 lung_aca：{val_recall[0]}, lung_n：{val_recall[1]}, lung_scc：{val_recall[2]}")
        print(f"测试集精确率 lung_aca：{val_precision[0]}, lung_n：{val_precision[1]}, lung_scc：{val_precision[2]}")

        # 画出每次迭代的图
        new_data = [
            [train_loss_his[-1], val_loss_his[-1]],            # 第一个子图数据
            [train_precision[0], train_precision[1], train_precision[2]],   # 第二个子图数据
            [val_precision[0], val_precision[1], val_precision[2]]      # 第三个子图数据
        ]
        plotter.update_plot(new_data)

        torch.save(net, "model.pt")
    plt.show()
    print(f"train acc:{max(train_his_acc)}, val acc:{max(val_his_acc)}")