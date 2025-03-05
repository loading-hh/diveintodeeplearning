import numpy as np
import matplotlib.pyplot as plt


class DynamicPlotter:
    def __init__(self):
        # 初始化画布和子图
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(10, 10))

        # 初始化数据存储结构
        self.x_data = [[] for _ in range(3)]  # 三个子图的x数据
        self.y_data = [
            [[] for _ in range(2)],   # 第一个子图2条线
            [[] for _ in range(3)],   # 第二个子图3条线
            [[] for _ in range(3)]    # 第三个子图3条线
        ]

        # 创建各子图的线对象
        self.lines = [
            [self.ax1.plot([], [], 'r-', label = "train_loss_his")[0],    # 红色实线
             self.ax1.plot([], [], 'b--', label = "val_loss_his")[0]],  # 蓝色虚线

            [self.ax2.plot([], [], 'g-', label = "train_class_1")[0],    # 绿色实线
             self.ax2.plot([], [], 'm--', label = "train_class_2")[0],   # 品红虚线
             self.ax2.plot([], [], 'c-.', label = "train_class_3")[0]],  # 青色点划线

            [self.ax3.plot([], [], 'y-', label = "val_class_1")[0],   # 黄色实线
             self.ax3.plot([], [], 'k--', label = "val_class_2")[0],  # 黑色虚线
             self.ax3.plot([], [], 'p-.', label = "val_class_3")[0]]  # 紫色点划线
        ]
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        # 设置交互模式
        plt.ion()
        self.fig.show()

    def update_plot(self, new_values):
        """
        更新所有子图的函数
        new_values: 包含三个子图数据的列表，每个子图数据对应其线条数
        """
        # 更新x轴数据（统一用时间步）
        for i in range(3):
            self.x_data[i].append(len(self.x_data[i]))

        # 更新各子图的y数据
        for subplot_idx in range(3):
            for line_idx in range(len(new_values[subplot_idx])):
                self.y_data[subplot_idx][line_idx].append(new_values[subplot_idx][line_idx])

                # 更新线条数据
                self.lines[subplot_idx][line_idx].set_data(
                    self.x_data[subplot_idx],
                    self.y_data[subplot_idx][line_idx]
                )

        # 自动调整坐标范围
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        # 重绘图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# 使用示例
if __name__ == "__main__":
    plotter = DynamicPlotter()

    for step in range(10):
        # 生成测试数据（可根据需要修改）
        new_data = [
            [np.sin(step / 5), np.cos(step / 5)],            # 第一个子图数据
            [np.sin(step / 5), np.cos(step / 5), step % 10],   # 第二个子图数据
            [step % 5, np.log(step + 1), np.sqrt(step)]      # 第三个子图数据
        ]

        plotter.update_plot(new_data)
        plt.pause(0.1)  # 控制刷新频率

    plt.ioff()
    plt.show()



# class PlotData(object):
#     def __init__(self, cols_num, epochs_num):
#         epochs_num = epochs_num + 1
#         # 训练集与验证集损失历史值
#         self.ax0 = plt.subplot2grid((1, cols_num), (0, 0), colspan=1, rowspan=1)
#         self.ax0.grid(ls = "-.")
#         self.ax0.set_xlim((0, epochs_num))
#         # 训练集精度历史值
#         self.ax1 = plt.subplot2grid((1, cols_num), (0, 1), colspan=1, rowspan=1)
#         self.ax1.grid(ls = "-.")
#         self.ax1.set_xlim((0, epochs_num))
#         # 验证集精度历史值
#         self.ax2 = plt.subplot2grid((1, cols_num), (0, 2), colspan=1, rowspan=1)
#         self.ax2.grid(ls = "-.")
#         self.ax2.set_xlim((0, epochs_num))

#     def plot_img(self, i, loss_his, train_his_acc, val_his_acc):
#         # 训练集与验证集损失
#         self.ax0.plot(range(i + 2), loss_his[0], label = "train_loss_his")
#         self.ax0.plot(range(i + 2), loss_his[1], label = "val_loss_his")
#         # 训练集精度历史值
#         self.ax1.plot(range(i + 2), train_his_acc[0], label = "class 1")
#         self.ax1.plot(range(i + 2), train_his_acc[1], label = "class 2")
#         self.ax1.plot(range(i + 2), train_his_acc[2], label = "class 3")
#         # 验证集精度历史值
#         self.ax2.plot(range(i + 2), val_his_acc[0], label = "class 1")
#         self.ax2.plot(range(i + 2), val_his_acc[1], label = "class 2")
#         self.ax2.plot(range(i + 2), val_his_acc[2], label = "class 3")

#         plt.draw()