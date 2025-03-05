import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import transformer


rand_seed_num = 2
# 为CPU中设置种子。
torch.manual_seed(rand_seed_num)
# 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
torch.cuda.manual_seed_all(rand_seed_num)


class PatchEmbeddding(nn.Module):
    """
        vit中的Patch Embedding

        Parameters:
            img_size:输入图片的大小.
            in_c:输入图片的通道数.
            patch_size:每个patch的大小.
            emb_dim:每个patch的长度,也就是输入到transformer中每个token的长度.

        Returns:
    """
    def __init__(self, img_size, in_c, patch_size, emb_dim):
        super().__init__()

        self.img_size = (img_size, img_size)
        self.in_c = in_c
        self.patch_size = patch_size
        self.emd_dim = emb_dim
        self.patch_num = (self.img_size[0]  // self.patch_size) * (self.img_size[1]  // self.patch_size)

        self.proj = nn.Conv2d(self.in_c, self.emd_dim, kernel_size=self.patch_size, stride=self.patch_size)


    """
        注意力函数,可以通过参数指定是多头还是单头,注意力评分函数用的是Dot-Product.

        Parameters:
            x:形状为(batch, 图像通道数, 每张图像高, 每张图像宽)

        Returns:
            x:维度为(batch, 卷积后w*h, 通道数)
    """
    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[1], "输入图像尺寸不与类传入的参数一样"

        # 将维度从(batch, 通道数, 卷积后wh),变为(batch, 卷积后w*h, 通道数)
        x = self.proj(x).flatten(2).transpose(-2, -1)

        return x


class VisionTransformer(nn.Module):
    """
        vit的实现,位置编码就只加在第一个transformer的encoder的输入上.并不是每个transformer的encoder的输入都要加.

        Parameters:
            block_num:是transformer的encoder有几个.
            img_size:输入图片的大小.
            in_c:输入图片的通道数.
            patch_size:每个patch的大小.
            token_dim:每个patch的长度,也就是输入到transformer中每个token的长度.
            class_num:有多少个类别.
            head_num:多头注意力中的几个头.
            mlp_hid_mul:MLP中的第一个全连接层输出的维度是输入维度的几倍.
            drop:dropout的概率.

        Returns:
    """
    def __init__(self, block_num, img_size, in_c, patch_size, token_dim, class_num, head_num, mul_hid_ratio, drop_ratio):
        super().__init__()

        self.block_num = block_num
        self.class_num = class_num
        self.patch_emb = PatchEmbeddding(img_size, in_c, patch_size, token_dim)

        # 可学习参数class_token,这个token向量的长度是与transformer中encoder每个token向量的长度是一样的.
        self.class_token = nn.Parameter(torch.randn(size=(1, 1, token_dim)))

        # 可学习参数位置编码,可以用sin和cos生成,但也能用可学习参数.
        # 最后用广播加到transformer中encoder的输入中.
        self.pos_emb = nn.Parameter(torch.randn(size=(1, self.patch_emb.patch_num + 1, token_dim)))

        # 动态添加属性
        for i in range(self.block_num):
            setattr(self, f"block{i}", transformer.TransformerEncoder(head_num, token_dim, mul_hid_ratio, drop_ratio))

        # 最后的MLP的head.
        # 模块注册的顺序会影响参数列表、模型结构展示等，但不直接影响前向传播的流程。
        self.MLP_head = nn.Linear(token_dim, self.class_num)


    """
        输入到transformer的encoder前对输入数据的处理.

        Parameters:
            x:形状为(batch, 图像通道数, 每张图像高, 每张图像宽)

        Returns:
    """
    def forward_feature(self, x):
        # 维度为(batch, 卷积后w*h, 通道数(就是每个token向量的长度))
        x = self.patch_emb(x)
        # 扩展维度,为了concate在x的前面.为什么不用repeat,因为repeat后是一个新变量,而expand不会重新分配内存，返回结果仅仅是原始张量上的一个视图.
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        # 将class_token与x,在维度1上concate在一块.维度为(batch, 卷积后w*h + 1, 通道数(就是每个token向量的长度))
        x = torch.cat([x, class_token], dim = 1)
        # 将x与位置编码通过广播加一块
        x = x + self.pos_emb

        return x

    """
        注意力函数,可以通过参数指定是多头还是单头,注意力评分函数用的是Dot-Product.

        Parameters:
            x:形状为(batch, 图像通道数, 每张图像高, 每张图像宽)

        Returns:
            x:维度为(batch, 每个token向量的长度)
    """
    def forward(self, x):
        x = self.forward_feature(x)
        for i in range(self.block_num):
            x = getattr(self, f"block{i}")(x)
        x_last = x[:, 0]
        y = self.MLP_head(x_last)

        return y


if __name__ == "__main__":
    x = torch.randn((8, 3, 224, 224))
    model = VisionTransformer(3, 224, 3, 16, 768, 2, 12, 4, 0)
    print(model(x))
