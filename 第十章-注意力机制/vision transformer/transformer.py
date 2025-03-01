import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Self_Attention(nn.Module):
    """
        注意力函数,可以通过参数指定是多头还是单头,注意力评分函数用的是Dot-Product.

        Parameters:
            toekn_size:每个token向量的长度.
            qk_size:每个q与k向量的长度.
            v_size:每个v向量的长度.当已有一个头时,qkv三向量的长度是一样的.多头时一般qkv长度也是一样的.
                    只要保证最后输出token的长度与输入token的长度一样的就可以.
            head_num:有几个头.

        Returns:经过多头注意力的输出,维度为(batch数, token数, 每个token的长度)
    """
    def __init__(self, token_size, qk_size, v_size, head_num):
        super().__init__()

        self.token_size = token_size
        self.qk_size = qk_size
        self.v_size = v_size
        self.head_num = head_num

        # 生成qkv的全连接层。
        self.W_q = nn.Linear(token_size, qk_size * head_num)
        self.W_k = nn.Linear(token_size, qk_size * head_num)
        self.W_v = nn.Linear(token_size, v_size * head_num)
        self.scale = 1 / torch.sqrt(torch.tensor(qk_size))

        # 如果是多头的话，最后还有一个可学习参数矩阵。
        self.W = nn.Linear(v_size * head_num, token_size)


    """
        注意力函数,可以通过参数指定是多头还是单头,注意力评分函数用的是Dot-Product.

        Parameters:
            x:形状为(batch, token数量, 每个token的长度)

        Returns:
    """
    def forward(self, x):
        batch, token_num, token_size = x.shape
        assert self.token_size == token_size, "判断类参数token长度与输入变量token长度不一样"

        # 变换维度为(batch数, token数, head头数, qkv向量的长度)
        # 又transpose交换维度后变为(batch数, head头数, token数, qkv向量长度)
        q = self.W_q(x).contiguous().view(batch, token_num, self.head_num, self.qk_size).transpose(1, 2)
        k = self.W_k(x).contiguous().view(batch, token_num, self.head_num, self.qk_size).transpose(1, 2)
        v = self.W_v(x).contiguous().view(batch, token_num, self.head_num, self.v_size).transpose(1, 2)

        # 得到相似度得分
        score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        score = torch.softmax(score, dim = -1)  # (batch, head头数, token数, token数)

        # 得到多头的b,维度为(batch, head头数, token数, v向量的长度)
        # transpose(1, 2)之后维度为(batch, token数, head头数, v向量的长度)
        b_t_h = torch.matmul(score, v).transpose(1, 2)

        # 将维度变为(bathch, token数, head头数 * v向量长度)
        b_concate = b_t_h.contiguous().view(batch, token_num, self.head_num * self.v_size)

        # 输出的维度为(batch, token数, b向量的长度)
        b = self.W(b_concate)

        return b


class Mlp(nn.Module):
    """
        transformer中的encoder中的最后的MLP块

        Parameters:
            in_features:输入到第一个全连接层的维度.
            hidden_features:第一个全连接层输出的维度.
            out_features:最后第二个全连接层输出的维度.
            drop:dropout的概率.

        Returns:
    """
    def __init__(self, in_features, hidden_features, out_features, drop_ratio):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(drop_ratio)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_ratio)


    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class TransformerEncoder(nn.Module):
    """
        transformer中的encoder中的最后的MLP块

        Parameters:
            head_num:多头注意力中的几个头.
            token_dim:输入到encoder中的每个token的长度.
            mlp_hid_mul:MLP中的第一个全连接层输出的维度.
            drop:dropout的概率.

        Returns:
    """
    def __init__(self, head_num, token_dim, mlp_hid_ratio, drop_ratio):
        super().__init__()

        self.norm1 = nn.LayerNorm(token_dim)
        self.att = Self_Attention(token_dim, token_dim, token_dim, head_num)
        self.norm2 = nn.LayerNorm(token_dim)
        self.mlp = Mlp(token_dim, token_dim * mlp_hid_ratio, token_dim, drop_ratio)


    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    print("asdf")