import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import transformer


class Patch_Embeddding(nn.Module):
    """
        vit中的Patch Embedding

        Parameters:
            img_size:输入图片的大小.
            in_c:输入图片的通道数.
            patch_num:将图片变为多少个patch.
            emb_dim:每个patch的长度,也就是输入到transformer中每个token的长度.

        Returns:
    """
    def __init__(self, img_size, in_c, patch_num, emb_dim):
        super().__init__()