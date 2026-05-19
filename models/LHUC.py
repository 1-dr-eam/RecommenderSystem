import torch
import torch.nn as nn
from models.MLP import MLP

# 封装类
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Network, self).__init__()
        self.mlp=MLP(input_dim, hidden_dims, output_dim)
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        return 2*self.output_layer(x)

class LHUC(nn.Module):
    def __init__(self, item_input_dim,user_input_dim, hidden_dims, output_dim):
        """
        Args:
            item_input_dim: 物品输入特征维度
            user_input_dim: 用户输入特征维度
            hidden_dims: 全连接网络的隐藏层维度
            output_dim: 用户和物品的输出维度
        """
        super(LHUC, self).__init__()
        self.linear1 = nn.Linear(item_input_dim, output_dim)
        self.linear2 = nn.Linear(item_input_dim, output_dim)
        self.network1 = Network(user_input_dim, hidden_dims, output_dim)
        self.network2 = Network(user_input_dim, hidden_dims, output_dim)

    def forward(self, item_feature, user_feature):
        item_output1 = self.linear1(item_feature)
        user_output1 = self.network1(user_feature)
        fusion_feature = item_output1 * user_output1 # 第一次哈达玛乘积

        item_output2 = self.linear2(fusion_feature)
        user_output2 = self.network2(user_feature)
        return item_output2 * user_output2

