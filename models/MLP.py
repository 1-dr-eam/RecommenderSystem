import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        输出的是特征向量，最后的输出层不带激活函数，便于进行封装
        Args:
            input_dim (int): input dimension
            hidden_dims (list): hidden dimensions of MLP
            output_dim (int): output dimension
        """
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim,hidden_dim),nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim,output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)