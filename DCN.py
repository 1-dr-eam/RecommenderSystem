import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self,n_layers,feature_dim):
        """
        Args:
            n_layer:交叉层数量
            feature_dim:特征维度
        """
        super().__init__()
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        layers = [nn.Linear(feature_dim,feature_dim)]*n_layers
        self.cross_layers = nn.Sequential(*layers)

    def hadamard(self,matrix1,matrix2):
        return torch.mul(matrix1,matrix2)

    def forward(self,x):
        x0=x.clone()
        for i in range(self.n_layers):
            x_cross=self.cross_layers[i](x)
            x=self.hadamard(x0,x_cross)+x
        return x

class DeepNetwork(nn.Module):
    def __init__(self,feature_dim,hidden_dims=[128,64]):
        super().__init__()
        self.feature_dim = feature_dim
        layers=[]
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim,hidden_dim),nn.ReLU()])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self,x):
        return self.mlp(x)

class DCN(nn.Module):
    def __init__(self,n_layers,feature_dim,hidden_dims=[128,64]):
        """
        Args:
            n_layers:交叉网络层数
            feature_dim:输入特征维度
            hidden_dims:MLP的隐藏层维度
        """
        super().__init__()
        self.cross_network = CrossNetwork(n_layers,feature_dim)
        self.deep_network = DeepNetwork(feature_dim,hidden_dims)
        self.output_layer = nn.Linear(feature_dim+hidden_dims[-1],hidden_dims[-1]) # 输出层降维

    def forward(self,x):
        deep_features = self.deep_network(x)
        cross_features = self.cross_network(x)
        features = torch.cat((deep_features,cross_features),dim=1)
        return self.output_layer(features)
