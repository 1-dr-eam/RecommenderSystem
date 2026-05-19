import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
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

# 封装类
class Gate(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Gate, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim)
        self.softmax = nn.Softmax(dim=1) # 第0维是batch维度

    def forward(self, x):
        return self.softmax(self.mlp(x))

# 封装类
class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Tower, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(x))

class MMoE(nn.Module):
    def __init__(self, num_experts, num_gates, input_dim, hidden_dims=[128, 64], expert_output_dim=64, tower_output_dim=1):
        """
        Args:
            num_experts:专家网络数
            num_gates:闸门数，等于进行预估的指标数
            hidden_dims:当前版本专家和闸门神经网络用 MLP 实现，这个是隐藏层维度数
            expert_output_dim:专家网络输出特征维度数
            tower_output_dim:输出塔输出维度，一般要得到点击率等就设为1
        """
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_gates = num_gates
        self.hidden_dims = hidden_dims

        self.experts = [MLP(input_dim,hidden_dims,expert_output_dim).to(device) for _ in range(self.num_experts)]
        self.gates = [Gate(input_dim,hidden_dims,num_experts).to(device) for _ in range(self.num_gates)] # 输出维度要和专家网络数相同，保证一个专家网络对应一个weight
        self.towers = [Tower(expert_output_dim, hidden_dims, tower_output_dim).to(device) for _ in range(self.num_gates)] # Tower的输入维度是Expert的输出维度

    def forward(self,x):
        """
        这里默认输出的点击率，点赞率等指标顺序与数据集中的标签顺序相同
        """
        batch_size = x.size(0)

        # Experts 输出特征
        experts_outputs = [] # 每一个元素是[256,64]
        for expert in self.experts:
            experts_outputs.append(expert(x))
        # Gate 输出 weight
        weights = []
        for gate in self.gates:
            weights.append(gate(x))
        # 由gate加权过的专家网络输出特征
        transformed_features = []
        for weight in weights:# weight:[256,num_experts]
            # 每个weight代表一种指标信息，带batch维度
            transformed_feature = []
            for i in range(batch_size):# [num_experts]
                sample_transformed_feature = []
                for j in range(self.num_experts):
                    sample_transformed_feature.append(weight[i][j] * experts_outputs[j][i])
                transformed_feature.append(sum(sample_transformed_feature))
            transformed_features.append(torch.stack(transformed_feature))

        # 经过Tower得到预估的点击率，点赞率等
        output_rates = []
        for tower,feature in zip(self.towers,transformed_features):
            output_rates.append(tower(feature))

        return output_rates
