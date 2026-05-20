import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self,r,num_features):
        """
        Args:
            r: 压缩比例
            num_features: 特征数量
        """
        super(SENet, self).__init__()
        self.r = r
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Linear(num_features, num_features//r),
            nn.ReLU(),
            nn.Linear(num_features//r, num_features),
            nn.Sigmoid()
        )

    def forward(self,features):
        """
        Args:
            features: 所有特征向量的列表，其中含有 m 个向量，每个向量代表一种特征，每个向量维度可以不同
        """
        input_features = torch.stack([feature.mean() for feature in features])
        weights = self.net(input_features)

        output = []
        for w,f in zip(weights,features):
            output.append(w*f)
        return output