import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MLP import MLP

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.net = MLP(input_dim, hidden_dims, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys):
        """
        Args:
            query: 候选物品特征 [batch_size, feature_dim]
            keys: 历史行为序列特征 [batch_size, seq_len, feature_dim]
        Returns:
            attention_weights: [batch_size, seq_len]
        batch可以理解为一次性做 batch_size 次推荐
        """
        batch_size, seq_len, feature_dim = keys.shape

        # 将query扩展到与keys相同的序列维度 [batch_size, seq_len, feature_dim]
        # 先unsqueeze加入一个维度，再将这个维度从1扩展为seq_len，-1表示该维度不变，实际执行的是复制操作，便于配对计算注意力分数
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)

        # 拼接query和keys [batch_size, seq_len, feature_dim * 2]
        concat_features = torch.cat([query_expanded, keys], dim=-1)

        # 展平后通过MLP计算注意力分数 [batch_size * seq_len, feature_dim * 2]
        flat_features = concat_features.view(-1, feature_dim * 2)
        attention_scores = self.net(flat_features)

        # 恢复形状 [batch_size, seq_len]
        attention_scores = attention_scores.view(batch_size, seq_len)

        # 用sigmoid激活，应用dropout
        attention_weights = torch.sigmoid(attention_scores)
        attention_weights = self.dropout(attention_weights)

        return attention_weights


class DIN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        Args:
            input_dim: 对应物品的特征维度
            hidden_dims: 注意力层中 MLP 的隐藏层维度
        """
        super(DIN, self).__init__()
        # 拼接两个物品特征后是input_dim*2
        self.attention = AttentionLayer(input_dim * 2, hidden_dims)

    def forward(self, history_item_features: torch.Tensor,
                selected_item_feature: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Args:
            history_item_features: [batch_size, seq_len, feature_dim]
            selected_item_feature: [batch_size, feature_dim]
            mask: [batch_size, seq_len] 可选的序列 mask，True表示有效位置，掩码可用SIM的思路生成
        Returns:
            weights_sum: [batch_size] 权重之和
            weighted_features: [batch_size, feature_dim] 加权和特征
        """
        # 计算注意力权重 [batch_size, seq_len]
        attention_weights = self.attention(selected_item_feature, history_item_features)

        # 应用mask
        if mask is not None:
            attention_weights = attention_weights * mask.float()

        # 扩展权重维度用于逐元素乘法 [batch_size, seq_len, 1]
        weights_expanded = attention_weights.unsqueeze(-1)

        # 加权求和 [batch_size, feature_dim]
        weighted_features = (weights_expanded * history_item_features).sum(dim=1)
        weights_sum = attention_weights.sum(dim=-1)

        return weights_sum, weighted_features