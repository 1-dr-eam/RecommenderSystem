from typing import List
from math import pow
import pandas as pd
from entities import *
import torch

# 数据样本转tensor
def collate_fn_two_towers(batch):
    """
    双塔
    数据加载器的批次处理函数
    将一批样本（每个含1正+2负）合并为张量
    """
    # === 用户特征（所有样本共享）===
    user_ids = torch.cat([x['pos'][0] for x in batch])
    user_discrete = torch.cat([x['pos'][1] for x in batch])
    user_continuous = torch.cat([x['pos'][2] for x in batch])
    user_feat = (user_ids, user_discrete, user_continuous)

    # === 正样本物品特征 ===
    pos_item_ids = torch.cat([x['pos'][3] for x in batch])
    pos_item_discrete = torch.cat([x['pos'][4] for x in batch])
    pos_item_continuous = torch.cat([x['pos'][5] for x in batch])
    pos_item_feat = (pos_item_ids, pos_item_discrete, pos_item_continuous)

    # === 负样本物品特征 ===
    n_neg = len(batch[0]['neg'])  # 一个正样本对应的负样本数，'neg'中存的是n个元组，每个元组是(neg_item_ids, neg_item_disc, neg_item_cont)
    neg_item_feats = []
    for i in range(n_neg):
        neg_ids = torch.cat([x['neg'][i][0] for x in batch])
        neg_disc = torch.cat([x['neg'][i][1] for x in batch])
        neg_cont = torch.cat([x['neg'][i][2] for x in batch])
        neg_item_feats.append((neg_ids, neg_disc, neg_cont))

    return {
        'user_feat': user_feat,
        'pos_item_feat': pos_item_feat,
        'neg_item_feats': neg_item_feats  # list of (ids, disc, cont)
    }

def collate_fn_three_towers(batch):
    """
    三塔
    数据加载器的批次处理函数
    """
    user_ids = torch.stack([x['user_ids'] for x in batch])
    user_discrete = torch.stack([x['user_discrete'] for x in batch])
    user_continuous = torch.stack([x['user_continuous'] for x in batch])
    scene_discrete = torch.stack([x['scene_discrete'] for x in batch])
    item_ids = torch.stack([x['item_ids'] for x in batch])
    item_discrete = torch.stack([x['item_discrete'] for x in batch])
    item_continuous = torch.stack([x['item_continuous'] for x in batch])
    stat_continuous = torch.stack([x['stat_continuous'] for x in batch])
    targets = torch.stack([x['targets'] for x in batch])

    return {
        'user_ids': user_ids,
        'user_discrete': user_discrete,
        'user_continuous': user_continuous,
        'scene_discrete': scene_discrete,
        'item_ids': item_ids,
        'item_discrete': item_discrete,
        'item_continuous': item_continuous,
        'stat_continuous': stat_continuous,
        'targets': targets
    }

def bpr_loss(pos_scores,neg_scores)->torch.Tensor:
    """BPR损失函数，用于LightGCN"""
    diff=pos_scores-neg_scores
    bpr=-1.0*torch.sum(torch.log(torch.sigmoid(diff)))
    return bpr

def fusion_formula(click_pred, cart_pred, forward_pred, buy_pred)->float:
    """
    融分公式
    """
    # return 0.4 * click_pred + 0.1 * cart_pred + 0.3 * forward_pred + 0.2 * buy_pred
    return (click_pred**0.7)*(cart_pred**0.6)*(buy_pred**0.5)*(forward_pred**0.5)

async def getItems(df_items)->List[Item]:
    """
    Dataframe 转 Item 对象列表
    """
    if df_items.empty:
        return []

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    new_items = []
    for row in df_items.itertuples(index=True):  # index=True 获取原始的 DataFrame 索引
        item_idx = row.Index
        # 构建 discrete_features 和 continuous_features 字典
        discrete_features = {
            'city': row.city,
            'name': row.name,
            'author': row.author,
            'item_categories': row.item_categories,
            'item_keywords': row.item_keywords
        }
        continuous_features = {'price': row.price}
        # 创建 Item 对象
        item = Item(
            item_idx,
            row.item_id,
            row.name,
            row.author,
            row.item_categories,
            row.item_keywords,
            discrete_features,
            continuous_features,
            row.create_time,
            row.image,
            row.description
        )
        # 计算特征
        item.calculate_content_feature(clip_model, preprocess)
        new_items.append(item)
    return new_items