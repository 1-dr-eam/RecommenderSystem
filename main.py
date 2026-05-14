import os
os.environ["OMP_NUM_THREADS"] = "1"
from typing import List
import pandas as pd
from entities import *
from recall import RecallRecommender
from rough_ranking import RoughRankingRecommender
from fine_ranking import FineRankingRecommender
from rearrangement import MmrDiversity

class RecommenderSystem:
    def __init__(self):
        # 需要用到的各种数据
        self.items=[]
        self.id_item_dict={}
        self.item_idx2id={}
        self.users=[]
        self.interactions=[] # [(user_id,item_id,rating)]

        self.df_items=pd.DataFrame()
        self.df_users=pd.DataFrame()
        self.df_interactions=pd.DataFrame() # 只包含 user_id 和 item_id 的 Dataframe
        self.df_train=pd.DataFrame()
        # 各阶段的推荐器
        self.rough_ranking_recommender=RoughRankingRecommender()
        self.fine_ranking_recommender=FineRankingRecommender()
        self.rearrangement_recommender=MmrDiversity()
        # 自动初始化数据
        self.prepare_data()

        self.recall_recommender = RecallRecommender(self.item_idx2id)

    def prepare_data(self):
        print("data preparing...")
        # clip用于提取图片和文本特征，拼接成为物品内容特征向量
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # 准备物品数据
        self.df_items = pd.read_csv("data/items_new.csv", encoding="utf-8")
        self.df_items['item_keywords'] = self.df_items['item_keywords'].apply(lambda x: tuple(x.split(';')))
        print("开始构建物品索引...")
        for row in self.df_items.itertuples(index=True):  # index=True 获取原始的 DataFrame 索引
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
            # 填充字典和列表
            self.id_item_dict[item.item_id] = item
            self.item_idx2id[item_idx] = item.item_id
            # 计算特征
            item.calculate_content_feature(clip_model, preprocess)
            self.items.append(item)
        print("物品索引构建完成")
        # 准备用户数据
        self.df_users = pd.read_csv("data/users_new.csv", encoding="utf-8")
        self.df_users['user_categories'] = self.df_users['user_categories'].fillna('').apply(lambda x: tuple(x.split(';')))
        self.df_users['user_keywords'] = self.df_users['user_keywords'].fillna('').apply(lambda x: tuple(x.split(';')))
        print("开始构建用户画像...")
        # itertuples 返回命名元组，访问速度极快
        for row in self.df_users.itertuples(index=True):
            # 获取原始索引
            user_idx = row.Index
            # 构建离散和连续特征字典
            discrete_features = {
                'gender': row.gender,
                'user_categories': row.user_categories,
                'user_keywords': row.user_keywords
            }
            continuous_features = {'age': row.age}
            # 创建 UserProfile 对象
            user = UserProfile(
                user_idx,
                row.user_id,
                row.user_categories,
                row.user_keywords,
                discrete_features,
                continuous_features,
                50
            )
            self.users.append(user)
        print("用户画像构建完成")
        # 准备交互数据
        self.df_interactions = pd.read_csv("data/interactions_new.csv", encoding="utf-8")
        self.interactions=list(zip(
            self.df_interactions['user_id'],
            self.df_interactions['item_id'],
            self.df_interactions['rating']
        ))
        # 准备双塔模型，三塔模型和精排多目标模型训练数据(根据交互记录)
        df_merge = pd.merge(self.df_interactions, self.df_users, how='left', on='user_id')
        self.df_train = pd.merge(df_merge, self.df_items, how='left', on='item_id')
        print("data finished\n")

    def fit(self):
        """离线计算"""
        # 召回
        self.recall_recommender.fit(self.df_train,self.items,self.interactions,self.df_interactions
                                    ,list(self.df_users['user_id']),list(self.df_items['item_id']))
        # 粗排
        self.rough_ranking_recommender.train_three_towers_model(self.df_train)
        self.rough_ranking_recommender.calculate_item_features(self.df_items)
        # 精排
        self.fine_ranking_recommender.train_multi_task_model(self.df_train)
        # 重排
        self.rearrangement_recommender.build_cosine_similarity_matrix(self.items)

    def fit_with_weights(self,twin_towers_model_weights_path,light_gcn_weights_path,
                         three_towers_model_weights_path,multi_task_model_path):
        """
        离线计算（模型直接加载训练好的权重）
        """
        # 召回
        self.recall_recommender.fit_with_weights(self.df_train, self.items, self.interactions, self.df_interactions
                                                ,list(self.df_users['user_id']), list(self.df_items['item_id'])
                                                ,twin_towers_model_weights_path,light_gcn_weights_path)
        # 粗排
        self.rough_ranking_recommender.load_three_towers_model(self.df_train,three_towers_model_weights_path)
        self.rough_ranking_recommender.calculate_item_features(self.df_items)
        # 精排
        self.fine_ranking_recommender.load_multi_task_model(self.df_train,multi_task_model_path)
        # 重排
        self.rearrangement_recommender.build_cosine_similarity_matrix(self.items)

    def recommend(self,user_id,hour,is_weekend,is_holiday)->List[int]:
        """
        在线推荐
        Args:
            user_id: 用户ID
            hour: 当前时间(小时)
            is_weekend: 当前是否是周末
            is_holiday: 当前是否是节假日
        """
        user_profile = next((user for user in self.users if user.user_id == user_id),None)
        df_user_profile=self.df_users.loc[self.df_users['user_id']==user_id]
        # ============= 召回 ================
        recall_items_ids=self.recall_recommender.recall(user_profile)
        # ============= 粗排 ================
        # 提取召回的id列表对应的行
        df_recall_items = self.df_items[self.df_items['item_id'].isin(recall_items_ids)].reset_index(drop=True)
        # 创建场景特征
        df_scene = pd.DataFrame([[hour,is_weekend,is_holiday]],columns=['hour','is_weekend','is_holiday'])
        rough_ranking_ids = self.rough_ranking_recommender.rough_ranking(df_user_profile.copy(),df_recall_items,df_scene)
        # ============= 精排 ================
        df_rough_items = df_recall_items[df_recall_items['item_id'].isin(rough_ranking_ids)].reset_index(drop=True)
        df_user = pd.concat([df_user_profile] * len(df_rough_items)).reset_index(drop=True)
        df_multi_task = pd.concat([df_rough_items, df_user], axis=1)
        # 添加上当前的场景特征
        df_multi_task['hour'] = [hour] * len(df_multi_task)
        df_multi_task['is_weekend'] = [is_weekend] * len(df_multi_task)
        df_multi_task['is_holiday'] = [is_holiday] * len(df_multi_task)
        item_id_score = self.fine_ranking_recommender.fine_ranking(df_multi_task)
        # ============= 重排 ================
        print("rearrangement...")
        rearrangement_items = []
        for item in self.items:
            if item.item_id in item_id_score:
                item.relevance_score = item_id_score[item.item_id]
                rearrangement_items.append(item)
        final_selected_items_ids=self.rearrangement_recommender.mmr_diversity_selection(rearrangement_items)
        print("最终推荐的物品ID列表：", final_selected_items_ids)
        print("recommend finished\n")
        return final_selected_items_ids

def main():
    """主流程"""
    recommender_system = RecommenderSystem()
    # 各模型权重路径
    twin_towers_model_weights_path = "model_weights/improved_twin_towers_model.pth"
    light_gcn_weights_path = "model_weights/lightgcn.pth"
    three_towers_model_weights_path = "model_weights/three_towers_model.pth"
    multi_task_model_path = "model_weights/multi_task_model.pth"
    # 采用预训练权重直接加载模型
    recommender_system.fit_with_weights(twin_towers_model_weights_path,light_gcn_weights_path, three_towers_model_weights_path, multi_task_model_path)
    # 假设对第一个用户做推荐，此时是15时，是周末，不是节假日
    recommender_system.recommend("U00001",15,1,0)

if __name__ == '__main__':
    main()