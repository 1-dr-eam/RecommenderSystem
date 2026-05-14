import math
from collections import defaultdict
from heapq import nlargest
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from entities import *
from twin_towers_model import TwoTowersModelRecommender
from LightGCN import LightGCNRecommender
from swing import Swing

class ItemCF:
    def __init__(self):
        self.item_sim_matrix = {}      # {item_i: {item_j: sim}}
        self.item_norm_sq = {}         # N_i = sum_u (r_ui)^2
        self.user_items_rating = {}    # 关键索引1：用户历史行为{user: {item: rating}}
        self.item_sim={}               # 关键索引2：每个物品最相似的n个物品{item:items}

    def fit(self, user_item_rating_list,top_k=10):
        """
        :param user_item_rating_list: List of (user_id, item_id, rating)
        :param top_k:每个物品索引topk个最相似的物品
        """
        # Step 1: 构建用户-物品-评分字典，并计算每个物品的 ||r_i||^2
        self.user_items_rating = defaultdict(dict)
        self.item_norm_sq = defaultdict(float)

        for user, item, rating in user_item_rating_list:
            self.user_items_rating[user][item] = rating # 构造关键索引1
            self.item_norm_sq[item] += rating * rating  # 求和

        # Step 2: 计算加权共现 C[i][j] = sum_u (r_ui * r_uj)，这部分是余弦相似度的分子
        cooccur = defaultdict(lambda: defaultdict(float)) # {item_i:{item_j:sum_score}}
        for user, item_rating_dict in self.user_items_rating.items():# {user: {item: rating}}
            items = list(item_rating_dict.keys()) # 当前用户交互过的所有物品列表
            n = len(items) # 数量
            for i in range(n):
                for j in range(i + 1, n):# 只遍历上三角，对称填充下三角
                    a, b = items[i], items[j]
                    r_ua = item_rating_dict[a] # user对a的rating
                    r_ub = item_rating_dict[b]
                    weight = r_ua * r_ub # 乘积
                    cooccur[a][b] += weight # 求和
                    cooccur[b][a] += weight  # 对称

        # Step 3: 计算余弦相似度矩阵
        self.item_sim_matrix = defaultdict(dict)
        for item_i, neighbors in cooccur.items():
            for item_j, score in neighbors.items():
                norm_i = self.item_norm_sq[item_i]
                norm_j = self.item_norm_sq[item_j]
                if norm_i > 0 and norm_j > 0:
                    # 分子与共同受众有关，分母无论是哪个用户都无所谓
                    sim = score / math.sqrt(norm_i * norm_j) # norm_i和norm_i已经是求和后的值了(这里是在做归一化)
                    self.item_sim_matrix[item_i][item_j] = sim

        # 构造关键索引2
        self.item_sim = defaultdict(dict)
        for item,neighbors in self.item_sim_matrix.items():
            top_neighbors = nlargest(top_k, neighbors.items(), key=lambda x: x[1])
            self.item_sim[item]=dict(top_neighbors)

        print("ItemCF training completed.")

    def itemCF_recommend(self, user_id, n_rec=100):
        """
        推荐时考虑用户对历史物品的评分强度
        score(j) = sum_{i in hist} r_ui * sim(i, j)
        """
        if user_id not in self.user_items_rating:
            return []

        user_hist = self.user_items_rating[user_id]  # {item: rating}
        item_scores = defaultdict(float) # {item: score}

        for item_i, r_ui in user_hist.items():
            if item_i not in self.item_sim:
                continue
            top_neighbors = self.item_sim[item_i]

            for item_j, sim_score in top_neighbors.items():
                if item_j in user_hist:
                    continue  # 不推荐已交互过的
                item_scores[item_j] += r_ui * sim_score  # 累加，用户对某物品的交互等级*物品与物品的相似度

        recs = nlargest(n_rec, item_scores.items(), key=lambda x: x[1]) # [(item:final_score)]
        res = [x[0] for x in recs]
        return set(res) # 只返回物品ID集合

class UserCF:
    def __init__(self):
        self.user_sim_matrix = {}       # {u: {v: sim}}
        self.user_top_sim={}                # 关键索引2：{u: {v: sim}}
        self.user_items_rating = {}     # 关键索引1：{user: {item: rating}}
        self.item_users = {}            # 反向索引：{item: {user: rating}}，用于快速找谁评过分

    def fit(self, user_item_rating_list,top_k=10):
        """
        训练 UserCF 模型（用户之间的相似度只用数量计算）
        :param user_item_rating_list: List of (user_id, item_id, rating)
        :param top_k:每个用户直接索引的邻居数
        """
        # Step 1: 构建正向和反向索引，并计算用户向量的 L2 范数平方
        self.user_items_rating = defaultdict(dict)
        self.item_users = defaultdict(dict)

        for user, item, rating in user_item_rating_list:
            self.user_items_rating[user][item] = rating
            self.item_users[item][user] = rating

        # Step 2: 计算用户共现（通过共同物品）
        # 优化：遍历每个物品，对该物品的所有用户两两组合
        cooccur = defaultdict(lambda: defaultdict(float))
        for item, user_rating_dict in self.item_users.items():
            users = list(user_rating_dict.keys())
            n = len(users)
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = users[i], users[j]
                    cooccur[u][v] += 1
                    cooccur[v][u] += 1  # 对称

        # Step 3: 计算相似度
        self.user_sim_matrix = defaultdict(dict)
        self.user_top_sim = defaultdict(dict)
        for u, neighbors in cooccur.items():
            for v, inner_prod in neighbors.items():
                sim = cooccur[u][v] / math.sqrt(len(self.user_items_rating[u].keys())*len(self.user_items_rating[v].keys()))
                self.user_sim_matrix[u][v] = sim

        for user,neighbors in self.user_sim_matrix.items():
            top_neighbors = nlargest(top_k, neighbors.items(), key=lambda x: x[1])
            self.user_top_sim[user]=dict(top_neighbors)

        print("UserCF training completed.")

    def userCF_recommend(self, user_id, n_rec=100):
        """
        为用户推荐物品
        :param user: 目标用户ID
        :param n_rec: 推荐 top-N 物品
        :return: set of item_id
        """
        if user_id not in self.user_items_rating:
            return []

        user_hist = set(self.user_items_rating[user_id].keys())
        item_scores = defaultdict(float) # {item:score}

        # 获取最相似的 top_k_sim_users 个用户
        if user_id not in self.user_top_sim:
            return []
        sim_users = self.user_top_sim[user_id] # {users:scores}
        # 遍历每个相似用户 v
        for v, sim_uv in sim_users.items():
            # 遍历 v 交互过但 user 未交互的物品
            for item, rating_v in self.user_items_rating[v].items():
                if item in user_hist:
                    continue
                # 加权累加：相似度 × v 对 item 的兴趣度分数
                item_scores[item] += sim_uv * rating_v

        # 返回 top-n_rec
        recs = nlargest(n_rec, item_scores.items(), key=lambda x: x[1])
        res=[x[0] for x in recs]
        return set(res)

class CFRecommender:
    """
    封装类，封装两种协同过滤召回方法
    """
    def __init__(self):
        self.user_cf=UserCF()
        self.item_cf=ItemCF()
        self.swing=Swing()
        self.n_rec=100

    def fit(self, user_item_rating_list,top_k=10):
        self.user_cf.fit(user_item_rating_list,top_k)
        self.item_cf.fit(user_item_rating_list,top_k)
        self.swing.fit(user_item_rating_list,top_k)

    def cf_recommend(self, user_id):
        user_cf_recalls=self.user_cf.userCF_recommend(user_id, self.n_rec)
        item_cf_recalls=self.item_cf.itemCF_recommend(user_id, self.n_rec)
        swing_recalls=self.swing.swing_recommend(user_id, self.n_rec)
        return user_cf_recalls | item_cf_recalls | swing_recalls

class ClassificationRecommender:
    """基于类目和关键词的召回"""
    def __init__(self):
        # 类目到物品列表的索引（按创建时间倒序排列）
        self.category_index = defaultdict(list)
        # 关键词到物品列表的索引（按创建时间倒序排列）
        self.keyword_index = defaultdict(list)
        # 所有物品的映射（用于快速查找）{item_id:item}
        self.items_map = {}

    def add_items(self, items):
        """添加物品到系统"""
        for item in items:
            self.items_map[item.item_id] = item
            # 添加到类目索引
            for category in item.categories:
                self.category_index[category].append(item)
            # 添加到关键词索引
            for keyword in item.keywords:
                self.keyword_index[keyword].append(item)

    def build_indices(self):
        """构建索引（按创建时间倒序排列）"""
        for category, items in self.category_index.items():
            # 按创建时间倒序排列
            self.category_index[category] = sorted(items, key=lambda x: x.create_time, reverse=True)

        for keyword, items in self.keyword_index.items():
            # 按创建时间倒序排列
            self.keyword_index[keyword] = sorted(items, key=lambda x: x.create_time, reverse=True)

    def category_recall(self, user_categories, topk=50)->set[Item]:
        """
        类目召回通道
        Args:
            user_categories: 用户感兴趣的类目列表
            topk: 每个类目召回的物品数量
        Returns:
            召回的物品id列表
        """
        recalled_items_ids = []

        for category in user_categories:
            if category in self.category_index:
                # 从对应类目中取出topk个物品
                category_items = self.category_index[category][:topk]
                recalled_items_ids.extend([item.item_id for item in category_items])

        # 去重并返回
        return set(recalled_items_ids)

    def keyword_recall(self, user_keywords, topk=50):
        """
        关键词召回通道
        Args:
            user_keywords: 用户感兴趣的关键词列表
            topk: 每个关键词召回的物品数量
        Returns:
            召回的物品列表
        """
        recalled_items_ids = []

        for keyword in user_keywords:
            if keyword in self.keyword_index:
                # 从对应关键词中取出topk个物品
                keyword_items = self.keyword_index[keyword][:topk]
                recalled_items_ids.extend([item.item_id for item in keyword_items])

        # 去重并返回
        return set(recalled_items_ids)

    def classification_recommend(self, user_profile, topk_per_channel=50):
        """
        基于类别的推荐主函数
        Args:
            user_profile: 用户画像对象
            topk_per_channel: 每个召回通道返回的物品数量
        Returns:
            召回池中的物品列表
        """
        # 从用户画像中获取兴趣类目和关键词
        user_categories = user_profile.categories
        user_keywords = user_profile.keywords

        # 类目召回
        category_recalled = self.category_recall(user_categories, topk=topk_per_channel)

        # 关键词召回
        keyword_recalled = self.keyword_recall(user_keywords, topk=topk_per_channel)

        # 合并召回结果（去重）
        return category_recalled.union(keyword_recalled)

class ClusteringRecommender:
    """基于聚类的推荐系统"""
    def __init__(self, n_clusters=50):
        """
        初始化聚类推荐器
        Args:
            n_clusters: 聚类数量
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_centers = None  # 聚类中心向量
        self.items_by_cluster = defaultdict(list)  # 每个聚类包含的物品{cluster:items}

        self.items_map = {}  # 物品ID到物品对象的映射
        self.all_item_features = []  # 所有物品的特征向量
        self.item_ids = []  # 物品ID列表，与特征向量顺序一致

    def add_items(self, items):
        """添加所有物品信息"""
        for item in items:
            self.items_map[item.item_id] = item
            self.all_item_features.append(item.content_feature.squeeze())
            self.item_ids.append(item.item_id)

    def fit_clustering(self):
        """训练聚类模型"""
        if not self.all_item_features:
            raise ValueError("没有物品数据可供聚类")

        # 将特征转换为numpy数组
        X = torch.stack(self.all_item_features).cpu().detach().numpy()

        # 训练K-means模型
        cluster_labels = self.kmeans.fit_predict(X)

        # 保存聚类中心
        self.cluster_centers = self.kmeans.cluster_centers_

        # 根据聚类标签组织物品
        for i, label in enumerate(cluster_labels):
            item_id = self.item_ids[i] # 取出id
            item = self.items_map[item_id] # 取出物品对象
            self.items_by_cluster[label].append(item)

    def get_nearest_cluster(self, seed_item):
        """
        找到种子物品最接近的聚类
        Args:
            seed_item: 种子物品对象
        Returns:
            最近聚类的索引
        """
        if self.cluster_centers is None:
            raise ValueError("聚类模型尚未训练")

        # 计算种子物品特征向量与各聚类中心的余弦相似度
        similarities = cosine_similarity([seed_item.content_feature], self.cluster_centers)[0]

        # 返回相似度最高的聚类索引
        nearest_cluster_idx = np.argmax(similarities)
        return nearest_cluster_idx

    def find_similar_items_in_cluster(self, seed_item, cluster_idx, m):
        """
        在指定聚类中找到与种子物品最相似的m篇物品
        Args:
            seed_item: 种子物品对象
            cluster_idx: 聚类索引
            m: 要返回的物品数量
        Returns:
            与种子物品最相似的物品列表
        """
        if cluster_idx not in self.items_by_cluster:
            return []

        cluster_items = self.items_by_cluster[cluster_idx] # 所有待选的同类物品

        # 计算种子物品与聚类中所有物品的余弦相似度
        similarities = []
        for item in cluster_items:
            # 避免返回种子物品本身
            if item.item_id == seed_item.item_id:
                continue

            sim = cosine_similarity([seed_item.content_feature], [item.content_feature])[0][0]
            similarities.append((item, sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前m个最相似的物品
        similar_items = [item[0] for item in similarities[:m]]
        return similar_items

    def clustering_recommend(self, user_profile, last_n=20, m=10)->set[int]:
        """
        基于聚类的召回方法
        Args:
            user_profile: 用户画像对象
            last_n: 使用用户最近交互的n个物品作为种子
            m: 每个种子物品在对应聚类中召回的物品数量
        Returns:
            召回的物品列表
        """
        if self.cluster_centers is None:
            raise ValueError("聚类模型尚未训练")

        # 获取用户最近交互的物品ID
        recent_item_ids = user_profile.get_last_n_interactions(last_n)

        if not recent_item_ids:
            return set()

        # 存储召回的物品（去重）
        recalled_items_ids = set()

        # 对每个种子物品进行处理
        for item_id in recent_item_ids:
            if item_id not in self.items_map:
                continue

            seed_item = self.items_map[item_id]

            # 找到种子物品所属的最近聚类
            nearest_cluster_idx = self.get_nearest_cluster(seed_item)

            # 在该聚类中找到最相似的m篇物品
            similar_items = self.find_similar_items_in_cluster(seed_item, nearest_cluster_idx, m)

            # 添加到召回结果中（去重）
            for item in similar_items:
                recalled_items_ids.add(item)

        return recalled_items_ids

class ColdStartRecommender:
    """
    封装类，封装用于冷启动的类目召回，关键词召回和内容向量聚类召回
    """
    def __init__(self,n_clusters=50):
        self.items=[]
        self.clustering_recommender = ClusteringRecommender(n_clusters=n_clusters)
        self.classification_recommender = ClassificationRecommender()

    def fit(self,items):
        self.items = items
        # 数据准备
        self.clustering_recommender.add_items(items)
        self.classification_recommender.add_items(items)
        # 聚类
        self.clustering_recommender.fit_clustering()
        # 类目和关键词
        self.classification_recommender.build_indices()

    def cold_start_recommend(self,user_profile)->set[int]:
        # 聚类
        clustering_recalls=self.clustering_recommender.clustering_recommend(user_profile=user_profile, last_n=20, m=10)
        # 类目和关键词
        classification_recalls=self.classification_recommender.classification_recommend(user_profile=user_profile,topk_per_channel=50)
        # 合并去重
        return classification_recalls.union(clustering_recalls)

class RecallRecommender:
    def __init__(self,item_idx2id):
        self.item_idx2id = item_idx2id
        self.items = None
        self.interactions = None
        # ============ 各项参数 ==============
        # 聚类簇数
        self.n_clusters = int(math.sqrt(len(item_idx2id)))
        # topk
        self.cf_topk=10 # 每个用户或物品索引topk个最相似的
        self.twin_towers_model_topk=50
        # LightGCN
        self.n_layers=2
        self.emb_dim=64
        # 各分推荐器
        self.cf_recommender = CFRecommender()
        self.cold_start_recommender = ColdStartRecommender(self.n_clusters)
        self.twin_towers_model_recommender=TwoTowersModelRecommender()
        self.light_gcn_recommender = LightGCNRecommender(item_idx2id)

    def fit(self, df_train, items, interactions, df_interactions, user_ids, item_ids):
        self.items = items
        self.interactions = interactions
        # 协同过滤和冷启动推荐器
        self.cf_recommender.fit(interactions, self.cf_topk)
        self.cold_start_recommender.fit(items)
        # 双塔模型推荐器
        self.twin_towers_model_recommender.train_twin_towers_model(df_train)
        self.twin_towers_model_recommender.fit(items)  # 计算物品特征向量，加入faiss
        # LightGCN
        self.light_gcn_recommender.train_light_gcn(self.n_layers, user_ids, item_ids, self.emb_dim, df_interactions)
        self.light_gcn_recommender.fit()

    def fit_with_weights(self, df_train, items, interactions, df_interactions, user_ids, item_ids
                         ,twin_towers_model_weights_path,light_gcn_weights_path):
        self.items = items
        self.interactions = interactions
        # 协同过滤和冷启动推荐器
        self.cf_recommender.fit(interactions, self.cf_topk)
        self.cold_start_recommender.fit(items)
        # 双塔模型推荐器
        self.twin_towers_model_recommender.load_twin_towers_model(df_train,twin_towers_model_weights_path)
        self.twin_towers_model_recommender.fit(items)  # 计算物品特征向量，加入faiss
        # LightGCN
        self.light_gcn_recommender.load_light_gcn(self.n_layers, user_ids, item_ids, self.emb_dim, df_interactions,light_gcn_weights_path)
        self.light_gcn_recommender.fit()

    def recall(self,user_profile:UserProfile)->set[int]:
        """
        只需传入当前要进行推荐的用户的用户画像
        """
        print("recall...")
        # 协同过滤召回
        cf_recall_items_ids = self.cf_recommender.cf_recommend(user_profile.user_id)
        # 冷启动召回
        cold_recall_items_ids = self.cold_start_recommender.cold_start_recommend(user_profile)
        # 双塔模型召回
        twin_recall_items_ids = self.twin_towers_model_recommender.recommend_for_user(
            user_profile,
            top_k=self.twin_towers_model_topk
        )
        # lightGCN召回
        light_gcn_recall_items_ids = self.light_gcn_recommender.recommend_for_user(user_profile.user_idx)
        # 合并去重
        middle_items_ids = cf_recall_items_ids.union(cold_recall_items_ids)
        middle_items_ids = middle_items_ids.union(twin_recall_items_ids)
        recall_items_ids = middle_items_ids.union(light_gcn_recall_items_ids)

        print(f"用户ID: {user_profile.user_id}")
        print(f"召回了共 {len(recall_items_ids)} 个物品,ID为:", end='')
        print(recall_items_ids)

        print("recall finished\n")

        return recall_items_ids

    def fine_tuning(self, df_items, df_users, df_interactions, new_df_interactions,items,flag_items,flag_users,flag_interactions):
        """
        Args:
            df_items: 全部物品数据
            df_users: 全部用户数据
            df_interactions: 全部交互数据
            new_df_interactions: 新增交互数据（可能为空）
            items: 全部物品对象列表
        """
        print("开始微调召回推荐器...")
        if flag_items:
            self.cold_start_recommender.fit(items)

        if flag_items or flag_users or flag_interactions:
            self.light_gcn_recommender.fine_tune_model_and_update_index(df_items, df_users,df_interactions)  # 包含了重建索引等fit过程

        if flag_interactions:
            self.twin_towers_model_recommender.fine_tune_model_and_update_processor(df_items, df_users, df_interactions,new_df_interactions)
            self.twin_towers_model_recommender.fit(items)
        elif flag_items:
            self.twin_towers_model_recommender.fit(items)

        if flag_interactions:
            self.interactions = list(zip(
                df_interactions['user_id'],
                df_interactions['item_id'],
                df_interactions['rating']
            ))
            self.cf_recommender.fit(self.interactions,10)


        print("召回推荐器微调完成")
