import logging
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from feature_processor import FeatureProcessor
from utils import collate_fn_two_towers
from dataset import TwoTowerDataset
import faiss
from DCN import DCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# ============== 双塔模型 ================
# ======================================
class TwoTowersModel(nn.Module):
    def __init__(self, n_users, n_items, user_discrete_sizes, item_discrete_sizes,
                 user_cont_dim, item_cont_dim, embed_dim=32, tower_hidden=[128, 64]):
        """
        双塔模型
        :param n_users:用户 ID 的总数（用于 Embedding 层）
        :param n_items:物品 ID 的总数（用于 Embedding 层）
        :param user_discrete_sizes:列表，每个元素是某个用户离散特征的类别数(如2表示用户性别有2类)
        :param item_discrete_sizes:同上，针对物品离散特征
        :param user_cont_dim:用户连续特征维度（如 age, income → 维度=2）
        :param item_cont_dim:同上
        :param embed_dim:所有 Embedding 层的输出维度
        :param tower_hidden:每个塔内部 MLP 的隐藏层结构
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 用户塔嵌入层
        self.user_id_embed = nn.Embedding(n_users, embed_dim) # (词典大小，输出向量维度)
        self.user_discrete_embeds = nn.ModuleList([
            nn.Embedding(size, embed_dim) for size in user_discrete_sizes # 一个离散特征一个Embedding层
        ])
        user_input_dim = embed_dim + len(user_discrete_sizes) * embed_dim + user_cont_dim # concatenate之后的输入向量维度

        # 物品塔嵌入层
        self.item_id_embed = nn.Embedding(n_items, embed_dim)
        self.item_discrete_embeds = nn.ModuleList([
            nn.Embedding(size, embed_dim) for size in item_discrete_sizes
        ])
        item_input_dim = embed_dim + len(item_discrete_sizes) * embed_dim + item_cont_dim

        # 用户塔
        self.user_tower = DCN(2,user_input_dim,[64,32])

        # 物品塔
        self.item_tower = DCN(2,item_input_dim,[64,32])

    def forward_user(self, user_ids, user_discrete, user_continuous):
        """
        用户塔前向传播
        :param user_ids:[B]
        :param user_discrete:[B, num_user_discrete_cols]
        :param user_continuous:[B, user_cont_dim]
        :return:用户塔输出用户特征向量
        """
        emb_list = [self.user_id_embed(user_ids)]
        # 遍历每个离散特征列
        for i in range(user_discrete.size(1)):
            emb_list.append(self.user_discrete_embeds[i](user_discrete[:, i]))
        emb_list.append(user_continuous)
        concat_feat = torch.cat(emb_list, dim=1) # 针对batch中的每个用户，cat其id，离散特征和连续特征
        u_out = self.user_tower(concat_feat)
        return nn.functional.normalize(u_out, p=2, dim=1)  # L2 归一化

    def forward_item(self, item_ids, item_discrete, item_continuous):
        """
        物品塔前向传播
        :param item_ids:[B]
        :param item_discrete:[B, num_item_discrete_cols]
        :param item_continuous:[B, item_cont_dim]
        :return:物品塔输出物品特征向量
        """
        emb_list = [self.item_id_embed(item_ids)]
        for i in range(item_discrete.size(1)):
            emb_list.append(self.item_discrete_embeds[i](item_discrete[:, i]))
        emb_list.append(item_continuous)
        concat_feat = torch.cat(emb_list, dim=1)
        i_out = self.item_tower(concat_feat)
        return nn.functional.normalize(i_out, p=2, dim=1) # L2 归一化

    def forward(self, user_ids, user_discrete, user_continuous, item_ids, item_discrete, item_continuous):
        """
        整体前向传播
        :params 与 forward_user和 forward_item 相同
        :return:用户与物品的余弦相似度
        """
        u_vec = self.forward_user(user_ids, user_discrete, user_continuous)
        i_vec = self.forward_item(item_ids, item_discrete, item_continuous)
        # 余弦相似度
        cos_sim = torch.sum(u_vec * i_vec, dim=1) # 因为用户塔和物品塔的输出已经归一化，所以这里内积相当于余弦相似度
        return cos_sim


class TwoTowersModelRecommender:
    """双塔模型在线召回器"""
    def __init__(self):
        self.model = None
        self.processor = None
        # 定义特征列名(双塔模型需要的特征与三塔模型不同)
        self.user_discrete_cols = ['gender', 'user_categories', 'user_keywords']
        self.user_continuous_cols = ['age']
        self.item_discrete_cols = ['name','author', 'city', 'item_categories', 'item_keywords']
        self.item_continuous_cols = ['price']
        # 推荐需要用到的数据结构
        self.all_item_vectors = None  # 预计算的所有物品特征向量
        self.all_item_ids = []  # 所有物品ID列表
        self.item_features = {}  # 物品ID到物品特征的映射
        self.faiss_index = None # faiss向量数据库索引

    def train_twin_towers_model(self,df_train):
        print("twin towers model training...")
        # ============ 预处理 ============
        processor = FeatureProcessor()
        processor.build_vocab_and_scale(
            df_train, self.user_discrete_cols, self.item_discrete_cols,
            self.user_continuous_cols, self.item_continuous_cols
        )

        # ========== 模型与数据加载器 ==========
        model = TwoTowersModel(
            n_users=len(processor.user_id_vocab),
            n_items=len(processor.item_id_vocab),
            user_discrete_sizes=[len(processor.user_discrete_vocab[col]) for col in self.user_discrete_cols],
            item_discrete_sizes=[len(processor.item_discrete_vocab[col]) for col in self.item_discrete_cols],
            user_cont_dim=len(self.user_continuous_cols),
            item_cont_dim=len(self.item_continuous_cols),
            embed_dim=64,
            tower_hidden=[128, 64]
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        # 训练时正样本对:负样本对=1:2
        dataset = TwoTowerDataset(
            df_train, processor,
            self.user_discrete_cols, self.item_discrete_cols,
            self.user_continuous_cols, self.item_continuous_cols,
            n_neg=2
        )
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn_two_towers)

        # ========== 训练循环 ==========
        for epoch in range(10):
            model.train()
            total_loss = 0
            for batch in dataloader:
                user_feat = [x.to(device) for x in batch['user_feat']]
                pos_item_feat = [x.to(device) for x in batch['pos_item_feat']]
                neg_item_feats = [[x.to(device) for x in neg] for neg in batch['neg_item_feats']]

                pos_score = model(*user_feat, *pos_item_feat)
                neg_scores = []
                for neg_feat in neg_item_feats:
                    neg_score = model(*user_feat, *neg_feat)
                    neg_scores.append(neg_score)
                neg_scores = torch.stack(neg_scores, dim=1)

                batch_size = pos_score.size(0)
                # 一个正样本对的标签是1，两个负样本对的标签是-1
                targets = torch.cat([
                    torch.ones(batch_size, 1, device=device),
                    -torch.ones(batch_size, 2, device=device)
                ], dim=1)
                all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
                loss = criterion(all_scores, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("epoch %d, loss %f" % (epoch, total_loss/len(dataloader)))

        self.model = model
        self.processor = processor
        # 保存模型权重
        torch.save(model.state_dict(),"model_weights/improved_twin_towers_model.pth")

        print("twin towers model training finished")

    def fine_tune_model_and_update_processor(self, df_items, df_users, df_interactions,new_df_interactions):
        """
        使用每日新数据对模型进行微调，并更新Processor
        Args:
            df_items: 全部物品数据
            df_users: 全部用户数据
            df_interactions: 全部交互数据
            new_df_interactions: 新增交互数据（不为空）
        """
        if df_interactions.empty:
            logging.info("新增交互记录为空，不需微调双塔模型")
            return
        print("开始双塔模型微调和 processor 更新...")

        df_merge = pd.merge(new_df_interactions, df_users, how='left', on='user_id')
        df_daily_data = pd.merge(df_merge, df_items, how='left', on='item_id') # 用来微调训练

        all_df_merge = pd.merge(df_interactions, df_users, how='left', on='user_id')
        all_df_information = pd.merge(all_df_merge, df_items, how='left', on='item_id') # 用来重建processor

        # 更新Processor的词汇表和标准化器
        print("更新Processor...")

        # 重新构建词汇表（增量更新）
        new_user_ids = set(df_interactions['user_id']) | set(self.processor.user_id_vocab.keys())
        self.processor.user_id_vocab = {uid: i for i, uid in enumerate(sorted(new_user_ids))}

        new_item_ids = set(df_interactions['item_id']) | set(self.processor.item_id_vocab.keys())
        self.processor.item_id_vocab = {iid: i for i, iid in enumerate(sorted(new_item_ids))}

        # 重新构建离散特征词汇表（包含新特征值）
        for col in self.user_discrete_cols:
            unique_vals = set(all_df_information[col].values) | set(self.processor.user_discrete_vocab[col].keys())
            self.processor.user_discrete_vocab[col] = {val: i for i, val in enumerate(sorted(unique_vals))}

        for col in self.item_discrete_cols:
            unique_vals = set(all_df_information[col].values) | set(self.processor.item_discrete_vocab[col].keys())
            self.processor.item_discrete_vocab[col] = {val: i for i, val in enumerate(sorted(unique_vals))}

        # 重新拟合标准化器（包含新数据）
        if self.user_continuous_cols:
            all_user_data = all_df_information[self.user_continuous_cols]
            self.processor.user_cont_scaler.partial_fit(all_user_data.values)

        if self.item_continuous_cols:
            all_item_data = all_df_information[self.item_continuous_cols]
            self.processor.item_cont_scaler.partial_fit(all_item_data.values)

        print(f"更新完成: 用户数 {len(self.processor.user_id_vocab)}, 物品数 {len(self.processor.item_id_vocab)}")

        # 3. 微调模型
        print("开始双塔模型微调...")
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # 使用较小的学习率
        criterion = nn.MSELoss()

        # 创建微调数据集
        dataset = TwoTowerDataset(
            df_daily_data, self.processor,
            self.user_discrete_cols, self.item_discrete_cols,
            self.user_continuous_cols, self.item_continuous_cols,
            n_neg=1  # 微调时可以减少负采样数量以加快速度
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_two_towers)

        self.model.train()
        total_loss = 0
        for batch in dataloader:
            user_feat = [x.to(device) for x in batch['user_feat']]
            pos_item_feat = [x.to(device) for x in batch['pos_item_feat']]
            neg_item_feats = [[x.to(device) for x in neg] for neg in batch['neg_item_feats']]

            pos_score = self.model(*user_feat, *pos_item_feat)
            neg_scores = []
            for neg_feat in neg_item_feats:
                neg_score = self.model(*user_feat, *neg_feat)
                neg_scores.append(neg_score)
            neg_scores = torch.stack(neg_scores, dim=1)

            batch_size = pos_score.size(0)
            targets = torch.cat([
                torch.ones(batch_size, 1, device=device),
                -torch.ones(batch_size, len(neg_item_feats), device=device)
            ], dim=1)
            all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
            loss = criterion(all_scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 保存模型权重
        # os.makedirs("model_weights", exist_ok=True)
        torch.save(self.model.state_dict(), "../model_weights/twin_towers_model.pth")
        print(f"双塔模型微调完成")


    def load_twin_towers_model(self,df_train,path):
        """用训练好的权重直接载入模型"""
        print("twin towers model loading...")
        # ============ 预处理 ============
        processor = FeatureProcessor()
        processor.build_vocab_and_scale(
            df_train, self.user_discrete_cols, self.item_discrete_cols,
            self.user_continuous_cols, self.item_continuous_cols
        )

        # ========== 模型与数据加载器 ==========
        model = TwoTowersModel(
            n_users=len(processor.user_id_vocab),
            n_items=len(processor.item_id_vocab),
            user_discrete_sizes=[len(processor.user_discrete_vocab[col]) for col in self.user_discrete_cols],
            item_discrete_sizes=[len(processor.item_discrete_vocab[col]) for col in self.item_discrete_cols],
            user_cont_dim=len(self.user_continuous_cols),
            item_cont_dim=len(self.item_continuous_cols),
            embed_dim=64,
            tower_hidden=[128, 64]
        ).to(device)

        model.load_state_dict(torch.load(path))
        self.model=model
        self.processor = processor

        print("twin towers model loading finished")

    def preprocess_user_feature(self, user_profile):
        """
        预处理用户特征，转换为模型输入格式
        Args:
            user_profile: 用户画像
        Returns:
            处理后的用户特征张量 (user_ids, user_discrete, user_continuous)
        """
        # 针对字典的**解包操作，构造单行Dataframe对象
        user_df = pd.DataFrame([{
            'user_id': user_profile.user_id,
            **user_profile.discrete_features,
            **user_profile.continuous_features
        }])

        user_ids, user_discrete, user_cont = self.processor.transform_user_features(
            user_df, self.user_discrete_cols, self.user_continuous_cols
        )

        return user_ids.to(device), user_discrete.to(device), user_cont.to(device)

    def preprocess_item_feature(self, item_feature):
        """
        预处理物品特征，转换为模型输入格式
        Args:
            item_feature: ItemFeature对象
        Returns:
            处理后的物品特征张量 (item_ids, item_discrete, item_continuous)
        """
        item_df = pd.DataFrame([{
            'item_id': item_feature.item_id,
            **item_feature.discrete_features,
            **item_feature.continuous_features
        }])

        item_ids, item_discrete, item_cont = self.processor.transform_item_features(
            item_df, self.item_discrete_cols, self.item_continuous_cols
        )

        return item_ids.to(device), item_discrete.to(device), item_cont.to(device)

    def compute_all_item_vectors(self, all_item_features):
        """
        离线计算所有物品的特征向量
        Args:
            all_item_features: 所有物品特征列表
        """
        print("正在计算所有物品的特征向量...")

        # 准备批量数据
        item_ids_list = []
        item_discrete_list = []
        item_continuous_list = []

        for item_feat in all_item_features:
            # 存储物品特征映射和所有物品ID列表
            self.item_features[item_feat.item_id] = item_feat
            self.all_item_ids.append(item_feat.item_id)
            item_ids, item_discrete, item_cont = self.preprocess_item_feature(item_feat)
            item_ids_list.append(item_ids)
            item_discrete_list.append(item_discrete)
            item_continuous_list.append(item_cont)

        # 批量处理
        all_item_ids = torch.cat(item_ids_list, dim=0)
        all_item_discrete = torch.cat(item_discrete_list, dim=0)
        all_item_continuous = torch.cat(item_continuous_list, dim=0)

        # 使用物品塔计算所有物品特征向量
        self.model.eval()
        with torch.no_grad():
            self.all_item_vectors = self.model.forward_item(
                all_item_ids, all_item_discrete, all_item_continuous
            )
        print(f"完成计算，共{len(self.all_item_ids)}个物品的特征向量")

    def initialize_faiss_index(self):
        """
        初始化 Faiss 索引
        计算好的物品特征向量存入向量数据库，后续用 ANN 高效匹配 topk
        """
        print("twin towers model faiss index building...")
        if self.all_item_vectors is None:
            raise ValueError("请先调用 compute_all_item_vectors 计算所有物品特征向量")

        # 获取向量维度
        vector_dim = self.all_item_vectors.shape[1]
        # 获取向量个数
        n_vectors = self.all_item_vectors.shape[0]
        # 转换为 numpy float32 格式，这是 Faiss 的标准格式
        item_vectors_np = self.all_item_vectors.cpu().numpy().astype('float32')

        # 根据物品数量选择索引类型
        if n_vectors < 1000:  # 小数据量使用精确搜索
            index = faiss.IndexFlatIP(vector_dim)  # 余弦相似度(向量已归一化)
            index.add(item_vectors_np)
        else:  # 大数据量使用近似搜索
            # IVF (Inverted File) 参数
            n_cluster = int(math.sqrt(n_vectors)) # 聚类中心数量
            # PQ (Product Quantization) 参数
            # M 是切分成的段数，因此向量维度要能整除M，物品塔的输出维度是32
            M = 8
            quantizer = faiss.IndexFlatIP(vector_dim)
            index = faiss.IndexIVFPQ(quantizer, vector_dim, n_cluster, M, 8)
            index.train(item_vectors_np)
            index.add(item_vectors_np)

        # 保存索引
        self.faiss_index = index

        print("twin towers model faiss index finished")

    def fit(self,all_item_features):
        """
        可用于系统初始化，也可用于每日的微调时重建索引，逻辑是一样的，只需传入全部的物品对象列表
        """
        self.compute_all_item_vectors(all_item_features)
        self.initialize_faiss_index()

    def recommend_for_user(self, user_profile, top_k=50):
        """
        使用 Faiss 向量数据库为特定用户进行召回
        Args:
            user_profile: 用户画像
            top_k: 召回物品数量
        Returns:
            召回的物品ID列表
        """
        if hasattr(self, 'faiss_index') and self.faiss_index is not None:
            # 计算用户特征向量
            user_ids, user_discrete, user_continuous = self.preprocess_user_feature(user_profile)

            self.model.eval()
            with torch.no_grad():
                user_vector = self.model.forward_user(user_ids, user_discrete, user_continuous)

            # 转换为 numpy 格式并确保是 float32
            user_vector_np = user_vector.cpu().numpy().astype('float32')

            # 使用 Faiss 进行快速相似度搜索
            # 设置搜索参数 (nprobe越大越精确但越慢)
            self.faiss_index.nprobe = 5
            # 搜索 top_k 个最相似的向量
            distances, indices = self.faiss_index.search(user_vector_np, top_k)

            # 将返回的索引映射回原始物品ID
            recommended_item_ids = [self.all_item_ids[i] for i in indices[0]]
            return set(recommended_item_ids)
        else:
            raise ValueError("请先调用 initialize_faiss_index 构建 Faiss 索引")

if __name__ == "__main__":
    # 加载训练数据
    print("data preparing...")
    # 准备物品数据
    df_items = pd.read_csv("data/items_new.csv", encoding="utf-8")
    df_items['item_keywords'] = df_items['item_keywords'].apply(lambda x: tuple(x.split(';')))
    # 准备用户数据
    df_users = pd.read_csv("data/users_new.csv", encoding="utf-8")
    df_users['user_categories'] = df_users['user_categories'].fillna('').apply(lambda x: tuple(x.split(';')))
    df_users['user_keywords'] = df_users['user_keywords'].fillna('').apply(lambda x: tuple(x.split(';')))
    # 准备交互数据
    df_interactions = pd.read_csv("data/interactions_new.csv", encoding="utf-8")
    # 准备双塔模型，三塔模型和精排多目标模型训练数据(根据交互记录)
    df_merge = pd.merge(df_interactions, df_users, how='left', on='user_id')
    df_train = pd.merge(df_merge, df_items, how='left', on='item_id')
    print("data finished\n")

    # 训练并保存权重
    recommender = TwoTowersModelRecommender()
    recommender.train_twin_towers_model(df_train)
