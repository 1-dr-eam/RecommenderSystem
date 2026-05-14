from collections import defaultdict
from heapq import nlargest

class Swing:
    def __init__(self):
        self.user_items_rating = {} # {user_id:{item_id:rating}}
        self.user_items = {}    # {user_id:item_ids}
        self.item_users = {} # {item_id:user_ids}
        self.item_sim_matrix = {}      # {item_i: {item_j: sim}}
        self.item_sim={}               # 关键索引2：每个物品最相似的n个物品{item:items}

    def fit(self, user_item_rating_list,top_k=10):
        """
        Args:
            user_item_rating_list: List of (user_id, item_id, rating)
            top_k:每个物品索引topk个最相似的物品
        """
        self.user_items_rating = defaultdict(dict) # {user_id:{item_id:rating}}
        # Step 1: 构建用户-物品字典和物品-用户字典，保存用户喜欢的物品列表和物品的受众
        self.user_items = defaultdict(set) # {user_id:item_ids}
        self.item_users = defaultdict(set) # {item_id:user_ids}

        for user_id, item_id, rating in user_item_rating_list:
            self.user_items[user_id].add(item_id)
            self.item_users[item_id].add(user_id)
            self.user_items_rating[user_id][item_id] = rating

        # Step 2: 计算用户喜欢的物品重合度
        n_users=len(self.user_items)
        overlap = defaultdict(dict) # {user:{user:overlap}}
        user_ids=list(self.user_items)
        for i in range(n_users):
            for j in range(i+1, n_users):
                _overlap = len(self.user_items[user_ids[i]] & self.user_items[user_ids[j]])
                overlap[user_ids[i]][user_ids[j]] = overlap[user_ids[j]][user_ids[i]] = _overlap

        # Step 3: 计算物品相似度
        self.item_sim_matrix = defaultdict(lambda: defaultdict(float))
        n_items=len(self.item_users)
        item_ids=list(self.item_users)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                item_i,item_j=item_ids[i],item_ids[j]
                co_users = self.item_users[item_i] & self.item_users[item_j]
                list_co_users=list(co_users)
                for m in range(len(list_co_users)):
                    for n in range(m+1, len(list_co_users)):
                        self.item_sim_matrix[item_i][item_j] += 1/(1.0+overlap[list_co_users[m]][list_co_users[n]])

        # 构造关键索引2
        self.item_sim = defaultdict(dict)
        for item,neighbors in self.item_sim_matrix.items():
            top_neighbors = nlargest(top_k, neighbors.items(), key=lambda x: x[1])
            self.item_sim[item]=dict(top_neighbors)

        print("ItemCF training completed.")

    def swing_recommend(self, user_id, n_rec=100):
        """
        推荐时考虑用户对历史物品的评分强度
        score(j) = sum{i in hist} r_ui * sim(i, j)
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