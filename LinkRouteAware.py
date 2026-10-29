import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as geom_nn

# ==================== 双视图图神经网络 ====================
class LinkRouteAwareGNN(nn.Module):
    """链路-路由感知的双视图图神经网络"""

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # 原始图转换层
        self.original_transform = nn.Linear(node_dim, hidden_dim)

        # 线图转换层
        self.line_transform = nn.Linear(edge_dim, hidden_dim)

        # 消息传递层
        self.edge_update_layers = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 节点更新层
        self.node_update_layers = nn.ModuleList([
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 边重要性预测层
        self.edge_importance = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def build_line_graph(self, edge_index, num_nodes):
        """构建线图（原始图的边变为线图的节点）"""
        num_edges = edge_index.size(1)

        # 创建线图节点映射: (src, dst) -> edge_id
        edge_mapping = {}
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            edge_mapping[(src, dst)] = i

        # 构建线图边索引
        line_edges = []
        for i in range(num_edges):
            src_i, dst_i = edge_index[:, i].tolist()

            # 找到所有以dst_i为起点的边
            for j in range(num_edges):
                if i == j: continue
                src_j, dst_j = edge_index[:, j].tolist()

                # 如果两条边共享一个节点（形成路径）
                if dst_i == src_j:
                    line_edges.append([i, j])

        if not line_edges:
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)

        return torch.tensor(line_edges, dtype=torch.long, device=edge_index.device).t()

    def encode_structure(self, node_states, edge_index, edge_features):
        """
        返回结构编码结果：
        - h_node: 原始图节点结构表示
        - h_edge: 线图边结构表示
        - importance_scores: 边重要性分数
        """
        num_nodes = node_states.size(0)
        num_edges = edge_index.size(1)

        # 原始图节点嵌入
        h_node = F.relu(self.original_transform(node_states))

        # 线图节点嵌入（原始图的边）
        h_edge = F.relu(self.line_transform(edge_features))

        # 构建线图
        line_edge_index = self.build_line_graph(edge_index, num_nodes)

        # 多轮消息传递
        for layer_idx in range(self.num_layers):
            # === 在原始图上聚合节点信息到边 ===
            src, dst = edge_index
            edge_messages = torch.cat([h_node[src], h_node[dst], h_edge], dim=1)
            h_edge_new = F.relu(self.edge_update_layers[layer_idx](edge_messages))

            # === 在线图上聚合路径信息 ===
            if line_edge_index.size(1) > 0:
                src_line, dst_line = line_edge_index
                path_messages = h_edge_new[src_line] + h_edge_new[dst_line]
                path_agg = geom_nn.global_mean_pool(path_messages, batch=None)
                path_agg = path_agg.repeat(num_edges, 1)
                h_edge_new = h_edge_new + path_agg

            # === 更新节点状态 ===
            # 聚合邻居边信息到节点
            edge_agg = geom_nn.global_add_pool(h_edge_new, dst)

            # 更新节点状态
            node_messages = torch.cat([h_node, edge_agg], dim=1)
            h_node = F.relu(self.node_update_layers[layer_idx](node_messages))

            # 保留残差连接
            h_edge = h_edge_new + h_edge

        # 计算边重要性分数
        src, dst = edge_index
        importance_input = torch.cat([h_node[src], h_node[dst], h_edge], dim=1)
        importance_scores = torch.sigmoid(self.edge_importance(importance_input))

        return h_node, h_edge, importance_scores.squeeze()

    def forward(self, node_states, edge_index, edge_features):
        _, _, importance_scores = self.encode_structure(
            node_states, edge_index, edge_features
        )
        return importance_scores
