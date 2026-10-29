import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as geom_nn
from torch_geometric.utils import degree, to_dense_adj
import gc
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


infection_rate=0.1

class GraphTransformer(nn.Module):
    """图Transformer系统"""

    def __init__(self, node_dim=4, time_dim=8, k_hop=2,
                 beta=-0.5, dropout=0.1, pre_ln=True):
        super().__init__()
        self.to(device)
        # 动态参数配置
        self.k_hop = k_hop
        self.beta = beta
        self.dropout = dropout

        # 增强时间编码
        self.time_enc = AdaptiveTimeEncoding(output_dim=time_dim).to(device)

        # 动态位置编码维度 = 聚类系数(1) + k_hop路径特征
        sw_dim = 1 + k_hop

        # Graph Transformer层
        self.gt_layer = EnhancedGraphTransformerLayer(
            in_dim=node_dim + time_dim + sw_dim,
            out_dim=16,
            num_heads=2,
            k_hop=k_hop,
            beta=beta,
            dropout=dropout,
            pre_ln=pre_ln
        ).to(device)

        # 动态边预测器
        self.edge_predictor = DynamicEdgePredictor(
            node_dim=node_dim,
            time_dim=time_dim,
            k_hop=k_hop,
            beta=beta
        ).to(device)

        # 自适应小世界正则化
        self.sw_reg = AdaptiveSWRegularizer().to(device)

        # 初始化参数
        self._init_weights()

        self.input_proj = nn.Linear(4, node_dim).to(device)
        self.local_proj = nn.Linear(16, 4).to(device)
        self.global_proj = nn.Linear(16, 4).to(device)

        self.register_buffer('delta_t',torch.tensor(1.0))

    def forward(self, data):
        # 解构输入数据
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_nodes = x.size(0)  # 动态获取节点数

        # 确保edge_attr是二维张量 [num_edges, 1]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 计算动态小世界特征
        sw_feat = self._compute_sw_features(edge_index, num_nodes)

        # 增强时间编码
        delta_t = self.delta_t.to(x.device)
        time_feat = self.time_enc(delta_t, x)

        # 统一维度处理
        proj_x = self.input_proj(x)
        sw_feat = sw_feat.view(num_nodes, -1)
        time_feat = time_feat.view(num_nodes, -1)

        # 确保特征维度匹配
        if proj_x.size(0) != sw_feat.size(0):
            sw_feat = sw_feat[:proj_x.size(0)]
        if proj_x.size(0) != time_feat.size(0):
            time_feat = time_feat[:proj_x.size(0)]

        # 特征拼接
        enhanced_x = torch.cat([
            proj_x,
            sw_feat,
            time_feat
        ], dim=-1)

        # 图Transformer处理
        h_local_gt, h_global_gt, h = self.gt_layer(enhanced_x, edge_index,edge_attr=edge_attr)

        # 返回局部和全局效应
        return self.local_proj(h_local_gt), self.global_proj(h_global_gt)

    def encode_representations(self, data, detach=False):
        """
        - H_struct: 节点级结构聚合图表示
        - H_dyn: 图级动力学表示
        - H_dyn_nodes: 节点级动力学表示，供需要节点粒度的模块使用
        """
        h_struct_nodes, h_dyn_nodes = self.forward(data)
        h_dyn_graph = h_dyn_nodes.mean(dim=0)
        if detach:
            return (
                h_struct_nodes.detach(),
                h_dyn_graph.detach(),
                h_dyn_nodes.detach(),
            )
        return h_struct_nodes, h_dyn_graph, h_dyn_nodes

    def _compute_sw_features(self, edge_index,num_nodes):
        """动态小世界特征计算"""
        # 确保邻接矩阵正确创建
        if edge_index.size(1) == 0:  # 如果没有边
            adj = torch.zeros(num_nodes, num_nodes, device=device)
        else:
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

        # 确保邻接矩阵是方阵
        if adj.dim() < 2:
            adj = adj.unsqueeze(0)
        if adj.size(0) != num_nodes or adj.size(1) != num_nodes:
            adj = torch.zeros(num_nodes, num_nodes, device=device)

        # 动态聚类系数
        with torch.no_grad():  # 不需要梯度，节省内存
            triu = torch.triu(adj)
            triangles = torch.matmul(triu, triu) * triu
            degrees = adj.sum(1)
            # 处理除零错误
            denominator = degrees * (degrees - 1)
            denominator[denominator == 0] = 1e-6  # 避免除以零
            clustering = triangles.sum(1) / (adj.sum(1) * (adj.sum(1) - 1) + 1e-6)
            clustering = torch.nan_to_num(clustering, nan=0.0)  # 处理NaN值

            # 多跳路径统计
            path_counts = []
            adj_power = adj.clone()
            max_hops = max(1, min(self.k_hop, 2))  # 至少1跳，最多2跳
            for _ in range(max_hops):
                adj_power = torch.matmul(adj_power, adj)
                path_counts.append(adj_power.sum(1))

            # 如果路径特征为空，添加零向量
            if len(path_counts) == 0:
                path_counts.append(torch.zeros(num_nodes, device=device))

            path_feat = torch.stack(path_counts, dim=-1)

        return torch.cat([clustering.unsqueeze(-1), path_feat], dim=-1)

    def _init_weights(self):
        # 参数初始化策略
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.constant_(param, val=0.1)


class RepresentationPretrainer(nn.Module):
    """
    对 GraphTransformer 产生的两类表示做预训练：
    - H_dyn 预测下一时刻宏观 SICR 密度；
    - H_struct 预测结构统计量（默认使用平均节点度）。
    """

    def __init__(self, struct_dim=4, dyn_dim=4, hidden_dim=64):
        super().__init__()
        self.dyn_head = nn.Sequential(
            nn.Linear(dyn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.struct_head = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_struct_nodes, h_dyn_graph):
        if h_dyn_graph.dim() == 1:
            h_dyn_graph = h_dyn_graph.unsqueeze(0)
        if h_struct_nodes.dim() == 2:
            h_struct_graph = h_struct_nodes.mean(dim=0, keepdim=True)
        else:
            h_struct_graph = h_struct_nodes
        return self.dyn_head(h_dyn_graph).squeeze(0), self.struct_head(h_struct_graph).squeeze()

    def compute_loss(
        self,
        h_struct_nodes,
        h_dyn_graph,
        z_next,
        structure_target,
        lambda_struct=0.5,
    ):
        pred_dyn, pred_struct = self.forward(h_struct_nodes, h_dyn_graph)
        z_next = z_next.to(pred_dyn.device).float().view_as(pred_dyn)
        structure_target = structure_target.to(pred_struct.device).float().view_as(pred_struct)
        loss_dyn = F.mse_loss(pred_dyn, z_next)
        loss_struct = F.mse_loss(pred_struct, structure_target)
        return loss_dyn + lambda_struct * loss_struct, loss_dyn, loss_struct


class EnhancedGraphTransformerLayer(nn.Module):
    """增强的图Transformer层（整合核注意力机制）"""

    def __init__(self, in_dim, out_dim=16, num_heads=2, k_hop=2,
                 beta=-0.5, dropout=0.1, pre_ln=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.to(device)

        #计算适配器后的实际输入维度
        self.actual_in_dim = num_heads * self.head_dim

        # 确保输入维度与输出维度兼容
        if in_dim != num_heads * self.head_dim:
            # 添加维度适配层
            self.dim_adapter = nn.Linear(in_dim, num_heads * self.head_dim).to(device)
            self.adapter_active = True
        else:
            self.adapter_active = False
            self.dim_adapter = None

        # 局部-全局双通路结构
        self.local_attn = geom_nn.TransformerConv(
            self.actual_in_dim,
            self.head_dim,
            heads=num_heads,
            edge_dim=1,
            dropout=dropout,
            root_weight=False
        ).to(device)

        # 核注意力全局通路
        self.global_attn = KernelGlobalAttention(
            hidden_channels=self.head_dim,
            heads=num_heads,
            num_layers=1,
            beta=beta,
            dropout=dropout
        ).to(device)

        # 动态参数系统
        self.beta = nn.Parameter(torch.zeros(1, self.actual_in_dim)) if beta < 0 \
            else nn.Parameter(torch.ones(1, self.actual_in_dim) * beta)

        # 归一化层
        self.pre_ln = nn.LayerNorm(out_dim) if pre_ln else None
        self.post_ln = nn.LayerNorm(out_dim).to(device)

        # 动态位置编码
        self.pos_enc = EvolvingPositionalEncoding(out_dim,max_hop=2).to(device)

        # 残差门控
        self.res_gate = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

        # 投影层
        self.proj = nn.Linear(self.actual_in_dim, out_dim).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index,edge_attr):
        # 确保edge_attr是二维的
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 如果存在维度不匹配，使用适配器
        if self.adapter_active:
            x = self.dim_adapter(x)

        # 动态位置增强
        orig_shape = x.shape
        x = self.pos_enc(x, edge_index)

        # 确保维度恢复
        if x.shape != orig_shape:
            x = x[:, :orig_shape[1]]

        # 预归一化
        identity = x
        if self.pre_ln:
            x = self.pre_ln(x)

        # 局部注意力通路
        h_local = self.local_attn(x, edge_index,edge_attr=edge_attr)
        # 确保投影层输入维度正确
        if h_local.size(-1) != self.num_heads * self.head_dim:
            h_local = self.proj(h_local)

        # 全局注意力通路
        h_global = self.global_attn(x)
        # 确保投影层输入维度正确
        if h_global.size(-1) != self.num_heads * self.head_dim:
            h_global = self.proj(h_global)

        # 动态融合
        beta = torch.sigmoid(self.beta) if self.beta.requires_grad else self.beta
        h = beta * h_local + (1 - beta) * h_global
        h = self.dropout(h)

        # 残差门控
        gate = self.res_gate(identity)
        normalized_h = F.layer_norm(h, (h.size(-1),))
        out = gate * identity + (1 - gate) * normalized_h

        return h_local, h_global, F.gelu(out)


class KernelGlobalAttention(nn.Module):
    """核驱动的全局注意力机制"""

    def __init__(self, hidden_channels, heads=2, num_layers=1,
                 beta=-0.5, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.layers = nn.ModuleList([
            KernelAttentionLayer(
                hidden_channels=hidden_channels,
                heads=heads,
                beta=beta,
                dropout=dropout,
                layer_idx=i,
                total_layers=num_layers
            ) for i in range(num_layers)
        ])
        self.to(device)

        # 输入维度适配器
        self.input_adapter = None
        self.adapter_active = False

    def forward(self, x):
        # 检查输入维度
        expected_dim = self.heads * self.hidden_channels
        if x.size(-1) != expected_dim:
            # 动态创建适配器
            if not self.adapter_active:
                self.input_adapter = nn.Linear(x.size(-1), expected_dim).to(device)
                self.adapter_active = True
            x = self.input_adapter(x)

        for layer in self.layers:
            x = layer(x)
        return x


class KernelAttentionLayer(nn.Module):
    """核注意力层"""

    def __init__(self, hidden_channels, heads=2, beta=-0.5, dropout=0.1,
                 layer_idx=0, total_layers=1):
        super().__init__()
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.layer_idx = layer_idx
        self.to(device)

        # 动态参数
        self.beta = nn.Parameter(torch.zeros(1, heads * hidden_channels)) if beta < 0 \
            else nn.Parameter(torch.ones(1, heads * hidden_channels) * beta)

        # 核变换网络
        self.qk_net = nn.Sequential(
            nn.Linear(heads * hidden_channels, 2 * heads * hidden_channels),
            nn.GELU(),
            nn.Linear(2 * heads * hidden_channels, 2 * heads * hidden_channels)
        ).to(device)

        self.v_net = nn.Sequential(
            nn.Linear(heads * hidden_channels, 2* heads * hidden_channels),
            nn.GLU()
        ).to(device)

        # 输出变换
        self.out_net = nn.Sequential(
            nn.Linear(heads * hidden_channels, heads * hidden_channels),
            nn.Dropout(dropout)
        ).to(device)

        # 归一化
        self.norm = nn.LayerNorm(heads * hidden_channels).to(device)
        self.path_norm = nn.LayerNorm(heads * hidden_channels) if layer_idx < total_layers - 1 else None

    def forward(self, x):

        # 强制垃圾回收释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        N, D = x.shape
        H, C = self.heads, self.hidden_channels

        # 核特征变换
        qk = self.qk_net(x).view(N, 2, H, C)
        q, k = qk.unbind(1)
        v = self.v_net(x).view(N, H, C)

        # 核函数应用
        q = torch.sigmoid(q)
        k = torch.sigmoid(k)

        # 高效注意力计算（内存优化）
        # 1. 计算分母部分
        k_sum = k.sum(dim=0)  # [H, C]
        qk_dot = torch.einsum('nhc,hc->nh', q, k_sum)  # [N, H]
        z = 1 / (qk_dot + 1e-6)  # [N, H]

        # 2. 计算分子部分（分步计算避免大张量）
        h = torch.zeros(N, H, C, device=x.device)  # 预分配输出

        # 按头部分步计算
        for head_idx in range(H):
            # 计算当前头的kv
            kv_head = torch.einsum('nc,nm->cm', k[:, head_idx], v[:, head_idx])

            # 计算当前头的输出
            q_head = q[:, head_idx]  # [N, C]
            h_head = torch.einsum('nc,cm->nm', q_head, kv_head)  # [N, M]
            h_head = h_head * z[:, head_idx].unsqueeze(-1)  # [N, M]

            # 存储结果
            h[:, head_idx] = h_head[:, :C]  # 截断到C维度

        # 3. 合并头
        h = h.reshape(N, -1)

        # 输出变换
        h = self.out_net(h)

        # 残差连接
        if self.layer_idx < 1:
            h = self.norm(h)
            beta = torch.sigmoid(self.beta) if self.beta.requires_grad else self.beta
            h = beta * h + (1 - beta) * x
            if self.path_norm is not None:
                h = self.path_norm(h)

        return h






class EvolvingPositionalEncoding(nn.Module):
    """动态演化位置编码系统"""

    def __init__(self, dim, max_hop=2, temporal_scale=0.1):
        super().__init__()
        self.temporal_scale = temporal_scale
        self.target_dim = dim
        self.pos_emb = nn.Embedding(max_hop + 2, dim)  # 保留0-hop和max_hop+1
        self.dist_adaptor = nn.Linear(dim, dim)
        self.time_adaptor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
        self.output_adapter = nn.Linear(dim, dim)
        self.max_hop = max_hop
        self.device=device

    def forward(self, x, edge_index):
        num_nodes = x.size(0)  # 动态获取节点数
        # 稀疏距离计算（使用邻接矩阵避免 BFS）
        with torch.no_grad():
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            # 计算距离矩阵
            dist = self.floyd_warshall(adj.cpu().numpy())
            dist = torch.tensor(dist, dtype=torch.long, device=x.device)

        # 限制距离不超过 max_hop
        dist = dist.clamp(0, self.max_hop)

        # 位置编码
        pe = self.pos_emb(dist)  # [N, N, D]
        pe = pe.mean(dim=1)  # [N, D] 平均距离编码

        # 通过适配器调整维度
        pe = self.dist_adaptor(pe)
        pe = self.output_adapter(pe)

        # 时间融合
        t_feat = torch.sigmoid(self.temporal_scale * x.mean(dim=-1)).unsqueeze(-1)
        pe = pe * t_feat
        pe = self.time_adaptor(pe)

        # 确保维度匹配
        if pe.size(-1) != x.size(-1):
            pe = pe[..., :x.size(-1)]
        if pe.size(0) != x.size(0):
            pe = pe[:x.size(0)]

        return x + pe

    def floyd_warshall(self, adj):
        """Floyd-Warshall 算法计算最短路径 (CPU)"""
        n = adj.shape[0]
        dist = np.where(adj > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # 将无穷大替换为 max_hop+1
        dist[dist == np.inf] = self.max_hop + 1
        return dist.astype(np.int32)


class DynamicEdgePredictor(nn.Module):
    """动态边预测器（带拓扑演化）"""

    def __init__(self, node_dim, time_dim, k_hop=3, beta=-0.5, threshold=0.65):
        super().__init__()
        self.k_hop = k_hop
        self.threshold = threshold

        # 多跳拓扑编码器
        self.top_enc = nn.Sequential(
            nn.Linear(k_hop * 2, 32),
            nn.ReLU(),
            nn.Linear(32, time_dim)
        )

        # 动态融合网络
        self.fusion = nn.Linear(2 * node_dim + 2 * time_dim, 128)

        # 概率预测
        self.prob_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 边属性生成
        self.attr_gen = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.beta = beta

    def forward(self,x, h, edge_index, delta_t):
        num_nodes = x.size(0)  # 动态获取节点数
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

        # 候选边生成（排除现有边）
        candidates = (adj == 0).nonzero(as_tuple=False)

        # 多跳拓扑特征
        top_feat = self._get_topological_feature(adj, candidates, num_nodes)

        # 时间编码融合
        time_feat = torch.sin(delta_t * math.pi / 24).unsqueeze(1)  # 周期编码

        # 特征拼接
        h_src = h[candidates[:, 0]]
        h_dst = h[candidates[:, 1]]
        fused = self.fusion(torch.cat([
            h_src, h_dst,
            top_feat,
            time_feat.expand(len(top_feat), -1)
        ], dim=-1))

        # 概率计算
        probs = self.prob_net(fused).squeeze()

        # 新边生成
        selected = probs > self.threshold
        new_edges = candidates[selected]
        attrs = self.attr_gen(fused[selected])

        return probs, (new_edges.t(), attrs)

    def _get_topological_feature(self, adj, candidates):
        """提取多跳拓扑特征"""
        features = []
        for i in range(self.k_hop):
            adj_power = torch.matrix_power(adj.float(), i + 1)
            src_feat = adj_power[candidates[:, 0]].sum(dim=1)
            dst_feat = adj_power[candidates[:, 1]].sum(dim=1)
            features.extend([src_feat, dst_feat])
        features = torch.stack(features, dim=-1)
        return self.top_enc(features)


class AdaptiveSWRegularizer(nn.Module):
    """自适应小世界正则化器"""

    def __init__(self, alpha=0.2, temp=0.5):
        super().__init__()
        self.alpha = alpha
        self.temp = temp

        # 特征适配器
        self.feat_adaptor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 预测clustering和path的目标值
        )

    def forward(self, h, new_edges, orig_edges):
        # 原始图特征
        orig_adj = to_dense_adj(orig_edges).squeeze(0)
        orig_clustering = self._clustering_coef(orig_adj)

        # 新图特征
        new_adj = orig_adj.clone()
        if new_edges and new_edges[0].numel() > 0:
            new_adj[new_edges[0][0], new_edges[0][1]] = 1
        new_clustering = self._clustering_coef(new_adj)

        # 特征预测
        pred_target = self.feat_adaptor(h.mean(dim=1)).sigmoid()

        # 损失计算
        loss = F.mse_loss(new_clustering, pred_target[:, 0]) + \
               F.kl_div(torch.log(new_clustering + 1e-6), orig_clustering, reduction='batchmean')

        return self.alpha * loss

    def _clustering_coef(self, adj):
        """动态聚类系数计算"""
        triu = adj.triu(diagonal=1)
        triangles = torch.mm(triu, triu) * triu
        degrees = adj.sum(dim=1)
        coef = triangles.sum(dim=1) / (degrees * (degrees - 1) + 1e-6)
        return torch.nan_to_num(coef, nan=0.0)


class AdaptiveTimeEncoding(nn.Module):
    """自适应时间编码（融合节点特征）"""

    def __init__(self, input_dim=4, output_dim=64, base=100):
        super().__init__()
        self.output_dim = output_dim
        self.base = base
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.feat_adaptor = nn.Linear(output_dim, output_dim//2)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, delta_t, node_feat):
        # 基础频率
        freqs = 1 / (self.base ** (2 * torch.arange(self.output_dim // 2, device=node_feat.device) / self.output_dim))

        # 特征自适应频率
        adapted = self.input_proj(node_feat)  # 保证输入一致
        feat_freq = self.feat_adaptor(adapted)
        freqs = freqs * (1 + torch.sigmoid(feat_freq))

        # 编码生成
        angles = delta_t * freqs
        encoding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return encoding / torch.sqrt(torch.tensor(self.output_dim // 2, device=node_feat.device))
