import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.distributions import Categorical
from collections import deque
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==================== 宏观节点控制策略网络====================
class MacroActor(nn.Module):
    """宏观Actor，输出传播参数调整动作 [Δβ_S, Δβ_SI, Δα_S, Δα_SC, Δδ_I, Δδ_IC]"""

    def __init__(self, input_dim, param_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
            nn.Tanh()
        )
        self.register_buffer('action_scale', torch.tensor(0.1))

    def forward(self, macro_obs):
        """
        macro_obs: [Batch, 宏观观测维度]
                    宏观观测 = [增强图表示 H(t), 宏观密度 Z(t), 节点聚合动作向量 \bar{A}^V]
        """
        raw_action = self.net(macro_obs)
        delta_params = raw_action * self.action_scale
        return delta_params


# ==================== 集中式分布型评论家====================
class DistributionalCritic(nn.Module):
    """分位数评论家网络，输出K个分位数估计值"""

    def __init__(self, obs_dim, action_dim, num_quantiles=32, hidden_dim=256):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quantiles)
        )

    def forward(self, obs, action):
        # 拼接宏观观测与宏观动作
        x = torch.cat([obs, action], dim=-1)
        # 输出每个样本的 num_quantiles 个分位数
        return self.net(x)


# ==================== 微观边策略网络 ====================
class EdgeWisePolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim)
        )

    def forward(self, edge_features):
        return self.net(edge_features)


# ==================== 智能体间通信模块 ====================
class CommunicationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=3,
            dropout=0.1,
            batch_first=True
        )
        self.aggregation_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, self_observation, other_observations):
        if len(other_observations) == 0:
            return self_observation

        batch_size = self_observation.size(0)
        all_observations = torch.cat([
            self_observation.unsqueeze(1),
            other_observations.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=1)

        attended_obs, attention_weights = self.attention(all_observations, all_observations, all_observations)
        self_attended = attended_obs[:, 0, :]
        other_aggregated = attended_obs[:, 1:, :].mean(dim=1)

        combined = torch.cat([self_attended, other_aggregated], dim=-1)
        enhanced_obs = self.aggregation_net(combined)
        return enhanced_obs

# ==================== 简化的对手模型 ====================
class SimpleOpponentModel(nn.Module):
    def __init__(self, node_state_dim, action_dim, structural_dim=4, hidden_dim=512):
        super().__init__()
        self.node_state_dim = node_state_dim
        self.structural_dim = structural_dim
        self.base_edge_dim = node_state_dim * 2 + 1
        input_dim = self.base_edge_dim + 2 * structural_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, edge_features):
        features = self._prepare_input(edge_features)
        return self.network(features)

    def _prepare_input(self, edge_features):
        """兼容 9 维旧边特征与 17 维结构增强边特征。"""
        if edge_features.size(1) < self.base_edge_dim:
            edge_features = F.pad(
                edge_features,
                (0, self.base_edge_dim - edge_features.size(1))
            )

        structural = edge_features[:, self.base_edge_dim:]
        expected_struct_dim = 2 * self.structural_dim
        if structural.size(1) < expected_struct_dim:
            structural = F.pad(structural, (0, expected_struct_dim - structural.size(1)))
        elif structural.size(1) > expected_struct_dim:
            structural = structural[:, :expected_struct_dim]

        return torch.cat([edge_features[:, :self.base_edge_dim], structural], dim=1)

    def predict(self, edge_features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(edge_features)
            probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()


# ==================== 微观 Actor ====================
class Actor(nn.Module):
    def __init__(self, node_state_dim, action_dim, agent_type, structural_dim=4):
        super().__init__()
        self.agent_type = agent_type
        self.action_dim = action_dim
        self.node_state_dim = node_state_dim
        self.structural_dim = structural_dim
        self.base_edge_dim = node_state_dim * 2 + 1
        input_dim = self.base_edge_dim + 2 * structural_dim
        self.communication = CommunicationModule(input_dim)
        self.edge_net = EdgeWisePolicyNetwork(input_dim, action_dim)

    def _prepare_edge_features(self, edge_features):
        """兼容 9 维旧边特征与 17 维结构增强边特征。"""
        if edge_features.size(1) < self.base_edge_dim:
            edge_features = F.pad(
                edge_features,
                (0, self.base_edge_dim - edge_features.size(1))
            )

        structural = edge_features[:, self.base_edge_dim:]
        expected_struct_dim = 2 * self.structural_dim
        if structural.size(1) < expected_struct_dim:
            structural = F.pad(structural, (0, expected_struct_dim - structural.size(1)))
        elif structural.size(1) > expected_struct_dim:
            structural = structural[:, :expected_struct_dim]

        return torch.cat([edge_features[:, :self.base_edge_dim], structural], dim=1)

    def forward(self, edge_features, other_agent_observations=None):
        edge_features = self._prepare_edge_features(edge_features)
        if other_agent_observations is None or len(other_agent_observations) == 0:
            enhanced_features = edge_features
        else:
            enhanced_features = []
            for i in range(edge_features.size(0)):
                single_edge_feat = edge_features[i:i + 1]
                other_obs_for_edge = []
                for other_obs in other_agent_observations:
                    if other_obs.size(0) > i:
                        other_obs_for_edge.append(other_obs[i:i + 1])
                if other_obs_for_edge:
                    other_obs_tensor = torch.cat(other_obs_for_edge, dim=0)
                    enhanced_edge = self.communication(single_edge_feat, other_obs_tensor)
                    enhanced_features.append(enhanced_edge)
                else:
                    enhanced_features.append(single_edge_feat)
            enhanced_features = torch.cat(enhanced_features, dim=0)

        action_logits = self.edge_net(enhanced_features)
        max_logit = action_logits.max(dim=-1, keepdim=True)[0]
        stable_logits = action_logits - max_logit.detach()
        action_probs = F.softmax(stable_logits, dim=-1)

        mean_node_state = edge_features[:, :self.node_state_dim].mean(dim=0)
        agent_type_tensor = torch.tensor([self.agent_type], device=edge_features.device)
        critic_input = torch.cat([mean_node_state, agent_type_tensor])
        state_value = self.critic(critic_input.unsqueeze(0))
        return action_probs, state_value

    def select_action(self, edge_features, other_agent_observations=None, critical_edges=None, epsilon=0.1,
                      difficulty_coefficient=None):
        """
        Args:
            edge_features: 边特征输入
            other_agent_observations: 其他智能体观测
            critical_edges: 关键边索引（为保持接口兼容保留，逻辑内不再使用）
            epsilon: 探索率
            difficulty_coefficient: 难度门控系数 eta(t) [0, 1]
        """
        with torch.no_grad():
            # 1. 获取基础动作概率分布
            action_probs, state_value = self.forward(edge_features, other_agent_observations)
            num_edges = edge_features.size(0)

            # 2. 定义离散动作空间分支
            # 动作: 0.大幅减少(0.5强), 1.小幅减少(0.95弱), 2.维持不变(1.0中性), 3.小幅增加(1.05弱), 4.大幅增加(1.5强)
            action_idx_strong = [0, 4]  # 大幅调整动作
            action_idx_weak = [1, 3]  # 小幅调整动作
            action_idx_neutral = [2]  # 中性动作

            # 3. 实施难度门控公式
            eta = torch.tensor(difficulty_coefficient, device=edge_features.device).clamp(0, 1)

            # 提取未归一化的概率
            probs_raw_strong = action_probs[:, action_idx_strong].sum(dim=-1, keepdim=True)
            probs_raw_weak = action_probs[:, action_idx_weak].sum(dim=-1, keepdim=True)
            probs_raw_neutral = action_probs[:, action_idx_neutral]

            # 通过 eta(t) 进行加权缩放
            probs_scaled_strong = probs_raw_strong * eta
            probs_scaled_weak = probs_raw_weak * (1 - eta)
            probs_scaled_neutral = probs_raw_neutral * 1.0  # 中性动作不改变权重

            # 将加权后的组概率拆分回对应的具体动作中 (保持每个动作的比例不变)
            total_strong = probs_raw_strong + 1e-8
            ratio_0 = action_probs[:, [0]] / total_strong
            ratio_4 = action_probs[:, [4]] / total_strong

            total_weak = probs_raw_weak + 1e-8
            ratio_1 = action_probs[:, [1]] / total_weak
            ratio_3 = action_probs[:, [3]] / total_weak

            # 组装新的加权概率分布
            weighted_probs = torch.cat([
                probs_scaled_strong * ratio_0,
                probs_scaled_weak * ratio_1,
                probs_scaled_neutral,
                probs_scaled_weak * ratio_3,
                probs_scaled_strong * ratio_4
            ], dim=-1)

            # 统一归一化，保证概率和为1
            final_probs = weighted_probs / (weighted_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # 4. 根据新的概率分布采样动作
            actions = []
            log_probs = []
            for i in range(num_edges):
                edge_probs = final_probs[i]

                if torch.isnan(edge_probs).any() or (edge_probs < 0).any():
                    edge_probs = torch.ones_like(edge_probs) / self.action_dim

                # ε-greedy 策略
                if random.random() < epsilon:
                    # 探索：在动作空间均匀随机
                    action_val = random.randint(0, self.action_dim - 1)
                    log_prob = torch.log(edge_probs[action_val] + 1e-10)
                else:
                    # 利用：基于加权修正后的概率分布进行采样
                    dist = Categorical(edge_probs)
                    action_tensor = dist.sample()
                    action_val = action_tensor.item()
                    log_prob = dist.log_prob(action_tensor)

                actions.append(action_val)
                log_probs.append(log_prob.item())

            state_value_scalar = state_value.item() if isinstance(state_value, torch.Tensor) else state_value
            return actions, log_probs, state_value_scalar


# ==================== 观测收集器 ====================
class ObservationCollector:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.suppress_observations = [None] * num_agents
        self.rumor_observations = [None] * num_agents

    def update_observations(self, agent_id, agent_type, edge_features, actions, state_value):
        if agent_type == 0:
            self.suppress_observations[agent_id] = {'edge_features': edge_features.clone().detach(), 'actions': actions,
                                                    'state_value': state_value}
        else:
            self.rumor_observations[agent_id] = {'edge_features': edge_features.clone().detach(), 'actions': actions,
                                                 'state_value': state_value}

    def get_other_agent_observations(self, agent_id, agent_type):
        if agent_type == 0:
            observations = []
            for i, obs in enumerate(self.suppress_observations):
                if i != agent_id and obs is not None:
                    observations.append(obs['edge_features'])
            return observations
        else:
            observations = []
            for i, obs in enumerate(self.rumor_observations):
                if i != agent_id and obs is not None:
                    observations.append(obs['edge_features'])
            return observations

    def clear(self):
        self.suppress_observations = [None] * self.num_agents
        self.rumor_observations = [None] * self.num_agents


# ==================== 网络划分工具 ====================
class NetworkPartitioner:
    def __init__(self, num_subnets=4):
        self.num_subnets = num_subnets
        self.subnet_centers = None

    def _distance_between(self, node_pos, center):
        if len(node_pos) >= 2:
            return math.sqrt((node_pos[0] - center[0]) ** 2 + (node_pos[1] - center[1]) ** 2)
        else:
            return random.random()

    def partition_network_balanced(self, nodes, edges, min_nodes_per_subnet=50):
        num_nodes = len(nodes)
        adjacency = [[] for _ in range(num_nodes)]
        for src, dst in edges:
            adjacency[src].append(dst)
            adjacency[dst].append(src)

        if self.subnet_centers is None:
            self.subnet_centers = []
            for i in range(self.num_subnets):
                center_x = random.gauss(0.5, 0.2)
                center_y = random.gauss(0.5, 0.2)
                self.subnet_centers.append((center_x, center_y))

        node_assignments = [-1] * num_nodes
        unassigned_nodes = set(range(num_nodes))

        for subnet_id in range(self.num_subnets):
            available_nodes = list(unassigned_nodes)
            if not available_nodes:
                break
            available_nodes.sort(
                key=lambda node_idx: self._distance_between(nodes[node_idx], self.subnet_centers[subnet_id]))
            nodes_to_assign = min(min_nodes_per_subnet, len(available_nodes))
            for i in range(nodes_to_assign):
                node_idx = available_nodes[i]
                node_assignments[node_idx] = subnet_id
                unassigned_nodes.remove(node_idx)

        for node_idx in unassigned_nodes:
            min_dist = float('inf')
            best_subnet = 0
            for i, center in enumerate(self.subnet_centers):
                dist = self._distance_between(nodes[node_idx], center)
                if dist < min_dist:
                    min_dist = dist
                    best_subnet = i
            node_assignments[node_idx] = best_subnet

        max_iterations = 50
        for iteration in range(max_iterations):
            has_isolated_nodes = False
            for subnet_id in range(self.num_subnets):
                subnet_nodes = [i for i, assign in enumerate(node_assignments) if assign == subnet_id]
                if len(subnet_nodes) < min_nodes_per_subnet:
                    candidate_nodes = {}
                    for node_idx in range(num_nodes):
                        if node_assignments[node_idx] != subnet_id:
                            connections_to_subnet = 0
                            for neighbor in adjacency[node_idx]:
                                if node_assignments[neighbor] == subnet_id:
                                    connections_to_subnet += 1
                            if connections_to_subnet > 0:
                                candidate_nodes[node_idx] = connections_to_subnet
                    if candidate_nodes:
                        best_node = max(candidate_nodes.items(), key=lambda x: x[1])[0]
                        original_subnet = node_assignments[best_node]
                        node_assignments[best_node] = subnet_id
                        has_isolated_nodes = True

                for node_idx in subnet_nodes:
                    intra_subnet_degree = 0
                    for neighbor in adjacency[node_idx]:
                        if node_assignments[neighbor] == subnet_id:
                            intra_subnet_degree += 1
                    if intra_subnet_degree == 0 and len(subnet_nodes) > 1:
                        has_isolated_nodes = True
                        connected_subnets = {}
                        for neighbor in adjacency[node_idx]:
                            neighbor_subnet = node_assignments[neighbor]
                            connected_subnets[neighbor_subnet] = connected_subnets.get(neighbor_subnet, 0) + 1
                        if connected_subnets:
                            best_new_subnet = max(connected_subnets.items(), key=lambda x: x[1])[0]
                            node_assignments[node_idx] = best_new_subnet

            if not has_isolated_nodes:
                break

        subnets = [[] for _ in range(self.num_subnets)]
        subnet_nodes = [[] for _ in range(self.num_subnets)]
        for node_idx, subnet_id in enumerate(node_assignments):
            subnet_nodes[subnet_id].append(node_idx)

        subnet_edges = [[] for _ in range(self.num_subnets)]
        for edge_idx, (src, dst) in enumerate(edges):
            src_subnet = node_assignments[src]
            dst_subnet = node_assignments[dst]
            if src_subnet == dst_subnet:
                subnet_edges[src_subnet].append(edge_idx)

        for i in range(self.num_subnets):
            subnets[i] = {
                'nodes': subnet_nodes[i],
                'edges': subnet_edges[i],
                'center': self.subnet_centers[i]
            }
        return subnets


# ==================== 多智能体控制器 ====================
class MultiAgentController:
    def __init__(self, num_agents=4, node_state_dim=4, action_dim=5, lr=0.001, weight_decay=1e-4, gamma=0.99,
                 min_opp_experience=50, opp_update_freq=1, initial_epsilon=0.5,
                 epsilon_decay=0.98, min_epsilon=0.05, structural_dim=4):
        self.num_agents = num_agents
        self.node_state_dim = node_state_dim
        self.action_dim = action_dim
        self.structural_dim = structural_dim
        self.macro_param_dim = 6  # 对应 Δβ_S, Δβ_SI, Δα_S, Δα_SC, Δδ_I, Δδ_IC

        self.freeze_opponent_models = False
        self.freeze_actor_models = False
        self.total_experiences = 0
        self.opp_update_counter = 0
        self.opp_update_freq = opp_update_freq

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_count = 0

        # ===== 1. 微观边智能体 (Actor) =====
        self.suppress_agents = [
            Actor(node_state_dim, action_dim, agent_type=0, structural_dim=structural_dim).to(device)
            for _ in range(num_agents)
        ]
        self.rumor_agents = [
            Actor(node_state_dim, action_dim, agent_type=1, structural_dim=structural_dim).to(device)
            for _ in range(num_agents)
        ]
        self.suppress_optims = [optim.Adam(agent.parameters(), lr=lr, weight_decay=weight_decay) for agent in
                                self.suppress_agents]
        self.rumor_optims = [optim.Adam(agent.parameters(), lr=lr, weight_decay=weight_decay) for agent in
                             self.rumor_agents]

        # ===== 2. 宏观节点智能体 (Macro Actor) =====
        # 宏观观测输入
        macro_obs_dim = 4 + 4 + 4
        self.macro_suppress_actor = MacroActor(macro_obs_dim, self.macro_param_dim).to(device)
        self.macro_rumor_actor = MacroActor(macro_obs_dim, self.macro_param_dim).to(device)
        self.macro_suppress_optim = optim.Adam(self.macro_suppress_actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.macro_rumor_optim = optim.Adam(self.macro_rumor_actor.parameters(), lr=lr, weight_decay=weight_decay)

        # ===== 3. 集中式分布型评论家 =====
        self.macro_suppress_critic = DistributionalCritic(macro_obs_dim, self.macro_param_dim).to(device)
        self.macro_rumor_critic = DistributionalCritic(macro_obs_dim, self.macro_param_dim).to(device)
        self.macro_critic_optims = [
            optim.Adam(self.macro_suppress_critic.parameters(), lr=lr),
            optim.Adam(self.macro_rumor_critic.parameters(), lr=lr)
        ]

        # ===== 4. 对手模型 =====
        self.suppress_opp = SimpleOpponentModel(
            node_state_dim, action_dim, structural_dim=structural_dim
        ).to(device)
        self.rumor_opp = SimpleOpponentModel(
            node_state_dim, action_dim, structural_dim=structural_dim
        ).to(device)
        self.suppress_opp_optim = optim.AdamW(self.suppress_opp.parameters(), lr=5e-4, weight_decay=1e-4)
        self.rumor_opp_optim = optim.AdamW(self.rumor_opp.parameters(), lr=5e-4, weight_decay=1e-4)

        # 内存管理
        self.network_partitioner = NetworkPartitioner(num_subnets=num_agents)
        self.subnets = None
        self.observation_collector = ObservationCollector(num_agents)
        self.subnet_node_counts = []  # 用于计算 omega_k

        self.suppress_memories = [deque(maxlen=10000) for _ in range(num_agents)]
        self.rumor_memories = [deque(maxlen=10000) for _ in range(num_agents)]
        self.macro_suppress_memory = deque(maxlen=10000)
        self.macro_rumor_memory = deque(maxlen=10000)
        self.suppress_opp_memory = deque(maxlen=20000)
        self.rumor_opp_memory = deque(maxlen=20000)

    def partition_network_balanced(self, nodes, edges, min_nodes_per_subnet=50):
        self.subnets = self.network_partitioner.partition_network_balanced(nodes, edges, min_nodes_per_subnet)
        self.subnet_node_counts = [len(sub['nodes']) for sub in self.subnets]
        return self.subnets

    def get_subnet_edge_features(self, all_edge_features, subnet_id):
        if self.subnets is None or subnet_id >= len(self.subnets):
            return all_edge_features
        subnet_edges = self.subnets[subnet_id]['edges']
        if not subnet_edges:
            return torch.empty(0, all_edge_features.size(1), device=all_edge_features.device)
        return all_edge_features[subnet_edges]

    def _aggregate_micro_actions(self, subnet_actions, subnet_edges_map, edge_index):
        """
        将微观边动作(加权乘子)按节点进行平均池化。
        生成节点级动作聚合向量 \bar{A}^V。
        """
        # 将列表格式的动作转换为张量
        if isinstance(subnet_actions, list):
            action_tensor = torch.tensor(subnet_actions, dtype=torch.float, device=device)
        else:
            action_tensor = subnet_actions

        if action_tensor.numel() == 0:
            return torch.zeros(self.total_nodes, device=device)


        agg_values = torch.zeros(self.total_nodes, device=device)
        degrees = torch.zeros(self.total_nodes, device=device)

        for idx, e_idx in enumerate(subnet_edges_map):
            src = edge_index[0, e_idx].item()
            dst = edge_index[1, e_idx].item()
            val = action_tensor[idx].item()
            agg_values[src] += val
            agg_values[dst] += val
            degrees[src] += 1.0
            degrees[dst] += 1.0

        degrees = torch.clamp(degrees, min=1.0)
        agg_values = agg_values / degrees
        return agg_values

    def get_actions(self, all_edge_features, H_dyn_rep, Z_t_macro, edge_index, critical_edges=None,difficulty_coefficient=0.5):
        """
        获取宏微观联合动作。
        H_dyn_rep: 增强图表示 H(t)
        Z_t_macro: 宏观密度 Z(t) = [S(t), I(t), C(t), R(t)]
        edge_index: 完整的图边索引
        """
        if self.subnets is None:
            raise ValueError("Network not partitioned. Call partition_network_balanced first.")

        if edge_index is not None and isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            self.total_nodes = int(edge_index.max().item()) + 1
        else:
            self.total_nodes = H_dyn_rep.size(0) if isinstance(H_dyn_rep, torch.Tensor) else H_dyn_rep.shape[0]

        if isinstance(H_dyn_rep, torch.Tensor):
            H_dyn_graph = H_dyn_rep.mean(dim=0) if H_dyn_rep.dim() > 1 else H_dyn_rep
        else:
            H_dyn_graph = torch.as_tensor(H_dyn_rep, dtype=torch.float, device=device).mean(dim=0)

        Z_t_macro = Z_t_macro.squeeze() if isinstance(Z_t_macro, torch.Tensor) and Z_t_macro.ndim > 1 else Z_t_macro
        if not isinstance(Z_t_macro, torch.Tensor):
            Z_t_macro = torch.as_tensor(Z_t_macro, dtype=torch.float, device=device)
        Z_t_macro = Z_t_macro.to(device).float().view(-1)
        if Z_t_macro.numel() < 4:
            Z_t_macro = F.pad(Z_t_macro, (0, 4 - Z_t_macro.numel()))
        elif Z_t_macro.numel() > 4:
            Z_t_macro = Z_t_macro[:4]

        H_dyn_graph = H_dyn_graph.to(device).float().view(-1)
        if H_dyn_graph.numel() < 4:
            H_dyn_graph = F.pad(H_dyn_graph, (0, 4 - H_dyn_graph.numel()))
        elif H_dyn_graph.numel() > 4:
            H_dyn_graph = H_dyn_graph[:4]

        all_suppress_actions = []
        all_rumor_actions = []
        all_suppress_log_probs = []
        all_rumor_log_probs = []
        all_suppress_opp_probs = []
        all_rumor_opp_probs = []
        all_suppress_values = []
        all_rumor_values = []

        self.observation_collector.clear()

        # 第一阶段：通信信息收集
        for i in range(self.num_agents):
            subnet_edge_features = self.get_subnet_edge_features(all_edge_features, i)
            if subnet_edge_features.size(0) == 0: continue
            subnet_edges = self.subnets[i]['edges']
            subnet_critical_edges = []
            if critical_edges is not None:
                subnet_critical_edges = [subnet_edges.index(edge) for edge in critical_edges if edge in subnet_edges]

            s_act, s_logp, s_val = self.suppress_agents[i].select_action(subnet_edge_features,
                                                                         critical_edges=subnet_critical_edges,
                                                                         epsilon=self.epsilon)
            r_act, r_logp, r_val = self.rumor_agents[i].select_action(subnet_edge_features,
                                                                      critical_edges=subnet_critical_edges,
                                                                      epsilon=self.epsilon)
            self.observation_collector.update_observations(i, 0, subnet_edge_features, s_act, s_val)
            self.observation_collector.update_observations(i, 1, subnet_edge_features, r_act, r_val)

        # 第二阶段：带通信的正式决策
        flat_suppress_actions = []
        flat_rumor_actions = []
        flat_suppress_edges = []
        flat_rumor_edges = []
        total_suppress_edge_count = 0
        total_rumor_edge_count = 0

        for i in range(self.num_agents):
            subnet_edge_features = self.get_subnet_edge_features(all_edge_features, i)
            if subnet_edge_features.size(0) == 0:
                all_suppress_actions.append([])
                all_rumor_actions.append([])
                all_suppress_log_probs.append(torch.tensor([], device=device))
                all_rumor_log_probs.append(torch.tensor([], device=device))
                all_suppress_opp_probs.append(np.array([]))
                all_rumor_opp_probs.append(np.array([]))
                all_suppress_values.append(0.0)
                all_rumor_values.append(0.0)
                continue

            subnet_edges = self.subnets[i]['edges']
            suppress_other_obs = self.observation_collector.get_other_agent_observations(i, 0)
            rumor_other_obs = self.observation_collector.get_other_agent_observations(i, 1)

            s_act, s_logp, s_val = self.suppress_agents[i].select_action(
                subnet_edge_features,
                other_agent_observations=suppress_other_obs,
                epsilon=self.epsilon,
                difficulty_coefficient=difficulty_coefficient
            )
            r_act, r_logp, r_val = self.rumor_agents[i].select_action(
                subnet_edge_features,
                other_agent_observations=rumor_other_obs,
                epsilon=self.epsilon,
                difficulty_coefficient=difficulty_coefficient
            )

            s_opp_probs = self.suppress_opp.predict(subnet_edge_features)
            r_opp_probs = self.rumor_opp.predict(subnet_edge_features)

            all_suppress_actions.append(s_act)
            all_rumor_actions.append(r_act)
            all_suppress_log_probs.append(torch.tensor(s_logp, device=device))
            all_rumor_log_probs.append(torch.tensor(r_logp, device=device))
            all_suppress_opp_probs.append(s_opp_probs)
            all_rumor_opp_probs.append(r_opp_probs)
            all_suppress_values.append(s_val)
            all_rumor_values.append(r_val)

            flat_suppress_actions.extend(s_act)
            flat_rumor_actions.extend(r_act)
            flat_suppress_edges.extend(subnet_edges)
            flat_rumor_edges.extend(subnet_edges)
            total_suppress_edge_count += len(s_act)
            total_rumor_edge_count += len(r_act)

        # ===== 3. 节点级聚合 =====
        agg_suppress_vec = self._aggregate_micro_actions(flat_suppress_actions, flat_suppress_edges, edge_index)
        agg_rumor_vec = self._aggregate_micro_actions(flat_rumor_actions, flat_rumor_edges, edge_index)

        bar_A_counter = agg_suppress_vec.mean()
        bar_A_rumor = agg_rumor_vec.mean()

        # ===== 4. 宏观Actor输出 =====
        bar_A_counter_vec = bar_A_counter.to(device).float().view(1).repeat(4)
        bar_A_rumor_vec = bar_A_rumor.to(device).float().view(1).repeat(4)
        macro_obs_counter = torch.cat([
            H_dyn_graph,
            Z_t_macro,
            bar_A_counter_vec
        ])
        macro_obs_rumor = torch.cat([
            H_dyn_graph,
            Z_t_macro,
            bar_A_rumor_vec
        ])

        macro_counter_action = self.macro_suppress_actor(macro_obs_counter.unsqueeze(0)).squeeze(0)
        macro_rumor_action = self.macro_rumor_actor(macro_obs_rumor.unsqueeze(0)).squeeze(0)

        return (all_suppress_actions, all_rumor_actions,
                all_suppress_log_probs, all_rumor_log_probs,
                all_suppress_opp_probs, all_rumor_opp_probs,
                all_suppress_values, all_rumor_values,
                macro_obs_counter, macro_obs_rumor,
                macro_counter_action, macro_rumor_action)

    def store_experience(self, agent_id, suppress_exp, rumor_exp, opp_actions, macro_exp_counter=None,
                         macro_exp_rumor=None):
        """存储经验：兼容微观经验与宏观经验"""
        suppress_opp_action = opp_actions[0]
        rumor_opp_action = opp_actions[1]
        self.total_experiences += 1

        if len(suppress_opp_action) > 0 and len(rumor_opp_action) > 0 and suppress_exp[0].size(0) > 0 and rumor_exp[
            0].size(0) > 0:
            s_actions = suppress_exp[1]
            r_actions = rumor_exp[1]
            if len(set(s_actions)) > 1 or len(set(r_actions)) > 1 or random.random() < 0.3:
                suppress_val = suppress_exp[5]
                rumor_val = rumor_exp[5]
                if isinstance(suppress_val, torch.Tensor):
                    suppress_val = suppress_val.item() if suppress_val.numel() == 1 else suppress_val.mean().item()
                if isinstance(rumor_val, torch.Tensor):
                    rumor_val = rumor_val.item() if rumor_val.numel() == 1 else rumor_val.mean().item()

                self.suppress_memories[agent_id].append((
                    suppress_exp[0],
                    torch.tensor(s_actions, device=device),
                    suppress_exp[2],
                    suppress_exp[3],
                    suppress_exp[4],
                    suppress_val
                ))
                self.rumor_memories[agent_id].append((
                    rumor_exp[0],
                    torch.tensor(r_actions, device=device),
                    rumor_exp[2],
                    rumor_exp[3],
                    rumor_exp[4],
                    rumor_val
                ))

                valid_suppress_opp = [a for a in suppress_opp_action if 0 <= a < 5]
                valid_rumor_opp = [a for a in rumor_opp_action if 0 <= a < 5]
                if len(valid_suppress_opp) > 0 and len(valid_suppress_opp) == suppress_exp[0].size(0):
                    self.suppress_opp_memory.append(
                        (suppress_exp[0], torch.tensor(valid_suppress_opp, dtype=torch.long, device=device)))
                if len(valid_rumor_opp) > 0 and len(valid_rumor_opp) == rumor_exp[0].size(0):
                    self.rumor_opp_memory.append(
                        (rumor_exp[0], torch.tensor(valid_rumor_opp, dtype=torch.long, device=device)))

        if macro_exp_counter is not None:
            obs_m, act_m, rew_m, next_obs_m = macro_exp_counter
            self.macro_suppress_memory.append((obs_m, act_m, rew_m, next_obs_m))
        if macro_exp_rumor is not None:
            obs_m, act_m, rew_m, next_obs_m = macro_exp_rumor
            self.macro_rumor_memory.append((obs_m, act_m, rew_m, next_obs_m))

        if self.opp_update_counter % 2 == 0:
            opp_loss1, opp_loss2 = self.update_opponent_models(batch_size=32)
            return opp_loss1, opp_loss2
        self.opp_update_counter += 1
        return 0.0, 0.0

    def update_opponent_models(self, batch_size=64):
        if self.freeze_opponent_models:
            return 0.0, 0.0
        suppress_loss = 0.0
        rumor_loss = 0.0

        if len(self.suppress_opp_memory) >= max(16, batch_size // 4):
            actual_batch_size = min(batch_size, len(self.suppress_opp_memory))
            indices = random.sample(range(len(self.suppress_opp_memory)), actual_batch_size)
            batch = [self.suppress_opp_memory[i] for i in indices]
            suppress_edge_feats, suppress_targets = zip(*batch)
            suppress_edge_feats = torch.cat(suppress_edge_feats, dim=0)
            suppress_targets = torch.cat(suppress_targets, dim=0)
            if suppress_edge_feats.size(0) > 0:
                self.suppress_opp.train()
                self.suppress_opp_optim.zero_grad()
                logits = self.suppress_opp(suppress_edge_feats)
                loss = F.cross_entropy(logits, suppress_targets)
                l2_norm = sum(p.pow(2.0).sum() for p in self.suppress_opp.parameters())
                loss = loss + 0.001 * l2_norm
                loss.backward()
                nn.utils.clip_grad_norm_(self.suppress_opp.parameters(), 1.0)
                self.suppress_opp_optim.step()
                suppress_loss = loss.item()

        if len(self.rumor_opp_memory) >= max(16, batch_size // 4):
            actual_batch_size = min(batch_size, len(self.rumor_opp_memory))
            indices = random.sample(range(len(self.rumor_opp_memory)), actual_batch_size)
            batch = [self.rumor_opp_memory[i] for i in indices]
            rumor_edge_feats, rumor_targets = zip(*batch)
            rumor_edge_feats = torch.cat(rumor_edge_feats, dim=0)
            rumor_targets = torch.cat(rumor_targets, dim=0)
            if rumor_edge_feats.size(0) > 0:
                self.rumor_opp.train()
                self.rumor_opp_optim.zero_grad()
                logits = self.rumor_opp(rumor_edge_feats)
                loss = F.cross_entropy(logits, rumor_targets)
                l2_norm = sum(p.pow(2.0).sum() for p in self.rumor_opp.parameters())
                loss = loss + 0.001 * l2_norm
                loss.backward()
                nn.utils.clip_grad_norm_(self.rumor_opp.parameters(), 1.0)
                self.rumor_opp_optim.step()
                rumor_loss = loss.item()
        return suppress_loss, rumor_loss

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def quantile_huber_loss(self, pred_quantiles, target_quantiles, taus):
        """
        公式 (43): 分位数 Huber 损失计算
        """
        error = target_quantiles - pred_quantiles
        huber = torch.where(error.abs() <= 1.0, 0.5 * error.pow(2), error.abs() - 0.5)
        rho = torch.abs(taus - (error < 0).float())
        return (rho * huber).mean()

    def update_agents(self, batch_size=32):
        """
        集中式分布型评论家更新，宏观策略更新，以及宏观优势向微观策略的传递。
        """
        if self.freeze_actor_models:
            return 0, 0

        total_suppress_loss = 0
        total_rumor_loss = 0
        updated_agents = 0

        # ====================================================
        # 1. 宏观层更新 
        # ====================================================
        if len(self.macro_suppress_memory) >= batch_size:
            indices = random.sample(range(len(self.macro_suppress_memory)), batch_size)
            macro_batch = [self.macro_suppress_memory[i] for i in indices]
            m_obs, m_act, m_rew, m_next_obs = zip(*macro_batch)
            m_obs = torch.stack(m_obs)
            m_act = torch.stack(m_act)
            m_rew = torch.tensor(m_rew, device=device).unsqueeze(-1)
            m_next_obs = torch.stack(m_next_obs)

            # 更新评论家
            with torch.no_grad():
                next_actions = self.macro_suppress_actor(m_next_obs)
                next_quantiles = self.macro_suppress_critic(m_next_obs, next_actions)
                target_quantiles = m_rew + self.gamma * next_quantiles

            curr_quantiles = self.macro_suppress_critic(m_obs, m_act)
            taus = torch.linspace(0.01, 0.99, self.macro_suppress_critic.num_quantiles).to(device)
            critic_loss = self.quantile_huber_loss(curr_quantiles, target_quantiles, taus)

            self.macro_critic_optims[0].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.macro_suppress_critic.parameters(), 0.5)
            self.macro_critic_optims[0].step()

            # 更新宏观 Actor
            # 计算全局优势 A_global = E[Z] - V_baseline
            mean_quantile_val = curr_quantiles.mean(dim=-1)
            baseline_val = self.macro_suppress_critic(m_obs, torch.zeros_like(m_act)).mean(dim=-1)
            global_advantage = mean_quantile_val - baseline_val

            # 宏观策略梯度
            log_prob = -F.mse_loss(m_act, self.macro_suppress_actor(m_obs), reduction='none').sum(dim=-1)
            macro_policy_loss = -(log_prob * global_advantage.detach()).mean()

            self.macro_suppress_optim.zero_grad()
            macro_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.macro_suppress_actor.parameters(), 0.5)
            self.macro_suppress_optim.step()

        # ====================================================
        # 2. 微观层更新以及优势回传
        # ====================================================
        total_omega = sum(self.subnet_node_counts)
        for agent_id in range(self.num_agents):
            if len(self.suppress_memories[agent_id]) < batch_size or len(self.rumor_memories[agent_id]) < batch_size:
                continue

            # 计算该智能体对应社区的规模优势分配权重 omega_k
            omega_k = self.subnet_node_counts[agent_id] / total_omega if total_omega > 0 else 1.0 / self.num_agents

            indices = random.sample(range(len(self.suppress_memories[agent_id])), batch_size)
            suppress_batch = [self.suppress_memories[agent_id][i] for i in indices]
            rumor_batch = [self.rumor_memories[agent_id][i] for i in indices]

            # 更新微观辟谣智能体
            s_policy_loss = 0
            s_value_loss = 0
            for exp in suppress_batch:
                edge_feat, action, log_prob, reward, next_edge_feat, state_value = exp
                edge_feat = edge_feat.to(device)
                next_edge_feat = next_edge_feat.to(device)
                action = action.to(device)
                log_prob = log_prob.to(device)

                new_probs, new_state_value = self.suppress_agents[agent_id](edge_feat)
                new_dist = Categorical(new_probs)
                new_log_prob = new_dist.log_prob(action)

                with torch.no_grad():
                    _, next_state_value = self.suppress_agents[agent_id](next_edge_feat)
                    td_target = reward + self.gamma * next_state_value * 0.99
                    td_target = torch.clamp(td_target, -10.0, 10.0)

                advantage = td_target.squeeze() - state_value

                community_advantage = advantage * omega_k

                ratio = (new_log_prob - log_prob).exp()
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(ratio * community_advantage, clipped_ratio * community_advantage)

                new_state_val_squeezed = new_state_value.squeeze()
                td_target_squeezed = td_target.detach().squeeze()
                if new_state_val_squeezed.dim() == 0 and td_target_squeezed.dim() == 0:
                    value_loss = F.smooth_l1_loss(new_state_val_squeezed, td_target_squeezed)
                else:
                    if new_state_val_squeezed.dim() == 0:
                        new_state_val_squeezed = new_state_val_squeezed.unsqueeze(0)
                    if td_target_squeezed.dim() == 0:
                        td_target_squeezed = td_target_squeezed.unsqueeze(0)
                    value_loss = F.smooth_l1_loss(new_state_val_squeezed, td_target_squeezed)

                s_policy_loss += policy_loss.mean()
                s_value_loss += value_loss

            self.suppress_optims[agent_id].zero_grad()
            total_s_loss_batch = (s_policy_loss / batch_size + s_value_loss / batch_size)
            total_s_loss_batch.backward()
            nn.utils.clip_grad_norm_(self.suppress_agents[agent_id].parameters(), 0.5)
            self.suppress_optims[agent_id].step()
            total_suppress_loss += total_s_loss_batch.item()

            # 更新微观传谣智能体
            r_policy_loss = 0
            r_value_loss = 0
            for exp in rumor_batch:
                edge_feat, action, log_prob, reward, next_edge_feat, state_value = exp
                edge_feat = edge_feat.to(device)
                next_edge_feat = next_edge_feat.to(device)
                action = action.to(device)
                log_prob = log_prob.to(device)

                new_probs, new_state_value = self.rumor_agents[agent_id](edge_feat)
                new_dist = Categorical(new_probs)
                new_log_prob = new_dist.log_prob(action)

                with torch.no_grad():
                    _, next_state_value = self.rumor_agents[agent_id](next_edge_feat)
                    td_target = reward + self.gamma * next_state_value * 0.99
                    td_target = torch.clamp(td_target, -10.0, 10.0)

                advantage = td_target.squeeze() - state_value
                community_advantage = advantage * omega_k

                ratio = (new_log_prob - log_prob).exp()
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(ratio * community_advantage, clipped_ratio * community_advantage)

                new_state_val_squeezed = new_state_value.squeeze()
                td_target_squeezed = td_target.detach().squeeze()
                if new_state_val_squeezed.dim() == 0 and td_target_squeezed.dim() == 0:
                    value_loss = F.smooth_l1_loss(new_state_val_squeezed, td_target_squeezed)
                else:
                    if new_state_val_squeezed.dim() == 0:
                        new_state_val_squeezed = new_state_val_squeezed.unsqueeze(0)
                    if td_target_squeezed.dim() == 0:
                        td_target_squeezed = td_target_squeezed.unsqueeze(0)
                    value_loss = F.smooth_l1_loss(new_state_val_squeezed, td_target_squeezed)

                r_policy_loss += policy_loss.mean()
                r_value_loss += value_loss

            self.rumor_optims[agent_id].zero_grad()
            total_r_loss_batch = (r_policy_loss / batch_size + r_value_loss / batch_size)
            total_r_loss_batch.backward()
            nn.utils.clip_grad_norm_(self.rumor_agents[agent_id].parameters(), 0.5)
            self.rumor_optims[agent_id].step()
            total_rumor_loss += total_r_loss_batch.item()

            updated_agents += 1

        if updated_agents == 0:
            return 0, 0
        return total_suppress_loss / updated_agents, total_rumor_loss / updated_agents

    def state_dict(self):
        state = {
            'suppress_optim_states': [optim.state_dict() for optim in self.suppress_optims],
            'rumor_optim_states': [optim.state_dict() for optim in self.rumor_optims],
            'macro_suppress_optim_state': self.macro_suppress_optim.state_dict(),
            'macro_rumor_optim_state': self.macro_rumor_optim.state_dict(),
            'macro_critic_optim_states': [optim.state_dict() for optim in self.macro_critic_optims],
            'total_experiences': self.total_experiences,
            'opp_update_counter': self.opp_update_counter,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'subnet_centers': self.network_partitioner.subnet_centers,
            'freeze_opponent_models': self.freeze_opponent_models,
            'freeze_actor_models': self.freeze_actor_models,
            'suppress_opp_optim_state': self.suppress_opp_optim.state_dict(),
            'rumor_opp_optim_state': self.rumor_opp_optim.state_dict()
        }
        return state

    def load_state_dict(self, state_dict):
        for i, optim_state in enumerate(state_dict['suppress_optim_states']):
            self.suppress_optims[i].load_state_dict(optim_state)
        for i, optim_state in enumerate(state_dict['rumor_optim_states']):
            self.rumor_optims[i].load_state_dict(optim_state)
        self.macro_suppress_optim.load_state_dict(state_dict['macro_suppress_optim_state'])
        self.macro_rumor_optim.load_state_dict(state_dict['macro_rumor_optim_state'])
        for i, optim_state in enumerate(state_dict['macro_critic_optim_states']):
            self.macro_critic_optims[i].load_state_dict(optim_state)

        self.suppress_opp_optim.load_state_dict(state_dict['suppress_opp_optim_state'])
        self.rumor_opp_optim.load_state_dict(state_dict['rumor_opp_optim_state'])

        self.total_experiences = state_dict['total_experiences']
        self.opp_update_counter = state_dict['opp_update_counter']
        self.epsilon = state_dict['epsilon']
        self.episode_count = state_dict['episode_count']
        self.network_partitioner.subnet_centers = state_dict['subnet_centers']
        self.freeze_opponent_models = state_dict.get('freeze_opponent_models', False)
        self.freeze_actor_models = state_dict.get('freeze_actor_models', False)

    def get_actor_state_dict(self):
        return {
            'suppress_agents': [agent.state_dict() for agent in self.suppress_agents],
            'rumor_agents': [agent.state_dict() for agent in self.rumor_agents],
            'macro_suppress_actor': self.macro_suppress_actor.state_dict(),
            'macro_rumor_actor': self.macro_rumor_actor.state_dict(),
            'suppress_optims': [optim.state_dict() for optim in self.suppress_optims],
            'rumor_optims': [optim.state_dict() for optim in self.rumor_optims],
            'macro_suppress_optim': self.macro_suppress_optim.state_dict(),
            'macro_rumor_optim': self.macro_rumor_optim.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }

    def load_actor_state_dict(self, state_dict):
        for i, agent_state in enumerate(state_dict['suppress_agents']):
            self.suppress_agents[i].load_state_dict(agent_state)
        for i, agent_state in enumerate(state_dict['rumor_agents']):
            self.rumor_agents[i].load_state_dict(agent_state)
        self.macro_suppress_actor.load_state_dict(state_dict['macro_suppress_actor'])
        self.macro_rumor_actor.load_state_dict(state_dict['macro_rumor_actor'])
        for i, optim_state in enumerate(state_dict['suppress_optims']):
            self.suppress_optims[i].load_state_dict(optim_state)
        for i, optim_state in enumerate(state_dict['rumor_optims']):
            self.rumor_optims[i].load_state_dict(optim_state)
        self.macro_suppress_optim.load_state_dict(state_dict['macro_suppress_optim'])
        self.macro_rumor_optim.load_state_dict(state_dict['macro_rumor_optim'])
        self.epsilon = state_dict.get('epsilon', self.epsilon)
        self.episode_count = state_dict.get('episode_count', self.episode_count)

    def get_opponent_state_dict(self):
        return {
            'suppress_opp': self.suppress_opp.state_dict(),
            'rumor_opp': self.rumor_opp.state_dict(),
            'suppress_opp_optim': self.suppress_opp_optim.state_dict(),
            'rumor_opp_optim': self.rumor_opp_optim.state_dict(),
        }

    def load_opponent_state_dict(self, state_dict):
        self.suppress_opp.load_state_dict(state_dict['suppress_opp'])
        self.rumor_opp.load_state_dict(state_dict['rumor_opp'])
        self.suppress_opp_optim.load_state_dict(state_dict['suppress_opp_optim'])
        self.rumor_opp_optim.load_state_dict(state_dict['rumor_opp_optim'])

    def clear_experiences(self):
        for i in range(self.num_agents):
            self.suppress_memories[i].clear()
            self.rumor_memories[i].clear()
        self.macro_suppress_memory.clear()
        self.macro_rumor_memory.clear()
        self.suppress_opp_memory.clear()
        self.rumor_opp_memory.clear()
        self.total_experiences = 0
        self.opp_update_counter = 0

    def freeze_opponent_models(self, freeze=True):
        self.freeze_opponent_models = freeze
        for param in self.suppress_opp.parameters(): param.requires_grad = not freeze
        for param in self.rumor_opp.parameters(): param.requires_grad = not freeze

    def freeze_actor_models(self, freeze=True):
        self.freeze_actor_models = freeze
        for model in self.suppress_agents + self.rumor_agents:
            for param in model.parameters(): param.requires_grad = not freeze
        for param in self.macro_suppress_actor.parameters(): param.requires_grad = not freeze
        for param in self.macro_rumor_actor.parameters(): param.requires_grad = not freeze
