import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torchdiffeq import odeint
from torch_geometric.utils import to_dense_adj
from GraphTransformer import GraphTransformer, RepresentationPretrainer
from LinkRouteAware import LinkRouteAwareGNN
import matplotlib as mpl
import subprocess
import sys

def setup_jupyter_fonts():
    """专门为JupyterLab环境设置的字体配置 - 修复版本"""
    
    # 方法1：直接设置matplotlib配置
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 在JupyterLab中特别设置
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    
    # 方法2：尝试不同的字体配置方法
    try:
        # 检查系统可用字体
        available_fonts = sorted([f.name for f in mpl.font_manager.fontManager.ttflist])
        print("可用的字体数量:", len(available_fonts))
        
        # 优先使用英文字体，避免中文问题
        english_fonts = [
            'DejaVu Sans', 
            'Arial',
            'Helvetica',
            'Liberation Sans',
            'Bitstream Vera Sans',
            'sans-serif'
        ]
        
        # 找到第一个可用的英文字体
        available_english_fonts = []
        for font_name in english_fonts:
            # 检查字体是否在可用字体列表中
            if any(font_name.lower() in f.lower() for f in available_fonts):
                available_english_fonts.append(font_name)
                print(f"✓ 找到英文字体: {font_name}")
        
        if available_english_fonts:
            plt.rcParams['font.family'] = available_english_fonts
            print(f"✓ 已设置字体: {available_english_fonts[0]}")
            return True
        else:
            # 使用系统默认字体
            plt.rcParams['font.family'] = ['sans-serif']
            print("⚠ 使用默认sans-serif字体")
            return True
            
    except Exception as e:
        print(f"字体设置过程中出现错误: {e}")
        # 使用最保守的设置
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return False

def setup_fonts_safe():
    """安全的字体设置，避免中文问题"""
    try:
        # 重置matplotlib默认设置
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        # 设置基本的英文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 12
        
        print("✓ 使用安全的英文字体配置")
        return True
    except Exception as e:
        print(f"安全字体设置失败: {e}")
        return False

# 初始化字体设置 - 使用安全版本
font_success = setup_fonts_safe()
if not font_success:
    print("使用最基本的字体配置")
    plt.rcParams['font.family'] = ['sans-serif']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 小世界网络生成模块 ====================
class SWNetworkGenerator(nn.Module):
    """小世界网络生成与维护模块（整合GraphTransformer）"""

    def __init__(self, num_nodes=1000, k=4, p=0.1, infection_rate=0.05, counter_rate=0.01):
        super().__init__()
        # 生成固定的小世界网络
        self.fixed_base_graph = nx.watts_strogatz_graph(num_nodes, k, p)
        self.num_nodes = num_nodes
        self.num_edges = len(self.fixed_base_graph.edges())

        # 初始化GraphTransformer
        self.graph_transformer = GraphTransformer(
            node_dim=4,
            time_dim=8,
            k_hop=2,
            beta=-0.5,
            dropout=0.1,
            pre_ln=True
        ).to(device)
        self.representation_pretrainer = RepresentationPretrainer(
            struct_dim=4,
            dyn_dim=4,
            hidden_dim=64
        ).to(device)
        self._H_struct_nodes = None
        self._H_dyn_graph = None
        self._H_dyn_nodes = None

        # 固定边索引 (不再变化)
        edges = list(self.fixed_base_graph.edges())
        self.fixed_edge_index = torch.tensor(edges).t().contiguous().to(device)
        self.to(device)

        # 保存初始边索引
        self.initial_edge_index = self.fixed_edge_index.clone()
        self.initial_num_edges = self.num_edges

        # 固定节点位置用于可视化 - 基于初始图计算
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from(range(num_nodes))
        initial_graph.add_edges_from(self.fixed_base_graph.edges())
        self.fixed_pos = nx.spring_layout(initial_graph, seed=42)

        # 初始化节点状态
        self.reset(infection_rate, counter_rate)

        # 初始化边权重 (可变的)
        self.edge_weights = torch.ones(self.num_edges, device=device)

        # 动态边相关参数
        self.edge_generation_interval = 150  # 每150步生成一次新边
        self.new_edge_weight = 1.0  # 新边初始权重
        self.edge_threshold = 0.9  # 边生成阈值

        # 添加双视图GNN用于关键边识别
        self.link_route_gnn = LinkRouteAwareGNN(
            node_dim=4,
            edge_dim=4 * 2 + 1,  # 源节点状态(4) + 目标节点状态(4) + 边权重(1)
            hidden_dim=64
        ).to(device)

        # 存储关键边
        self.critical_edges = None
        self.high_infection_threshold = 0.8  # 高感染阶段阈值

        # ODE参数化传播模型
        self.ode_params = nn.ParameterDict({
            'beta': nn.Parameter(torch.randn(num_nodes).to(device)),  # 感染率
            'gamma': nn.Parameter(torch.randn(num_nodes).to(device)),  # 恢复率
            'mu': nn.Parameter(torch.randn(num_nodes).to(device)),  # 辟谣率
            'omega': nn.Parameter(torch.randn(num_nodes).to(device)),  # S→C
            'delta_I': nn.Parameter(torch.randn(num_nodes).to(device)),  # I→C
            'delta_IC': nn.Parameter(torch.randn(num_nodes).to(device)),  # I→C 的结构继承项
        })

        # 用于存放每步来自宏观Actor的输出动作 (Delta参数)
        self.register_buffer('delta_beta', torch.zeros(1, device=device))
        self.register_buffer('delta_beta_SI', torch.zeros(1, device=device))
        self.register_buffer('delta_alpha_S', torch.zeros(1, device=device))
        self.register_buffer('delta_alpha_SC', torch.zeros(1, device=device))
        self.register_buffer('delta_delta_I', torch.zeros(1, device=device))
        self.register_buffer('delta_delta_IC', torch.zeros(1, device=device))

        # 固定节点位置用于可视化
        self.fixed_pos = nx.spring_layout(self.fixed_base_graph, seed=42)

        # 用于记录时间步
        self.time_step = 0

    def reset(self, infection_rate=0.05, counter_rate=0.01):
        """重置节点状态到初始值"""
        # 重置节点状态
        self._init_node_states(infection_rate, counter_rate)

        # 重置边索引和权重到初始状态
        self.fixed_edge_index = self.initial_edge_index.clone()
        self.edge_weights = torch.ones(self.initial_num_edges, device=device)
        self.num_edges = self.initial_num_edges
        # 重置时间步
        self.time_step = 0

        # 清除当前图结构（确保下次可视化重建）
        if hasattr(self, 'current_graph'):
            del self.current_graph

    def _init_node_states(self, infection_rate, counter_rate):
        self.node_states = torch.zeros(self.num_nodes, 4, device=device)
        self.node_states[:, 0] = 1  # 所有节点初始为易感态(S)

        # 确保感染节点和辟谣节点不重叠
        all_indices = torch.randperm(self.num_nodes)
        num_infected = max(1, int(self.num_nodes * infection_rate))
        num_counter = max(1, int(self.num_nodes * counter_rate))

        infected = all_indices[:num_infected]
        counter_candidates = all_indices[num_infected:]
        counter_indices = counter_candidates[:num_counter]

        self.node_states[infected, 0] = 0
        self.node_states[infected, 1] = 1

        self.node_states[counter_indices, 0] = 0
        self.node_states[counter_indices, 2] = 1

    def get_graph_representation(self):
        """获取图的状态表示"""
        # 返回节点状态的平均
        return self.node_states.mean(dim=0).detach()

    def get_macro_density(self):
        """返回下一步宏观 Actor 使用的 SICR 密度 [S(t), I(t), C(t), R(t)]。"""
        return self.node_states.mean(dim=0).detach()

    def refresh_representations(self, detach=True):
        """
        计算 H_struct 与 H_dyn。
        H_struct: 节点级结构聚合图表示，供社团级智能体观测。
        H_dyn: 图级动力学表示，供网络级智能体观测。
        """
        if self.graph_transformer is None:
            self._H_struct_nodes = torch.zeros(self.num_nodes, 4, device=device)
            self._H_dyn_graph = torch.zeros(4, device=device)
            self._H_dyn_nodes = torch.zeros(self.num_nodes, 4, device=device)
            return self._H_struct_nodes, self._H_dyn_graph, self._H_dyn_nodes

        data = Data(
            x=self.node_states if not detach else self.node_states.detach(),
            edge_index=self.fixed_edge_index,
            edge_attr=self.edge_weights.detach().unsqueeze(1)
        ).to(device)

        if detach:
            self.graph_transformer.eval()
            with torch.no_grad():
                h_struct, h_dyn_graph, h_dyn_nodes = self.graph_transformer.encode_representations(
                    data, detach=True
                )
        else:
            self.graph_transformer.train()
            h_struct, h_dyn_graph, h_dyn_nodes = self.graph_transformer.encode_representations(
                data, detach=False
            )

        self._H_struct_nodes = h_struct
        self._H_dyn_graph = h_dyn_graph
        self._H_dyn_nodes = h_dyn_nodes
        return self._H_struct_nodes, self._H_dyn_graph, self._H_dyn_nodes

    def get_structure_and_dynamics_representation(self, detach=True):
        if (
            not detach
            or self._H_struct_nodes is None
            or self._H_dyn_graph is None
        ):
            return self.refresh_representations(detach=detach)
        return self._H_struct_nodes, self._H_dyn_graph, self._H_dyn_nodes

    def get_edge_features(self, include_structural=False):
        """获取所有边的特征: [源节点状态, 目标节点状态, 边权重]"""
        if include_structural and self._H_struct_nodes is None:
            self.refresh_representations(detach=True)

        edge_features = []
        for i in range(self.fixed_edge_index.shape[1]):
            src = self.fixed_edge_index[0, i]
            dst = self.fixed_edge_index[1, i]
            weight = self.edge_weights[i]

            src_state = self.node_states[src].detach()
            dst_state = self.node_states[dst].detach()
            edge_feat = torch.cat([src_state, dst_state, weight.unsqueeze(0)])
            if include_structural:
                src_struct = self._H_struct_nodes[src].detach()
                dst_struct = self._H_struct_nodes[dst].detach()
                edge_feat = torch.cat([edge_feat, src_struct, dst_struct])
            edge_features.append(edge_feat)

        return torch.stack(edge_features).to(device)

    def identify_critical_edges(self):
        """识别关键传播边"""
        node_states = self.node_states.detach()
        edge_features = self.get_edge_features()

        with torch.no_grad():
            importance_scores = self.link_route_gnn(
                node_states,
                self.fixed_edge_index,
                edge_features
            )

        # 选择重要性得分最高的Top-K边
        k = max(1, int(len(importance_scores) * 0.2))  # 选择前20%的边
        topk_values, topk_indices = torch.topk(importance_scores, k)
        self.critical_edges = topk_indices.cpu().numpy()

        print(f"识别到 {len(self.critical_edges)} 条关键边，平均重要性: {topk_values.mean().item():.4f}")
        return self.critical_edges

    def _generate_new_edges(self):
        """使用GraphTransformer生成新边"""
        if self.graph_transformer is None:
            return None, None

        # 准备图数据
        data = Data(
            x=self.node_states.clone().detach(),
            edge_index=self.fixed_edge_index,
            edge_attr=self.edge_weights.clone().detach().unsqueeze(1)
        ).to(device)

        with torch.no_grad():
            local_effect, global_effect = self.graph_transformer(data)
            node_representations = local_effect + global_effect

            # 计算余弦相似度
            norm_rep = F.normalize(node_representations, p=2, dim=1)
            similarity = torch.mm(norm_rep, norm_rep.t())

            # 获取当前邻接矩阵
            adj_matrix = to_dense_adj(self.fixed_edge_index,
                                      max_num_nodes=self.num_nodes).squeeze(0)

            triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
            non_edges_mask = (adj_matrix == 0) & triu_mask

            # 获取候选边索引
            non_edges = non_edges_mask.nonzero(as_tuple=False)

            # 无候选边时返回
            if len(non_edges) == 0:
                return None, None

            # 提取候选边相似度
            edge_scores = similarity[non_edges[:, 0], non_edges[:, 1]]

            k = max(1, len(edge_scores) // 20)  # 取前5%
            topk_values = torch.topk(edge_scores, k=k).values
            dynamic_threshold = topk_values[-1].item() if k > 0 else 0.95

            threshold = max(dynamic_threshold, 0.95)
            selected_mask = edge_scores > threshold

            # 额外限制最大生成数量
            MAX_NEW_EDGES = 5  # 每次最多生成5条新边
            if selected_mask.sum() > MAX_NEW_EDGES:
                # 取相似度最高的前MAX_NEW_EDGES条边
                top_indices = torch.topk(edge_scores, k=MAX_NEW_EDGES).indices
                selected_mask = torch.zeros_like(selected_mask)
                selected_mask[top_indices] = True

            selected_edges = non_edges[selected_mask]

            if len(selected_edges) == 0:
                return None, None

            new_edges = selected_edges.t().contiguous()
            reverse_edges = torch.stack([new_edges[1], new_edges[0]], dim=0)
            all_new_edges = torch.cat([new_edges, reverse_edges], dim=1)

            # 新边权重（初始值较低）
            num_new_edges = all_new_edges.size(1)
            new_weights = torch.full((num_new_edges,),
                                     self.new_edge_weight * 0.5,  # 初始权重减半
                                     device=device)

            print(f"生成 {len(selected_edges)} 条新边，相似度阈值={threshold:.4f}")
            return all_new_edges, new_weights

    def apply_actions(self, suppress_actions, rumor_actions, edge_indices=None):
        """应用智能体动作到边权重

        Args:
            suppress_actions: 辟谣者动作列表
            rumor_actions: 传谣者动作列表
            edge_indices: 应用动作的边索引列表，如果为None则应用到所有边
        """
        with torch.no_grad():
            # 动作映射：0-大幅减少, 1-小幅减少, 2-维持不变, 3-小幅增加, 4-大幅增加
            action_effects = torch.tensor([0.5, 0.95, 1.0, 1.05, 1.5], device=device)

            if edge_indices is None:
                # 应用到所有边
                suppress_effects = action_effects[torch.tensor(suppress_actions, device=device)]
                rumor_effects = action_effects[torch.tensor(rumor_actions, device=device)]
                self.edge_weights = self.edge_weights * suppress_effects * rumor_effects
            else:
                # 只应用到指定的边
                suppress_effects = action_effects[torch.tensor(suppress_actions, device=device)]
                rumor_effects = action_effects[torch.tensor(rumor_actions, device=device)]

                # 确保索引在范围内
                valid_indices = [idx for idx in edge_indices if idx < len(self.edge_weights)]
                if valid_indices:
                    valid_indices_tensor = torch.tensor(valid_indices, device=device)
                    combined_effects = suppress_effects * rumor_effects

                    # 只更新指定边的权重
                    self.edge_weights[valid_indices_tensor] = \
                        self.edge_weights[valid_indices_tensor] * combined_effects

            # 限制边权重范围
            self.edge_weights = torch.clamp(self.edge_weights, 0.1, 5.0)

    def ode_func(self, t, states):
        """ODE函数 """
        device = states.device

        # 提取基础传播参数
        beta = F.softplus(self.ode_params['beta'])
        gamma = F.softplus(self.ode_params['gamma'])
        mu = F.softplus(self.ode_params['mu'])
        omega = F.softplus(self.ode_params['omega'])
        delta_I = F.softplus(self.ode_params['delta_I'])
        delta_IC = F.softplus(self.ode_params['delta_IC'])

        # 提取控制动作产生的参数调整
        delta_beta = self.delta_beta
        delta_beta_SI = self.delta_beta_SI
        delta_alpha_S = self.delta_alpha_S
        delta_alpha_SC = self.delta_alpha_SC
        delta_delta_I = self.delta_delta_I
        delta_delta_IC = self.delta_delta_IC

        S = states[:, 0]
        I = states[:, 1]
        C = states[:, 2]
        R = states[:, 3]

        edge_index = self.fixed_edge_index
        edge_weights = self.edge_weights

        total_I_effect = torch.zeros(self.num_nodes, device=device)
        total_C_effect = torch.zeros(self.num_nodes, device=device)

        # 稀疏聚合计算邻居诱导项
        batch_size = 1000
        for i in range(0, edge_index.size(1), batch_size):
            end_idx = min(i + batch_size, edge_index.size(1))
            batch_edges = edge_index[:, i:end_idx]
            batch_weights = edge_weights[i:end_idx]

            source_nodes = batch_edges[0]
            target_nodes = batch_edges[1]

            I_contrib = I[source_nodes] * batch_weights
            total_I_effect.index_add_(0, target_nodes, I_contrib)
            C_contrib = C[source_nodes] * batch_weights
            total_C_effect.index_add_(0, target_nodes, C_contrib)

        # 受控动力方程 (公式5) - 采用节点级聚合 (微观-宏观的桥梁)
        # dS_i / dt = - (β_S + Δβ_S + (β_SI + Δβ_SI)*Σ(a*I)) * S_i - (α_S + Δα_S + (α_SC + Δα_SC)*Σ(a*C)) * S_i
        dS = - (beta + delta_beta + (beta + delta_beta_SI) * total_I_effect) * S \
             - (omega + delta_alpha_S + (omega + delta_alpha_SC) * total_C_effect) * S

        # dI_i / dt = (β_S + Δβ_S + (β_SI + Δβ_SI)*Σ(a*I)) * S_i - (δ_I + Δδ_I + (δ_IC + Δδ_IC)*Σ(a*C) + γ) * I_i
        dI = (beta + delta_beta + (beta + delta_beta_SI) * total_I_effect) * S \
             - (delta_I + delta_delta_I + (delta_IC + delta_delta_IC) * total_C_effect + gamma) * I

        # dC_i / dt = (α_S + Δα_S + (α_SC + Δα_SC)*Σ(a*C)) * S_i + (δ_I + Δδ_I + (δ_IC + Δδ_IC)*Σ(a*C)) * I_i
        dC = (omega + delta_alpha_S + (omega + delta_alpha_SC) * total_C_effect) * S \
             + (delta_I + delta_delta_I + (delta_IC + delta_delta_IC) * total_C_effect) * I

        # dR_i / dt = γ * I_i
        dR = gamma * I

        return torch.stack([dS, dI, dC, dR], dim=1)

    def step(self, macro_actions=None, controlled_edge_weights=None):
        """执行一步ODE求解 - 优化内存版本"""
        # 1. 更新宏动作参数
        if macro_actions is not None:
            self.delta_beta = macro_actions[0]  # Δβ_S
            self.delta_beta_SI = macro_actions[1]  # Δβ_SI
            self.delta_alpha_S = macro_actions[2]  # Δα_S
            self.delta_alpha_SC = macro_actions[3]  # Δα_SC
            self.delta_delta_I = macro_actions[4]  # Δδ_I
            self.delta_delta_IC = macro_actions[5]  # Δδ_IC

        # 2. 更新受控边权重 (w_rumor * w_counter)
        if controlled_edge_weights is not None:
            self.edge_weights = torch.clamp(controlled_edge_weights, 0.1, 5.0)

        # 3. 使用ODE求解器演化
        t_span = [0, 1]
        t_tensor = torch.tensor(t_span, dtype=torch.float32).to(self.node_states.device)

        # 使用带梯度的 ODE 求解 (支持反向传播)
        states = odeint(
            self.ode_func,
            self.node_states,
            t_tensor,
            method='euler',
            options={'step_size': 0.1}
        )

        new_states = states[-1]
        new_states = torch.clamp(new_states, 0.0, 1.0)
        row_sums = new_states.sum(dim=1, keepdim=True)
        self.node_states = new_states / (row_sums + 1e-8)

        self.time_step += 1
        infection_rate = self.node_states[:, 1].mean().item()
        counter_rate = self.node_states[:, 2].mean().item()
        return infection_rate, counter_rate

    def visualize_states(self, block=True):
        """可视化状态分布，包括边权重 - 修复字体问题版本"""
        cpu_states = self.node_states.detach().cpu().numpy()
        cpu_weights = self.edge_weights.detach().cpu().numpy()

        # 使用指定颜色
        color_map = {
            0: 'blue',   # S: 蓝色
            1: 'red',    # I: 红色
            2: 'green',  # C: 绿色
            3: 'yellow'  # R: 黄色
        }

        # 获取主导状态
        dominant_states = cpu_states.argmax(axis=1)

        # 计算每种状态的节点数量
        state_counts = {
            'S': (dominant_states == 0).sum(),
            'I': (dominant_states == 1).sum(),
            'C': (dominant_states == 2).sum(),
            'R': (dominant_states == 3).sum()
        }

        # 创建节点颜色列表
        node_colors = [color_map[s] for s in dominant_states]

        # 创建/更新当前图结构
        if not hasattr(self, 'current_graph') or self.time_step == 0:
            # 从固定边索引创建图
            edges = self.fixed_edge_index.t().cpu().numpy()
            self.current_graph = nx.Graph()
            self.current_graph.add_nodes_from(range(self.num_nodes))
            # 确保边为整数类型
            self.current_graph.add_edges_from([(int(u), int(v)) for u, v in edges])
            print(f"创建新图结构，包含 {len(edges)} 条边")

            # 计算固定布局
            self.fixed_pos = nx.spring_layout(self.current_graph, seed=42)

        # 如果有动态生成的新边，添加到图中
        if self.fixed_edge_index.shape[1] > len(self.current_graph.edges):
            new_edges = []
            # 获取图中已有边
            existing_edges = set(self.current_graph.edges())

            # 检查所有边
            for i in range(self.fixed_edge_index.shape[1]):
                u = int(self.fixed_edge_index[0, i].item())
                v = int(self.fixed_edge_index[1, i].item())
                # 跳过已有边和自环
                if (u, v) in existing_edges or (v, u) in existing_edges or u == v:
                    continue
                # 添加新边
                self.current_graph.add_edge(u, v)
                new_edges.append((u, v))

            if new_edges:
                print(f"添加 {len(new_edges)} 条新边到图结构中")

        # 创建图表 - 使用英文标签避免字体问题
        plt.figure(figsize=(15, 10))

        # 子图1: 网络可视化
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        nx.draw_networkx_nodes(
            self.current_graph, self.fixed_pos,
            node_color=node_colors,
            node_size=50,
            ax=ax1
        )

        if len(cpu_weights) > 0:
            if np.ptp(cpu_weights) > 0:  # 权重有变化
                edge_widths = 0.5 + 2.0 * (cpu_weights - np.min(cpu_weights)) / np.ptp(cpu_weights)
            else:  # 所有权重相同
                edge_widths = 1.0 * np.ones_like(cpu_weights)
        else:
            edge_widths = 1.0

        # 绘制边
        nx.draw_networkx_edges(
            self.current_graph, self.fixed_pos,
            edge_color='gray',
            width=edge_widths,
            alpha=0.7,
            ax=ax1
        )

        # 添加图例 - 使用英文标签
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                       label=f'Susceptible(S): {state_counts["S"]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                       label=f'Infected(I): {state_counts["I"]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
                       label=f'Counter(C): {state_counts["C"]}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10,
                       label=f'Recovered(R): {state_counts["R"]}')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', title="Node States")
        ax1.set_title(f"Network State (Time Step: {self.time_step})", fontsize=12)
        ax1.axis('off')

        # 子图2: 状态分布饼图
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        sizes = [state_counts['S'], state_counts['I'], state_counts['C'], state_counts['R']]
        labels = ['Susceptible(S)', 'Infected(I)', 'Counter(C)', 'Recovered(R)']
        colors = ['blue', 'red', 'green', 'yellow']

        # 解决标签重合问题
        patches, texts, autotexts = ax2.pie(
            sizes,
            colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # 只显示大于5%的百分比
            startangle=90,
            pctdistance=0.85,  # 百分比标签离圆心更远
            labeldistance=1.1  # 文本标签离圆心更远
        )

        # 添加图例显示所有标签
        ax2.legend(patches, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.axis('equal')
        ax2.set_title('Node State Distribution', fontsize=12)

        # 子图3: 边权重直方图
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        ax3.hist(cpu_weights, bins=20, alpha=0.7, color='purple')
        ax3.set_title('Edge Weight Distribution')
        ax3.set_xlabel('Weight')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle("SICR Model State Distribution", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 为标题留出空间

        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.5)  # 显示0.5秒
            plt.close()

    def save_ode_params(self, file_path=None):
        """保存训练好的ODE参数"""
        if file_path is None:
            # 默认保存路径
            import time
            timestamp = int(time.time())
            file_path = f"ode_params_{timestamp}.pth"

        # 准备保存的数据
        save_data = {
            'ode_params': {key: param.data.cpu() for key, param in self.ode_params.items()},
            'num_nodes': self.num_nodes,
            'training_time': getattr(self, 'training_time', None),
            'final_loss': getattr(self, 'final_loss', None)
        }

        torch.save(save_data, file_path)
        print(f"ODE参数已保存到: {file_path}")

        # 同时保存为文本格式便于查看
        txt_file = file_path.replace('.pth', '_stats.txt')
        with open(txt_file, 'w') as f:
            f.write("ODE参数统计:\n")
            f.write("=" * 50 + "\n")
            for key, param in save_data['ode_params'].items():
                f.write(f"{key}:\n")
                f.write(f"  范围: [{param.min().item():.6f}, {param.max().item():.6f}]\n")
                f.write(f"  均值: {param.mean().item():.6f}\n")
                f.write(f"  标准差: {param.std().item():.6f}\n\n")

            if save_data['training_time']:
                f.write(f"训练时间: {save_data['training_time']:.2f}秒\n")
            if save_data['final_loss']:
                f.write(f"最终损失: {save_data['final_loss']:.6f}\n")

        print(f"参数统计已保存到: {txt_file}")
        return file_path

    def load_ode_params(self, file_path):
        """加载预训练的ODE参数"""
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return False

        try:
            checkpoint = torch.load(file_path, map_location=device)

            # 检查节点数是否匹配
            if checkpoint['num_nodes'] != self.num_nodes:
                print(f"警告: 保存的参数是针对 {checkpoint['num_nodes']} 个节点的，"
                      f"当前网络有 {self.num_nodes} 个节点")
                # 可以选择进行参数插值，这里简单处理为不加载
                return False

            # 加载参数
            for key in self.ode_params.keys():
                if key in checkpoint['ode_params']:
                    self.ode_params[key].data.copy_(checkpoint['ode_params'][key].to(device))

            # 加载训练信息
            if 'training_time' in checkpoint:
                self.training_time = checkpoint['training_time']
            if 'final_loss' in checkpoint:
                self.final_loss = checkpoint['final_loss']

            print(f"ODE参数已从 {file_path} 加载")

            # 验证加载的参数
            self._print_ode_params_stats()
            return True

        except Exception as e:
            print(f"加载ODE参数时出错: {e}")
            return False

    def _print_ode_params_stats(self):
        """打印ODE参数统计信息"""
        print("ODE参数统计:")
        print("-" * 40)
        with torch.no_grad():
            for key, param in self.ode_params.items():
                param_pos = F.softplus(param)  # 确保参数为正
                print(f"{key}:")
                print(f"  原始范围: [{param.min().item():.4f}, {param.max().item():.4f}]")
                print(f"  正数范围: [{param_pos.min().item():.4f}, {param_pos.max().item():.4f}]")
                print(f"  均值: {param_pos.mean().item():.4f} ± {param_pos.std().item():.4f}")
        print("-" * 40)

    def get_ode_params_summary(self):
        """获取ODE参数的摘要信息"""
        summary = {}
        with torch.no_grad():
            for key, param in self.ode_params.items():
                param_pos = F.softplus(param)
                summary[key] = {
                    'raw_min': param.min().item(),
                    'raw_max': param.max().item(),
                    'positive_min': param_pos.min().item(),
                    'positive_max': param_pos.max().item(),
                    'mean': param_pos.mean().item(),
                    'std': param_pos.std().item()
                }
        return summary

    def pretrain_ode(self, pretrain_steps=1000, lr=0.01, save_path=None, verbose=True):
        """预训练ODE参数以拟合网络动力过程"""
        print("开始ODE参数预训练...")

        import time
        start_time = time.time()

        # 只训练ODE参数，固定其他部分
        optimizer = torch.optim.Adam([
            {'params': self.ode_params.parameters(), 'lr': lr}
        ])

        # 记录损失
        train_losses = []

        for step in range(pretrain_steps):
            optimizer.zero_grad()

            # 保存当前状态作为起始点
            initial_states = self.node_states.clone().detach()

            # 使用ODE进行一步演化
            t_span = [0, 1]
            t_tensor = torch.tensor(t_span, dtype=torch.float32).to(device)

            # 前向传播
            states = odeint(self.ode_func, initial_states, t_tensor,
                            method='rk4', options={'step_size': 0.1})

            predicted_states = states[-1]

            # 计算物理约束损失
            loss = self._compute_pretrain_loss(initial_states, predicted_states)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if verbose and (step % 10 == 0 or step == pretrain_steps - 1):
                print(f"预训练步数 [{step + 1}/{pretrain_steps}], 损失: {loss.item():.6f}")

        # 记录训练时间
        self.training_time = time.time() - start_time
        self.final_loss = train_losses[-1] if train_losses else 0.0

        # 绘制损失曲线
        self._plot_pretrain_loss(train_losses)

        print(f"ODE参数预训练完成! 耗时: {self.training_time:.2f}秒")

        # 自动保存参数
        if save_path is not None:
            self.save_ode_params(save_path)
        else:
            # 默认保存
            self.save_ode_params()

        return train_losses

    def _compute_pretrain_loss(self, initial_states, predicted_states):
        """计算预训练的复合损失函数"""
        # 1. 状态守恒损失 - 确保状态总和接近1
        sum_loss = torch.mean((predicted_states.sum(dim=1) - 1.0) ** 2)

        # 2. 非负性损失 - 确保状态值非负
        non_neg_loss = torch.mean(F.relu(-predicted_states) ** 2)

        # 3. 物理合理性损失 - 确保状态变化符合物理规律
        physical_loss = 0.0

        # S应该减少或保持不变
        s_decrease = F.relu(predicted_states[:, 0] - initial_states[:, 0])
        physical_loss += torch.mean(s_decrease ** 2) * 0.1

        # I的变化应该有合理的范围
        i_change = torch.abs(predicted_states[:, 1] - initial_states[:, 1])
        physical_loss += torch.mean(F.relu(i_change - 0.5) ** 2) * 0.1

        # 4. 平滑性损失 - 相邻节点状态应该相似
        smoothness_loss = self._compute_smoothness_loss(predicted_states)

        # 5. 参数正则化
        param_reg = 0.0
        for param in self.ode_params.values():
            param_reg += torch.mean(param ** 2) * 0.001

        # 组合损失
        total_loss = (sum_loss + non_neg_loss + physical_loss +
                      smoothness_loss * 0.01 + param_reg)

        return total_loss

    def _compute_smoothness_loss(self, states):
        """计算图结构上的平滑性损失 - 优化内存版本"""
        if self.fixed_edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=states.device)

        smoothness_loss = 0.0
        edge_index = self.fixed_edge_index
        edge_weights = self.edge_weights

        # 分批处理边
        batch_size = 5000
        for i in range(0, edge_index.shape[1], batch_size):
            end_idx = min(i + batch_size, edge_index.shape[1])
            batch_edges = edge_index[:, i:end_idx]
            batch_weights = edge_weights[i:end_idx]

            for j in range(batch_edges.shape[1]):
                src = batch_edges[0, j]
                dst = batch_edges[1, j]
                weight = batch_weights[j]

                # 相邻节点状态差异
                state_diff = torch.norm(states[src] - states[dst], p=2)
                smoothness_loss += state_diff * weight

        return smoothness_loss / edge_index.shape[1]

    def _plot_pretrain_loss(self, losses):
        """绘制预训练损失曲线"""
        plt.figure(figsize=(10, 6))

        # 原始损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', alpha=0.7, linewidth=1)
        plt.title('ODE Pretraining Loss Curve')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)

        # 平滑后的损失曲线
        plt.subplot(1, 2, 2)
        window_size = max(1, len(losses) // 50)
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size - 1, len(losses)), smoothed_losses, 'r-', linewidth=2)
        else:
            plt.plot(losses, 'r-', linewidth=2)

        plt.title('Smoothed Loss Curve')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def validate_ode_parameters(self, validation_steps=100):
        """验证ODE参数的合理性"""
        print("\n验证ODE参数...")

        with torch.no_grad():
            # 检查参数范围
            beta = F.softplus(self.ode_params['beta'])
            gamma = F.softplus(self.ode_params['gamma'])
            mu = F.softplus(self.ode_params['mu'])
            omega = F.softplus(self.ode_params['omega'])

            print(f"beta范围: [{beta.min().item():.4f}, {beta.max().item():.4f}]")
            print(f"gamma范围: [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
            print(f"mu范围: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"omega范围: [{omega.min().item():.4f}, {omega.max().item():.4f}]")

            # 模拟验证
            initial_states = self.node_states.clone()
            total_infection_change = 0.0

            for step in range(validation_steps):
                t_span = [0, 1]
                t_tensor = torch.tensor(t_span, dtype=torch.float32).to(device)

                states = odeint(self.ode_func, initial_states, t_tensor,
                                method='rk4', options={'step_size': 0.1})

                final_states = states[-1]
                infection_change = (final_states[:, 1] - initial_states[:, 1]).abs().mean().item()
                total_infection_change += infection_change

                initial_states = final_states

            avg_infection_change = total_infection_change / validation_steps
            print(f"平均感染状态变化: {avg_infection_change:.6f}")

            return {
                'beta_range': (beta.min().item(), beta.max().item()),
                'gamma_range': (gamma.min().item(), gamma.max().item()),
                'mu_range': (mu.min().item(), mu.max().item()),
                'omega_range': (omega.min().item(), omega.max().item()),
                'avg_infection_change': avg_infection_change
            }


def pretrain_with_save():
    """演示完整的预训练和保存流程"""
    # 创建网络
    network = SWNetworkGenerator(num_nodes=2000, k=8, p=0.1)

    print("=== 预训练前参数 ===")
    network._print_ode_params_stats()

    # 进行预训练并保存
    print("\n=== 开始预训练 ===")
    save_path = "pretrained_ode_params.pth"
    losses = network.pretrain_ode(
        pretrain_steps=100,
        lr=0.1,
        save_path=save_path,
        verbose=True
    )

    print("\n=== 预训练后参数 ===")
    network._print_ode_params_stats()

    # 演示加载功能
    print("\n=== 测试参数加载 ===")
    network2 = SWNetworkGenerator(num_nodes=2000, k=8, p=0.1)
    print("新网络初始参数:")
    network2._print_ode_params_stats()

    if network2.load_ode_params(save_path):
        print("加载后参数:")
        network2._print_ode_params_stats()

    return network, losses, save_path

if __name__ == "__main__":
    trained_network, loss_history, param_file = pretrain_with_save()
