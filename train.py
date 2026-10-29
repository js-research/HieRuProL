import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from SWNetwork import SWNetworkGenerator
from MultiAgentController import MultiAgentController
from scipy.spatial.distance import jensenshannon

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
        
        # 检查可用字体
        available_fonts = sorted([f.name for f in mpl.font_manager.fontManager.ttflist])
        print("可用的字体数量:", len(available_fonts))
        
        # 优先使用英文字体
        english_fonts = [
            'DejaVu Sans', 
            'Arial',
            'Helvetica',
            'Liberation Sans',
            'Bitstream Vera Sans',
            'sans-serif'
        ]
        
        available_english_fonts = []
        for font_name in english_fonts:
            if any(font_name.lower() in f.lower() for f in available_fonts):
                available_english_fonts.append(font_name)
                print(f"✓ 找到英文字体: {font_name}")
        
        if available_english_fonts:
            plt.rcParams['font.family'] = available_english_fonts
            print(f"✓ 已设置字体: {available_english_fonts[0]}")
        else:
            plt.rcParams['font.family'] = ['sans-serif']
            print("⚠ 使用默认sans-serif字体")
            
        return True
    except Exception as e:
        print(f"字体设置过程中出现错误: {e}")
        # 使用最保守的设置
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 初始化字体设置
font_success = setup_fonts_safe()
if not font_success:
    print("使用最基本的字体配置")
    plt.rcParams['font.family'] = ['sans-serif']

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

# 创建结果图片目录
if not os.path.exists('results'):
    os.makedirs('results')

def calculate_distribution_similarity(pred_probs, true_actions, action_dim=5):
    """
    计算预测分布和实际动作分布的相似度
    Args:
        pred_probs: 预测的概率分布 [num_edges, action_dim]
        true_actions: 实际选择的动作 [num_edges]
        action_dim: 动作维度
    Returns:
        similarity: 分布相似度 (0-1)
        true_distribution: 实际动作的分布
        pred_distribution: 预测动作的分布
    """
    if len(pred_probs) == 0 or len(true_actions) == 0:
        return 0.0, np.zeros(action_dim), np.zeros(action_dim)
    
    # 将实际动作转换为one-hot分布
    true_one_hot = np.eye(action_dim)[true_actions]
    
    # 计算实际动作的分布（平均）
    true_distribution = np.mean(true_one_hot, axis=0)
    
    # 计算预测动作的分布（平均）
    pred_distribution = np.mean(pred_probs, axis=0)
    
    # 使用Jensen-Shannon散度计算分布相似度
    js_distance = jensenshannon(pred_distribution, true_distribution)
    
    # JS散度范围为0-1，转换为相似度 (1 - JS距离)
    similarity = 1.0 - js_distance
    
    return similarity, true_distribution, pred_distribution

def compute_structure_loss(h_v, edge_index, num_nodes):
        """
        InfoNCE 损失
        输入:
            h_v: 节点嵌入 (num_nodes, hidden_dim)
            edge_index: 图的边索引 (2, num_edges)
            num_nodes: 节点总数
        """
        # 归一化节点嵌入以计算余弦相似度
        h_v_norm = F.normalize(h_v, p=2, dim=-1)

        # 计算节点间相似度矩阵 (N, N)
        similarity = torch.mm(h_v_norm, h_v_norm.t())

        # 提取正样本对的相似度 (邻接矩阵为1的边)
        src, dst = edge_index
        pos_sim = similarity[src, dst]  # [num_edges]

        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(similarity).sum(dim=-1)[src] - 1e-8)).mean()
        # 或者简化为 GraphCL 类似的局部对比结构：
        loss_structural = -torch.log(torch.sigmoid(pos_sim) + 1e-8).mean()

        return loss_structural

def compute_fusion_loss(H_struct, H_dyn, Z_true_next, A_true, node_degree_true):
        """
        融合对齐损失
        需要定义辅助解码器 f_dyn 和 f_struct
        输入:
            H_struct: 结构图表示 (Batch, dim)
            H_dyn: 动力学图表示 (Batch, dim)
            Z_true_next: 真实下一时刻宏观密度 (Batch, 4)
            A_true: 原始图邻接矩阵
            node_degree_true: 真实节点度统计 (num_nodes)
        """

        f_dyn = nn.Linear(H_dyn.shape[-1], 4).to(device)  # 映射到 4 维 [S, I, C, R]
        f_struct = nn.Linear(H_struct.shape[-1], 1).to(device)  # 映射到 1 维，用于预测节点度的聚合值

        # 1. 融合为增强图表示 H(t)
        H_fuse = H_dyn + H_struct  # 简单相加融合，实际可改为 MLP 融合
        # 加上 batch 维度
        H_fuse = H_fuse.unsqueeze(0)

        # 2. 通过辅助解码器进行映射
        pred_dyn = f_dyn(H_fuse)  # 预测 Z(t+1)

        pred_struct = f_struct(H_fuse.mean(dim=1))  # 预测平均度

        # 3. 计算 MSE 损失
        loss_fusion = F.mse_loss(pred_dyn.squeeze(), Z_true_next) + \
                      args.lambda_weight * F.mse_loss(pred_struct.squeeze(), node_degree_true)

        return loss_fusion


def pretrain_representations(network, epochs=20, lr=1e-3, lambda_struct=0.5):
    """
    预训练 H_struct 与 H_dyn：
    - H_dyn 预测下一时刻宏观 SICR 密度；
    - H_struct 预测网络平均度。
    """
    parameters = list(network.graph_transformer.parameters()) + list(
        network.representation_pretrainer.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=lr)

    network.train()
    last_loss = None
    for epoch in range(epochs):
        saved_states = network.node_states.clone()
        saved_weights = network.edge_weights.clone()
        saved_time_step = network.time_step
        saved_deltas = {
            'delta_beta': network.delta_beta.clone(),
            'delta_beta_SI': network.delta_beta_SI.clone(),
            'delta_alpha_S': network.delta_alpha_S.clone(),
            'delta_alpha_SC': network.delta_alpha_SC.clone(),
            'delta_delta_I': network.delta_delta_I.clone(),
            'delta_delta_IC': network.delta_delta_IC.clone(),
        }

        with torch.no_grad():
            network.step()
            z_next = network.get_macro_density().detach()

        network.node_states = saved_states
        network.edge_weights = saved_weights
        network.time_step = saved_time_step
        for name, value in saved_deltas.items():
            setattr(network, name, value)

        h_struct_nodes, h_dyn_graph, _ = network.get_structure_and_dynamics_representation(
            detach=False
        )
        degree = torch.bincount(
            network.fixed_edge_index[0],
            minlength=network.num_nodes
        ).float().mean()

        loss, loss_dyn, loss_struct = network.representation_pretrainer.compute_loss(
            h_struct_nodes,
            h_dyn_graph,
            z_next,
            degree,
            lambda_struct=lambda_struct,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        last_loss = (float(loss.item()), float(loss_dyn.item()), float(loss_struct.item()))

    network.refresh_representations(detach=True)
    print(
        "Representation pretraining finished: "
        f"loss={last_loss[0]:.4f}, dyn={last_loss[1]:.4f}, struct={last_loss[2]:.4f}"
    )
    return last_loss


# ==================== 训练模块 ====================
def train_simulation():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
    gc.collect()  # 强制Python垃圾回收
    print("内存清理完成，准备开始训练...")

    # 环境参数
    NUM_NODES = 2000
    INIT_INFECTION = 0.1
    INIT_COUNTER = 0.01  # 初始辟谣节点比例
    EPISODES = 3
    STEPS_PER_EPISODE = 50
    NUM_AGENTS = 4  # 智能体数量

    # 奖励函数参数
    BASE_REWARD = 0  # 基础奖励确保非负
    INFECTION_CHANGE_WEIGHT = 3.0  # 感染率变化权重
    COUNTER_CHANGE_WEIGHT = 3.0  # 辟谣率变化权重 (提高权重)
    ACTION_COST_WEIGHT = 0.1  # 动作成本权重
    OPP_ACCURACY_BONUS = 0.5  # 对手预测准确率奖励系数（提高权重）

    # 课程学习参数
    K = 5.0  # 难度调节系数
    MAX_INFECTION_RATE = 1.0  # 最大感染率
    HIGH_INFECTION_THRESHOLD = 0.6  # 高感染阶段阈值
    LOW_INFECTION_THRESHOLD = 0.4  # 低感染阶段阈值

    PRETRAINED_ODE_PATH = "pretrained_ode_params.pth"  # 根据实际文件路径修改
    # 初始化小世界网络 (结构固定)
    network = SWNetworkGenerator(num_nodes=NUM_NODES, infection_rate=INIT_INFECTION, counter_rate=INIT_COUNTER)
    print(f"固定网络生成完成，节点数: {NUM_NODES}, 边数: {network.num_edges}")
    print(f"初始感染节点比例: {INIT_INFECTION}, 初始辟谣节点比例: {INIT_COUNTER}")

    if os.path.exists(PRETRAINED_ODE_PATH):
        print(f"发现预训练ODE参数文件: {PRETRAINED_ODE_PATH}")
        if network.load_ode_params(PRETRAINED_ODE_PATH):
            print("成功加载预训练ODE参数！")
            # 打印加载后的参数统计
            ode_summary = network.get_ode_params_summary()
            for param_name, stats in ode_summary.items():
                print(
                    f"{param_name}: 均值={stats['mean']:.4f}, 范围=[{stats['positive_min']:.4f}, {stats['positive_max']:.4f}]")
        else:
            print("警告：预训练ODE参数加载失败，将使用随机初始化参数")
    else:
        print(f"未找到预训练ODE参数文件: {PRETRAINED_ODE_PATH}，将使用随机初始化参数")

    # 预训练结构表示 H_struct 与动力学表示 H_dyn
    pretrain_representations(network, epochs=20, lr=1e-3)

    # 初始化多智能体控制器
    node_state_dim = 4  # S,I,C,R 状态
    action_dim = 5  # 五种动作

    # 初始化多智能体控制器（4个智能体）
    controller = MultiAgentController(
        num_agents=NUM_AGENTS,
        node_state_dim=node_state_dim,
        action_dim=action_dim,
        lr=1e-4,
        weight_decay=1e-4,
        gamma=0.95,
        min_opp_experience=10,
        opp_update_freq=1,  # 降低更新频率但增加每次的数据量
        initial_epsilon=0.3,  # 初始探索率
        epsilon_decay=0.98,  # 衰减率
        min_epsilon=0.05  # 最小探索率
    )

    # 划分网络为4个子网，确保无孤立节点且平衡
    nodes = [(random.random(), random.random()) for _ in range(NUM_NODES)]

    edge_index = network.fixed_edge_index.cpu().numpy()
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]

    # 使用平衡划分方法，确保每个子网至少有200个节点
    min_nodes_per_subnet = max(200, NUM_NODES // (NUM_AGENTS * 2))
    subnets = controller.partition_network_balanced(nodes, edges, min_nodes_per_subnet)

    print("平衡网络划分完成:")
    for i, subnet in enumerate(subnets):
        # 计算子网内节点的平均度
        subnet_degrees = []
        for node_idx in subnet['nodes']:
            degree = 0
            for edge_idx in subnet['edges']:
                src, dst = edges[edge_idx]
                if src == node_idx or dst == node_idx:
                    degree += 1
            subnet_degrees.append(degree)

        avg_degree = sum(subnet_degrees) / len(subnet_degrees) if subnet_degrees else 0
        isolated_nodes = sum(1 for d in subnet_degrees if d == 0)

        print(f"子网 {i}: {len(subnet['nodes'])} 个节点, {len(subnet['edges'])} 条边, "
              f"平均度: {avg_degree:.2f}, 孤立节点: {isolated_nodes}")

    # 保存子网规模信息到文件 - 使用英文避免编码问题
    subnet_info_file = 'results/subnet_info.txt'
    with open(subnet_info_file, 'w', encoding='utf-8') as f:
        f.write("Network Partition Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Nodes: {NUM_NODES}\n")
        f.write(f"Total Edges: {network.num_edges}\n")
        f.write(f"Number of Agents: {NUM_AGENTS}\n")
        f.write(f"Initial Infection Rate: {INIT_INFECTION}\n")
        f.write(f"Initial Counter Rate: {INIT_COUNTER}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Subnet Details:\n")
        f.write("-" * 50 + "\n")

        total_subnet_edges = 0
        total_subnet_nodes = 0

        for i, subnet in enumerate(subnets):
            # 计算子网内节点的平均度
            subnet_degrees = []
            for node_idx in subnet['nodes']:
                degree = 0
                for edge_idx in subnet['edges']:
                    src, dst = edges[edge_idx]
                    if src == node_idx or dst == node_idx:
                        degree += 1
                subnet_degrees.append(degree)

            avg_degree = sum(subnet_degrees) / len(subnet_degrees) if subnet_degrees else 0
            isolated_nodes = sum(1 for d in subnet_degrees if d == 0)
            max_degree = max(subnet_degrees) if subnet_degrees else 0
            min_degree = min(subnet_degrees) if subnet_degrees else 0

            f.write(f"Subnet {i}:\n")
            f.write(f"  Nodes: {len(subnet['nodes'])}\n")
            f.write(f"  Edges: {len(subnet['edges'])}\n")
            f.write(f"  Node Percentage: {len(subnet['nodes']) / NUM_NODES * 100:.2f}%\n")
            f.write(f"  Edge Percentage: {len(subnet['edges']) / network.num_edges * 100:.2f}%\n")
            f.write(f"  Average Degree: {avg_degree:.2f}\n")
            f.write(f"  Max Degree: {max_degree}\n")
            f.write(f"  Min Degree: {min_degree}\n")
            f.write(f"  Isolated Nodes: {isolated_nodes}\n")
            f.write(f"  Isolated Node Percentage: {isolated_nodes / len(subnet['nodes']) * 100:.2f}%\n")
            f.write("-" * 30 + "\n")

            total_subnet_edges += len(subnet['edges'])
            total_subnet_nodes += len(subnet['nodes'])

        f.write("\nOverall Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Subnet Nodes: {total_subnet_nodes} (Coverage: {total_subnet_nodes / NUM_NODES * 100:.2f}%)\n")
        f.write(f"Total Subnet Edges: {total_subnet_edges} (Coverage: {total_subnet_edges / network.num_edges * 100:.2f}%)\n")
        f.write(f"Average Subnet Size: {total_subnet_nodes / NUM_AGENTS:.2f} nodes/subnet\n")
        f.write(f"Average Subnet Edges: {total_subnet_edges / NUM_AGENTS:.2f} edges/subnet\n")

    print(f"子网规模信息已保存到: {subnet_info_file}")

    # 同时创建一个子网规模的可视化图表 - 使用英文标签
    plt.figure(figsize=(12, 8))

    # 1. 子网节点分布饼图
    plt.subplot(2, 2, 1)
    node_sizes = [len(subnet['nodes']) for subnet in subnets]
    plt.pie(node_sizes, labels=[f'Subnet {i}' for i in range(NUM_AGENTS)], autopct='%1.1f%%', startangle=90)
    plt.title('Subnet Node Distribution')

    # 2. 子网边分布饼图
    plt.subplot(2, 2, 2)
    edge_sizes = [len(subnet['edges']) for subnet in subnets]
    plt.pie(edge_sizes, labels=[f'Subnet {i}' for i in range(NUM_AGENTS)], autopct='%1.1f%%', startangle=90)
    plt.title('Subnet Edge Distribution')

    # 3. 子网平均度柱状图
    plt.subplot(2, 2, 3)
    avg_degrees = []
    for i, subnet in enumerate(subnets):
        subnet_degrees = []
        for node_idx in subnet['nodes']:
            degree = 0
            for edge_idx in subnet['edges']:
                src, dst = edges[edge_idx]
                if src == node_idx or dst == node_idx:
                    degree += 1
            subnet_degrees.append(degree)
        avg_degree = sum(subnet_degrees) / len(subnet_degrees) if subnet_degrees else 0
        avg_degrees.append(avg_degree)

    plt.bar(range(NUM_AGENTS), avg_degrees, color=plt.cm.Set3(range(NUM_AGENTS)))
    plt.xlabel('Subnet ID')
    plt.ylabel('Average Degree')
    plt.title('Average Degree per Subnet')
    plt.xticks(range(NUM_AGENTS))

    # 4. 子网孤立节点比例柱状图
    plt.subplot(2, 2, 4)
    isolated_ratios = []
    for i, subnet in enumerate(subnets):
        subnet_degrees = []
        for node_idx in subnet['nodes']:
            degree = 0
            for edge_idx in subnet['edges']:
                src, dst = edges[edge_idx]
                if src == node_idx or dst == node_idx:
                    degree += 1
            subnet_degrees.append(degree)
        isolated_nodes = sum(1 for d in subnet_degrees if d == 0)
        isolated_ratio = isolated_nodes / len(subnet['nodes']) * 100 if len(subnet['nodes']) > 0 else 0
        isolated_ratios.append(isolated_ratio)

    plt.bar(range(NUM_AGENTS), isolated_ratios, color=plt.cm.Set3(range(NUM_AGENTS)))
    plt.xlabel('Subnet ID')
    plt.ylabel('Isolated Nodes (%)')
    plt.title('Isolated Node Ratio per Subnet')
    plt.xticks(range(NUM_AGENTS))

    plt.tight_layout()
    plt.savefig('results/subnet_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("子网分布可视化图已保存到: results/subnet_distribution.png")

    # 新的数据结构：按episode存储数据
    episode_data = {
        'infections': [],  # 每个episode的感染率序列 [5, 50]
        'counters': [],  # 每个episode的辟谣率序列 [5, 50]
        'suppress_rewards': [[] for _ in range(NUM_AGENTS)],  # [5, 50] 每个智能体
        'rumor_rewards': [[] for _ in range(NUM_AGENTS)],  # [5, 50] 每个智能体
        'suppress_values': [[] for _ in range(NUM_AGENTS)],  # [5, 50] 每个智能体
        'rumor_values': [[] for _ in range(NUM_AGENTS)],  # [5, 50] 每个智能体
        'difficulties': [],  # [5, 50]
        'suppress_opp_acc': [],  # [5, 50] 按角色存储，不再是按智能体
        'rumor_opp_acc': [],  # [5, 50] 按角色存储，不再是按智能体
        'suppress_opp_losses': [],  # [5, 50]
        'rumor_opp_losses': [],  # [5, 50]
        
        # ==================== 新增：C1和C2评价指标 ====================
        'c1_metrics': {
            'LE': [],       # Lyapunov指数
            'CT': [],       # 收敛时间
            'SSE': []       # 稳态误差
        },
        'c2_metrics': {
            'FV': [],       # 感染峰值下降速度
            'IDR_UC': [],   # 单位成本感染下降率
            'PE': [],       # 传播范围
            'AE': []        # 策略动作熵
        },
        'time_step_metrics': {
            'infection_rates': [],      # 每个episode的感染率时间序列
            'counter_rates': [],        # 每个episode的辟谣率时间序列
            'suppress_rewards': [],     # 每个episode的辟谣奖励时间序列
            'rumor_rewards': [],        # 每个episode的传谣奖励时间序列
            'suppress_values': [],      # 每个episode的辟谣状态价值时间序列
            'rumor_values': [],         # 每个episode的传谣状态价值时间序列
            'c1_le': [],                # 每个episode的LE时间序列
            'c1_sse': [],               # 每个episode的SSE时间序列
            'c2_fv': [],                # 每个episode的FV时间序列
            'c2_idr_uc': [],            # 每个episode的IDR_UC时间序列
            'c2_pe': [],                # 每个episode的PE时间序列
            'c2_ae': []                 # 每个episode的AE时间序列
        },
        
        # ==================== 新增：动作分布数据 ====================
        'action_distributions': {
            'suppress_pred': [],  # 辟谣者预测分布
            'suppress_true': [],  # 辟谣者实际分布
            'rumor_pred': [],     # 传谣者预测分布
            'rumor_true': []      # 传谣者实际分布
        }
    }

    # 最佳模型跟踪
    best_suppress_reward = -float('inf')
    best_rumor_reward = -float('inf')
    best_episode = -1

    # ==================== 第一阶段：锁定对手模型，训练Actor ====================
    print("\n" + "=" * 60)
    print("第一阶段：锁定对手模型，训练Actor")
    print("=" * 60)

    # 锁定对手模型（不更新）
    controller.freeze_opponent_models = True

    for episode in range(EPISODES):
        # 重置节点状态 (网络结构保持不变)
        network.reset(INIT_INFECTION, INIT_COUNTER)

        # 重置最大感染率
        MAX_INFECTION_RATE = 0.0

        # 初始化当前episode的统计
        episode_infections = []
        episode_counters = []
        episode_difficulties = []
        episode_suppress_rewards = [[] for _ in range(NUM_AGENTS)]
        episode_rumor_rewards = [[] for _ in range(NUM_AGENTS)]
        episode_suppress_values = [[] for _ in range(NUM_AGENTS)]
        episode_rumor_values = [[] for _ in range(NUM_AGENTS)]
        episode_suppress_opp_acc = []  # 按角色存储
        episode_rumor_opp_acc = []     # 按角色存储
        episode_suppress_opp_losses = []
        episode_rumor_opp_losses = []

        # ==================== 新增：C1和C2指标时间序列 ====================
        episode_c1_le = []        # LE时间序列
        episode_c1_sse = []       # SSE时间序列
        episode_c2_fv = []        # FV时间序列
        episode_c2_idr_uc = []    # IDR_UC时间序列
        episode_c2_pe = []        # PE时间序列
        episode_c2_ae = []        # AE时间序列

        # ==================== 新增：动作分布数据 ====================
        episode_suppress_pred_dists = []
        episode_suppress_true_dists = []
        episode_rumor_pred_dists = []
        episode_rumor_true_dists = []

        # 初始化统计
        total_suppress_rewards = [0] * NUM_AGENTS
        total_rumor_rewards = [0] * NUM_AGENTS
        avg_infection = 0
        avg_counter = 0

        # 对手预测统计（按角色）
        suppress_correct = 0
        rumor_correct = 0
        total_predictions = 0

        # 对手模型损失 - 按角色累计
        episode_suppress_opp_loss = 0
        episode_rumor_opp_loss = 0

        # 状态价值统计 - 按智能体区分
        suppress_value_sum = [0] * NUM_AGENTS
        rumor_value_sum = [0] * NUM_AGENTS

        # 记录初始状态
        prev_infection_rate = network.node_states[:, 1].mean().item()
        prev_counter_rate = network.node_states[:, 2].mean().item()

        print(f"\n===== Phase 1 Episode {episode} Start =====")

        # 记录动作历史用于C2指标计算
        suppress_actions_history = []
        rumor_actions_history = []

        for step in range(STEPS_PER_EPISODE):
            # 刷新 H_struct 与 H_dyn，并构造结构增强后的边特征
            network.refresh_representations(detach=True)
            H_struct_nodes, H_dyn_graph, _ = network.get_structure_and_dynamics_representation(detach=True)
            all_edge_features = network.get_edge_features(include_structural=True)
            Z_t_macro = network.get_macro_density()
            time_step = controller.total_experiences

            # 获取关键边信息
            critical_edges = network.critical_edges

            current_infection_rate = network.node_states[:, 1].mean().item()

            # 智能体选择动作（传入关键边信息）
            (all_suppress_actions, all_rumor_actions,
             all_suppress_log_probs, all_rumor_log_probs,
             all_suppress_opp_probs, all_rumor_opp_probs,
             all_suppress_values, all_rumor_values,
             macro_obs_counter, macro_obs_rumor,
             macro_counter_action, macro_rumor_action) = controller.get_actions(
                all_edge_features,
                H_dyn_graph,
                Z_t_macro,
                network.fixed_edge_index,
                critical_edges
            )

            # 合并所有智能体的动作
            suppress_actions = []
            rumor_actions = []
            applied_edge_indices = []  # 记录应用动作的边索引

            for i in range(NUM_AGENTS):
                # 获取子网的边索引
                subnet_edges = subnets[i]['edges']
                suppress_actions.extend(all_suppress_actions[i])
                rumor_actions.extend(all_rumor_actions[i])
                applied_edge_indices.extend(subnet_edges)

            # 记录动作历史
            suppress_actions_history.append(suppress_actions)
            rumor_actions_history.append(rumor_actions)

            # 应用动作到网络 (只修改子网中的边)
            network.apply_actions(suppress_actions, rumor_actions, applied_edge_indices)

            # 单步传播更新节点状态，宏动作以两队差值作用于动力学参数
            macro_actions = macro_counter_action - macro_rumor_action
            infection_rate, counter_rate = network.step(macro_actions=macro_actions)
            avg_infection += infection_rate
            avg_counter += counter_rate

            # 更新最大感染率
            MAX_INFECTION_RATE = max(MAX_INFECTION_RATE, infection_rate)

            # 计算感染比例
            infection_ratio = infection_rate / (MAX_INFECTION_RATE + 1e-5)  # 避免除以0
            infection_ratio = np.clip(infection_ratio, 0.0, 1.0)

            # 计算难度系数 (课程学习机制)
            difficulty = 1 / (1 + math.exp(-K * infection_ratio))
            episode_difficulties.append(difficulty)

            # 确定感染阶段 - 使用英文描述
            if infection_ratio > HIGH_INFECTION_THRESHOLD:
                infection_stage = "High Infection"
            elif infection_ratio < LOW_INFECTION_THRESHOLD:
                infection_stage = "Low Infection"
            else:
                infection_stage = "Transition"

            # ====== 奖励函数设计 ======
            # 计算感染率和辟谣率的变化
            delta_infection = infection_rate - prev_infection_rate
            delta_counter = counter_rate - prev_counter_rate

            # 辟谣智能体奖励函数 (根据感染阶段弹性调整)
            suppress_reward = BASE_REWARD + 0.2 
            suppress_reward += INFECTION_CHANGE_WEIGHT * (-delta_infection)  # 感染率降低时奖励

            # 根据感染阶段调整奖励权重
            if infection_stage == "High Infection":
                # 高感染阶段：减少动作成本，鼓励快速压制
                suppress_reward += COUNTER_CHANGE_WEIGHT * 0.8 * delta_counter  # 辟谣率提高时奖励
                action_cost = ACTION_COST_WEIGHT * 0.7 * np.mean(suppress_actions)  # 降低动作成本
            elif infection_stage == "Low Infection":
                # 低感染阶段：增加动作成本，鼓励精准控制
                suppress_reward += COUNTER_CHANGE_WEIGHT * 1.2 * delta_counter  # 提高辟谣率奖励
                action_cost = ACTION_COST_WEIGHT * 1.3 * np.mean(suppress_actions)  # 增加动作成本
            else:
                # 过渡阶段：使用标准权重
                suppress_reward += COUNTER_CHANGE_WEIGHT * delta_counter
                action_cost = ACTION_COST_WEIGHT * np.mean(suppress_actions)

            suppress_reward -= action_cost

            # 传谣智能体奖励函数 (根据感染阶段弹性调整)
            rumor_reward = BASE_REWARD
            rumor_reward += INFECTION_CHANGE_WEIGHT * delta_infection  # 感染率提高时奖励

            # 根据感染阶段调整奖励权重
            if infection_stage == "High Infection":
                # 高感染阶段：降低伪装难度，鼓励快速传播
                rumor_reward += COUNTER_CHANGE_WEIGHT * 0.8 * (-delta_counter)  # 辟谣率降低时奖励
                action_cost = ACTION_COST_WEIGHT * 0.7 * np.mean(rumor_actions)  # 降低动作成本
            elif infection_stage == "Low Infection":
                # 低感染阶段：提升隐蔽性要求，增加动作成本
                rumor_reward += COUNTER_CHANGE_WEIGHT * 1.2 * (-delta_counter)  # 提高辟谣率降低奖励
                action_cost = ACTION_COST_WEIGHT * 1.3 * np.mean(rumor_actions)  # 增加动作成本
            else:
                # 过渡阶段：使用标准权重
                rumor_reward += COUNTER_CHANGE_WEIGHT * (-delta_counter)
                action_cost = ACTION_COST_WEIGHT * np.mean(rumor_actions)

            competition_penalty = 0.3
            suppress_reward -= competition_penalty * rumor_reward
            rumor_reward -= competition_penalty * suppress_reward

            rumor_reward -= action_cost

            # ==================== 使用分布相似度计算对手预测准确率 ====================
            suppress_similarity = 0
            rumor_similarity = 0
            total_suppress_edges = 0
            total_rumor_edges = 0

            # 存储分布数据
            step_suppress_pred_dists = []
            step_suppress_true_dists = []
            step_rumor_pred_dists = []
            step_rumor_true_dists = []

            for i in range(NUM_AGENTS):
                if len(all_suppress_opp_probs[i]) > 0 and len(all_rumor_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_suppress_opp_probs[i], all_rumor_actions[i])
                    suppress_similarity += sim * len(all_rumor_actions[i])
                    total_suppress_edges += len(all_rumor_actions[i])
                    
                    # 存储分布数据
                    step_suppress_pred_dists.append(pred_dist)
                    step_suppress_true_dists.append(true_dist)

                if len(all_rumor_opp_probs[i]) > 0 and len(all_suppress_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_rumor_opp_probs[i], all_suppress_actions[i])
                    rumor_similarity += sim * len(all_suppress_actions[i])
                    total_rumor_edges += len(all_suppress_actions[i])
                    
                    # 存储分布数据
                    step_rumor_pred_dists.append(pred_dist)
                    step_rumor_true_dists.append(true_dist)

            suppress_acc = suppress_similarity / total_suppress_edges if total_suppress_edges > 0 else 0
            rumor_acc = rumor_similarity / total_rumor_edges if total_rumor_edges > 0 else 0

            # 计算平均分布
            avg_suppress_pred_dist = np.mean(step_suppress_pred_dists, axis=0) if step_suppress_pred_dists else np.zeros(5)
            avg_suppress_true_dist = np.mean(step_suppress_true_dists, axis=0) if step_suppress_true_dists else np.zeros(5)
            avg_rumor_pred_dist = np.mean(step_rumor_pred_dists, axis=0) if step_rumor_pred_dists else np.zeros(5)
            avg_rumor_true_dist = np.mean(step_rumor_true_dists, axis=0) if step_rumor_true_dists else np.zeros(5)
            
            # 存储到episode数据
            episode_suppress_pred_dists.append(avg_suppress_pred_dist)
            episode_suppress_true_dists.append(avg_suppress_true_dist)
            episode_rumor_pred_dists.append(avg_rumor_pred_dist)
            episode_rumor_true_dists.append(avg_rumor_true_dist)

            suppress_reward += OPP_ACCURACY_BONUS * suppress_acc
            rumor_reward += OPP_ACCURACY_BONUS * rumor_acc

            # ====== 奖励函数结束 ======

            # 记录当前step的数据到episode缓存
            episode_infections.append(infection_rate)
            episode_counters.append(counter_rate)
            episode_suppress_opp_acc.append(suppress_acc)  # 按角色记录
            episode_rumor_opp_acc.append(rumor_acc)        # 按角色记录

            # ==================== 新增：计算C1和C2指标 ====================
            # 计算Lyapunov指数(LE)
            if len(episode_infections) > 1:
                le = np.mean(np.diff(np.log(np.array(episode_infections) + 1e-10)))
                episode_c1_le.append(le)

            # 计算稳态误差(SSE)
            target_infection = 0.0
            sse = abs(infection_rate - target_infection)
            episode_c1_sse.append(sse)

            # 计算感染峰值下降速度(FV)
            peak_infection = max(episode_infections) if episode_infections else 0
            fv = (peak_infection - infection_rate) / (len(episode_infections) + 1e-10)
            episode_c2_fv.append(fv)

            # 计算单位成本感染下降率(IDR_UC)
            all_suppress_actions_flat = [action for step_actions in suppress_actions_history for action in step_actions]
            all_rumor_actions_flat = [action for step_actions in rumor_actions_history for action in step_actions]

            if all_suppress_actions_flat and all_rumor_actions_flat:
                total_action_cost = np.mean(all_suppress_actions_flat) + np.mean(all_rumor_actions_flat)
            else:
                total_action_cost = 0
            idr_uc = (INIT_INFECTION - infection_rate) / (total_action_cost + 1e-10)
            episode_c2_idr_uc.append(idr_uc)

            # 计算传播范围(PE)
            ir_count = 0
            for node_state in network.node_states:
                if node_state[1] > 0.5 or node_state[3] > 0.5:
                    ir_count += 1
            pe = ir_count / len(network.node_states)
            episode_c2_pe.append(pe)

            # 计算策略动作熵(AE)
            def calculate_entropy(probs_list):
                total_entropy = 0
                total_count = 0
                for probs in probs_list:
                    if len(probs) == 0:
                        continue
                    if probs.ndim == 1:
                        probs = probs.reshape(1, -1)
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                    if isinstance(entropy, np.ndarray):
                        total_entropy += np.sum(entropy)
                        total_count += len(entropy)
                    else:
                        total_entropy += entropy
                        total_count += 1
                return total_entropy / total_count if total_count > 0 else 0

            # 计算所有智能体的动作概率熵
            suppress_entropies = []
            rumor_entropies = []
            
            for i in range(NUM_AGENTS):
                if len(all_suppress_log_probs[i]) > 0:
                    suppress_probs = torch.softmax(all_suppress_log_probs[i], dim=-1)
                    suppress_entropy = calculate_entropy([suppress_probs.cpu().detach().numpy()])
                    suppress_entropies.append(suppress_entropy)
                
                if len(all_rumor_log_probs[i]) > 0:
                    rumor_probs = torch.softmax(all_rumor_log_probs[i], dim=-1)
                    rumor_entropy = calculate_entropy([rumor_probs.cpu().detach().numpy()])
                    rumor_entropies.append(rumor_entropy)

            avg_suppress_entropy = np.mean(suppress_entropies) if suppress_entropies else 0
            avg_rumor_entropy = np.mean(rumor_entropies) if rumor_entropies else 0
            ae = (avg_suppress_entropy + avg_rumor_entropy) / 2
            episode_c2_ae.append(ae)

            # ==================== 指标计算结束 ====================

            # 获取新边特征，并保持与当前观测相同的结构增强维度
            network.refresh_representations(detach=True)
            next_edge_features = network.get_edge_features(include_structural=True)

            # 为每个智能体存储经验
            for agent_id in range(NUM_AGENTS):
                subnet_edge_features = controller.get_subnet_edge_features(all_edge_features, agent_id)
                subnet_next_edge_features = controller.get_subnet_edge_features(next_edge_features, agent_id)

                if subnet_edge_features.size(0) == 0:
                    continue

                # 计算智能体的子网奖励（考虑子网规模比例和子网边平均动作成本）
                subnet_edge_count = len(all_suppress_actions[agent_id])
                total_edge_count = len(suppress_actions)

                # 基础比例：子网边数占总边数的比例
                subnet_ratio = subnet_edge_count / total_edge_count if total_edge_count > 0 else 0

                # 计算子网边平均动作成本
                subnet_suppress_action_mean = np.mean(all_suppress_actions[agent_id]) if len(
                    all_suppress_actions[agent_id]) > 0 else 0
                subnet_rumor_action_mean = np.mean(all_rumor_actions[agent_id]) if len(
                    all_rumor_actions[agent_id]) > 0 else 0

                # 计算全局边平均动作成本
                global_suppress_action_mean = np.mean(suppress_actions) if len(suppress_actions) > 0 else 0
                global_rumor_action_mean = np.mean(rumor_actions) if len(rumor_actions) > 0 else 0

                # 计算动作成本效率因子（子网边平均动作成本相对于全局平均动作成本的比例）
                # 如果子网动作成本低于全局平均，则效率因子>1，反之<1
                suppress_efficiency = max(0.5,
                                          min(2.0, global_suppress_action_mean / (subnet_suppress_action_mean + 1e-5)))
                rumor_efficiency = max(0.5, min(2.0, global_rumor_action_mean / (subnet_rumor_action_mean + 1e-5)))

                # 综合奖励分配权重：基础比例 × 动作成本效率
                suppress_weight = subnet_ratio * suppress_efficiency
                rumor_weight = subnet_ratio * rumor_efficiency

                # 归一化权重
                total_suppress_weight = sum([len(all_suppress_actions[i]) / total_edge_count *
                                             (global_suppress_action_mean / (np.mean(all_suppress_actions[i]) + 1e-5)
                                              if len(all_suppress_actions[i]) > 0 else 1.0)
                                             for i in range(NUM_AGENTS)])
                total_rumor_weight = sum([len(all_rumor_actions[i]) / total_edge_count *
                                          (global_rumor_action_mean / (np.mean(all_rumor_actions[i]) + 1e-5)
                                           if len(all_rumor_actions[i]) > 0 else 1.0)
                                          for i in range(NUM_AGENTS)])

                # 避免除零
                total_suppress_weight = total_suppress_weight if total_suppress_weight > 0 else 1
                total_rumor_weight = total_rumor_weight if total_rumor_weight > 0 else 1

                normalized_suppress_weight = suppress_weight / total_suppress_weight
                normalized_rumor_weight = rumor_weight / total_rumor_weight

                subnet_suppress_reward = suppress_reward * normalized_suppress_weight
                subnet_rumor_reward = rumor_reward * normalized_rumor_weight

                # 记录每个智能体的奖励
                total_suppress_rewards[agent_id] += subnet_suppress_reward
                total_rumor_rewards[agent_id] += subnet_rumor_reward
                episode_suppress_rewards[agent_id].append(subnet_suppress_reward)
                episode_rumor_rewards[agent_id].append(subnet_rumor_reward)

                # 记录每个智能体的状态价值
                if isinstance(all_suppress_values[agent_id], (list, np.ndarray)):
                    suppress_val = np.mean(all_suppress_values[agent_id]) if len(
                        all_suppress_values[agent_id]) > 0 else 0
                else:
                    suppress_val = all_suppress_values[agent_id]
                suppress_value_sum[agent_id] += suppress_val
                episode_suppress_values[agent_id].append(suppress_val)

                if isinstance(all_rumor_values[agent_id], (list, np.ndarray)):
                    rumor_val = np.mean(all_rumor_values[agent_id]) if len(all_rumor_values[agent_id]) > 0 else 0
                else:
                    rumor_val = all_rumor_values[agent_id]
                rumor_value_sum[agent_id] += rumor_val
                episode_rumor_values[agent_id].append(rumor_val)

                # 存储经验（包含同一子网内的对手动作）
                suppress_opp_action = all_rumor_actions[agent_id]  # 同一子网内的传谣者动作
                rumor_opp_action = all_suppress_actions[agent_id]  # 同一子网内的辟谣者动作

                controller.store_experience(
                    agent_id,
                    (subnet_edge_features, all_suppress_actions[agent_id], all_suppress_log_probs[agent_id],
                     subnet_suppress_reward, subnet_next_edge_features, all_suppress_values[agent_id]),
                    (subnet_edge_features, all_rumor_actions[agent_id], all_rumor_log_probs[agent_id],
                     subnet_rumor_reward, subnet_next_edge_features, all_rumor_values[agent_id]),
                    (suppress_opp_action, rumor_opp_action)  # 同一子网内的对手动作
                )

            # 第一阶段不更新对手模型，只更新Actor
            # 更新智能体策略
            controller.update_agents(batch_size=4)

            # 更新前一状态
            prev_infection_rate = infection_rate
            prev_counter_rate = counter_rate

            # 打印每一步的详细信息
            if step % 1 == 0:  # 每10步打印一次，减少输出
                print(f"Step {step}: [{infection_stage}] Infection Rate: {infection_rate:.4f} (Ratio: {infection_ratio:.2f}), "
                      f"Difficulty: {difficulty:.2f}, Counter Rate: {counter_rate:.4f}, "
                      f"Suppress Reward: {suppress_reward:.4f}, Rumor Reward: {rumor_reward:.4f}, "
                      f"Opponent Accuracy[Suppress:{suppress_acc:.3f}, Rumor:{rumor_acc:.3f}]")

        # episode结束后保存数据到总结构
        episode_data['infections'].append(episode_infections)
        episode_data['counters'].append(episode_counters)
        episode_data['difficulties'].append(episode_difficulties)
        episode_data['suppress_opp_acc'].append(episode_suppress_opp_acc)  # 按角色存储
        episode_data['rumor_opp_acc'].append(episode_rumor_opp_acc)        # 按角色存储

        for i in range(NUM_AGENTS):
            episode_data['suppress_rewards'][i].append(episode_suppress_rewards[i])
            episode_data['rumor_rewards'][i].append(episode_rumor_rewards[i])
            episode_data['suppress_values'][i].append(episode_suppress_values[i])
            episode_data['rumor_values'][i].append(episode_rumor_values[i])

        # ==================== 新增：保存动作分布数据 ====================
        # 计算episode平均分布
        avg_episode_suppress_pred = np.mean(episode_suppress_pred_dists, axis=0) if episode_suppress_pred_dists else np.zeros(5)
        avg_episode_suppress_true = np.mean(episode_suppress_true_dists, axis=0) if episode_suppress_true_dists else np.zeros(5)
        avg_episode_rumor_pred = np.mean(episode_rumor_pred_dists, axis=0) if episode_rumor_pred_dists else np.zeros(5)
        avg_episode_rumor_true = np.mean(episode_rumor_true_dists, axis=0) if episode_rumor_true_dists else np.zeros(5)
        
        episode_data['action_distributions']['suppress_pred'].append(avg_episode_suppress_pred)
        episode_data['action_distributions']['suppress_true'].append(avg_episode_suppress_true)
        episode_data['action_distributions']['rumor_pred'].append(avg_episode_rumor_pred)
        episode_data['action_distributions']['rumor_true'].append(avg_episode_rumor_true)

        # ==================== 新增：保存C1和C2指标 ====================
        # 计算收敛时间(CT)
        conv_time = STEPS_PER_EPISODE
        for i, inf_rate in enumerate(episode_infections):
            if inf_rate <= 0.05:  # 感染率低于5%认为收敛
                conv_time = i
                break

        # 保存C1指标
        episode_data['c1_metrics']['LE'].append(np.mean(episode_c1_le) if episode_c1_le else 0)
        episode_data['c1_metrics']['CT'].append(conv_time)
        episode_data['c1_metrics']['SSE'].append(np.mean(episode_c1_sse) if episode_c1_sse else 0)

        # 保存C2指标
        episode_data['c2_metrics']['FV'].append(np.mean(episode_c2_fv) if episode_c2_fv else 0)
        episode_data['c2_metrics']['IDR_UC'].append(np.mean(episode_c2_idr_uc) if episode_c2_idr_uc else 0)
        episode_data['c2_metrics']['PE'].append(np.mean(episode_c2_pe) if episode_c2_pe else 0)
        episode_data['c2_metrics']['AE'].append(np.mean(episode_c2_ae) if episode_c2_ae else 0)

        # 保存时间序列指标
        episode_data['time_step_metrics']['infection_rates'].append(episode_infections)
        episode_data['time_step_metrics']['counter_rates'].append(episode_counters)
        
        # 计算平均奖励和状态价值
        avg_suppress_rewards = np.mean([episode_suppress_rewards[i] for i in range(NUM_AGENTS)], axis=0)
        avg_rumor_rewards = np.mean([episode_rumor_rewards[i] for i in range(NUM_AGENTS)], axis=0)
        avg_suppress_values = np.mean([episode_suppress_values[i] for i in range(NUM_AGENTS)], axis=0)
        avg_rumor_values = np.mean([episode_rumor_values[i] for i in range(NUM_AGENTS)], axis=0)
        
        episode_data['time_step_metrics']['suppress_rewards'].append(avg_suppress_rewards)
        episode_data['time_step_metrics']['rumor_rewards'].append(avg_rumor_rewards)
        episode_data['time_step_metrics']['suppress_values'].append(avg_suppress_values)
        episode_data['time_step_metrics']['rumor_values'].append(avg_rumor_values)
        
        episode_data['time_step_metrics']['c1_le'].append(episode_c1_le)
        episode_data['time_step_metrics']['c1_sse'].append(episode_c1_sse)
        episode_data['time_step_metrics']['c2_fv'].append(episode_c2_fv)
        episode_data['time_step_metrics']['c2_idr_uc'].append(episode_c2_idr_uc)
        episode_data['time_step_metrics']['c2_pe'].append(episode_c2_pe)
        episode_data['time_step_metrics']['c2_ae'].append(episode_c2_ae)

        controller.decay_epsilon()  # 衰减探索率
        print(f"Phase 1 Episode {episode} Complete, New Epsilon: {controller.epsilon:.4f}")

        # 计算平均值
        avg_infection /= STEPS_PER_EPISODE
        avg_counter /= STEPS_PER_EPISODE

        # 计算总奖励用于最佳模型判断
        total_suppress_reward = sum(total_suppress_rewards)
        total_rumor_reward = sum(total_rumor_rewards)

        # 保存第一阶段最佳模型
        if total_suppress_reward > best_suppress_reward:
            best_suppress_reward = total_suppress_reward
            best_rumor_reward = total_rumor_reward
            best_episode = episode

            # 保存第一阶段模型（只保存Actor参数）
            torch.save({
                'episode': episode,
                'controller_actor_state_dict': controller.get_actor_state_dict(),  # 只保存Actor参数
                'suppress_reward': total_suppress_reward,
                'rumor_reward': total_rumor_reward,
                'infection_rate': avg_infection,
                'counter_rate': avg_counter,
                'epsilon': controller.epsilon
            }, f'models/phase1_best_actor_episode_{episode}.pth')

            print(f"Saved Phase 1 Best Actor Model at Episode {episode}, Suppress Reward: {best_suppress_reward:.2f}")

        # 打印本轮统计和C1/C2指标
        print(f"Phase 1 Episode {episode} Complete: Avg Infection Rate {avg_infection:.4f}, Avg Counter Rate {avg_counter:.4f}")
        print(f"Suppress Agent Rewards: {[f'{r:.2f}' for r in total_suppress_rewards]}, Total Reward: {total_suppress_reward:.2f}")
        print(f"Rumor Agent Rewards: {[f'{r:.2f}' for r in total_rumor_rewards]}, Total Reward: {total_rumor_reward:.2f}")
        print(f"Opponent Accuracy - Suppress: {np.mean(episode_suppress_opp_acc):.4f}, Rumor: {np.mean(episode_rumor_opp_acc):.4f}")
        
        # 打印C1和C2指标
        print(f"C1指标 - LE: {np.mean(episode_c1_le) if episode_c1_le else 0:.6f}, CT: {conv_time}, SSE: {np.mean(episode_c1_sse) if episode_c1_sse else 0:.6f}")
        print(f"C2指标 - FV: {np.mean(episode_c2_fv) if episode_c2_fv else 0:.6f}, IDR_UC: {np.mean(episode_c2_idr_uc) if episode_c2_idr_uc else 0:.6f}, PE: {np.mean(episode_c2_pe) if episode_c2_pe else 0:.6f}, AE: {np.mean(episode_c2_ae) if episode_c2_ae else 0:.6f}")
        print("-" * 80)

    # 保存第一阶段最终模型
    torch.save({
        'episode': EPISODES - 1,
        'controller_actor_state_dict': controller.get_actor_state_dict(),
        'epsilon': controller.epsilon
    }, 'models/phase1_final_actor.pth')
    print("Phase 1 Training Complete, Saved Final Actor Model")

    # ==================== 第二阶段：锁定Actor，训练对手模型 ====================
    print("\n" + "=" * 60)
    print("Phase 2: Lock Actor, Train Opponent Models")
    print("=" * 60)

    # 解锁对手模型，锁定Actor
    controller.freeze_opponent_models = False
    controller.freeze_actor_models = True

    # 重置经验池，专注于对手模型训练
    controller.clear_experiences()

    for episode in range(EPISODES):
        # 重置节点状态
        network.reset(INIT_INFECTION, INIT_COUNTER)
        MAX_INFECTION_RATE = 0.0

        # 初始化统计
        episode_suppress_opp_losses = []
        episode_rumor_opp_losses = []
        episode_suppress_opp_acc = []  # 按角色存储
        episode_rumor_opp_acc = []     # 按角色存储

        # ==================== 新增：第二阶段也需要记录动作分布 ====================
        episode_suppress_pred_dists = []
        episode_suppress_true_dists = []
        episode_rumor_pred_dists = []
        episode_rumor_true_dists = []

        # 对手模型损失 - 按角色累计
        episode_suppress_opp_loss = 0
        episode_rumor_opp_loss = 0

        print(f"\n===== Phase 2 Episode {episode} Start =====")

        for step in range(STEPS_PER_EPISODE):
            # 刷新 H_struct 与 H_dyn，并构造结构增强后的边特征
            network.refresh_representations(detach=True)
            H_struct_nodes, H_dyn_graph, _ = network.get_structure_and_dynamics_representation(detach=True)
            all_edge_features = network.get_edge_features(include_structural=True)
            Z_t_macro = network.get_macro_density()
            time_step = controller.total_experiences

            # 获取关键边信息
            critical_edges = network.critical_edges
            current_infection_rate = network.node_states[:, 1].mean().item()

            # 智能体选择动作（使用第一阶段训练好的Actor）
            (all_suppress_actions, all_rumor_actions,
             all_suppress_log_probs, all_rumor_log_probs,
             all_suppress_opp_probs, all_rumor_opp_probs,
             all_suppress_values, all_rumor_values,
             macro_obs_counter, macro_obs_rumor,
             macro_counter_action, macro_rumor_action) = controller.get_actions(
                all_edge_features,
                H_dyn_graph,
                Z_t_macro,
                network.fixed_edge_index,
                critical_edges
            )

            # 合并所有智能体的动作并应用到网络
            suppress_actions = []
            rumor_actions = []
            applied_edge_indices = []

            for i in range(NUM_AGENTS):
                subnet_edges = subnets[i]['edges']
                suppress_actions.extend(all_suppress_actions[i])
                rumor_actions.extend(all_rumor_actions[i])
                applied_edge_indices.extend(subnet_edges)

            network.apply_actions(suppress_actions, rumor_actions, applied_edge_indices)

            # 单步传播
            macro_actions = macro_counter_action - macro_rumor_action
            infection_rate, counter_rate = network.step(macro_actions=macro_actions)

            # 更新最大感染率
            MAX_INFECTION_RATE = max(MAX_INFECTION_RATE, infection_rate)

            # ==================== 修改：使用分布相似度计算对手预测准确率 ====================
            suppress_similarity = 0
            rumor_similarity = 0
            total_suppress_edges = 0
            total_rumor_edges = 0

            # 存储分布数据
            step_suppress_pred_dists = []
            step_suppress_true_dists = []
            step_rumor_pred_dists = []
            step_rumor_true_dists = []

            for i in range(NUM_AGENTS):
                if len(all_suppress_opp_probs[i]) > 0 and len(all_rumor_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_suppress_opp_probs[i], all_rumor_actions[i])
                    suppress_similarity += sim * len(all_rumor_actions[i])
                    total_suppress_edges += len(all_rumor_actions[i])

                    # 存储分布数据
                    step_suppress_pred_dists.append(pred_dist)
                    step_suppress_true_dists.append(true_dist)

                if len(all_rumor_opp_probs[i]) > 0 and len(all_suppress_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_rumor_opp_probs[i], all_suppress_actions[i])
                    rumor_similarity += sim * len(all_suppress_actions[i])
                    total_rumor_edges += len(all_suppress_actions[i])

                    # 存储分布数据
                    step_rumor_pred_dists.append(pred_dist)
                    step_rumor_true_dists.append(true_dist)

            suppress_acc = suppress_similarity / total_suppress_edges if total_suppress_edges > 0 else 0
            rumor_acc = rumor_similarity / total_rumor_edges if total_rumor_edges > 0 else 0

            # 计算平均分布
            avg_suppress_pred_dist = np.mean(step_suppress_pred_dists, axis=0) if step_suppress_pred_dists else np.zeros(5)
            avg_suppress_true_dist = np.mean(step_suppress_true_dists, axis=0) if step_suppress_true_dists else np.zeros(5)
            avg_rumor_pred_dist = np.mean(step_rumor_pred_dists, axis=0) if step_rumor_pred_dists else np.zeros(5)
            avg_rumor_true_dist = np.mean(step_rumor_true_dists, axis=0) if step_rumor_true_dists else np.zeros(5)

            # 存储到episode数据
            episode_suppress_pred_dists.append(avg_suppress_pred_dist)
            episode_suppress_true_dists.append(avg_suppress_true_dist)
            episode_rumor_pred_dists.append(avg_rumor_pred_dist)
            episode_rumor_true_dists.append(avg_rumor_true_dist)

            episode_suppress_opp_acc.append(suppress_acc)
            episode_rumor_opp_acc.append(rumor_acc)

            # 获取新边特征，并保持与当前观测相同的结构增强维度
            network.refresh_representations(detach=True)
            next_edge_features = network.get_edge_features(include_structural=True)

            # 为每个智能体存储经验（只用于对手模型训练）
            for agent_id in range(NUM_AGENTS):
                subnet_edge_features = controller.get_subnet_edge_features(all_edge_features, agent_id)
                subnet_next_edge_features = controller.get_subnet_edge_features(next_edge_features, agent_id)

                if subnet_edge_features.size(0) == 0:
                    continue

                # 计算基础奖励（简化版，主要用于对手模型训练）
                delta_infection = infection_rate - prev_infection_rate
                suppress_reward = BASE_REWARD + INFECTION_CHANGE_WEIGHT * (-delta_infection)
                rumor_reward = BASE_REWARD + INFECTION_CHANGE_WEIGHT * delta_infection

                # 存储经验
                suppress_opp_action = all_rumor_actions[agent_id]
                rumor_opp_action = all_suppress_actions[agent_id]

                controller.store_experience(
                    agent_id,
                    (subnet_edge_features, all_suppress_actions[agent_id], all_suppress_log_probs[agent_id],
                     suppress_reward, subnet_next_edge_features, all_suppress_values[agent_id]),
                    (subnet_edge_features, all_rumor_actions[agent_id], all_rumor_log_probs[agent_id],
                     rumor_reward, subnet_next_edge_features, all_rumor_values[agent_id]),
                    (suppress_opp_action, rumor_opp_action)
                )

            # 第二阶段：只更新对手模型
            step_suppress_opp_loss, step_rumor_opp_loss = controller.update_opponent_models(batch_size=4)

            episode_suppress_opp_loss += step_suppress_opp_loss
            episode_rumor_opp_loss += step_rumor_opp_loss

            # 记录对手模型损失
            episode_suppress_opp_losses.append(step_suppress_opp_loss)
            episode_rumor_opp_losses.append(step_rumor_opp_loss)

            # 更新前一状态
            prev_infection_rate = infection_rate
            prev_counter_rate = counter_rate

            # 打印训练信息
            if step % 1 == 0:
                print(
                    f"Phase 2 Step {step}: Opponent Model Loss[Suppress:{step_suppress_opp_loss:.4f}, Rumor:{step_rumor_opp_loss:.4f}], "
                    f"Accuracy[Suppress:{suppress_acc:.3f}, Rumor:{rumor_acc:.3f}]")

        # 记录第二阶段数据
        episode_data['suppress_opp_losses'].append(episode_suppress_opp_losses)
        episode_data['rumor_opp_losses'].append(episode_rumor_opp_losses)
        episode_data['suppress_opp_acc'].append(episode_suppress_opp_acc)  # 按角色存储
        episode_data['rumor_opp_acc'].append(episode_rumor_opp_acc)        # 按角色存储

        # ==================== 新增：保存第二阶段动作分布数据 ====================
        # 计算episode平均分布
        avg_episode_suppress_pred = np.mean(episode_suppress_pred_dists, axis=0) if episode_suppress_pred_dists else np.zeros(5)
        avg_episode_suppress_true = np.mean(episode_suppress_true_dists, axis=0) if episode_suppress_true_dists else np.zeros(5)
        avg_episode_rumor_pred = np.mean(episode_rumor_pred_dists, axis=0) if episode_rumor_pred_dists else np.zeros(5)
        avg_episode_rumor_true = np.mean(episode_rumor_true_dists, axis=0) if episode_rumor_true_dists else np.zeros(5)

        episode_data['action_distributions']['suppress_pred'].append(avg_episode_suppress_pred)
        episode_data['action_distributions']['suppress_true'].append(avg_episode_suppress_true)
        episode_data['action_distributions']['rumor_pred'].append(avg_episode_rumor_pred)
        episode_data['action_distributions']['rumor_true'].append(avg_episode_rumor_true)

        print(f"Phase 2 Episode {episode} Complete")
        print(
            f"Avg Opponent Model Loss - Suppress: {episode_suppress_opp_loss / STEPS_PER_EPISODE:.4f}, Rumor: {episode_rumor_opp_loss / STEPS_PER_EPISODE:.4f}")
        print(
            f"Avg Opponent Accuracy - Suppress: {np.mean(episode_suppress_opp_acc):.4f}, Rumor: {np.mean(episode_rumor_opp_acc):.4f}")

    # 保存第二阶段最终对手模型
    torch.save({
        'controller_opponent_state_dict': controller.get_opponent_state_dict(),
    }, 'models/phase2_final_opponent.pth')
    print("Phase 2 Training Complete, Saved Final Opponent Model")

    # ==================== 第三阶段：同步训练Actor和对手模型 ====================
    print("\n" + "=" * 60)
    print("Phase 3: Joint Training of Actor and Opponent Models")
    print("=" * 60)

    # 解锁所有模型进行同步训练
    controller.freeze_opponent_models = False
    controller.freeze_actor_models = False

    # 重置经验池
    controller.clear_experiences()

    for episode in range(EPISODES):
        # 重置节点状态
        network.reset(INIT_INFECTION, INIT_COUNTER)
        MAX_INFECTION_RATE = 0.0

        # 初始化当前episode的统计（完整统计）
        episode_infections = []
        episode_counters = []
        episode_difficulties = []
        episode_suppress_rewards = [[] for _ in range(NUM_AGENTS)]
        episode_rumor_rewards = [[] for _ in range(NUM_AGENTS)]
        episode_suppress_values = [[] for _ in range(NUM_AGENTS)]
        episode_rumor_values = [[] for _ in range(NUM_AGENTS)]
        episode_suppress_opp_acc = []  # 按角色存储
        episode_rumor_opp_acc = []     # 按角色存储
        episode_suppress_opp_losses = []
        episode_rumor_opp_losses = []

        # ==================== 新增：C1和C2指标时间序列 ====================
        episode_c1_le = []        # LE时间序列
        episode_c1_sse = []       # SSE时间序列
        episode_c2_fv = []        # FV时间序列
        episode_c2_idr_uc = []    # IDR_UC时间序列
        episode_c2_pe = []        # PE时间序列
        episode_c2_ae = []        # AE时间序列

        # ==================== 新增：动作分布数据 ====================
        episode_suppress_pred_dists = []
        episode_suppress_true_dists = []
        episode_rumor_pred_dists = []
        episode_rumor_true_dists = []

        # 初始化统计
        total_suppress_rewards = [0] * NUM_AGENTS
        total_rumor_rewards = [0] * NUM_AGENTS
        avg_infection = 0
        avg_counter = 0

        # 对手预测统计（按角色）
        suppress_correct = 0
        rumor_correct = 0
        total_predictions = 0

        # 对手模型损失 - 按角色累计
        episode_suppress_opp_loss = 0
        episode_rumor_opp_loss = 0

        # 状态价值统计 - 按智能体区分
        suppress_value_sum = [0] * NUM_AGENTS
        rumor_value_sum = [0] * NUM_AGENTS

        # 记录初始状态
        prev_infection_rate = network.node_states[:, 1].mean().item()
        prev_counter_rate = network.node_states[:, 2].mean().item()

        print(f"\n===== Phase 3 Episode {episode} Start =====")

        # 记录动作历史用于C2指标计算
        suppress_actions_history = []
        rumor_actions_history = []

        for step in range(STEPS_PER_EPISODE):
            # 刷新 H_struct 与 H_dyn，并构造结构增强后的边特征
            network.refresh_representations(detach=True)
            H_struct_nodes, H_dyn_graph, _ = network.get_structure_and_dynamics_representation(detach=True)
            all_edge_features = network.get_edge_features(include_structural=True)
            Z_t_macro = network.get_macro_density()
            time_step = controller.total_experiences

            # 获取关键边信息
            critical_edges = network.critical_edges

            current_infection_rate = network.node_states[:, 1].mean().item()

            # 智能体选择动作
            (all_suppress_actions, all_rumor_actions,
             all_suppress_log_probs, all_rumor_log_probs,
             all_suppress_opp_probs, all_rumor_opp_probs,
             all_suppress_values, all_rumor_values,
             macro_obs_counter, macro_obs_rumor,
             macro_counter_action, macro_rumor_action) = controller.get_actions(
                all_edge_features,
                H_dyn_graph,
                Z_t_macro,
                network.fixed_edge_index,
                critical_edges
            )

            # 合并所有智能体的动作
            suppress_actions = []
            rumor_actions = []
            applied_edge_indices = []

            for i in range(NUM_AGENTS):
                subnet_edges = subnets[i]['edges']
                suppress_actions.extend(all_suppress_actions[i])
                rumor_actions.extend(all_rumor_actions[i])
                applied_edge_indices.extend(subnet_edges)

            # 记录动作历史
            suppress_actions_history.append(suppress_actions)
            rumor_actions_history.append(rumor_actions)

            # 应用动作到网络
            network.apply_actions(suppress_actions, rumor_actions, applied_edge_indices)

            # 单步传播
            macro_actions = macro_counter_action - macro_rumor_action
            infection_rate, counter_rate = network.step(macro_actions=macro_actions)
            avg_infection += infection_rate
            avg_counter += counter_rate

            # 更新最大感染率
            MAX_INFECTION_RATE = max(MAX_INFECTION_RATE, infection_rate)

            # 计算感染比例和难度
            infection_ratio = infection_rate / (MAX_INFECTION_RATE + 1e-5)
            infection_ratio = np.clip(infection_ratio, 0.0, 1.0)
            difficulty = 1 / (1 + math.exp(-K * infection_ratio))
            episode_difficulties.append(difficulty)

            # 确定感染阶段
            if infection_ratio > HIGH_INFECTION_THRESHOLD:
                infection_stage = "High Infection"
            elif infection_ratio < LOW_INFECTION_THRESHOLD:
                infection_stage = "Low Infection"
            else:
                infection_stage = "Transition"

            # 计算奖励（使用完整奖励函数）
            delta_infection = infection_rate - prev_infection_rate
            delta_counter = counter_rate - prev_counter_rate

            # 辟谣智能体奖励
            suppress_reward = BASE_REWARD + 0.2 + INFECTION_CHANGE_WEIGHT * (-delta_infection)
            if infection_stage == "High Infection":
                suppress_reward += COUNTER_CHANGE_WEIGHT * 0.8 * delta_counter
                action_cost = ACTION_COST_WEIGHT * 0.7 * np.mean(suppress_actions)
            elif infection_stage == "Low Infection":
                suppress_reward += COUNTER_CHANGE_WEIGHT * 1.2 * delta_counter
                action_cost = ACTION_COST_WEIGHT * 1.3 * np.mean(suppress_actions)
            else:
                suppress_reward += COUNTER_CHANGE_WEIGHT * delta_counter
                action_cost = ACTION_COST_WEIGHT * np.mean(suppress_actions)
            suppress_reward -= action_cost

            # 传谣智能体奖励
            rumor_reward = BASE_REWARD + INFECTION_CHANGE_WEIGHT * delta_infection
            if infection_stage == "High Infection":
                rumor_reward += COUNTER_CHANGE_WEIGHT * 0.8 * (-delta_counter)
                action_cost = ACTION_COST_WEIGHT * 0.7 * np.mean(rumor_actions)
            elif infection_stage == "Low Infection":
                rumor_reward += COUNTER_CHANGE_WEIGHT * 1.2 * (-delta_counter)
                action_cost = ACTION_COST_WEIGHT * 1.3 * np.mean(rumor_actions)
            else:
                rumor_reward += COUNTER_CHANGE_WEIGHT * (-delta_counter)
                action_cost = ACTION_COST_WEIGHT * np.mean(rumor_actions)
            rumor_reward -= action_cost

            # ==================== 修改：使用分布相似度计算对手预测准确率 ====================
            suppress_similarity = 0
            rumor_similarity = 0
            total_suppress_edges = 0
            total_rumor_edges = 0

            # 存储分布数据
            step_suppress_pred_dists = []
            step_suppress_true_dists = []
            step_rumor_pred_dists = []
            step_rumor_true_dists = []

            for i in range(NUM_AGENTS):
                if len(all_suppress_opp_probs[i]) > 0 and len(all_rumor_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_suppress_opp_probs[i], all_rumor_actions[i])
                    suppress_similarity += sim * len(all_rumor_actions[i])
                    total_suppress_edges += len(all_rumor_actions[i])
                    
                    # 存储分布数据
                    step_suppress_pred_dists.append(pred_dist)
                    step_suppress_true_dists.append(true_dist)

                if len(all_rumor_opp_probs[i]) > 0 and len(all_suppress_actions[i]) > 0:
                    sim, true_dist, pred_dist = calculate_distribution_similarity(
                        all_rumor_opp_probs[i], all_suppress_actions[i])
                    rumor_similarity += sim * len(all_suppress_actions[i])
                    total_rumor_edges += len(all_suppress_actions[i])
                    
                    # 存储分布数据
                    step_rumor_pred_dists.append(pred_dist)
                    step_rumor_true_dists.append(true_dist)

            suppress_acc = suppress_similarity / total_suppress_edges if total_suppress_edges > 0 else 0
            rumor_acc = rumor_similarity / total_rumor_edges if total_rumor_edges > 0 else 0

            # 计算平均分布
            avg_suppress_pred_dist = np.mean(step_suppress_pred_dists, axis=0) if step_suppress_pred_dists else np.zeros(5)
            avg_suppress_true_dist = np.mean(step_suppress_true_dists, axis=0) if step_suppress_true_dists else np.zeros(5)
            avg_rumor_pred_dist = np.mean(step_rumor_pred_dists, axis=0) if step_rumor_pred_dists else np.zeros(5)
            avg_rumor_true_dist = np.mean(step_rumor_true_dists, axis=0) if step_rumor_true_dists else np.zeros(5)
            
            # 存储到episode数据
            episode_suppress_pred_dists.append(avg_suppress_pred_dist)
            episode_suppress_true_dists.append(avg_suppress_true_dist)
            episode_rumor_pred_dists.append(avg_rumor_pred_dist)
            episode_rumor_true_dists.append(avg_rumor_true_dist)

            suppress_reward += OPP_ACCURACY_BONUS * suppress_acc
            rumor_reward += OPP_ACCURACY_BONUS * rumor_acc

            # 记录数据
            episode_infections.append(infection_rate)
            episode_counters.append(counter_rate)
            episode_suppress_opp_acc.append(suppress_acc)  # 按角色记录
            episode_rumor_opp_acc.append(rumor_acc)        # 按角色记录

            # ==================== 新增：计算C1和C2指标 ====================
            # 计算Lyapunov指数(LE)
            if len(episode_infections) > 1:
                le = np.mean(np.diff(np.log(np.array(episode_infections) + 1e-10)))
                episode_c1_le.append(le)

            # 计算稳态误差(SSE)
            target_infection = 0.0
            sse = abs(infection_rate - target_infection)
            episode_c1_sse.append(sse)

            # 计算感染峰值下降速度(FV)
            peak_infection = max(episode_infections) if episode_infections else 0
            fv = (peak_infection - infection_rate) / (len(episode_infections) + 1e-10)
            episode_c2_fv.append(fv)

            # 计算单位成本感染下降率(IDR_UC)
            all_suppress_actions_flat = [action for step_actions in suppress_actions_history for action in step_actions]
            all_rumor_actions_flat = [action for step_actions in rumor_actions_history for action in step_actions]

            if all_suppress_actions_flat and all_rumor_actions_flat:
                total_action_cost = np.mean(all_suppress_actions_flat) + np.mean(all_rumor_actions_flat)
            else:
                total_action_cost = 0
            idr_uc = (INIT_INFECTION - infection_rate) / (total_action_cost + 1e-10)
            episode_c2_idr_uc.append(idr_uc)

            # 计算传播范围(PE)
            ir_count = 0
            for node_state in network.node_states:
                if node_state[1] > 0.5 or node_state[3] > 0.5:
                    ir_count += 1
            pe = ir_count / len(network.node_states)
            episode_c2_pe.append(pe)

            # 计算策略动作熵(AE)
            suppress_entropies = []
            rumor_entropies = []
            
            for i in range(NUM_AGENTS):
                if len(all_suppress_log_probs[i]) > 0:
                    suppress_probs = torch.softmax(all_suppress_log_probs[i], dim=-1)
                    suppress_entropy = calculate_entropy([suppress_probs.cpu().detach().numpy()])
                    suppress_entropies.append(suppress_entropy)
                
                if len(all_rumor_log_probs[i]) > 0:
                    rumor_probs = torch.softmax(all_rumor_log_probs[i], dim=-1)
                    rumor_entropy = calculate_entropy([rumor_probs.cpu().detach().numpy()])
                    rumor_entropies.append(rumor_entropy)

            avg_suppress_entropy = np.mean(suppress_entropies) if suppress_entropies else 0
            avg_rumor_entropy = np.mean(rumor_entropies) if rumor_entropies else 0
            ae = (avg_suppress_entropy + avg_rumor_entropy) / 2
            episode_c2_ae.append(ae)

            # ==================== 指标计算结束 ====================

            # 获取新边特征，并保持与当前观测相同的结构增强维度
            network.refresh_representations(detach=True)
            next_edge_features = network.get_edge_features(include_structural=True)

            # 为每个智能体存储经验
            for agent_id in range(NUM_AGENTS):
                subnet_edge_features = controller.get_subnet_edge_features(all_edge_features, agent_id)
                subnet_next_edge_features = controller.get_subnet_edge_features(next_edge_features, agent_id)

                if subnet_edge_features.size(0) == 0:
                    continue

                # 计算智能体的子网奖励（使用完整的奖励分配逻辑）
                subnet_edge_count = len(all_suppress_actions[agent_id])
                total_edge_count = len(suppress_actions)
                subnet_ratio = subnet_edge_count / total_edge_count if total_edge_count > 0 else 0

                subnet_suppress_action_mean = np.mean(all_suppress_actions[agent_id]) if len(
                    all_suppress_actions[agent_id]) > 0 else 0
                subnet_rumor_action_mean = np.mean(all_rumor_actions[agent_id]) if len(
                    all_rumor_actions[agent_id]) > 0 else 0

                global_suppress_action_mean = np.mean(suppress_actions) if len(suppress_actions) > 0 else 0
                global_rumor_action_mean = np.mean(rumor_actions) if len(rumor_actions) > 0 else 0

                suppress_efficiency = max(0.5,
                                          min(2.0, global_suppress_action_mean / (subnet_suppress_action_mean + 1e-5)))
                rumor_efficiency = max(0.5, min(2.0, global_rumor_action_mean / (subnet_rumor_action_mean + 1e-5)))

                suppress_weight = subnet_ratio * suppress_efficiency
                rumor_weight = subnet_ratio * rumor_efficiency

                total_suppress_weight = sum([len(all_suppress_actions[i]) / total_edge_count *
                                             (global_suppress_action_mean / (np.mean(all_suppress_actions[i]) + 1e-5)
                                              if len(all_suppress_actions[i]) > 0 else 1.0)
                                             for i in range(NUM_AGENTS)])
                total_rumor_weight = sum([len(all_rumor_actions[i]) / total_edge_count *
                                          (global_rumor_action_mean / (np.mean(all_rumor_actions[i]) + 1e-5)
                                           if len(all_rumor_actions[i]) > 0 else 1.0)
                                          for i in range(NUM_AGENTS)])

                total_suppress_weight = total_suppress_weight if total_suppress_weight > 0 else 1
                total_rumor_weight = total_rumor_weight if total_rumor_weight > 0 else 1

                normalized_suppress_weight = suppress_weight / total_suppress_weight
                normalized_rumor_weight = rumor_weight / total_rumor_weight

                subnet_suppress_reward = suppress_reward * normalized_suppress_weight
                subnet_rumor_reward = rumor_reward * normalized_rumor_weight

                # 记录奖励
                total_suppress_rewards[agent_id] += subnet_suppress_reward
                total_rumor_rewards[agent_id] += subnet_rumor_reward
                episode_suppress_rewards[agent_id].append(subnet_suppress_reward)
                episode_rumor_rewards[agent_id].append(subnet_rumor_reward)

                # 记录状态价值
                if isinstance(all_suppress_values[agent_id], (list, np.ndarray)):
                    suppress_val = np.mean(all_suppress_values[agent_id]) if len(
                        all_suppress_values[agent_id]) > 0 else 0
                else:
                    suppress_val = all_suppress_values[agent_id]
                suppress_value_sum[agent_id] += suppress_val
                episode_suppress_values[agent_id].append(suppress_val)

                if isinstance(all_rumor_values[agent_id], (list, np.ndarray)):
                    rumor_val = np.mean(all_rumor_values[agent_id]) if len(all_rumor_values[agent_id]) > 0 else 0
                else:
                    rumor_val = all_rumor_values[agent_id]
                rumor_value_sum[agent_id] += rumor_val
                episode_rumor_values[agent_id].append(rumor_val)

                # 存储经验
                suppress_opp_action = all_rumor_actions[agent_id]
                rumor_opp_action = all_suppress_actions[agent_id]

                controller.store_experience(
                    agent_id,
                    (subnet_edge_features, all_suppress_actions[agent_id], all_suppress_log_probs[agent_id],
                     subnet_suppress_reward, subnet_next_edge_features, all_suppress_values[agent_id]),
                    (subnet_edge_features, all_rumor_actions[agent_id], all_rumor_log_probs[agent_id],
                     subnet_rumor_reward, subnet_next_edge_features, all_rumor_values[agent_id]),
                    (suppress_opp_action, rumor_opp_action)
                )

            # 第三阶段：同时更新Actor和对手模型
            # 更新对手模型
            step_suppress_opp_loss, step_rumor_opp_loss = controller.update_opponent_models(batch_size=4)

            episode_suppress_opp_loss += step_suppress_opp_loss
            episode_rumor_opp_loss += step_rumor_opp_loss
            episode_suppress_opp_losses.append(step_suppress_opp_loss)
            episode_rumor_opp_losses.append(step_rumor_opp_loss)

            # 更新Actor
            controller.update_agents(batch_size=4)

            # 更新前一状态
            prev_infection_rate = infection_rate
            prev_counter_rate = counter_rate

            # 打印训练信息
            if step % 1 == 0:
                print(f"Phase 3 Step {step}: [{infection_stage}] Infection Rate: {infection_rate:.4f}, "
                      f"Suppress Reward: {suppress_reward:.4f}, Rumor Reward: {rumor_reward:.4f}")
                print(
                    f"Opponent Model Loss[Suppress:{step_suppress_opp_loss:.4f}, Rumor:{step_rumor_opp_loss:.4f}], "
                    f"Accuracy[Suppress:{suppress_acc:.3f}, Rumor:{rumor_acc:.3f}]")


        # 记录第三阶段数据
        episode_data['infections'].append(episode_infections)
        episode_data['counters'].append(episode_counters)
        episode_data['difficulties'].append(episode_difficulties)
        episode_data['suppress_opp_losses'].append(episode_suppress_opp_losses)
        episode_data['rumor_opp_losses'].append(episode_rumor_opp_losses)
        episode_data['suppress_opp_acc'].append(episode_suppress_opp_acc)  # 按角色存储
        episode_data['rumor_opp_acc'].append(episode_rumor_opp_acc)        # 按角色存储

        for i in range(NUM_AGENTS):
            episode_data['suppress_rewards'][i].append(episode_suppress_rewards[i])
            episode_data['rumor_rewards'][i].append(episode_rumor_rewards[i])
            episode_data['suppress_values'][i].append(episode_suppress_values[i])
            episode_data['rumor_values'][i].append(episode_rumor_values[i])

        # ==================== 新增：保存动作分布数据 ====================
        # 计算episode平均分布
        avg_episode_suppress_pred = np.mean(episode_suppress_pred_dists, axis=0) if episode_suppress_pred_dists else np.zeros(5)
        avg_episode_suppress_true = np.mean(episode_suppress_true_dists, axis=0) if episode_suppress_true_dists else np.zeros(5)
        avg_episode_rumor_pred = np.mean(episode_rumor_pred_dists, axis=0) if episode_rumor_pred_dists else np.zeros(5)
        avg_episode_rumor_true = np.mean(episode_rumor_true_dists, axis=0) if episode_rumor_true_dists else np.zeros(5)
        
        episode_data['action_distributions']['suppress_pred'].append(avg_episode_suppress_pred)
        episode_data['action_distributions']['suppress_true'].append(avg_episode_suppress_true)
        episode_data['action_distributions']['rumor_pred'].append(avg_episode_rumor_pred)
        episode_data['action_distributions']['rumor_true'].append(avg_episode_rumor_true)

        # ==================== 新增：保存C1和C2指标 ====================
        # 计算收敛时间(CT)
        conv_time = STEPS_PER_EPISODE
        for i, inf_rate in enumerate(episode_infections):
            if inf_rate <= 0.05:  # 感染率低于5%认为收敛
                conv_time = i
                break

        # 保存C1指标
        episode_data['c1_metrics']['LE'].append(np.mean(episode_c1_le) if episode_c1_le else 0)
        episode_data['c1_metrics']['CT'].append(conv_time)
        episode_data['c1_metrics']['SSE'].append(np.mean(episode_c1_sse) if episode_c1_sse else 0)

        # 保存C2指标
        episode_data['c2_metrics']['FV'].append(np.mean(episode_c2_fv) if episode_c2_fv else 0)
        episode_data['c2_metrics']['IDR_UC'].append(np.mean(episode_c2_idr_uc) if episode_c2_idr_uc else 0)
        episode_data['c2_metrics']['PE'].append(np.mean(episode_c2_pe) if episode_c2_pe else 0)
        episode_data['c2_metrics']['AE'].append(np.mean(episode_c2_ae) if episode_c2_ae else 0)

        # 保存时间序列指标
        episode_data['time_step_metrics']['infection_rates'].append(episode_infections)
        episode_data['time_step_metrics']['counter_rates'].append(episode_counters)
        
        # 计算平均奖励和状态价值
        avg_suppress_rewards = np.mean([episode_suppress_rewards[i] for i in range(NUM_AGENTS)], axis=0)
        avg_rumor_rewards = np.mean([episode_rumor_rewards[i] for i in range(NUM_AGENTS)], axis=0)
        avg_suppress_values = np.mean([episode_suppress_values[i] for i in range(NUM_AGENTS)], axis=0)
        avg_rumor_values = np.mean([episode_rumor_values[i] for i in range(NUM_AGENTS)], axis=0)
        
        episode_data['time_step_metrics']['suppress_rewards'].append(avg_suppress_rewards)
        episode_data['time_step_metrics']['rumor_rewards'].append(avg_rumor_rewards)
        episode_data['time_step_metrics']['suppress_values'].append(avg_suppress_values)
        episode_data['time_step_metrics']['rumor_values'].append(avg_rumor_values)
        
        episode_data['time_step_metrics']['c1_le'].append(episode_c1_le)
        episode_data['time_step_metrics']['c1_sse'].append(episode_c1_sse)
        episode_data['time_step_metrics']['c2_fv'].append(episode_c2_fv)
        episode_data['time_step_metrics']['c2_idr_uc'].append(episode_c2_idr_uc)
        episode_data['time_step_metrics']['c2_pe'].append(episode_c2_pe)
        episode_data['time_step_metrics']['c2_ae'].append(episode_c2_ae)

        controller.decay_epsilon()
        print(f"Phase 3 Episode {episode} Complete, New Epsilon: {controller.epsilon:.4f}")

        # 计算平均值
        avg_infection /= STEPS_PER_EPISODE
        avg_counter /= STEPS_PER_EPISODE

        # 计算总奖励
        total_suppress_reward = sum(total_suppress_rewards)
        total_rumor_reward = sum(total_rumor_rewards)

        # 保存第三阶段最佳模型
        if total_suppress_reward > best_suppress_reward:
            best_suppress_reward = total_suppress_reward
            best_rumor_reward = total_rumor_reward
            best_episode = episode

            torch.save({
                'episode': episode,
                'controller_state_dict': controller.state_dict(),
                'suppress_reward': total_suppress_reward,
                'rumor_reward': total_rumor_reward,
                'infection_rate': avg_infection,
                'counter_rate': avg_counter,
                'epsilon': controller.epsilon
            }, f'models/phase3_best_model_episode_{episode}.pth')

            print(f"Saved Phase 3 Best Model at Episode {episode}, Suppress Reward: {best_suppress_reward:.2f}")

        print(f"Phase 3 Episode {episode} Complete: Avg Infection Rate {avg_infection:.4f}, Avg Counter Rate {avg_counter:.4f}")
        print(f"Suppress Agent Rewards: {[f'{r:.2f}' for r in total_suppress_rewards]}, Total Reward: {total_suppress_reward:.2f}")
        print(f"Rumor Agent Rewards: {[f'{r:.2f}' for r in total_rumor_rewards]}, Total Reward: {total_rumor_reward:.2f}")
        print(f"Opponent Accuracy - Suppress: {np.mean(episode_suppress_opp_acc):.4f}, Rumor: {np.mean(episode_rumor_opp_acc):.4f}")
        
        # 打印C1和C2指标
        print(f"C1指标 - LE: {np.mean(episode_c1_le) if episode_c1_le else 0:.6f}, CT: {conv_time}, SSE: {np.mean(episode_c1_sse) if episode_c1_sse else 0:.6f}")
        print(f"C2指标 - FV: {np.mean(episode_c2_fv) if episode_c2_fv else 0:.6f}, IDR_UC: {np.mean(episode_c2_idr_uc) if episode_c2_idr_uc else 0:.6f}, PE: {np.mean(episode_c2_pe) if episode_c2_pe else 0:.6f}, AE: {np.mean(episode_c2_ae) if episode_c2_ae else 0:.6f}")
        print("-" * 80)

    # 保存第三阶段最终模型
    torch.save({
        'episode': EPISODES - 1,
        'controller_state_dict': controller.state_dict(),
        'epsilon': controller.epsilon
    }, 'models/phase3_final_model.pth')
    print("Phase 3 Training Complete, Saved Final Complete Model")

    # ==================== 计算每个step的均值和标准差 ====================
    def compute_step_stats(data_2d):
        """
        Args:
            data_2d: 形状为 [num_episodes, num_steps] 的数据

        Returns:
            means: 形状为 [num_steps] 的均值
            stds: 形状为 [num_steps] 的标准差
        """
        # 转换为 numpy 数组
        data_array = np.array(data_2d)  # [episodes, steps]

        # 计算每个step的均值和标准差
        means = np.mean(data_array, axis=0)  # [steps]
        stds = np.std(data_array, axis=0)  # [steps]

        return means, stds

    def compute_agent_step_stats(data_3d):
        """
        处理智能体相关的三维数据 [num_agents, num_episodes, num_steps]

        Args:
            data_3d: 形状为 [num_agents, num_episodes, num_steps] 的数据

        Returns:
            means: 形状为 [num_agents, num_steps] 的均值
            stds: 形状为 [num_agents, num_steps] 的标准差
        """
        num_agents = len(data_3d)
        num_steps = len(data_3d[0][0]) if num_agents > 0 and len(data_3d[0]) > 0 else 0

        means = np.zeros((num_agents, num_steps))
        stds = np.zeros((num_agents, num_steps))

        for i in range(num_agents):
            # 将智能体i的数据转换为 [episodes, steps]
            agent_data = np.array(data_3d[i])  # [episodes, steps]
            means[i] = np.mean(agent_data, axis=0)
            stds[i] = np.std(agent_data, axis=0)

        return means, stds

    # 计算所有指标的统计量
    infection_means, infection_stds = compute_step_stats(episode_data['infections'])
    counter_means, counter_stds = compute_step_stats(episode_data['counters'])
    difficulty_means, difficulty_stds = compute_step_stats(episode_data['difficulties'])

    suppress_reward_means, suppress_reward_stds = compute_agent_step_stats(episode_data['suppress_rewards'])
    rumor_reward_means, rumor_reward_stds = compute_agent_step_stats(episode_data['rumor_rewards'])
    suppress_value_means, suppress_value_stds = compute_agent_step_stats(episode_data['suppress_values'])
    rumor_value_means, rumor_value_stds = compute_agent_step_stats(episode_data['rumor_values'])

    # 对于对手准确率，使用按角色存储的数据
    suppress_opp_acc_means, suppress_opp_acc_stds = compute_step_stats(episode_data['suppress_opp_acc'])
    rumor_opp_acc_means, rumor_opp_acc_stds = compute_step_stats(episode_data['rumor_opp_acc'])

    # 对于对手模型损失，检查数据是否存在且不为空
    if episode_data['suppress_opp_losses'] and len(episode_data['suppress_opp_losses']) > 0:
        suppress_opp_loss_means, suppress_opp_loss_stds = compute_step_stats(episode_data['suppress_opp_losses'])
    else:
        suppress_opp_loss_means = np.zeros(STEPS_PER_EPISODE)
        suppress_opp_loss_stds = np.zeros(STEPS_PER_EPISODE)

    if episode_data['rumor_opp_losses'] and len(episode_data['rumor_opp_losses']) > 0:
        rumor_opp_loss_means, rumor_opp_loss_stds = compute_step_stats(episode_data['rumor_opp_losses'])
    else:
        rumor_opp_loss_means = np.zeros(STEPS_PER_EPISODE)
        rumor_opp_loss_stds = np.zeros(STEPS_PER_EPISODE)

    # ==================== 新增：计算C1和C2指标的统计量 ====================
    # C1指标统计
    c1_le_means, c1_le_stds = compute_step_stats(episode_data['time_step_metrics']['c1_le'])
    c1_sse_means, c1_sse_stds = compute_step_stats(episode_data['time_step_metrics']['c1_sse'])
    
    # C2指标统计
    c2_fv_means, c2_fv_stds = compute_step_stats(episode_data['time_step_metrics']['c2_fv'])
    c2_idr_uc_means, c2_idr_uc_stds = compute_step_stats(episode_data['time_step_metrics']['c2_idr_uc'])
    c2_pe_means, c2_pe_stds = compute_step_stats(episode_data['time_step_metrics']['c2_pe'])
    c2_ae_means, c2_ae_stds = compute_step_stats(episode_data['time_step_metrics']['c2_ae'])

    # ==================== 可视化部分 - 使用英文标签 ====================
    # 定义颜色和标签
    colors_suppress = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_rumor = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    agent_labels_suppress = [f'Suppress Agent {i}' for i in range(NUM_AGENTS)]
    agent_labels_rumor = [f'Rumor Agent {i}' for i in range(NUM_AGENTS)]

    # 定义步数范围
    steps_full = list(range(STEPS_PER_EPISODE))
    steps_20 = list(range(20))  # 前20步

    # ==================== 修复可视化函数 ====================
    def create_visualizations(steps_range, suffix):
        """
        创建可视化图表
        Args:
            steps_range: 步数范围
            suffix: 文件名后缀
        """
        # 1. 感染率和辟谣率变化趋势（均值±标准差）
        plt.figure(figsize=(12, 6))
        plt.plot(steps_range, infection_means[:len(steps_range)], label='Infection Rate', color='red', linewidth=2)
        plt.fill_between(steps_range, 
                         infection_means[:len(steps_range)] - infection_stds[:len(steps_range)], 
                         infection_means[:len(steps_range)] + infection_stds[:len(steps_range)], 
                         alpha=0.3, color='red')

        plt.plot(steps_range, counter_means[:len(steps_range)], label='Counter Rate', color='blue', linewidth=2)
        plt.fill_between(steps_range, 
                         counter_means[:len(steps_range)] - counter_stds[:len(steps_range)], 
                         counter_means[:len(steps_range)] + counter_stds[:len(steps_range)], 
                         alpha=0.3, color='blue')

        plt.title('Infection and Counter Rate Trends')
        plt.xlabel('Step')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/infection_counter_trend_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 智能体奖励变化趋势 - 按智能体分开
        plt.figure(figsize=(14, 8))

        # 绘制辟谣智能体奖励
        for i in range(NUM_AGENTS):
            plt.plot(steps_range, suppress_reward_means[i][:len(steps_range)],
                     label=agent_labels_suppress[i],
                     color=colors_suppress[i],
                     linewidth=1.5)
            plt.fill_between(steps_range,
                             suppress_reward_means[i][:len(steps_range)] - suppress_reward_stds[i][:len(steps_range)],
                             suppress_reward_means[i][:len(steps_range)] + suppress_reward_stds[i][:len(steps_range)],
                             alpha=0.2, color=colors_suppress[i])

        # 绘制传谣智能体奖励
        for i in range(NUM_AGENTS):
            plt.plot(steps_range, rumor_reward_means[i][:len(steps_range)],
                     label=agent_labels_rumor[i],
                     color=colors_rumor[i],
                     linewidth=1.5, linestyle='--')
            plt.fill_between(steps_range,
                             rumor_reward_means[i][:len(steps_range)] - rumor_reward_stds[i][:len(steps_range)],
                             rumor_reward_means[i][:len(steps_range)] + rumor_reward_stds[i][:len(steps_range)],
                             alpha=0.2, color=colors_rumor[i])

        plt.title('Agent Reward Trends')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/agents_rewards_individual_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 对手预测准确率（按角色）
        plt.figure(figsize=(12, 6))

        plt.plot(steps_range, suppress_opp_acc_means[:len(steps_range)],
                 label='Suppress Opponent Accuracy',
                 color='blue',
                 linewidth=2)
        plt.fill_between(steps_range,
                         suppress_opp_acc_means[:len(steps_range)] - suppress_opp_acc_stds[:len(steps_range)],
                         suppress_opp_acc_means[:len(steps_range)] + suppress_opp_acc_stds[:len(steps_range)],
                         alpha=0.3, color='blue')

        plt.plot(steps_range, rumor_opp_acc_means[:len(steps_range)],
                 label='Rumor Opponent Accuracy',
                 color='red',
                 linewidth=2)
        plt.fill_between(steps_range,
                         rumor_opp_acc_means[:len(steps_range)] - rumor_opp_acc_stds[:len(steps_range)],
                         rumor_opp_acc_means[:len(steps_range)] + rumor_opp_acc_stds[:len(steps_range)],
                         alpha=0.3, color='red')

        plt.title('Opponent Prediction Accuracy (by Role)')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/opponent_accuracy_by_role_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 智能体状态价值变化
        plt.figure(figsize=(14, 8))

        for i in range(NUM_AGENTS):
            plt.plot(steps_range, suppress_value_means[i][:len(steps_range)],
                     label=f'{agent_labels_suppress[i]} State Value',
                     color=colors_suppress[i],
                     linewidth=1.5)

        for i in range(NUM_AGENTS):
            plt.plot(steps_range, rumor_value_means[i][:len(steps_range)],
                     label=f'{agent_labels_rumor[i]} State Value',
                     color=colors_rumor[i],
                     linewidth=1.5, linestyle='--')

        plt.title('Agent State Value Changes')
        plt.xlabel('Step')
        plt.ylabel('State Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/state_values_individual_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. 智能体净收益趋势
        plt.figure(figsize=(14, 8))

        for i in range(NUM_AGENTS):
            # 辟谣智能体净收益 = 奖励 * (1 - 感染率) * 对手准确率
            suppress_net_rewards = [r * (1 - e) * a for r, e, a in
                                    zip(suppress_reward_means[i][:len(steps_range)], 
                                        infection_means[:len(steps_range)], 
                                        suppress_opp_acc_means[:len(steps_range)])]
            plt.plot(steps_range, suppress_net_rewards,
                     label=f'{agent_labels_suppress[i]} Net Reward',
                     color=colors_suppress[i],
                     linewidth=2)

        for i in range(NUM_AGENTS):
            # 传谣智能体净收益 = 奖励 * 感染率 * 对手准确率
            rumor_net_rewards = [r * e * a for r, e, a in
                                 zip(rumor_reward_means[i][:len(steps_range)], 
                                     infection_means[:len(steps_range)], 
                                     rumor_opp_acc_means[:len(steps_range)])]
            plt.plot(steps_range, rumor_net_rewards,
                     label=f'{agent_labels_rumor[i]} Net Reward',
                     color=colors_rumor[i],
                     linewidth=2, linestyle='--')

        plt.title('Agent Net Reward Trends')
        plt.xlabel('Step')
        plt.ylabel('Net Reward')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/net_rewards_individual_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. 课程学习难度变化
        plt.figure(figsize=(12, 6))
        plt.plot(steps_range, difficulty_means[:len(steps_range)], label='Difficulty Coefficient', color='black', linewidth=2)
        plt.fill_between(steps_range, 
                         difficulty_means[:len(steps_range)] - difficulty_stds[:len(steps_range)], 
                         difficulty_means[:len(steps_range)] + difficulty_stds[:len(steps_range)], 
                         alpha=0.3, color='black')
        plt.title('Curriculum Learning Difficulty Changes')
        plt.xlabel('Step')
        plt.ylabel('Difficulty Coefficient')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/difficulty_curve_avg{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 7. C1和C2指标变化趋势 - 修复维度问题
        plt.figure(figsize=(15, 10))

        # C1指标 - Lyapunov指数(LE) - 修复：使用正确的维度
        plt.subplot(2, 3, 1)
        # 由于LE使用np.diff，数据点少一个，所以需要调整x轴
        le_steps = steps_range[1:]  # 跳过第一个点
        le_data_length = min(len(le_steps), len(c1_le_means))
        plt.plot(le_steps[:le_data_length], c1_le_means[:le_data_length], 
                 label='Lyapunov Exponent (LE)', color='blue', linewidth=2)
        if le_data_length > 0:
            plt.fill_between(le_steps[:le_data_length], 
                             c1_le_means[:le_data_length] - c1_le_stds[:le_data_length], 
                             c1_le_means[:le_data_length] + c1_le_stds[:le_data_length], 
                             alpha=0.3, color='blue')
        plt.title('Lyapunov Exponent (LE) Trend')
        plt.xlabel('Step')
        plt.ylabel('LE Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # C1指标 - 稳态误差(SSE)
        plt.subplot(2, 3, 2)
        plt.plot(steps_range, c1_sse_means[:len(steps_range)], label='Steady State Error (SSE)', color='red', linewidth=2)
        plt.fill_between(steps_range, 
                         c1_sse_means[:len(steps_range)] - c1_sse_stds[:len(steps_range)], 
                         c1_sse_means[:len(steps_range)] + c1_sse_stds[:len(steps_range)], 
                         alpha=0.3, color='red')
        plt.title('Steady State Error (SSE) Trend')
        plt.xlabel('Step')
        plt.ylabel('SSE Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # C2指标 - 感染峰值下降速度(FV)
        plt.subplot(2, 3, 3)
        plt.plot(steps_range, c2_fv_means[:len(steps_range)], label='Infection Peak Drop Speed (FV)', color='green', linewidth=2)
        plt.fill_between(steps_range, 
                         c2_fv_means[:len(steps_range)] - c2_fv_stds[:len(steps_range)], 
                         c2_fv_means[:len(steps_range)] + c2_fv_stds[:len(steps_range)], 
                         alpha=0.3, color='green')
        plt.title('Infection Peak Drop Speed (FV)')
        plt.xlabel('Step')
        plt.ylabel('FV Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # C2指标 - 单位成本感染下降率(IDR_UC)
        plt.subplot(2, 3, 4)
        plt.plot(steps_range, c2_idr_uc_means[:len(steps_range)], label='Infection Drop Rate per Unit Cost (IDR_UC)', 
                 color='purple', linewidth=2)
        plt.fill_between(steps_range, 
                         c2_idr_uc_means[:len(steps_range)] - c2_idr_uc_stds[:len(steps_range)], 
                         c2_idr_uc_means[:len(steps_range)] + c2_idr_uc_stds[:len(steps_range)], 
                         alpha=0.3, color='purple')
        plt.title('Infection Drop Rate per Unit Cost (IDR_UC)')
        plt.xlabel('Step')
        plt.ylabel('IDR_UC Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # C2指标 - 传播范围(PE)
        plt.subplot(2, 3, 5)
        plt.plot(steps_range, c2_pe_means[:len(steps_range)], label='Propagation Extent (PE)', color='orange', linewidth=2)
        plt.fill_between(steps_range, 
                         c2_pe_means[:len(steps_range)] - c2_pe_stds[:len(steps_range)], 
                         c2_pe_means[:len(steps_range)] + c2_pe_stds[:len(steps_range)], 
                         alpha=0.3, color='orange')
        plt.title('Propagation Extent (PE)')
        plt.xlabel('Step')
        plt.ylabel('PE Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # C2指标 - 策略动作熵(AE)
        plt.subplot(2, 3, 6)
        plt.plot(steps_range, c2_ae_means[:len(steps_range)], label='Action Entropy (AE)', color='brown', linewidth=2)
        plt.fill_between(steps_range, 
                         c2_ae_means[:len(steps_range)] - c2_ae_stds[:len(steps_range)], 
                         c2_ae_means[:len(steps_range)] + c2_ae_stds[:len(steps_range)], 
                         alpha=0.3, color='brown')
        plt.title('Action Entropy (AE)')
        plt.xlabel('Step')
        plt.ylabel('AE Value')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'results/c1_c2_metrics_trends{suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 分别创建前20步和完整步数的可视化
    print("生成前20步的可视化结果...")
    create_visualizations(steps_20, '_first20')
    
    print("生成完整步数的可视化结果...")
    create_visualizations(steps_full, '_full')

    # 8. 对手模型训练损失 - 直接按训练顺序展开
    plt.figure(figsize=(12, 6))

    # 将所有的对手模型损失数据展平为一维数组
    all_suppress_opp_losses = []
    all_rumor_opp_losses = []

    # 遍历所有episode的损失数据
    for episode_losses in episode_data['suppress_opp_losses']:
        all_suppress_opp_losses.extend(episode_losses)

    for episode_losses in episode_data['rumor_opp_losses']:
        all_rumor_opp_losses.extend(episode_losses)

    # 创建横轴 - 按训练顺序的step编号
    training_steps = list(range(len(all_suppress_opp_losses)))

    # 绘制损失曲线
    plt.plot(training_steps, all_suppress_opp_losses,
             label='Suppress Opponent Model Loss',
             color='blue',
             linewidth=1,
             alpha=0.7)

    plt.plot(training_steps, all_rumor_opp_losses,
             label='Rumor Opponent Model Loss',
             color='purple',
             linewidth=1,
             alpha=0.7)

    plt.title('Opponent Model Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/opponent_losses_avg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 9. 智能体奖励分布箱线图 - 按8个智能体分开
    plt.figure(figsize=(16, 8))

    # 准备数据：8个智能体的所有奖励数据
    all_rewards_data = []
    all_labels = []

    # 添加辟谣智能体奖励
    for i in range(NUM_AGENTS):
        # 将所有episode的数据展平
        all_agent_rewards = []
        for episode in range(EPISODES):
            all_agent_rewards.extend(episode_data['suppress_rewards'][i][episode - 1])
        all_rewards_data.append(all_agent_rewards)
        all_labels.append(f'Suppress{i}')

    # 添加传谣智能体奖励
    for i in range(NUM_AGENTS):
        all_agent_rewards = []
        for episode in range(EPISODES):
            all_agent_rewards.extend(episode_data['rumor_rewards'][i][episode - 1])
        all_rewards_data.append(all_agent_rewards)
        all_labels.append(f'Rumor{i}')

    # 创建箱线图
    box = plt.boxplot(all_rewards_data, widths=0.7, patch_artist=True, labels=all_labels)

    # 设置颜色：辟谣用蓝色系，传谣用红色系
    colors = colors_suppress + colors_rumor
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('Agent Reward Distribution Boxplot')
    plt.ylabel('Reward Value')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reward_distribution_individual_avg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. C1和C2指标综合对比图
    plt.figure(figsize=(14, 10))

    # 计算每个episode的平均C1和C2指标
    c1_le_episode = episode_data['c1_metrics']['LE']
    c1_ct_episode = episode_data['c1_metrics']['CT']
    c1_sse_episode = episode_data['c1_metrics']['SSE']

    c2_fv_episode = episode_data['c2_metrics']['FV']
    c2_idr_uc_episode = episode_data['c2_metrics']['IDR_UC']
    c2_pe_episode = episode_data['c2_metrics']['PE']
    c2_ae_episode = episode_data['c2_metrics']['AE']

    # 确保所有列表长度一致
    episodes = list(range(len(c1_le_episode)))

    # C1指标按episode变化
    plt.subplot(2, 2, 1)
    plt.plot(episodes, c1_le_episode, label='LE', marker='o', color='blue')
    plt.plot(episodes, c1_ct_episode, label='CT', marker='s', color='red')
    plt.plot(episodes, c1_sse_episode, label='SSE', marker='^', color='green')
    plt.title('C1 Metrics by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C2指标按episode变化
    plt.subplot(2, 2, 2)
    plt.plot(episodes, c2_fv_episode, label='FV', marker='o', color='purple')
    plt.plot(episodes, c2_idr_uc_episode, label='IDR_UC', marker='s', color='orange')
    plt.plot(episodes, c2_pe_episode, label='PE', marker='^', color='brown')
    plt.plot(episodes, c2_ae_episode, label='AE', marker='d', color='pink')
    plt.title('C2 Metrics by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C1指标最终值比较
    plt.subplot(2, 2, 3)
    c1_final_values = [c1_le_episode[-1], c1_ct_episode[-1], c1_sse_episode[-1]]
    c1_labels = ['LE', 'CT', 'SSE']
    plt.bar(c1_labels, c1_final_values, color=['blue', 'red', 'green'])
    plt.title('Final C1 Metrics Values')
    plt.ylabel('Metric Value')

    # C2指标最终值比较
    plt.subplot(2, 2, 4)
    c2_final_values = [c2_fv_episode[-1], c2_idr_uc_episode[-1], c2_pe_episode[-1], c2_ae_episode[-1]]
    c2_labels = ['FV', 'IDR_UC', 'PE', 'AE']
    plt.bar(c2_labels, c2_final_values, color=['purple', 'orange', 'brown', 'pink'])
    plt.title('Final C2 Metrics Values')
    plt.ylabel('Metric Value')

    plt.tight_layout()
    plt.savefig('results/c1_c2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 11. 动作分布可视化
    plt.figure(figsize=(16, 12))

    # 获取最后一个episode的分布数据
    # 确保索引不超出范围
    last_episode = min(len(episode_data['action_distributions']['suppress_pred']) - 1, 
                       len(episode_data['action_distributions']['suppress_true']) - 1,
                       len(episode_data['action_distributions']['rumor_pred']) - 1,
                       len(episode_data['action_distributions']['rumor_true']) - 1)

    if last_episode >= 0:
        suppress_pred_dist = episode_data['action_distributions']['suppress_pred'][last_episode]
        suppress_true_dist = episode_data['action_distributions']['suppress_true'][last_episode]
        rumor_pred_dist = episode_data['action_distributions']['rumor_pred'][last_episode]
        rumor_true_dist = episode_data['action_distributions']['rumor_true'][last_episode]
    else:
        # 如果没有数据，使用零数组
        suppress_pred_dist = np.zeros(5)
        suppress_true_dist = np.zeros(5)
        rumor_pred_dist = np.zeros(5)
        rumor_true_dist = np.zeros(5)

    # 动作标签
    action_labels = ['Action 0', 'Action 1', 'Action 2', 'Action 3', 'Action 4']
    x_pos = np.arange(len(action_labels))

    # 辟谣者动作分布
    plt.subplot(2, 2, 1)
    width = 0.35
    plt.bar(x_pos - width/2, suppress_pred_dist, width, label='Predicted', alpha=0.7, color='blue')
    plt.bar(x_pos + width/2, suppress_true_dist, width, label='Actual', alpha=0.7, color='red')
    plt.title('Suppress Agent Action Distribution\n(Predicted vs Actual)')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.xticks(x_pos, action_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 传谣者动作分布
    plt.subplot(2, 2, 2)
    plt.bar(x_pos - width/2, rumor_pred_dist, width, label='Predicted', alpha=0.7, color='blue')
    plt.bar(x_pos + width/2, rumor_true_dist, width, label='Actual', alpha=0.7, color='red')
    plt.title('Rumor Agent Action Distribution\n(Predicted vs Actual)')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.xticks(x_pos, action_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 分布相似度随时间变化
    plt.subplot(2, 2, 3)
    # 使用动作分布数据的长度作为范围
    episodes_range = range(min(len(episode_data['action_distributions']['suppress_pred']),
                              len(episode_data['action_distributions']['suppress_true']),
                              len(episode_data['action_distributions']['rumor_pred']),
                              len(episode_data['action_distributions']['rumor_true'])))

    suppress_similarities = []
    rumor_similarities = []

    for i in episodes_range:
        suppress_pred = episode_data['action_distributions']['suppress_pred'][i]
        suppress_true = episode_data['action_distributions']['suppress_true'][i]
        rumor_pred = episode_data['action_distributions']['rumor_pred'][i]
        rumor_true = episode_data['action_distributions']['rumor_true'][i]

        suppress_sim = 1.0 - jensenshannon(suppress_pred, suppress_true)
        rumor_sim = 1.0 - jensenshannon(rumor_pred, rumor_true)

        suppress_similarities.append(suppress_sim)
        rumor_similarities.append(rumor_sim)

    plt.plot(episodes_range, suppress_similarities, label='Suppress Similarity', marker='o', color='blue')
    plt.plot(episodes_range, rumor_similarities, label='Rumor Similarity', marker='s', color='red')
    plt.title('Action Distribution Similarity Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Similarity (1 - JS Distance)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 动作选择趋势
    plt.subplot(2, 2, 4)
    # 计算每个动作的选择频率随时间的变化
    if len(episode_data['action_distributions']['suppress_true']) > 0:
        action_trends_suppress = np.array(episode_data['action_distributions']['suppress_true'])
        action_trends_rumor = np.array(episode_data['action_distributions']['rumor_true'])

        for action in range(5):
            if len(episodes_range) == len(action_trends_suppress):
                plt.plot(episodes_range, action_trends_suppress[:, action], 
                         label=f'Suppress Action {action}', linestyle='-', alpha=0.7)
                plt.plot(episodes_range, action_trends_rumor[:, action], 
                         label=f'Rumor Action {action}', linestyle='--', alpha=0.7)

        plt.title('Action Selection Trends Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Selection Frequency')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No action distribution data available', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Action Selection Trends Over Episodes')

    plt.tight_layout()
    plt.savefig('results/action_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 12. 详细分布对比（最后一个episode）
    plt.figure(figsize=(15, 10))

    # 辟谣者详细分布
    plt.subplot(2, 2, 1)
    plt.bar(x_pos, suppress_pred_dist, alpha=0.7, color='lightblue', label='Predicted')
    plt.bar(x_pos, suppress_true_dist, alpha=0.7, color='lightcoral', label='Actual')
    for i, (pred, true) in enumerate(zip(suppress_pred_dist, suppress_true_dist)):
        plt.text(i, max(pred, true) + 0.01, f'{abs(pred-true):.3f}', 
                 ha='center', va='bottom', fontsize=8)
    plt.title('Suppress Agent: Detailed Action Distribution')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.xticks(x_pos, action_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 传谣者详细分布
    plt.subplot(2, 2, 2)
    plt.bar(x_pos, rumor_pred_dist, alpha=0.7, color='lightblue', label='Predicted')
    plt.bar(x_pos, rumor_true_dist, alpha=0.7, color='lightcoral', label='Actual')
    for i, (pred, true) in enumerate(zip(rumor_pred_dist, rumor_true_dist)):
        plt.text(i, max(pred, true) + 0.01, f'{abs(pred-true):.3f}', 
                 ha='center', va='bottom', fontsize=8)
    plt.title('Rumor Agent: Detailed Action Distribution')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.xticks(x_pos, action_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 分布差异热力图
    plt.subplot(2, 2, 3)
    suppress_diff = np.abs(suppress_pred_dist - suppress_true_dist).reshape(1, -1)
    rumor_diff = np.abs(rumor_pred_dist - rumor_true_dist).reshape(1, -1)

    plt.imshow(suppress_diff, cmap='Reds', aspect='auto')
    plt.colorbar(label='Absolute Difference')
    plt.title('Suppress Agent: Distribution Differences')
    plt.xticks(x_pos, action_labels)
    plt.yticks([0], ['Difference'])

    plt.subplot(2, 2, 4)
    plt.imshow(rumor_diff, cmap='Reds', aspect='auto')
    plt.colorbar(label='Absolute Difference')
    plt.title('Rumor Agent: Distribution Differences')
    plt.xticks(x_pos, action_labels)
    plt.yticks([0], ['Difference'])

    plt.tight_layout()
    plt.savefig('results/detailed_action_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最终的C1和C2指标统计
    print("\n" + "=" * 60)
    print("最终C1和C2指标统计")
    print("=" * 60)
    print(f"C1指标:")
    print(f"  Lyapunov指数(LE): {np.mean(episode_data['c1_metrics']['LE']):.6f} ± {np.std(episode_data['c1_metrics']['LE']):.6f}")
    print(f"  收敛时间(CT): {np.mean(episode_data['c1_metrics']['CT']):.2f} ± {np.std(episode_data['c1_metrics']['CT']):.2f}")
    print(f"  稳态误差(SSE): {np.mean(episode_data['c1_metrics']['SSE']):.6f} ± {np.std(episode_data['c1_metrics']['SSE']):.6f}")
    
    print(f"\nC2指标:")
    print(f"  感染峰值下降速度(FV): {np.mean(episode_data['c2_metrics']['FV']):.6f} ± {np.std(episode_data['c2_metrics']['FV']):.6f}")
    print(f"  单位成本感染下降率(IDR_UC): {np.mean(episode_data['c2_metrics']['IDR_UC']):.6f} ± {np.std(episode_data['c2_metrics']['IDR_UC']):.6f}")
    print(f"  传播范围(PE): {np.mean(episode_data['c2_metrics']['PE']):.6f} ± {np.std(episode_data['c2_metrics']['PE']):.6f}")
    print(f"  策略动作熵(AE): {np.mean(episode_data['c2_metrics']['AE']):.6f} ± {np.std(episode_data['c2_metrics']['AE']):.6f}")

    # 打印分布相似度统计
    print("\n动作分布相似度统计:")
    print(f"辟谣者平均分布相似度: {np.mean(suppress_similarities):.4f} ± {np.std(suppress_similarities):.4f}")
    print(f"传谣者平均分布相似度: {np.mean(rumor_similarities):.4f} ± {np.std(rumor_similarities):.4f}")
    print(f"总体平均分布相似度: {(np.mean(suppress_similarities) + np.mean(rumor_similarities)) / 2:.4f}")

    # 打印最常见的动作
    suppress_most_common = np.argmax(suppress_true_dist)
    rumor_most_common = np.argmax(rumor_true_dist)
    print(f"\n最常见的动作:")
    print(f"辟谣者: Action {suppress_most_common} (频率: {suppress_true_dist[suppress_most_common]:.3f})")
    print(f"传谣者: Action {rumor_most_common} (频率: {rumor_true_dist[rumor_most_common]:.3f})")

    return best_episode, episode_data


# 启动训练
if __name__ == "__main__":
    best_episode, training_data = train_simulation()
    print(f"\n训练完成！最佳模型在Episode {best_episode}")
