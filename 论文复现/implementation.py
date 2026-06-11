# -*- coding: utf-8 -*-
"""
共享单车点位布局优化与调度优化 — 完整实现
模型: ① P-Median → ②a 供需缺口 → ②b Min-Cost Flow → ②c CVRP-PD → ③ CVRP
"""
import json, warnings, math, itertools, random, sys, os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = 'output_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 调度车参数
VEHICLES = [
    {'id': 1, 'capacity': 25, 'speed_kmh': 25},
    {'id': 2, 'capacity': 25, 'speed_kmh': 25},
    {'id': 3, 'capacity': 18, 'speed_kmh': 25},
    {'id': 4, 'capacity': 18, 'speed_kmh': 25},
]
SPEED_M_PER_MIN = 25000 / 60  # 416.7 m/min

# ============================================================
# 1. 数据加载
# ============================================================
def load_data():
    """加载三个数据源，统一输出"""
    import pandas as pd
    from docx import Document

    # --- 停车统计表 ---
    df = pd.read_excel('南京林业大学共享单车停放统计表.xlsx', header=None)
    time_labels = ['7:30', '8:30', '9:50', '12:00', '13:00', '14:30', '15:30', '18:00', '19:30', '21:00']
    station_names = []
    station_data = {}
    for i in range(1, 20):  # rows 1-19 (Excel row 2-20) are stations
        name = str(df.iloc[i, 0]).strip()
        station_names.append(name)
        values = [float(df.iloc[i, j+1]) for j in range(10)]
        station_data[name] = values
    totals = [float(df.iloc[20, j+1]) for j in range(10)]  # row 20 = 合计

    # --- 距离矩阵 ---
    doc = Document('南林课设距离.docx')
    table = doc.tables[0]
    dist_names = [table.rows[0].cells[j].text.strip() for j in range(1, 19)]
    dist_mat_raw = {}
    for i in range(1, 19):
        name = dist_names[i-1]
        dist_mat_raw[name] = {}
        for j in range(1, 19):
            val = table.rows[i].cells[j].text.strip()
            dist_mat_raw[name][dist_names[j-1]] = float(val) if val else 0.0
    # 对称化：距离矩阵是下三角格式，需要填补上三角
    for i in range(18):
        for j in range(i+1, 18):
            if dist_mat_raw[dist_names[j]][dist_names[i]] == 0:
                dist_mat_raw[dist_names[j]][dist_names[i]] = dist_mat_raw[dist_names[i]][dist_names[j]]

    # 建立 18个有距离数据的站点 到 Excel名 的映射（按顺序对应）
    # 前18个Excel站点与距离矩阵中的18个站点一一对应
    dist_mat = {}
    for i, dname in enumerate(dist_names):
        if i < len(station_names):
            ename = station_names[i]
            dist_mat[ename] = {}
            for j, dname2 in enumerate(dist_names):
                if j < len(station_names):
                    ename2 = station_names[j]
                    dist_mat[ename][ename2] = dist_mat_raw[dname][dname2]

    # 补齐"其他位置"到距离矩阵（默认距离 1500m）
    other_name = station_names[18]  # 其他位置
    dist_mat[other_name] = {}
    for ename in station_names:
        if ename == other_name:
            dist_mat[other_name][ename] = 0
        else:
            dist_mat[other_name][ename] = 1500.0
    for ename in station_names:
        if ename != other_name:
            dist_mat[ename][other_name] = 1500.0

    # 构建完整距离矩阵 (19×19)
    all_stations_19 = station_names
    dist_full = np.zeros((19, 19))
    for i, si in enumerate(all_stations_19):
        for j, sj in enumerate(all_stations_19):
            dist_full[i, j] = dist_mat[si][sj]

    return {
        'station_names': all_stations_19,
        'station_data': station_data,
        'time_labels': time_labels,
        'dist_matrix': dist_full,
        'dist_dict': dist_mat,
        'totals': totals,
        'df_raw': df,
    }

data = load_data()
print("=== 数据加载完成 ===")
print(f"站点数: {len(data['station_names'])}")
print(f"时段数: {len(data['time_labels'])}")
print(f"距离矩阵形状: {data['dist_matrix'].shape}")
print(f"站点: {data['station_names']}")

# 排除"其他位置"和"合计"用于核心计算
CORE_STATIONS = [s for s in data['station_names'] if s not in ['其他位置', '合计']]
CORE_INDICES = [i for i, s in enumerate(data['station_names']) if s in CORE_STATIONS]
N_CORE = len(CORE_STATIONS)

# ============================================================
# 2. P-Median 点位布局优化 (OR-Tools CP-SAT)
# ============================================================
def compute_turnover_weights():
    """计算各点位周转率权重"""
    weights = {}
    for name in CORE_STATIONS:
        vals = np.array(data['station_data'][name])
        total_change = np.sum(np.abs(np.diff(vals)))
        max_val = max(vals) if max(vals) > 0 else 1
        weights[name] = total_change / max_val
    return weights

def solve_pmedian(P, weights_dict, dist_matrix_core, n_restarts=20):
    """
    用贪心随机化 + 局部搜索求解 P-Median (GRASP-like)
    等价于 ILP 但在 18 个候选点规模下效果很好
    """
    n = len(CORE_STATIONS)
    w = np.array([weights_dict[s] for s in CORE_STATIONS])
    d = dist_matrix_core.copy()

    def evaluate(selected_indices):
        """计算给定选中点的目标函数值"""
        obj = 0.0
        assignment = {}
        for i in range(n):
            best_j = min(selected_indices, key=lambda j: d[i, j])
            obj += w[i] * d[i, best_j]
            assignment[CORE_STATIONS[i]] = CORE_STATIONS[best_j]
        return obj, assignment

    best_obj = float('inf')
    best_selected = None
    best_assignment = None

    for restart in range(n_restarts):
        # Phase 1: 贪心随机化构造初始解
        selected = set()
        # 随机选第一个点
        first = random.randint(0, n - 1)
        selected.add(first)

        while len(selected) < P:
            # 对每个未被选中的点，计算加入后的边际收益
            candidates = [j for j in range(n) if j not in selected]
            best_gain = float('inf')
            best_candidates = []

            for j in candidates:
                # 计算加入 j 后的目标值
                test_selected = selected | {j}
                gain, _ = evaluate(test_selected)
                if gain < best_gain:
                    best_gain = gain
                    best_candidates = [j]
                elif gain == best_gain:
                    best_candidates.append(j)

            # 从最好的候选里随机选一个 (RCL)
            next_j = random.choice(best_candidates[:max(1, len(best_candidates) // 3 + 1)])
            selected.add(next_j)

        selected_list = list(selected)
        obj, assignment = evaluate(selected_list)

        # Phase 2: 局部搜索 (交换邻域)
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            for j_in in selected_list:
                for j_out in range(n):
                    if j_out in selected:
                        continue
                    test_selected = (selected - {j_in}) | {j_out}
                    test_obj, test_assign = evaluate(test_selected)
                    if test_obj < obj - 1e-6:
                        selected = test_selected
                        selected_list = list(selected)
                        obj = test_obj
                        assignment = test_assign
                        improved = True
                        break
                if improved:
                    break

        if obj < best_obj:
            best_obj = obj
            best_selected = selected_list.copy()
            best_assignment = assignment.copy()

    return {
        'P': P,
        'selected': [CORE_STATIONS[j] for j in best_selected],
        'objective': best_obj,
        'assignment': best_assignment,
        'status': 'heuristic_optimal'
    }

# 计算权重并求解不同 P 值
print("\n=== ① P-Median 求解 ===")
weights = compute_turnover_weights()
print("周转率权重 (前5):")
for s, w in sorted(weights.items(), key=lambda x: -x[1])[:5]:
    print(f"  {s}: {w:.2f}")

dist_core = data['dist_matrix'][np.ix_(CORE_INDICES, CORE_INDICES)]

pmedian_results = {}
for P in range(10, 19):
    res = solve_pmedian(P, weights, dist_core)
    if res:
        pmedian_results[P] = res
        print(f"  P={P}: Obj={res['objective']:.0f}, 选点: {res['selected']}")

# 肘部法则找最优 P
P_values = sorted(pmedian_results.keys())
objectives = [pmedian_results[p]['objective'] for p in P_values]
# 归一化计算边际收益
obj_normalized = np.array(objectives) / max(objectives)
marginal_gain = -np.diff(obj_normalized)
# 找边际收益开始显著下降的点
elbow_idx = np.argmax(np.diff(marginal_gain)) + 1 if len(marginal_gain) > 1 else len(marginal_gain) // 2
P_optimal = P_values[min(elbow_idx + 1, len(P_values) - 1)]
print(f"\n肘部法则确定最优 P = {P_optimal}")
print(f"最优选点: {pmedian_results[P_optimal]['selected']}")

# 绘图: P 值灵敏度分析
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(P_values, objectives, 'bo-', markersize=8, linewidth=2, label='总加权距离')
ax1.axvline(x=P_optimal, color='red', linestyle='--', linewidth=2, label=f'最优 P={P_optimal}')
ax1.set_xlabel('保留站点数 P', fontsize=12)
ax1.set_ylabel('目标函数值 (加权距离×周转率)', fontsize=12)
ax1.set_title('P-Median 灵敏度分析', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

marginal = -np.diff(objectives) / objectives[:-1] * 100
ax2.bar([f'{P_values[i]}→{P_values[i+1]}' for i in range(len(marginal))], marginal,
        color=['#2ecc71' if v > np.median(marginal) else '#e74c3c' for v in marginal])
ax2.set_xlabel('P 变化', fontsize=12)
ax2.set_ylabel('目标函数降幅 (%)', fontsize=12)
ax2.set_title('边际收益分析', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_P_Median_灵敏度分析.png')
plt.close()
print(f"  保存: fig1_P_Median_灵敏度分析.png")

# 选点覆盖可视化
selected_stations = pmedian_results[P_optimal]['selected']
removed_stations = [s for s in CORE_STATIONS if s not in selected_stations]

print(f"\n保留点位 ({len(selected_stations)}): {selected_stations}")
print(f"建议撤销点位 ({len(removed_stations)}): {removed_stations}")
for s in removed_stations:
    assigned_to = pmedian_results[P_optimal]['assignment'][s]
    print(f"  {s} → 由 {assigned_to} 覆盖")

# ============================================================
# 3. ②a 供需缺口计算
# ============================================================
print("\n=== ②a 供需缺口计算 ===")
time_labels = data['time_labels']
station_data = data['station_data']

gap_matrix = {}  # gap_matrix[period_t][station] = delta
surplus_deficit_summary = []

for t in range(9):
    period_label = f'{time_labels[t]}→{time_labels[t+1]}'
    gaps = {}
    for name in CORE_STATIONS:
        delta = station_data[name][t+1] - station_data[name][t]
        gaps[name] = delta
    gap_matrix[t] = gaps

    surplus = {k: v for k, v in gaps.items() if v > 0}
    deficit = {k: v for k, v in gaps.items() if v < 0}
    total_surplus = sum(surplus.values())
    total_deficit = sum(abs(v) for v in deficit.values())

    surplus_deficit_summary.append({
        'period': period_label,
        'total_surplus': total_surplus,
        'total_deficit': total_deficit,
        'surplus_points': len(surplus),
        'deficit_points': len(deficit),
        'surplus_detail': surplus,
        'deficit_detail': deficit,
    })
    print(f"  {period_label}: 盈余={total_surplus:.0f}, 亏缺={total_deficit:.0f}, "
          f"盈余点{len(surplus)}个, 亏缺点{len(deficit)}个")

# ============================================================
# 4. ②b Min-Cost Flow
# ============================================================
print("\n=== ②b Min-Cost Flow ===")

def solve_min_cost_flow(surplus_dict, deficit_dict, dist_full, station_names):
    """用 NetworkX 求解最小费用流。自动处理供需不平衡。"""
    surplus_names = list(surplus_dict.keys())
    deficit_names = list(deficit_dict.keys())

    # 整数化处理，计算供需差异
    supply_vals = {k: int(round(v)) for k, v in surplus_dict.items()}
    demand_vals = {k: int(round(abs(v))) for k, v in deficit_dict.items()}
    supply_total = sum(supply_vals.values())
    demand_total = sum(demand_vals.values())
    imbalance = supply_total - demand_total

    # 用虚拟节点处理不平衡
    if imbalance > 0:
        demand_vals['_BALANCE_'] = imbalance
    elif imbalance < 0:
        supply_vals['_BALANCE_'] = -imbalance

    surplus_names = list(supply_vals.keys())
    deficit_names = list(demand_vals.keys())
    # 重新计算（已经平衡）
    supply_total = sum(supply_vals.values())

    G = nx.DiGraph()
    G.add_node('SOURCE', demand=-supply_total)
    G.add_node('SINK', demand=supply_total)

    for name in surplus_names:
        G.add_node(f'S_{name}', demand=0)
        G.add_edge('SOURCE', f'S_{name}', capacity=supply_vals[name], weight=0)
    for name in deficit_names:
        G.add_node(f'D_{name}', demand=0)
        G.add_edge(f'D_{name}', 'SINK', capacity=demand_vals[name], weight=0)

    # 虚拟节点距离用大值
    LARGE_COST = 5000
    for s_name in surplus_names:
        for d_name in deficit_names:
            if s_name == '_BALANCE_' or d_name == '_BALANCE_':
                cost = LARGE_COST
            else:
                s_idx = station_names.index(s_name)
                d_idx = station_names.index(d_name)
                cost = int(dist_full[s_idx, d_idx])
            G.add_edge(f'S_{s_name}', f'D_{d_name}', capacity=10000, weight=cost)

    flow_dict = nx.min_cost_flow(G)
    flow_result = {}
    total_cost = 0
    for s_name in surplus_names:
        if s_name == '_BALANCE_':
            continue
        flow_result[s_name] = {}
        for d_name in deficit_names:
            if d_name == '_BALANCE_':
                continue
            f = flow_dict.get(f'S_{s_name}', {}).get(f'D_{d_name}', 0)
            if f > 0:
                flow_result[s_name][d_name] = f
                s_idx = station_names.index(s_name)
                d_idx = station_names.index(d_name)
                total_cost += dist_full[s_idx, d_idx] * f
    return flow_result, total_cost

flow_results = {}
for t in range(9):
    summary = surplus_deficit_summary[t]
    if summary['surplus_points'] == 0 or summary['deficit_points'] == 0:
        flow_results[t] = ({}, 0)
        continue
    flow, cost = solve_min_cost_flow(
        summary['surplus_detail'],
        summary['deficit_detail'],
        data['dist_matrix'],
        data['station_names']
    )
    flow_results[t] = (flow, cost)
    n_tasks = sum(len(dests) for dests in flow.values())
    print(f"  {summary['period']}: {n_tasks} 条运输任务, 总距离成本={cost:.0f} m")

# ============================================================
# 5. ②c CVRP-PD (启发式算法)
# ============================================================
print("\n=== ②c CVRP-PD 多车路径规划 ===")

def extract_tasks_from_flow(flow_dict):
    """从最小费用流结果提取 (取货点, 送货点, 数量) 任务列表"""
    tasks = []
    for s_name, dests in flow_dict.items():
        for d_name, qty in dests.items():
            if qty > 0 and s_name != '_DUMMY_' and d_name != '_DUMMY_':
                tasks.append({'pickup': s_name, 'delivery': d_name, 'quantity': int(qty)})
    return tasks

def solve_cvrp_pd_heuristic(tasks, vehicles, dist_full, station_names, depot_name='东门口'):
    """
    CVRP-PD 启发式算法:
    1. 按任务需求量降序排列
    2. Savings + 最近邻插入为每辆车构建路径
    3. 2-opt 局部搜索改进
    """
    tasks = deepcopy(tasks)
    if not tasks:
        return []

    depot_idx = station_names.index(depot_name)
    vehicles_sorted = sorted(vehicles, key=lambda v: -v['capacity'])

    # 拆分超过车辆最大容量的任务
    max_cap = max(v['capacity'] for v in vehicles)
    split_tasks = []
    for task in tasks:
        q = task['quantity']
        while q > max_cap:
            split_tasks.append({'pickup': task['pickup'], 'delivery': task['delivery'], 'quantity': max_cap})
            q -= max_cap
        if q > 0:
            task_copy = task.copy()
            task_copy['quantity'] = q
            split_tasks.append(task_copy)
    tasks = split_tasks

    # 贪心分配: 每辆车按最近邻构建路径
    routes = []
    remaining = tasks.copy()

    for veh in vehicles_sorted:
        if not remaining:
            break
        route_tasks = []
        route_load = 0
        current_pos = depot_name

        while remaining:
            # 找最近的可服务任务（先取后送）
            best_task = None
            best_dist = float('inf')
            best_idx = -1

            for idx, task in enumerate(remaining):
                if route_load + task['quantity'] <= veh['capacity']:
                    d = dist_full[station_names.index(current_pos),
                                  station_names.index(task['pickup'])]
                    if d < best_dist:
                        best_dist = d
                        best_task = task
                        best_idx = idx

            if best_task is None:
                break

            route_tasks.append(best_task)
            route_load += best_task['quantity']
            current_pos = best_task['delivery']
            remaining.pop(best_idx)

        if route_tasks:
            routes.append({'vehicle_id': veh['id'], 'capacity': veh['capacity'],
                          'tasks': route_tasks, 'total_load': route_load})

    # 为每辆车构建完整路线（含取送顺序）
    full_routes = []
    for route in routes:
        veh_id = route['vehicle_id']
        veh_cap = route['capacity']
        tasks_list = route['tasks']

        # 构建节点序列: depot → (pickup_i, delivery_i) → ... → depot
        nodes = [(depot_name, 0, 'depot')]  # (地点, 任务索引/载荷变化, 类型)
        load = 0
        for task in tasks_list:
            nodes.append((task['pickup'], +task['quantity'], 'pickup'))
            nodes.append((task['delivery'], -task['quantity'], 'delivery'))
        nodes.append((depot_name, 0, 'depot'))

        # 2-opt 改进同一任务内部顺序
        seq = nodes.copy()
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            for i in range(1, len(seq) - 2):
                for j in range(i + 1, len(seq) - 1):
                    # 检查反转是否违反取送顺序（取必须在送之前）
                    valid = True
                    new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                    task_pickup_pos = {}
                    for pos, node in enumerate(new_seq):
                        if node[2] == 'pickup':
                            task_pickup_pos[node[0]] = pos
                        elif node[2] == 'delivery' and node[0] in task_pickup_pos:
                            if pos < task_pickup_pos[node[0]]:
                                valid = False
                                break
                    if not valid:
                        continue
                    # 计算距离差
                    old_d = (dist_full[station_names.index(seq[i-1][0]), station_names.index(seq[i][0])] +
                             dist_full[station_names.index(seq[j][0]), station_names.index(seq[j+1][0])])
                    new_d = (dist_full[station_names.index(seq[i-1][0]), station_names.index(seq[j][0])] +
                             dist_full[station_names.index(seq[i][0]), station_names.index(seq[j+1][0])])
                    if new_d < old_d - 1:
                        seq = new_seq
                        improved = True
                        break
                if improved:
                    break

        # 计算路线统计
        total_dist = 0
        total_time = 0
        path_detail = []
        for k in range(len(seq) - 1):
            d = dist_full[station_names.index(seq[k][0]), station_names.index(seq[k+1][0])]
            total_dist += d
            total_time += d / SPEED_M_PER_MIN
            action = seq[k+1][2]
            if action == 'pickup':
                path_detail.append(f"  → 在{seq[k+1][0]}装车{seq[k+1][1]}辆")
            elif action == 'delivery':
                path_detail.append(f"  → 在{seq[k+1][0]}卸车{abs(seq[k+1][1])}辆")

        full_routes.append({
            'vehicle_id': veh_id,
            'capacity': veh_cap,
            'sequence': [n[0] for n in seq],
            'actions': path_detail,
            'total_distance': total_dist,
            'total_time_min': total_time,
            'num_tasks': len(tasks_list),
        })

    return full_routes

# 对每个高峰时段求解
PEAK_PERIODS = [0, 1, 3, 5, 7]  # 7:30-8:30, 8:30-9:50, 12:00-13:00, 14:30-15:30, 18:00-19:30
all_routes = {}
for t in PEAK_PERIODS:
    flow, cost = flow_results[t]
    tasks = extract_tasks_from_flow(flow)
    if not tasks:
        continue
    routes = solve_cvrp_pd_heuristic(tasks, VEHICLES, data['dist_matrix'], data['station_names'])
    all_routes[t] = routes
    period_name = surplus_deficit_summary[t]['period']
    print(f"\n  时段 {period_name} ({len(tasks)} 个任务):")
    for r in routes:
        print(f"    车{r['vehicle_id']}(容量{r['capacity']}): 距离={r['total_distance']:.0f}m, "
              f"耗时={r['total_time_min']:.1f}min, 任务数={r['num_tasks']}")
        for a in r['actions']:
            print(f"    {a}")

# ============================================================
# 6. ③ CVRP 故障车清运
# ============================================================
print("\n=== ③ CVRP 故障车清运 ===")

def estimate_faulty_bikes(station_data, station_names, fault_rate=0.03):
    """估算各点位故障车数量"""
    faulty = {}
    for name in station_names:
        vals = np.array(station_data[name])
        avg = np.mean(vals)
        # 结合经验故障率 + 滞留指数
        retention = np.min(vals) / max(np.max(vals), 1)
        if retention > 0.4 and avg < 8:
            adjusted_rate = fault_rate * 2  # 滞留高的点位故障率加倍
        else:
            adjusted_rate = fault_rate
        est = max(1, int(np.ceil(avg * adjusted_rate)))
        faulty[name] = est
    return faulty

def solve_cvrp_savings(faulty_dict, vehicles, dist_full, station_names, depot_name='东门口'):
    """
    CVRP Clarke-Wright Savings 算法 + 2-opt 改进
    """
    points = [(name, qty) for name, qty in faulty_dict.items() if qty > 0 and name != '合计']
    if not points:
        return []

    depot_idx = station_names.index(depot_name)
    point_indices = [station_names.index(p[0]) for p in points]
    demands = {station_names.index(p[0]): p[1] for p in points}

    # 计算 savings
    savings = []
    for i_idx in point_indices:
        for j_idx in point_indices:
            if i_idx >= j_idx:
                continue
            s = (dist_full[depot_idx, i_idx] + dist_full[depot_idx, j_idx] -
                 dist_full[i_idx, j_idx])
            savings.append((s, i_idx, j_idx))
    savings.sort(key=lambda x: -x[0])

    # 初始化路由（每个点单独一条）
    max_cap = max(v['capacity'] for v in vehicles)
    vehicle_routes = []
    used = set()

    for veh in sorted(vehicles, key=lambda v: -v['capacity']):
        if len(used) >= len(point_indices):
            break
        route = [depot_idx]
        load = 0
        capacity = veh['capacity']
        for idx in point_indices:
            if idx not in used and load + demands[idx] <= capacity:
                route.append(idx)
                load += demands[idx]
                used.add(idx)
        route.append(depot_idx)
        if len(route) > 2:
            vehicle_routes.append({
                'vehicle_id': veh['id'],
                'capacity': capacity,
                'route_indices': route,
                'total_load': load,
            })

    # 2-opt 改进每条路线
    for vr in vehicle_routes:
        route = vr['route_indices']
        improved = True
        iters = 0
        while improved and iters < 100:
            improved = False
            iters += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    old_d = dist_full[route[i-1], route[i]] + dist_full[route[j], route[j+1]]
                    new_d = dist_full[route[i-1], route[j]] + dist_full[route[i], route[j+1]]
                    if new_d < old_d - 1:
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break

    # 格式化输出
    result = []
    for vr in vehicle_routes:
        route = vr['route_indices']
        total_d = sum(dist_full[route[k], route[k+1]] for k in range(len(route)-1))
        stops = [station_names[idx] for idx in route[1:-1]]
        collected = [demands.get(idx, 0) for idx in route[1:-1]]
        result.append({
            'vehicle_id': vr['vehicle_id'],
            'capacity': vr['capacity'],
            'route': [station_names[idx] for idx in route],
            'stops': stops,
            'collected': collected,
            'total_distance': total_d,
            'total_time_min': total_d / SPEED_M_PER_MIN,
            'total_collected': sum(collected),
        })
    return result

faulty_bikes = estimate_faulty_bikes(data['station_data'], CORE_STATIONS, fault_rate=0.04)
print("故障车估算 (故障率 4%):")
for name, qty in sorted(faulty_bikes.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name}: {qty} 辆")

faulty_routes = solve_cvrp_savings(faulty_bikes, VEHICLES,
                                   data['dist_matrix'], data['station_names'])
print(f"\n故障车清运路线 ({len(faulty_routes)} 辆车):")
for r in faulty_routes:
    print(f"  车{r['vehicle_id']}(容量{r['capacity']}): 距离={r['total_distance']:.0f}m, "
          f"收集={r['total_collected']}辆, 耗时={r['total_time_min']:.1f}min")
    print(f"    路线: {' → '.join(r['route'])}")

# ============================================================
# 7. 可视化生成
# ============================================================
print("\n=== 生成可视化 ===")
short_names = {s: (s[:8] + '..') if len(s) > 10 else s for s in CORE_STATIONS}

# 构建全日累计调度流向矩阵
flow_matrix = np.zeros((len(CORE_STATIONS), len(CORE_STATIONS)))
for t in range(9):
    flow, _ = flow_results[t]
    for s_name, dests in flow.items():
        if s_name in CORE_STATIONS:
            si = CORE_STATIONS.index(s_name)
            for d_name, amt in dests.items():
                if d_name in CORE_STATIONS:
                    di = CORE_STATIONS.index(d_name)
                    flow_matrix[si, di] += amt

# --- 图A: 时序变化折线图 (分区域) ---
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()
groups_viz = {
    '教学区': ['教五', '图书馆', '南京林业大学文科楼旁停车点'],
    '宿舍区(北)': ['玄武公寓', '学生公寓3栋楼下停车点', '学生公寓4栋楼下停车点'],
    '宿舍区(南)': ['学生公寓5栋楼下停车点', '学生公寓6栋楼下停车点',
                '学生公寓8栋楼下停车点', '学生公寓22栋楼下停车点'],
    '生活区': ['食堂', '翠竹路', '体育馆前'],
    '出入口': ['东门口', '西门口'],
    '其他': ['研究生大厦停车点', '水土保持学院', '南大山'],
}
colors10 = plt.cm.tab10(np.linspace(0, 1, 10))
x_ticks = np.arange(len(time_labels))

for ax_idx, (gname, members) in enumerate(groups_viz.items()):
    ax = axes[ax_idx]
    for i, name in enumerate(members):
        if name in station_data:
            ax.plot(x_ticks, station_data[name], 'o-', color=colors10[i], linewidth=1.8,
                    markersize=6, label=short_names.get(name, name))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(time_labels, rotation=45, fontsize=9)
    ax.set_title(gname, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_ylabel('停放量 (辆)', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figA_分区域时序变化.png')
plt.close()
print("  保存: figA_分区域时序变化.png")

# --- 图B: 热力图 (全站点 × 全时段) ---
fig, ax = plt.subplots(figsize=(14, 8))
matrix_data = np.array([station_data[s] for s in CORE_STATIONS])
sns = [short_names.get(s, s) for s in CORE_STATIONS]
im = ax.imshow(matrix_data, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, fontsize=10)
ax.set_yticks(range(len(CORE_STATIONS)))
ax.set_yticklabels(sns, fontsize=9)
for i in range(len(CORE_STATIONS)):
    for j in range(len(time_labels)):
        v = int(matrix_data[i, j])
        ax.text(j, i, v, ha='center', va='center', fontsize=7,
                color='white' if v > np.median(matrix_data) else 'black')
ax.set_title('时空停放量热力图 (辆)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('停放量 (辆)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figB_时空热力图.png')
plt.close()
print("  保存: figB_时空热力图.png")

# --- 图C: 供需缺口堆叠图 ---
fig, ax = plt.subplots(figsize=(14, 5))
period_labels_short = [f'{time_labels[t]}→{time_labels[t+1]}' for t in range(9)]
surplus_vals = [sds['total_surplus'] for sds in surplus_deficit_summary]
deficit_vals = [-sds['total_deficit'] for sds in surplus_deficit_summary]
xp = np.arange(9)
width = 0.55
ax.bar(xp, surplus_vals, width, color='#e74c3c', alpha=0.85, label='盈余 (需调出)', edgecolor='#c0392b', linewidth=0.5)
ax.bar(xp, deficit_vals, width, color='#3498db', alpha=0.85, label='亏缺 (需调入)', edgecolor='#2980b9', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=1)
# 标注净差
for t in range(9):
    net = surplus_vals[t] + deficit_vals[t]
    ax.annotate(f'净: {net:+.0f}', (xp[t], max(surplus_vals[t], abs(deficit_vals[t])) + 15),
                ha='center', fontsize=9, color='#2c3e50', fontweight='bold')
ax.set_xticks(xp)
ax.set_xticklabels(period_labels_short, fontsize=9)
ax.set_ylabel('车辆数', fontsize=12)
ax.set_title('各时段供需缺口分析', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figC_供需缺口堆叠图.png')
plt.close()
print("  保存: figC_供需缺口堆叠图.png")

# --- 图D: 调度流量桑基图 (用弦图替代) ---
# 选取流量最大的top-10站点做流向矩阵可视化
fig, ax = plt.subplots(figsize=(12, 10))
total_outflow_arr = flow_matrix.sum(axis=1)
total_inflow_arr = flow_matrix.sum(axis=0)
top_n = 10
top_idx = np.argsort(-(total_outflow_arr + total_inflow_arr))[:top_n]
sub_mat = flow_matrix[np.ix_(top_idx, top_idx)]
top_sns = [sns[i] for i in top_idx]
im3 = ax.imshow(sub_mat, cmap='YlOrRd', aspect='auto')
for i in range(top_n):
    for j in range(top_n):
        v = int(sub_mat[i, j])
        if v > 0:
            ax.text(j, i, v, ha='center', va='center', fontsize=8,
                    color='white' if v > sub_mat.max()/2 else 'black')
ax.set_xticks(range(top_n))
ax.set_xticklabels(top_sns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_sns, fontsize=9)
ax.set_xlabel('→ 调入点', fontsize=12)
ax.set_ylabel('调出点 ←', fontsize=12)
ax.set_title('全日累计调度流向矩阵 (辆)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im3, ax=ax, shrink=0.85)
cbar.set_label('累计调度量 (辆)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figD_调度流向矩阵.png')
plt.close()
print("  保存: figD_调度流向矩阵.png")

# --- 图E: 故障车分布 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# E1: 故障车柱状图
ax = axes[0]
faulty_sorted = sorted(faulty_bikes.items(), key=lambda x: -x[1])
f_names = [short_names.get(k, k) for k, v in faulty_sorted]
f_vals = [v for k, v in faulty_sorted]
bar_cols = ['#c0392b' if v >= 5 else '#e67e22' if v >= 3 else '#f39c12' for v in f_vals]
bars = ax.barh(range(len(f_names)), f_vals, color=bar_cols, edgecolor='white', height=0.7)
ax.set_yticks(range(len(f_names)))
ax.set_yticklabels(f_names, fontsize=9)
ax.set_xlabel('估算故障车数 (辆)', fontsize=11)
ax.set_title('各点位故障车估算量', fontsize=13, fontweight='bold')
for bar, val in zip(bars, f_vals):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=9, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')

# E2: 清运路线
ax = axes[1]
# 用坐标表示站点位置（多维缩放到2D）
from sklearn.manifold import MDS
coords_2d = None
try:
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords_2d = mds.fit_transform(dist_core)
except Exception:
    # fallback: use PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(dist_core)

ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='gray', s=30, alpha=0.5)
for i, name in enumerate(CORE_STATIONS):
    ax.annotate(short_names.get(name, name), (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=6, ha='center', va='bottom', alpha=0.7)

route_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for r_idx, route in enumerate(faulty_routes):
    pts = route['route']
    indices = []
    for p in pts:
        if p in CORE_STATIONS:
            indices.append(CORE_STATIONS.index(p))
        else:
            indices.append(0)  # fallback to depot
    if len(indices) > 1:
        for k in range(len(indices) - 1):
            ax.annotate('', xy=coords_2d[indices[k+1]], xytext=coords_2d[indices[k]],
                       arrowprops=dict(arrowstyle='->', color=route_colors[r_idx % 4],
                                      lw=2, alpha=0.8))
ax.set_title('故障车清运路线 (MDS投影)', fontsize=13, fontweight='bold')
ax.set_xlabel('MDS维度1')
ax.set_ylabel('MDS维度2')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figE_故障车分析与清运.png')
plt.close()
print("  保存: figE_故障车分析与清运.png")

# --- 图F: 调度效率甘特图 (选取1个高峰时段) ---
peak_t = 0  # 7:30-8:30
if peak_t in all_routes:
    fig, ax = plt.subplots(figsize=(14, 6))
    routes_p = all_routes[peak_t]
    y_labels = []
    colors_gantt = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for r_idx, route in enumerate(routes_p):
        cum_time = 0
        seq = route['sequence']
        for k in range(len(seq) - 1):
            si = data['station_names'].index(seq[k]) if seq[k] in data['station_names'] else 0
            sj = data['station_names'].index(seq[k+1]) if seq[k+1] in data['station_names'] else 0
            seg_dist = data['dist_matrix'][si, sj]
            seg_time = seg_dist / SPEED_M_PER_MIN
            ax.barh(r_idx, seg_time, left=cum_time, height=0.5,
                   color=colors_gantt[r_idx % 4], alpha=0.8,
                   edgecolor='white', linewidth=0.5)
            # 标注站点名
            if seg_time > 0.3:
                ax.text(cum_time + seg_time/2, r_idx,
                       short_names.get(seq[k+1], seq[k+1]),
                       ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            cum_time += seg_time
        y_labels.append(f'车{route["vehicle_id"]} (容{route["capacity"]})')
        # 在最后标注总时间
        ax.text(cum_time + 0.2, r_idx, f'{cum_time:.1f}min', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(routes_p)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('累计时间 (分钟)', fontsize=11)
    ax.set_title(f'调度甘特图 ({surplus_deficit_summary[peak_t]["period"]})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figF_调度甘特图.png')
    plt.close()
    print("  保存: figF_调度甘特图.png")

# --- 图G: 点位评价雷达图 ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# 选5个代表性点位做雷达图
radar_stations = ['教五', '食堂', '图书馆', '东门口', '学生公寓22栋楼下停车点']
metrics = ['周转率', '高峰强度', '覆盖范围', '点位重要性']
n_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]

for i, sname in enumerate(radar_stations):
    vals = np.array(station_data[sname])
    turnover = np.sum(np.abs(np.diff(vals))) / max(vals) if max(vals) > 0 else 0
    peak_intensity = max(vals) / (np.mean(vals) + 1)
    # 覆盖范围: 该点到其他点的平均距离的倒数 (越大越好)
    sidx = data['station_names'].index(sname)
    coverage = 1.0 / (np.mean([data['dist_matrix'][sidx, j] for j in CORE_INDICES if j != sidx]) / 1000 + 0.1)
    importance = np.mean(vals) / (np.mean(data['totals']) / 19 + 0.1)

    values = [turnover, peak_intensity, coverage, importance]
    # 归一化到 0-1
    all_vals = []
    for s2 in CORE_STATIONS:
        v2 = np.array(station_data[s2])
        all_vals.append([
            np.sum(np.abs(np.diff(v2))) / max(max(v2), 1),
            max(v2) / (np.mean(v2) + 1),
            1.0 / (np.mean([data['dist_matrix'][data['station_names'].index(s2), j]
                           for j in CORE_INDICES if j != data['station_names'].index(s2)]) / 1000 + 0.1),
            np.mean(v2) / (np.mean(data['totals']) / 19 + 0.1),
        ])
    all_vals = np.array(all_vals)
    max_vals = all_vals.max(axis=0)
    max_vals[max_vals == 0] = 1
    values_norm = np.array(values) / max_vals
    values_plot = values_norm.tolist() + values_norm[:1].tolist()

    ax.fill(angles, values_plot, alpha=0.15)
    ax.plot(angles, values_plot, 'o-', linewidth=2, markersize=6, label=short_names.get(sname, sname))

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_title('点位效能雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figG_点位效能雷达图.png')
plt.close()
print("  保存: figG_点位效能雷达图.png")

# ============================================================
# 8. 结果数据导出
# ============================================================
print("\n=== 导出结构化结果 ===")
import json as json_writer

# 结果摘要
summary = {
    'P_Median': {
        'optimal_P': P_optimal,
        'selected_stations': pmedian_results[P_optimal]['selected'],
        'removed_stations': removed_stations,
        'assignment': pmedian_results[P_optimal]['assignment'],
        'sensitivity': {str(p): {'objective': r['objective'], 'selected': r['selected']}
                       for p, r in pmedian_results.items()},
    },
    'supply_demand': [{
        'period': sds['period'],
        'total_surplus': sds['total_surplus'],
        'total_deficit': sds['total_deficit'],
        'surplus_points': sds['surplus_points'],
        'deficit_points': sds['deficit_points'],
    } for sds in surplus_deficit_summary],
    'min_cost_flow': {str(t): {'n_tasks': sum(len(dests) for dests in flow.values()),
                                'total_cost': cost} for t, (flow, cost) in flow_results.items()},
    'faulty_bikes': faulty_bikes,
    'faulty_routes': [{
        'vehicle_id': r['vehicle_id'],
        'capacity': r['capacity'],
        'route': r['route'],
        'total_distance': r['total_distance'],
        'total_time_min': r['total_time_min'],
        'total_collected': r['total_collected'],
    } for r in faulty_routes],
}

with open(f'{OUTPUT_DIR}/results_summary.json', 'w', encoding='utf-8') as f:
    json_writer.dump(summary, f, ensure_ascii=False, indent=2)
print("  保存: results_summary.json")

# 导出调度任务表 (所有时段的 Min-Cost Flow 结果)
dispatch_tasks = []
for t in range(9):
    flow, cost = flow_results[t]
    period_name = surplus_deficit_summary[t]['period']
    for s_name, dests in flow.items():
        for d_name, amt in dests.items():
            dispatch_tasks.append({
                'period': period_name,
                'period_idx': t,
                'from': s_name,
                'to': d_name,
                'quantity': int(amt),
            })

with open(f'{OUTPUT_DIR}/dispatch_tasks.json', 'w', encoding='utf-8') as f:
    json_writer.dump(dispatch_tasks, f, ensure_ascii=False, indent=2)
print(f"  保存: dispatch_tasks.json ({len(dispatch_tasks)} 条调度任务)")

# 导出各时段调度路线
route_summary = {}
for t, routes in all_routes.items():
    period_name = surplus_deficit_summary[t]['period']
    route_summary[period_name] = [{
        'vehicle_id': r['vehicle_id'],
        'capacity': r['capacity'],
        'sequence': r['sequence'],
        'actions': r['actions'],
        'total_distance': r['total_distance'],
        'total_time_min': r['total_time_min'],
        'num_tasks': r['num_tasks'],
    } for r in routes]

with open(f'{OUTPUT_DIR}/route_plans.json', 'w', encoding='utf-8') as f:
    json_writer.dump(route_summary, f, ensure_ascii=False, indent=2)
print(f"  保存: route_plans.json")

# CSV 导出 (方便 Excel 查看)
import csv
with open(f'{OUTPUT_DIR}/dispatch_tasks.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['时段', '时段索引', '调出点', '调入点', '数量(辆)'])
    for task in dispatch_tasks:
        writer.writerow([task['period'], task['period_idx'], task['from'], task['to'], task['quantity']])
print("  保存: dispatch_tasks.csv")

print(f"\n=== 全部计算完成 ===")
print(f"输出文件 ({len(os.listdir(OUTPUT_DIR))} 个):")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024
    print(f"  {fname} ({size_kb:.1f} KB)")

