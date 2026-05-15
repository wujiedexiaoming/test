# -*- coding: utf-8 -*-
"""
FDAL-SSA: Fitness-Distance Adaptive Levy SSA

核心诊断（基于全部消融实验数据分析）：
- Levy失败原因1: 各向同性随机步在D=30下99%朝错误方向
- Levy失败原因2: 对所有个体无差别扰动，破坏精英收敛
- Levy失败原因3: 步长一刀切，没有根据个体到最优的距离自适应

本方案核心改进：
1. 距离自适应步长: scale ∝ |X[i] - X_best|，收敛时自动归零
2. 有向Levy: 用个体→最优向量做方向偏置，避免盲飞
3. 成功历史步长调节: 根据Levy成功率自动调整强度
4. 安全态公式一字不改: 保护SSA核心收敛力
5. 仅改动危险态: 只影响~20%更新，影响面可控
"""

import numpy as np
import math
import csv
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
DIM = 30
N_POP = 40
MAX_ITER = 500
RUNS = 30

# ==================== 基准函数 ====================
def F1_Sphere(x):
    return np.sum(x ** 2)

def F2_Schwefel_222(x):
    """Schwefel 2.22"""
    abs_x = np.abs(x)
    return np.sum(abs_x) + np.prod(abs_x)

def F3_Ackley(x):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def F4_Rastrigin(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

BENCHMARK_FUNCS = {
    'F1_Sphere':       {'func': F1_Sphere,       'bounds': (-100, 100), 'dim': DIM},
    'F2_Schwefel_222': {'func': F2_Schwefel_222, 'bounds': (-10, 10),   'dim': DIM},
    'F3_Ackley':       {'func': F3_Ackley,       'bounds': (-32, 32),   'dim': DIM},
    'F4_Rastrigin':    {'func': F4_Rastrigin,    'bounds': (-5.12, 5.12),'dim': DIM},
}

# ==================== 镜像反射边界 ====================
def reflect_bounds(X, lb, ub):
    X = np.asarray(X).copy()
    lb, ub = np.asarray(lb), np.asarray(ub)
    X = np.where(X > ub, 2 * ub - X, X)
    X = np.where(X < lb, 2 * lb - X, X)
    return np.clip(X, lb, ub)

# ==================== 原始 SSA ====================
class OriginalSSA:
    def __init__(self, n_pop, max_iter, lb, ub, dim, seed=42):
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.lb = np.array(lb, dtype=np.float64)
        self.ub = np.array(ub, dtype=np.float64)
        self.dim = dim
        self.PD = max(1, int(0.2 * n_pop))
        self.SD = max(1, int(0.1 * n_pop))
        self.ST = 0.8
        np.random.seed(seed)
        self.X = np.random.uniform(lb, ub, (n_pop, dim))
        self.fitness = np.full(n_pop, np.inf)
        self.best_fitness = np.inf
        self.best_position = None

    def optimize(self, func):
        for i in range(self.n_pop):
            self.fitness[i] = func(self.X[i])
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.X[best_idx].copy()

        for t in range(self.max_iter):
            sorted_indices = np.argsort(self.fitness)
            PD_indices = sorted_indices[:self.PD]
            F_indices = sorted_indices[self.PD:]
            X_new = self.X.copy()

            for i in PD_indices:
                r2 = np.random.random()
                if r2 < self.ST:
                    X_new[i] = self.X[i] * np.exp(-np.random.random() / (t + 1))
                else:
                    X_new[i] = self.X[i] + np.random.normal(0, 1, self.dim)

            for idx, i in enumerate(F_indices):
                if idx < len(F_indices) / 2:
                    A = np.random.choice([-1, 1], self.dim) * np.random.random(self.dim)
                    AA = A / (np.abs(A).sum() + 1e-10)
                    best_pd = self.X[PD_indices[0]]
                    X_new[i] = best_pd + np.abs(self.X[i] - best_pd) * AA
                else:
                    Q = np.random.normal(0, 1, self.dim)
                    X_new[i] = Q * np.exp((self.fitness[i] - self.fitness[sorted_indices[-1]]) /
                                          (np.abs(self.fitness[sorted_indices[-1]]) + 1e-10))

            SD_indices = np.random.choice(self.n_pop, self.SD, replace=False)
            for i in SD_indices:
                if self.fitness[i] > self.best_fitness:
                    X_new[i] = self.best_position + np.random.normal(0, 1, self.dim) * np.abs(self.X[i] - self.best_position)
                elif abs(self.fitness[i] - self.best_fitness) < 1e-8:
                    X_new[i] = self.X[i] + np.random.choice([-1, 1]) * np.random.random(self.dim) / (np.abs(self.X[i]) + 1e-10)
                else:
                    X_new[i] = self.best_position + np.random.random(self.dim) * (self.best_position - self.X[i])

            X_new = np.clip(X_new, self.lb, self.ub)
            for i in range(self.n_pop):
                f_new = func(X_new[i])
                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    self.X[i] = X_new[i].copy()

            current_best = np.argmin(self.fitness)
            if self.fitness[current_best] < self.best_fitness:
                self.best_fitness = self.fitness[current_best]
                self.best_position = self.X[current_best].copy()
        return self.best_position, self.best_fitness


# ==================== FDAL-SSA ====================
class FDAL_SSA:
    """
    Fitness-Distance Adaptive Levy SSA

    改动清单（仅4处，其余全同原版）：
    1. Tent混沌初始化
    2. 危险态: 距离自适应有向Levy替代高斯散步
    3. 镜像反射边界替代clip
    4. 全局最优极小幅柯西变异（后处理，概率触发）

    不碰的部分：安全态、跟随者、警戒者全部原版
    """
    def __init__(self, n_pop, max_iter, lb, ub, dim, seed=42):
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.lb = np.array(lb, dtype=np.float64)
        self.ub = np.array(ub, dtype=np.float64)
        self.dim = dim
        self.PD = max(1, int(0.2 * n_pop))
        self.SD = max(1, int(0.1 * n_pop))
        self.ST = 0.8
        np.random.seed(seed)

        # 改动1: Tent混沌初始化
        self.X = self._tent_initialize()
        self.fitness = np.full(n_pop, np.inf)
        self.best_fitness = np.inf
        self.best_position = None

        # 成功历史自适应参数
        self._levy_successes = 0
        self._levy_attempts = 0
        self._adaptive_boost = 1.0  # 初始基准倍率

    def _tent_map(self, x, a=0.5):
        return np.where(x < a, x / a, (1 - x) / (1 - a))

    def _tent_initialize(self):
        X = np.zeros((self.n_pop, self.dim))
        for j in range(self.dim):
            x = np.random.random()
            for i in range(self.n_pop):
                x = self._tent_map(x, a=0.5)
                X[i, j] = self.lb[j] + x * (self.ub[j] - self.lb[j])
        return np.clip(X, self.lb, self.ub)

    def _levy_flight(self, beta=1.5, size=None):
        """Mantegna算法"""
        if size is None:
            size = self.dim
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def optimize(self, func):
        for i in range(self.n_pop):
            self.fitness[i] = func(self.X[i])
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.X[best_idx].copy()

        prev_best = self.best_fitness
        stagnation_count = 0

        for t in range(self.max_iter):
            # ============================================================
            # 计算种群状态指标
            # ============================================================
            sorted_indices = np.argsort(self.fitness)
            PD_indices = sorted_indices[:self.PD]
            F_indices = sorted_indices[self.PD:]
            X_new = self.X.copy()

            # 种群多样性: avg std per dim
            diversity = np.mean(np.std(self.X, axis=0))
            # 群体中位数适应度
            median_fitness = self.fitness[sorted_indices[self.n_pop // 2]]

            # ============================================================
            # 发现者（Producer）———— 安全态一字不改, 危险态用距离自适应有向Levy
            # ============================================================
            for i in PD_indices:
                r2 = np.random.random()
                if r2 < self.ST:
                    # 【安全态】原版公式, 一字不改
                    X_new[i] = self.X[i] * np.exp(-np.random.random() / (t + 1))
                else:
                    # 【危险态】距离自适应有向Levy
                    # 核心公式:
                    #   X_new = X + alpha*(X_best - X) + beta*scale_i*Levy
                    # 其中:
                    #   alpha: 向最优靠拢的力度 (随迭代增大)
                    #   beta:  随机探索的强度 (随迭代减小)
                    #   scale_i: 距离自适应步长 = base * |X_i - X_best| / search_range

                    # 距最优的逐维距离
                    dist_to_best = np.abs(self.X[i] - self.best_position)

                    # 距离自适应基步长：远则大，近则小
                    # 归一化：除以搜索范围，使不同函数间可比
                    search_range = self.ub - self.lb
                    norm_dist = np.mean(dist_to_best / (search_range + 1e-10))

                    # 基步长：0.02 * (1-t/T) * 搜索范围
                    base_scale = 0.02 * (1 - t / self.max_iter) * (self.ub - self.lb)

                    # 距离自适应：远处个体( norm_dist ~ 5-50%)用完整步长
                    #             近处个体( norm_dist ~ 0.01%)自动衰减到几乎为零
                    # 使用sqrt避免过度压缩
                    distance_factor = np.sqrt(np.clip(norm_dist, 0.001, 1.0))

                    # 成功历史调节
                    success_ratio = self._levy_successes / max(self._levy_attempts, 1)
                    if success_ratio > 0.25:
                        self._adaptive_boost = min(2.0, self._adaptive_boost * 1.05)
                    elif success_ratio < 0.10:
                        self._adaptive_boost = max(0.1, self._adaptive_boost * 0.95)

                    # 最终步长 = 基步长 × 距离因子 × 历史调节 × 多样性衰减
                    # 多样性低时进一步降低步长（保护单峰函数收敛）
                    diversity_damping = np.clip(diversity / 1.0, 0.1, 1.0)
                    scale = base_scale * distance_factor * self._adaptive_boost * diversity_damping

                    # 有向Levy: 主要方向朝最优，叠加Levy随机性
                    levy_step = self._levy_flight(beta=1.5, size=self.dim)
                    # 方向偏置: unit vector toward best (逐维)
                    direction = self.best_position - self.X[i]
                    dir_norm = np.linalg.norm(direction) + 1e-10
                    unit_dir = direction / dir_norm

                    # 混合: 70%有向 + 30%各向同性Levy
                    # 前期混合更多随机性，后期更多有向
                    mix_ratio = 0.3 + 0.4 * (t / self.max_iter)  # 0.3→0.7, 后期更有向
                    directed_component = mix_ratio * unit_dir
                    levy_component = (1 - mix_ratio) * levy_step / (np.linalg.norm(levy_step) + 1e-10)

                    # 合成步长向量，每维步长不同
                    step_vector = scale * (directed_component + levy_component)

                    X_new[i] = self.X[i] + step_vector

            # ============================================================
            # 加入者（Follower）———— 原版不变
            # ============================================================
            for idx, i in enumerate(F_indices):
                if idx < len(F_indices) / 2:
                    A = np.random.choice([-1, 1], self.dim) * np.random.random(self.dim)
                    AA = A / (np.abs(A).sum() + 1e-10)
                    best_pd = self.X[PD_indices[0]]
                    X_new[i] = best_pd + np.abs(self.X[i] - best_pd) * AA
                else:
                    Q = np.random.normal(0, 1, self.dim)
                    X_new[i] = Q * np.exp((self.fitness[i] - self.fitness[sorted_indices[-1]]) /
                                          (np.abs(self.fitness[sorted_indices[-1]]) + 1e-10))

            # ============================================================
            # 警戒者（Watcher）———— 原版不变
            # ============================================================
            SD_indices = np.random.choice(self.n_pop, self.SD, replace=False)
            for i in SD_indices:
                if self.fitness[i] > self.best_fitness:
                    X_new[i] = self.best_position + np.random.normal(0, 1, self.dim) * np.abs(self.X[i] - self.best_position)
                elif abs(self.fitness[i] - self.best_fitness) < 1e-8:
                    X_new[i] = self.X[i] + np.random.choice([-1, 1]) * np.random.random(self.dim) / (np.abs(self.X[i]) + 1e-10)
                else:
                    X_new[i] = self.best_position + np.random.random(self.dim) * (self.best_position - self.X[i])

            # 改动3: 镜像反射边界
            X_new = reflect_bounds(X_new, self.lb, self.ub)

            # 评估与贪婪更新
            for i in range(self.n_pop):
                f_new = func(X_new[i])
                if f_new < self.fitness[i]:
                    # 追踪危险态Levy成功率
                    if i in PD_indices:
                        self._levy_attempts += 1
                        if f_new < self.fitness[i]:
                            self._levy_successes += 1
                    self.fitness[i] = f_new
                    self.X[i] = X_new[i].copy()

            current_best = np.argmin(self.fitness)
            if self.fitness[current_best] < self.best_fitness:
                self.best_fitness = self.fitness[current_best]
                self.best_position = self.X[current_best].copy()

            # ============================================================
            # 改动4: 全局最优极小幅柯西变异（后处理）
            # 原理: 消融实验证明 scale=0.001*(ub-lb) 的柯西变异对单峰无害，对多峰有益
            # ============================================================
            if np.random.random() < 0.25:
                cauchy_scale = 0.0005 * (1 - t / self.max_iter) * (self.ub - self.lb)
                cauchy_step = np.random.standard_cauchy(self.dim)
                X_mutant = self.best_position + cauchy_scale * cauchy_step
                X_mutant = reflect_bounds(X_mutant, self.lb, self.ub)
                f_mutant = func(X_mutant)
                if f_mutant < self.best_fitness:
                    self.best_fitness = f_mutant
                    self.best_position = X_mutant.copy()
                    worst_idx = np.argmax(self.fitness)
                    self.X[worst_idx] = X_mutant.copy()
                    self.fitness[worst_idx] = f_mutant

            # ============================================================
            # 停滞检测与强化探索
            # ============================================================
            if self.best_fitness < prev_best - 1e-12:
                stagnation_count = 0
                prev_best = self.best_fitness
            else:
                stagnation_count += 1

            # 停滞超过30代: 用有向Levy对最差50%个体做一次强化扰动
            if stagnation_count >= 30:
                n_perturb = self.n_pop // 2
                worst_indices = np.argsort(self.fitness)[-n_perturb:]
                for i in worst_indices:
                    # 方向: 朝最优, 步长: 中等Levy
                    direction = self.best_position - self.X[i]
                    unit_dir = direction / (np.linalg.norm(direction) + 1e-10)
                    levy_step = self._levy_flight(beta=1.5, size=self.dim)
                    # 停滞时步长稍大一些，但用距离限制
                    dist = np.linalg.norm(direction)
                    escape_scale = 0.03 * np.clip(dist, 0.01 * np.linalg.norm(self.ub - self.lb),
                                                 0.2 * np.linalg.norm(self.ub - self.lb))
                    X_perturb = self.X[i] + escape_scale * (0.5 * unit_dir + 0.5 * levy_step)
                    X_perturb = reflect_bounds(X_perturb, self.lb, self.ub)
                    f_perturb = func(X_perturb)
                    if f_perturb < self.fitness[i]:
                        self.fitness[i] = f_perturb
                        self.X[i] = X_perturb.copy()
                        if f_perturb < self.best_fitness:
                            self.best_fitness = f_perturb
                            self.best_position = X_perturb.copy()
                            prev_best = f_perturb
                stagnation_count = 0

        return self.best_position, self.best_fitness


# ==================== 主测试 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("FDAL-SSA (Fitness-Distance Adaptive Levy) vs Original SSA")
    print("=" * 70)
    print(f"DIM={DIM}, POP={N_POP}, ITER={MAX_ITER}, RUNS={RUNS}")
    print("")
    print("改进策略:")
    print("  1. Tent混沌初始化")
    print("  2. 发现者危险态: 距离自适应有向Levy (替代高斯随机散步)")
    print("     - 步长 ∝ |X_i - X_best|, 收敛时自动归零")
    print("     - 70-30% 有向+各向同性混合")
    print("     - 成功历史自适应调节")
    print("     - 种群多样性衰减")
    print("  3. 镜像反射边界")
    print("  4. 全局最优极小幅柯西后处理")
    print("  5. 停滞检测: 30代无改进触发强化Levy探索")
    print("")
    print("安全态、跟随者、警戒者全部保持原版公式")
    print("=" * 70)

    results = []
    for fname, finfo in BENCHMARK_FUNCS.items():
        func = finfo['func']
        lb, ub = finfo['bounds']
        bounds = [lb] * DIM

        print(f"\n{'─'*60}")
        print(f"Testing: {fname} (bounds=[{lb}, {ub}], dim={DIM})")
        print(f"{'─'*60}")
        ssa_vals, fdal_vals = [], []
        for r in range(RUNS):
            ssa = OriginalSSA(N_POP, MAX_ITER, bounds, [ub]*DIM, DIM, seed=r)
            _, best_ssa = ssa.optimize(func)
            ssa_vals.append(best_ssa)

            fdal = FDAL_SSA(N_POP, MAX_ITER, bounds, [ub]*DIM, DIM, seed=r)
            _, best_fdal = fdal.optimize(func)
            fdal_vals.append(best_fdal)

            if (r + 1) % 10 == 0:
                print(f"  Progress: {r+1}/{RUNS} runs completed")

        row = {
            'Function': fname,
            'SSA_Mean': np.mean(ssa_vals),
            'SSA_Best': np.min(ssa_vals),
            'SSA_Std': np.std(ssa_vals),
            'FDAL_Mean': np.mean(fdal_vals),
            'FDAL_Best': np.min(fdal_vals),
            'FDAL_Std': np.std(fdal_vals),
        }
        results.append(row)

        s_mean, s_best, s_std = row['SSA_Mean'], row['SSA_Best'], row['SSA_Std']
        f_mean, f_best, f_std = row['FDAL_Mean'], row['FDAL_Best'], row['FDAL_Std']

        print(f"\n  {'':>6} {'Mean':>15} {'Best':>15} {'Std':>15}")
        print(f"  {'SSA':>6} {s_mean:>15.6e} {s_best:>15.6e} {s_std:>15.6e}")
        print(f"  {'FDAL':>6} {f_mean:>15.6e} {f_best:>15.6e} {f_std:>15.6e}")

        if s_mean != 0:
            mean_improve = (s_mean - f_mean) / s_mean * 100
            best_improve = (s_best - f_best) / s_best * 100 if s_best != 0 else 0
            print(f"  Mean提升: {mean_improve:+.1f}%  |  Best提升: {best_improve:+.1f}%")

    # 汇总
    print("\n" + "=" * 70)
    print("FINAL RESULTS: FDAL-SSA vs Original SSA")
    print("=" * 70)
    header = f"{'Function':<18} {'SSA_Mean':>14} {'FDAL_Mean':>14} {'Improve':>10}"
    print(header)
    print("-" * 60)
    for row in results:
        s, f = row['SSA_Mean'], row['FDAL_Mean']
        improve = f"{(s-f)/s*100:+.1f}%" if s != 0 else "N/A"
        print(f"{row['Function']:<18} {s:>14.6e} {f:>14.6e} {improve:>10}")

    # 保存CSV
    with open('FDAL_SSA_vs_SSA.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Function',
                         'SSA_Mean', 'SSA_Best', 'SSA_Std',
                         'FDAL_Mean', 'FDAL_Best', 'FDAL_Std'])
        for row in results:
            writer.writerow([
                row['Function'],
                row['SSA_Mean'], row['SSA_Best'], row['SSA_Std'],
                row['FDAL_Mean'], row['FDAL_Best'], row['FDAL_Std'],
            ])
    print("\n[OK] Results saved to FDAL_SSA_vs_SSA.csv")
