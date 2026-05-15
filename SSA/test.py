# -*- coding: utf-8 -*-
"""
ISSA vs SSA 基准函数对比测试（多样性自适应版）
改进策略融合：Tent混沌初始化 + 自适应莱维飞行 + 自适应柯西变异
"""

import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
DIM = 30
N_POP = 40
MAX_ITER = 500
RUNS = 30

# 莱维飞行参数
LEVY_BETA = 1.5

# 柯西变异参数
CAUCHY_SCALE = 0.1
STAGNATION_THRESHOLD = 20

# ==================== 基准函数 ====================
def F1_Sphere(x):
    return np.sum(x ** 2)

def F2_Schwefel_222(x):
    """Schwefel 2.22 函数: f(x) = sum(|x_i|) + prod(|x_i|)"""
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
    'F1_Sphere':          {'func': F1_Sphere,         'bounds': (-100, 100), 'dim': DIM},
    'F2_Schwefel_222':    {'func': F2_Schwefel_222,   'bounds': (-10, 10),   'dim': DIM},
    'F3_Ackley':          {'func': F3_Ackley,         'bounds': (-32, 32),   'dim': DIM},
    'F4_Rastrigin':       {'func': F4_Rastrigin,      'bounds': (-5.12, 5.12),'dim': DIM},
}

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

# ==================== 改进 ISSA（多样性自适应版）====================
class ISSA:
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
        # 策略1：Tent混沌初始化
        self.X = self._tent_initialize()
        self.fitness = np.full(n_pop, np.inf)
        self.best_fitness = np.inf
        self.best_position = None

    def _tent_map(self, x, a=0.5):
        x = np.asarray(x)
        return np.where(x < a, x / a, (1 - x) / (1 - a))

    def _tent_initialize(self):
        X = np.zeros((self.n_pop, self.dim))
        for j in range(self.dim):
            x = np.random.random()
            for i in range(self.n_pop):
                x = self._tent_map(x, a=0.5)
                X[i, j] = self.lb[j] + x * (self.ub[j] - self.lb[j])
        return np.clip(X, self.lb, self.ub)

    def _levy_flight(self, beta=LEVY_BETA, size=None):
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

        # 自适应停滞计数
        stagnation_count = 0
        prev_best = self.best_fitness

        for t in range(self.max_iter):
            # ========== 多样性自适应：计算种群平均标准差 ==========
            diversity = np.mean(np.std(self.X, axis=0))
            # 软阈值归一化：diversity 大时 ratio→1，diversity 小时 ratio→0
            # 1.0 是经验阈值，当 diversity < 1 时步长急剧缩小
            adaptive_ratio = diversity / (diversity + 1.0)

            sorted_indices = np.argsort(self.fitness)
            PD_indices = sorted_indices[:self.PD]
            F_indices = sorted_indices[self.PD:]
            X_new = self.X.copy()

            # 策略2：莱维飞行（危险状态 + 多样性自适应步长）
            for i in PD_indices:
                r2 = np.random.random()
                if r2 < self.ST:
                    # 安全状态：纯原始 SSA 指数衰减收敛
                    X_new[i] = self.X[i] * np.exp(-np.random.random() / (t + 1))
                else:
                    # 危险状态：Levy 飞行，步长按种群多样性自适应
                    # 前期 diversity 大（~50），scale ≈ 0.05 * 1 * (ub-lb)，大步探索
                    # 后期 diversity 小（~0.01），scale ≈ 0.05 * 0.01 * (ub-lb)，极小步不干扰
                    scale = 0.05 * adaptive_ratio * (self.ub - self.lb)
                    levy_step = self._levy_flight(beta=LEVY_BETA, size=self.dim)
                    X_new[i] = self.X[i] + scale * levy_step

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

            # ========== 策略3：自适应柯西变异（停滞触发）==========
            if self.best_fitness < prev_best - 1e-12:
                stagnation_count = 0
                prev_best = self.best_fitness
            else:
                stagnation_count += 1

            if stagnation_count >= STAGNATION_THRESHOLD:
                # 柯西步长也按多样性自适应：聚集时炸得轻，分散时炸得猛
                sigma = CAUCHY_SCALE * adaptive_ratio * (self.ub - self.lb)
                cauchy_step = np.random.standard_cauchy(self.dim)
                X_mutant = self.best_position + sigma * cauchy_step
                X_mutant = np.clip(X_mutant, self.lb, self.ub)
                f_mutant = func(X_mutant)
                if f_mutant < self.best_fitness:
                    self.best_fitness = f_mutant
                    self.best_position = X_mutant.copy()
                    worst_idx = np.argmax(self.fitness)
                    self.X[worst_idx] = X_mutant.copy()
                    self.fitness[worst_idx] = f_mutant
                    stagnation_count = 0
                    prev_best = self.best_fitness

        return self.best_position, self.best_fitness


# ==================== 主测试 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("ISSA vs SSA 基准函数对比测试（多样性自适应版）")
    print("=" * 70)
    print(f"维度: {DIM}, 种群: {N_POP}, 迭代: {MAX_ITER}, 独立运行: {RUNS}")
    print("ISSA策略: Tent混沌初始化 + 自适应莱维飞行 + 自适应柯西变异")
    print("=" * 70)

    results = []
    for fname, finfo in BENCHMARK_FUNCS.items():
        func = finfo['func']
        lb, ub = finfo['bounds']
        dim = finfo['dim']
        bounds = [lb] * dim

        print(f"\n测试函数: {fname} (维度={dim})")
        ssa_vals, issa_vals = [], []
        for r in range(RUNS):
            ssa = OriginalSSA(N_POP, MAX_ITER, bounds, [ub]*dim, dim, seed=r)
            _, best_ssa = ssa.optimize(func)
            ssa_vals.append(best_ssa)

            issa = ISSA(N_POP, MAX_ITER, bounds, [ub]*dim, dim, seed=r)
            _, best_issa = issa.optimize(func)
            issa_vals.append(best_issa)

            if (r + 1) % 10 == 0:
                print(f"  已完成 {r+1}/{RUNS} 次...")

        results.append({
            'Function': fname,
            'SSA_Mean': np.mean(ssa_vals), 'SSA_Best': np.min(ssa_vals), 'SSA_Std': np.std(ssa_vals),
            'ISSA_Mean': np.mean(issa_vals), 'ISSA_Best': np.min(issa_vals), 'ISSA_Std': np.std(issa_vals),
        })
        print(f"  SSA:  Mean={np.mean(ssa_vals):.6e}, Best={np.min(ssa_vals):.6e}, Std={np.std(ssa_vals):.6e}")
        print(f"  ISSA: Mean={np.mean(issa_vals):.6e}, Best={np.min(issa_vals):.6e}, Std={np.std(issa_vals):.6e}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("【测试结果】ISSA vs SSA（多样性自适应版）")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv('issa_vs_ssa_benchmark_diversity.csv', index=False, encoding='utf-8-sig')
    print("\n[OK] 结果已保存: issa_vs_ssa_benchmark_diversity.csv")