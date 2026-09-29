# -*- coding: utf-8 -*-
"""HSGDA 实验绘图：生成 latex/images/hsgda_*.pdf（同时保存 PNG 到 results/）。"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
IMG_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'latex', 'images'))

COLORS = {'HSGDA': 'tab:red', 'SSGDA': 'tab:blue', 'TSGDA-1': 'tab:green'}


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name + '.pdf'))
    fig.savefig(os.path.join(RESULTS_DIR, name + '.png'), dpi=150)
    plt.close(fig)
    print('saved', name)


def smooth_mean_std(df, key, value, steps):
    """按 key 分组求各 step 的均值与标准差。"""
    out = {}
    for cv, sub in df.groupby(key):
        g = sub.groupby('step')[value]
        out[cv] = (g.mean(), g.std())
    return out


def fig_convergence():
    """图 1：真实超梯度范数的收敛性（轨迹 + min-T 网格）。"""
    data = np.load(os.path.join(RESULTS_DIR, 'hsgda_conv_gphi2.npz'))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    t = np.arange(1, data['hsgda'].shape[1] + 1)
    for name, key in [('HSGDA', 'hsgda'), ('SSGDA', 'ssgda'),
                      ('TSGDA-1', 'tsgda')]:
        arr = np.clip(data[key], 1e-8, None)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.loglog(t, mean, label=name, color=COLORS[name])
        ax.fill_between(t, np.clip(mean - std, 1e-8, None),
                        mean + std, alpha=0.18, color=COLORS[name])
    ref = 2.0 * (t / t[0]) ** (-0.5)
    ax.loglog(t, ref, 'k--', lw=1, label=r'$\propto t^{-1/2}$')
    ax.set_xlabel('Outer iteration $t$')
    ax.set_ylabel(r'$\|\nabla \Phi(x^t)\|_2^2$')
    ax.set_title('(a) True hypergradient norm')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_conv_min.csv'))
    g = df.groupby('T')['min_gphi2']
    Tg = np.array(sorted(df['T'].unique()), dtype=float)
    m = g.mean().values
    s = g.std().values
    ax.loglog(Tg, m, 'o-', color='tab:red', label=r'$\min_{t<T}\|\nabla\Phi(x^t)\|^2$')
    ax.fill_between(Tg, np.clip(m - s, 1e-8, None), m + s, alpha=0.2,
                    color='tab:red')
    ref = m[0] * (Tg / Tg[0]) ** (-0.5)
    ax.loglog(Tg, ref, 'k--', lw=1, label=r'$\propto T^{-1/2}$')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\min_{t<T}\ \|\nabla\Phi(x^t)\|_2^2$')
    ax.set_title('(b) Trajectory minimum versus $T$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'hsgda_convergence')


def fig_accuracy():
    """图 2：预测精度（population 目标终值对比）。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_conv_accuracy.csv'))
    order = ['HSGDA', 'SSGDA', 'TSGDA-1']
    means = [df[df['solver'] == s]['phi_pop_mean'].values[0] for s in order]
    stds = [df[df['solver'] == s]['phi_pop_std'].values[0] for s in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(order, means, yerr=stds, capsize=4,
                  color=[COLORS[s] for s in order], alpha=0.85)
    ax.axhline(means[0], color='tab:red', ls=':', lw=1)
    ax.set_ylabel(r'$\Phi(x^T)$ (population hyper-objective)')
    ax.set_title('Accuracy of the returned solution')
    ax.grid(alpha=0.3, axis='y')
    for b, mv in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mv, f'{mv:.3f}',
                ha='center', va='bottom' if mv < 0 else 'center', fontsize=8)
    fig.tight_layout()
    save(fig, 'hsgda_accuracy')


def fig_gen_tk():
    """图 3：K 的影响（HSGDA 曲线 + 两求解器的最终 Gap 对比）。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_tk_detail.csv'))
    tab = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_tk_final.csv'),
                      header=[0, 1], index_col=[0, 1])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    # (a) HSGDA 验证误差曲线
    ax = axes[0]
    for K in sorted(df['K'].unique()):
        sub = df[(df['solver'] == 'HSGDA') & (df['K'] == K)]
        g = sub.groupby('step')['val']
        ax.plot(g.mean().index, g.mean().values, label=f'$K={K}$', lw=1.5)
    ax.set_xlabel('Outer iteration $t$')
    ax.set_ylabel('Validation loss (meta set)')
    ax.set_title('(a) HSGDA validation loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (b) HSGDA 泛化 Gap 曲线
    ax = axes[1]
    for K in sorted(df['K'].unique()):
        sub = df[(df['solver'] == 'HSGDA') & (df['K'] == K)]
        g = sub.groupby('step')['gap']
        mean, std = g.mean(), g.std()
        ax.plot(mean.index, mean.values, label=f'$K={K}$', lw=1.5)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel('Outer iteration $t$')
    ax.set_ylabel('Generalization gap')
    ax.set_title('(b) HSGDA generalization gap')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (c) 两求解器的最终 Gap 随 K 的变化
    ax = axes[2]
    for solver, color, mk in [('HSGDA', 'tab:red', 'o'),
                              ('TSGDA-1', 'tab:green', 's')]:
        sub = tab.loc[solver]
        m = sub[('gap', 'mean')]
        s = sub[('gap', 'std')]
        ax.errorbar(sub.index, m, yerr=s, marker=mk, capsize=3,
                    color=color, label=solver)
    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Final generalization gap')
    ax.set_title('(c) Effect of $K$ on the gap')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'hsgda_gen_tk')


def fig_gen_T():
    """图 4：外层迭代数 T 的影响（最终误差与 Gap）。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_T_final.csv'),
                     header=[0, 1], index_col=0)
    metrics = [('val', 'Final validation loss'),
               ('test', 'Final testing loss'),
               ('gap', 'Final generalization gap')]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, (metric, label) in zip(axes, metrics):
        mean, std = df[(metric, 'mean')], df[(metric, 'std')]
        ax.errorbar(df.index, mean, yerr=std, marker='o', capsize=3,
                    color='tab:red')
        ax.set_xlabel(r'$T$')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'hsgda_gen_T')


def fig_gen_m1():
    """图 5：meta 集大小 m1 的影响（最终误差与 Gap，含 1/m1 参考线）。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_m1_final.csv'),
                     header=[0, 1], index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, metric, label in [
            (axes[0], 'val', 'Final validation loss'),
            (axes[1], 'test', 'Final testing loss'),
            (axes[2], 'gap', 'Final generalization gap')]:
        mean, std = df[(metric, 'mean')], df[(metric, 'std')]
        ax.errorbar(df.index, mean, yerr=std, marker='o', capsize=3,
                    color='tab:red')
        ax.set_xscale('log')
        ax.set_xlabel(r'$m_1$')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which='both')
    ax = axes[2]
    ref = df[('gap', 'mean')].abs().max() \
        * (df.index / df.index[0]) ** (-1)
    ax.plot(df.index, ref, 'k--', lw=1, label=r'$\propto 1/m_1$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'hsgda_gen_m1')


def fig_gen_eta():
    """图 6：步长 eta 与调度的影响。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_eta_detail.csv'))
    metrics = [('val', 'Validation loss (meta set)'),
               ('test', 'Testing loss (test set)'),
               ('gap', 'Generalization gap')]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, (metric, label) in zip(axes, metrics):
        for name, sub in df.groupby('sched'):
            g = sub.groupby('step')[metric]
            mean, std = g.mean(), g.std()
            ax.plot(mean.index, mean.values, label=name, lw=1.5)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.12)
        ax.set_xlabel('Outer iteration $t$')
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'hsgda_gen_eta')


def fig_digits():
    """图 7：digits 重加权任务的精度与权重行为。"""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'hsgda_digits_detail.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for solver in ['HSGDA', 'SSGDA', 'TSGDA-1']:
        sub = df[df['solver'] == solver]
        g = sub.groupby('step')['acc']
        mean, std = g.mean(), g.std()
        ax.plot(mean.index, mean.values, label=solver, color=COLORS[solver])
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15,
                        color=COLORS[solver])
    ax.set_xlabel('Outer iteration $t$')
    ax.set_ylabel('Test accuracy')
    ax.set_title('(a) Test accuracy on digits')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sub = df[df['solver'] == 'HSGDA']
    g = sub.groupby('step')
    for col, lab in [('w_clean_mean', 'mean weight of clean samples'),
                     ('w_noise_mean', 'mean weight of noisy samples')]:
        m, s = g[col].mean(), g[col].std()
        ax.plot(m.index, m.values, label=lab)
        ax.fill_between(m.index, m - s, m + s, alpha=0.15)
    ax.set_xlabel('Outer iteration $t$')
    ax.set_ylabel(r'Mean sample weight $\sigma(x_i)$')
    ax.set_title('(b) Learned sample weights (HSGDA)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'hsgda_digits')


def main():
    fig_convergence()
    fig_accuracy()
    fig_gen_tk()
    fig_gen_T()
    fig_gen_m1()
    fig_gen_eta()
    fig_digits()


if __name__ == '__main__':
    main()
