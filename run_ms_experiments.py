# -*- coding: utf-8 -*-
"""MS-BMO 二阶算法的收敛与泛化验证实验（对应 new2.tex 第 7 节）。

实验 C（收敛验证，对应 new2.tex 图 7.1）：
    MS-BMO vs SSGDA vs TSGDA-1，外层迭代为横轴，
    记录元集误差、训练误差与超梯度范数 ‖d‖（ε-驻点性证据）。
实验 M（元集大小 m1，对应主文图 1 的协议）：
    MS-BMO，m1 ∈ {100, 500, 1000}，横轴为外层迭代，
    记录验证误差、测试误差、泛化 GAP（test − val）与未加权 GAP。
实验 T（外层迭代数 T，对应主文图 1 的 T 影响结论）：
    MS-BMO，T ∈ {25, 50, 100, 200}，取最终误差，观察过拟合与 GAP 增长。
实验 L（学习率 η，对应主文图 3）：
    MS-BMO，η ∈ {3e-4, 1e-3, 3e-3}，横轴为外层迭代。
实验 S（步长调度，对应主文图 4）：
    MS-BMO，Fixed / Decaying(0.95) / Faster Decaying(0.85)。

用法：
    python run_ms_experiments.py --exp conv --repeats 3
    python run_ms_experiments.py --exp all --quick     # 快速冒烟
    python run_ms_experiments.py --exp all             # 全量
结果输出 results/msbmo_*.csv 与 ../latex/images/msbmo_*.pdf（同时存 PNG）。
"""
import os
import sys
import argparse
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bmo_core import (generate_simulation_data, make_dataloaders,
                      WeightNet, ConvGenerator, ConvDiscriminator,
                      BMOTask, MSBMO, SSGDA, TSGDA1)
from bmo_core.solver_ms import plugin_hypergrad_norm

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'Create_real_data', 'Chaplin', 'frames')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
IMG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'latex', 'images'))


def build_task(args, device):
    netG = ConvGenerator(latent_dim=args.latent_dim, img_size=args.img_size,
                         width=args.width_g).to(device)
    netD = ConvDiscriminator(img_size=args.img_size, width=args.width_d).to(device)
    weight_net = WeightNet(hidden_size=10, num_layers=2).to(device)
    task = BMOTask(netG, netD, weight_net, noise_std=args.noise_std,
                   latent_dim=args.latent_dim, device=device)
    return task


def init_nets(task, seed):
    torch.manual_seed(seed)
    for m in (task.netG, task.netD, task.weight_net):
        for p in m.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
            else:
                torch.nn.init.zeros_(p)


def run_ms(args, device, seed, m1, eta=1e-3, eta_mode='exp', eta_decay=0.95,
           T=100, eval_every=5, probe=None):
    data = generate_simulation_data(FRAMES_DIR, img_size=args.img_size,
                                    num_copies=args.num_copies,
                                    noise_std=args.noise_std,
                                    m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=args.batch_size, device=device)
    task = build_task(args, device)
    init_nets(task, seed)
    solver = MSBMO(task, eta_x=eta, eta_w=eta,
                   eta_x_schedule=eta_mode, eta_x_decay=eta_decay,
                   eta_w_schedule=eta_mode, eta_w_decay=eta_decay,
                   beta_x=args.beta_x, beta_w=args.beta_w,
                   cg_iters=args.cg_iters, damping=args.damping,
                   grad_clip=args.grad_clip, seed=seed)
    hist = solver.run(loaders['train'], loaders['meta'], loaders['test'],
                      T=T, eval_every=eval_every, probe=probe)
    df = pd.DataFrame(hist.records)
    step_map = dict(zip(solver.extra['step'], range(len(solver.extra['step']))))
    for k in ('grad_norm', 'cg_res', 'corr_norm'):
        df[k] = [solver.extra[k][step_map[s]] if s in step_map else np.nan
                 for s in df['step']]
    if 'hyper_norm' in solver.extra:
        hn = [np.nan] + solver.extra['hyper_norm']
        df['hyper_norm'] = hn[:len(df)]
    df['seed'] = seed
    df['m1'] = m1
    df['eta'] = eta
    df['eta_mode'] = eta_mode
    df['solver'] = 'MS-BMO'
    return df


class ProbedSSGDA(SSGDA):
    """带探针钩子的 SSGDA（在评估点计算超梯度范数等指标）。"""

    def run_with_probe(self, train_loader, meta_loader, test_loader, T=200,
                       eval_every=10, probe=None):
        self.evaluate(train_loader, meta_loader, test_loader, 0)
        for t in range(self.t, self.t + T):
            meta_batch = self._sample(meta_loader)
            train_batch = self._sample(train_loader)
            self._update_upper(meta_batch, t)
            self._update_lower_yz(train_batch, t)
            if (t + 1) % eval_every == 0:
                if probe is not None:
                    self.extra_probe.append(probe(self.task))
                self.evaluate(train_loader, meta_loader, test_loader, t + 1)
        self.t += T
        return self.history


class ProbedTSGDA1(TSGDA1):
    """带探针钩子的 TSGDA-1：每个外层步结束后评估（对齐外层迭代横轴）。"""

    def run_with_probe(self, train_loader, meta_loader, test_loader, T=50, K=50,
                       eval_every=10, probe=None):
        self.evaluate(train_loader, meta_loader, test_loader, 0)
        for t in range(self.t, self.t + T):
            meta_batch = self._sample(meta_loader)
            self._update_upper(meta_batch, t)
            for k in range(K):
                train_batch = self._sample(train_loader)
                self._update_lower_yz(train_batch, k)
            if (t + 1) % eval_every == 0:
                if probe is not None:
                    self.extra_probe.append(probe(self.task))
                self.evaluate(train_loader, meta_loader, test_loader, t + 1)
        self.t += T
        return self.history


def run_first_order(args, device, seed, solver_name, m1, T, eval_every=5, K=1,
                    eta=1e-3, probe=None):
    data = generate_simulation_data(FRAMES_DIR, img_size=args.img_size,
                                    num_copies=args.num_copies,
                                    noise_std=args.noise_std,
                                    m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=args.batch_size, device=device)
    task = build_task(args, device)
    init_nets(task, seed)
    if solver_name == 'SSGDA':
        solver = ProbedSSGDA(task, eta=eta, gamma1=eta, gamma2=eta,
                             eta_schedule='fixed',
                             gamma_schedule='fixed', seed=seed)
        solver.extra_probe = []
        hist = solver.run_with_probe(loaders['train'], loaders['meta'], loaders['test'],
                                     T=T, eval_every=eval_every, probe=probe)
    else:
        solver = ProbedTSGDA1(task, eta=eta, gamma1=eta, gamma2=eta,
                              eta_schedule='fixed', eta_decay=0.95,
                              gamma_schedule='fixed', seed=seed)
        solver.extra_probe = []
        hist = solver.run_with_probe(loaders['train'], loaders['meta'], loaders['test'],
                                     T=T, K=K, eval_every=eval_every, probe=probe)
    df = pd.DataFrame(hist.records)
    if probe is not None:
        df['hyper_norm'] = [np.nan] + solver.extra_probe
    df['seed'] = seed
    df['m1'] = m1
    df['solver'] = solver_name
    return df


def make_probe(args, loaders):
    """返回 probe(task)：用多个 batch 估计 ‖𝒢(x,y,z)‖。"""
    it_meta = [iter(loaders['meta'])]
    it_train = [iter(loaders['train'])]

    def _batches(it_box, loader, n):
        out = []
        while len(out) < n:
            try:
                out.append(next(it_box[0]))
            except StopIteration:
                it_box[0] = iter(loader)
        return out

    def probe(task):
        mbs = _batches(it_meta, loaders['meta'], args.probe_micro)
        tbs = _batches(it_train, loaders['train'], args.probe_micro)
        mb = torch.cat([b[0] if isinstance(b, (tuple, list)) else b for b in mbs])
        tb = torch.cat([b[0] if isinstance(b, (tuple, list)) else b for b in tbs])
        return plugin_hypergrad_norm(task, mb, tb,
                                     cg_iters=max(args.cg_iters, 15),
                                     damping=args.probe_damping)
    return probe


def agg_curves(details, keys):
    df = pd.concat(details, ignore_index=True)
    cols = [c for c in ['train_error', 'val_error', 'test_error', 'gap',
                        'gap_raw', 'grad_norm'] if c in df.columns]
    summary = df.groupby(keys + ['step'])[cols].agg(['mean', 'std']).reset_index()
    return df, summary


def plot_curves(summary, xkey, curve_key, metrics, out_base, title, xlabel):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, (metric, label) in zip(np.atleast_1d(axes), metrics):
        for cv in sorted(summary[curve_key].unique()):
            sub = summary[summary[curve_key] == cv].sort_values('step')
            mean, std = sub[(metric, 'mean')], sub[(metric, 'std')].fillna(0)
            ax.plot(sub['step'], mean, marker='o', markersize=3, label=str(cv))
            ax.fill_between(sub['step'], mean - std, mean + std, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_base + '.pdf')
    fig.savefig(out_base + '.png', dpi=150)
    plt.close(fig)


def final_table(details, sweep_key):
    df = pd.concat(details, ignore_index=True)
    tab = df.groupby([sweep_key, 'seed']).last().reset_index()
    tab = tab.groupby(sweep_key)[['train_error', 'val_error', 'test_error', 'gap']] \
             .agg(['mean', 'std']).reset_index()
    return tab


def plot_final(tab, sweep_key, metrics, out_base, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, (metric, label) in zip(np.atleast_1d(axes), metrics):
        mean, std = tab[(metric, 'mean')], tab[(metric, 'std')].fillna(0)
        ax.errorbar(tab[sweep_key], mean, yerr=std, marker='o', capsize=3)
        ax.set_xlabel(sweep_key)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_base + '.pdf')
    fig.savefig(out_base + '.png', dpi=150)
    plt.close(fig)


def exp_conv(args, device):
    details_ms, details_ssgda, details_tsgda = [], [], []
    for r in range(args.repeats):
        seed = 100 + r
        t0 = time.time()
        # 先构造数据以获得 probe 所需的 loader（与 run_ms 内部一致）
        data = generate_simulation_data(FRAMES_DIR, img_size=args.img_size,
                                        num_copies=args.num_copies,
                                        noise_std=args.noise_std, m1=args.m1, seed=seed)
        loaders = make_dataloaders(data, batch_size=args.batch_size, device=device)
        probe = make_probe(args, loaders)
        details_ms.append(run_ms(args, device, seed, m1=args.m1, eta=args.eta,
                                 eta_mode='fixed', T=args.T,
                                 eval_every=args.eval_every, probe=probe))
        details_ssgda.append(run_first_order(args, device, seed, 'SSGDA',
                                             m1=args.m1, T=args.T,
                                             eval_every=args.eval_every, eta=args.eta,
                                             probe=probe))
        details_tsgda.append(run_first_order(args, device, seed, 'TSGDA-1',
                                             m1=args.m1, T=max(args.T // 2, 10),
                                             eval_every=args.eval_every, eta=args.eta,
                                             K=args.k_tsgda))
        print(f'[conv] seed={seed} done ({time.time()-t0:.0f}s)')
    df_ms = pd.concat(details_ms, ignore_index=True)
    df_ssgda = pd.concat(details_ssgda, ignore_index=True)
    df_tsgda = pd.concat(details_tsgda, ignore_index=True)
    df_ms.to_csv(os.path.join(RESULTS_DIR, 'msbmo_conv_detail.csv'), index=False)
    df_ssgda.to_csv(os.path.join(RESULTS_DIR, 'msbmo_conv_ssgda_detail.csv'), index=False)
    df_tsgda.to_csv(os.path.join(RESULTS_DIR, 'msbmo_conv_tsgda_detail.csv'), index=False)

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.6))
    for df, name, color in [(df_ms, 'MS-BMO', 'tab:red'),
                            (df_ssgda, 'SSGDA', 'tab:blue'),
                            (df_tsgda, 'TSGDA-1', 'tab:green')]:
        g = df.groupby('step')[['val_error', 'train_error', 'gap']].agg(['mean', 'std'])
        axes[0].plot(g.index, g[('val_error', 'mean')], label=name, color=color)
        axes[0].fill_between(g.index, g[('val_error', 'mean')] - g[('val_error', 'std')],
                             g[('val_error', 'mean')] + g[('val_error', 'std')], alpha=0.15, color=color)
        axes[1].plot(g.index, g[('train_error', 'mean')], label=name, color=color)
        axes[1].fill_between(g.index, g[('train_error', 'mean')] - g[('train_error', 'std')],
                             g[('train_error', 'mean')] + g[('train_error', 'std')], alpha=0.15, color=color)
        axes[2].plot(g.index, g[('gap', 'mean')], label=name, color=color)
        axes[2].fill_between(g.index, g[('gap', 'mean')] - g[('gap', 'std')],
                             g[('gap', 'mean')] + g[('gap', 'std')], alpha=0.15, color=color)
    gn = df_ms.groupby('step')[['grad_norm']].agg(['mean', 'std'])
    axes[3].semilogy(gn.index, gn[('grad_norm', 'mean')], color='tab:red', label='MS-BMO')
    axes[3].fill_between(gn.index, (gn[('grad_norm', 'mean')] - gn[('grad_norm', 'std')]).clip(lower=1e-8),
                         gn[('grad_norm', 'mean')] + gn[('grad_norm', 'std')], alpha=0.2, color='tab:red')
    for df, name, color in [(df_ms, 'MS-BMO', 'tab:red'), (df_ssgda, 'SSGDA', 'tab:blue'),
                            (df_tsgda, 'TSGDA-1', 'tab:green')]:
        if 'hyper_norm' in df.columns:
            gp = df.dropna(subset=['hyper_norm']).groupby('step')[['hyper_norm']].agg(['mean', 'std'])
            axes[4].semilogy(gp.index, gp[('hyper_norm', 'mean')], label=name, color=color, marker='o', markersize=3)
    for ax, lb in zip(axes, ['Errors in Meta Set', 'Errors in Training Set',
                             'Generalization Gap', 'Momentum Norm of MS-BMO',
                             'Plug-in Hyper-gradient Norm']):
        ax.set_xlabel('Outer Iteration t')
        ax.set_ylabel(lb)
        ax.set_title(lb)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[4].legend()
    fig.tight_layout()
    base = os.path.join(IMG_DIR, 'msbmo_conv')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=150)
    plt.close(fig)
    print('saved', base)


def exp_m1(args, device):
    details = []
    for m1 in args.m1_list:
        for r in range(args.repeats):
            seed = 100 + r
            t0 = time.time()
            df = run_ms(args, device, seed, m1=m1, eta=args.eta, eta_mode='fixed',
                        T=args.T, eval_every=args.eval_every)
            details.append(df)
            last = df.iloc[-1]
            print(f'[m1] m1={m1} seed={seed} val={last["val_error"]:.4f} '
                  f'test={last["test_error"]:.4f} gap={last["gap"]:.4f} ({time.time()-t0:.0f}s)')
    df, summary = agg_curves(details, ['m1'])
    df.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_m1_detail.csv'), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_m1_summary.csv'), index=False)
    plot_curves(summary, 'step', 'm1',
                [('val_error', 'Errors in Meta Set'), ('test_error', 'Errors in Test Set'),
                 ('gap', 'Generalization Gap'), ('gap_raw', 'Generalization Gap (raw)')],
                os.path.join(IMG_DIR, 'msbmo_gen_m1'),
                f'MS-BMO: effect of the meta-set size m1 (T={args.T})', 'Outer Iteration t')
    tab = final_table(details, 'm1')
    tab.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_m1_final.csv'), index=False)
    print(tab)


def exp_T(args, device):
    details = []
    for Tv in args.T_list:
        for r in range(args.repeats):
            seed = 100 + r
            t0 = time.time()
            df = run_ms(args, device, seed, m1=args.m1, eta=args.eta, eta_mode='fixed',
                        T=Tv, eval_every=args.eval_every)
            df['T'] = Tv
            details.append(df)
            last = df.iloc[-1]
            print(f'[T] T={Tv} seed={seed} val={last["val_error"]:.4f} '
                  f'test={last["test_error"]:.4f} gap={last["gap"]:.4f} ({time.time()-t0:.0f}s)')
    df = pd.concat(details, ignore_index=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_T_detail.csv'), index=False)
    tab = final_table(details, 'T')
    tab.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_T_final.csv'), index=False)
    plot_final(tab, 'T',
               [('val_error', 'Final Meta Error'), ('test_error', 'Final Test Error'),
                ('gap', 'Final Generalization Gap')],
               os.path.join(IMG_DIR, 'msbmo_gen_T'),
               f'MS-BMO: effect of the number of iterations T (m1={args.m1})')
    print(tab)


def exp_lr(args, device):
    details = []
    for eta in args.eta_list:
        for r in range(args.repeats):
            seed = 100 + r
            t0 = time.time()
            df = run_ms(args, device, seed, m1=args.m1, eta=eta, eta_mode='fixed',
                        eta_decay=0.95, T=args.T, eval_every=args.eval_every)
            df['eta_name'] = f'eta={eta:g}'
            details.append(df)
            last = df.iloc[-1]
            print(f'[lr] eta={eta:g} seed={seed} val={last["val_error"]:.4f} '
                  f'test={last["test_error"]:.4f} gap={last["gap"]:.4f} ({time.time()-t0:.0f}s)')
    df, summary = agg_curves(details, ['eta_name'])
    df.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_lr_detail.csv'), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_lr_summary.csv'), index=False)
    plot_curves(summary, 'step', 'eta_name',
                [('val_error', 'Errors in Meta Set'), ('test_error', 'Errors in Test Set'),
                 ('gap', 'Generalization Gap')],
                os.path.join(IMG_DIR, 'msbmo_gen_lr'),
                f'MS-BMO: effect of the step size eta (m1={args.m1})', 'Outer Iteration t')
    tab = final_table(details, 'eta_name')
    tab.to_csv(os.path.join(RESULTS_DIR, 'msbmo_gen_lr_final.csv'), index=False)
    print(tab)


def exp_sched(args, device):
    details = []
    scheds = [('fixed', 0.95, 'Fixed'), ('exp', 0.95, 'Decaying'), ('exp', 0.85, 'Faster Decaying')]
    for mode, decay, name in scheds:
        for r in range(args.repeats):
            seed = 100 + r
            t0 = time.time()
            df = run_ms(args, device, seed, m1=args.m1, eta=args.eta, eta_mode=mode,
                        eta_decay=decay, T=args.T, eval_every=args.eval_every)
            df['eta_name'] = name
            details.append(df)
            last = df.iloc[-1]
            print(f'[sched] {name} seed={seed} val={last["val_error"]:.4f} '
                  f'test={last["test_error"]:.4f} gap={last["gap"]:.4f} ({time.time()-t0:.0f}s)')
    df, summary = agg_curves(details, ['eta_name'])
    df.to_csv(os.path.join(RESULTS_DIR, 'msbmo_sched_detail.csv'), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, 'msbmo_sched_summary.csv'), index=False)
    plot_curves(summary, 'step', 'eta_name',
                [('val_error', 'Errors in Meta Set'), ('gap', 'Generalization Gap')],
                os.path.join(IMG_DIR, 'msbmo_sched'),
                f'MS-BMO: effect of step-size schedules (m1={args.m1})', 'Outer Iteration t')
    tab = final_table(details, 'eta_name')
    tab.to_csv(os.path.join(RESULTS_DIR, 'msbmo_sched_final.csv'), index=False)
    print(tab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', choices=['conv', 'm1', 'T', 'lr', 'sched', 'all'], default='all')
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--width_g', type=int, default=64)
    p.add_argument('--width_d', type=int, default=16)
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--num_copies', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--m1', type=int, default=500)
    p.add_argument('--T', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--beta_x', type=float, default=0.5)
    p.add_argument('--beta_w', type=float, default=0.5)
    p.add_argument('--cg_iters', type=int, default=10)
    p.add_argument('--damping', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--eta', type=float, default=3e-3, help='MS-BMO 与基线的统一固定步长')
    p.add_argument('--probe_micro', type=int, default=4, help='超梯度探针的 micro-batch 数')
    p.add_argument('--probe_damping', type=float, default=1e-2, help='探针 CG 的阻尼')
    p.add_argument('--k_tsgda', type=int, default=50)
    p.add_argument('--m1_list', type=int, nargs='+', default=[100, 500, 1000])
    p.add_argument('--T_list', type=int, nargs='+', default=[25, 50, 100, 200])
    p.add_argument('--eta_list', type=float, nargs='+', default=[3e-4, 1e-3, 3e-3])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.T = 30
        args.repeats = 1
        args.eval_every = 10
        args.m1_list = [200]
        args.T_list = [15, 30]
        args.eta_list = [1e-3]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    t0 = time.time()
    print(f'device={device} exp={args.exp} T={args.T} repeats={args.repeats}')

    exps = ['conv', 'm1', 'T', 'lr', 'sched'] if args.exp == 'all' else [args.exp]
    for e in exps:
        print(f'===== exp {e} =====')
        dict(conv=exp_conv, m1=exp_m1, T=exp_T, lr=exp_lr, sched=exp_sched)[e](args, device)
    print(f'all done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
