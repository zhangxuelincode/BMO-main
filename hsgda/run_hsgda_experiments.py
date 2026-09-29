# -*- coding: utf-8 -*-
"""HSGDA 的收敛与泛化验证实验（对应主文图 1 到图 4 的协议， 针对 new.tex 的 HSGDA）。

实验 E1（收敛与精度验证， 对应 new.tex Theorem thm_conv_hsgda）：
    HSGDA vs SSGDA vs TSGDA-1，T=3000，横轴为外层迭代，
    记录真实超梯度范数平方 ||grad Phi(x^t)||^2（解析可求）与 population 目标值，
    并输出 min_{t<T} ||grad Phi||^2 随 T 的网格曲线用于对照 O(1/sqrt(T)) 理论界。
实验 E2（迭代数 T 与内层步数 K 的影响， 对应主文图 1）：
    HSGDA，K in {1,5,10}，T=300，记录 meta 集（验证）误差、测试误差与泛化 Gap。
实验 E3（meta 集大小 m1 的影响， 对应主文图 2）：
    HSGDA，m1 in {25,50,100,200,400,800}，T=1500，最终 Gap 随 m1 的变化对照 1/m1。
实验 E4（步长 eta 与调度的影响， 对应主文图 3 和图 4）：
    固定 eta 与指数衰减调度（0.95 与 0.85），记录三条误差曲线与最终 Gap。

泛化 Gap 定义与主文一致：Gap = 测试误差 − 验证（meta 集）误差。
问题实例固定（seed=0 生成），各重复实验只变化 meta 集、初始化与随机噪声，
对应理论中固定问题、重抽 D_{m1} 的 on-average argument stability 设定。

用法：
    python run_hsgda_experiments.py --exp probe      # 快速调参探针
    python run_hsgda_experiments.py --exp all        # 全量
    python run_hsgda_experiments.py --exp conv|m1|tk|eta
结果输出 results/hsgda_*.csv 与 npz。
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsgda_core import SyntheticBMO, run_hsgda, run_ssgda, run_tsgda1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 默认步长（smoke test 验证过：确定性跟踪几何收敛，HSGDA 稳定下降）
ETA_DEF = 0.01
GAMMA_DEF = 0.02
ALPHA_DEF = 0.02


def make_prob(**kw):
    return SyntheticBMO(seed=0, **kw)


def rows_to_df(all_rows):
    return pd.DataFrame([r for rows in all_rows for r in rows])


def records_to_rows(rec, extra):
    n = len(rec['step'])
    rows = []
    for i in range(n):
        r = dict(step=rec['step'][i], val=rec['val'][i], test=rec['test'][i],
                 gap=rec['gap'][i], phi_pop=rec['phi_pop'][i])
        r.update(extra)
        rows.append(r)
    return rows


# ---------------- E1: 收敛与精度 ----------------

def exp_conv(args):
    T = args.T_conv
    seeds = range(args.repeats)
    rows_h, rows_s, rows_t = [], [], []
    gphi_h, gphi_s, gphi_t = [], [], []
    for seed in seeds:
        t0 = time.time()
        prob = make_prob()
        prob.make_meta(args.m1, np.random.default_rng(1000 + seed))
        out = run_hsgda(prob, T=T, K=1, eta=ETA_DEF, gamma=GAMMA_DEF,
                        alpha=ALPHA_DEF, seed=seed, eval_every=args.eval_every,
                        track_phi=True, x0=_x0(prob, seed))
        rows_h += records_to_rows(out['records'], dict(solver='HSGDA', seed=seed))
        gphi_h.append(out['gphi2_all'])
        out = run_ssgda(prob, T=T, eta=ETA_DEF, gamma=GAMMA_DEF, seed=seed,
                        eval_every=args.eval_every, track_phi=True,
                        x0=_x0(prob, seed))
        rows_s += records_to_rows(out['records'], dict(solver='SSGDA', seed=seed))
        gphi_s.append(out['gphi2_all'])
        out = run_tsgda1(prob, T=T, K=args.k_tsgda, eta=ETA_DEF,
                         gamma=GAMMA_DEF, seed=seed, eval_every=args.eval_every,
                         track_phi=True, x0=_x0(prob, seed))
        rows_t += records_to_rows(out['records'], dict(solver='TSGDA-1', seed=seed))
        gphi_t.append(out['gphi2_all'])
        print(f'[conv] seed={seed} done ({time.time()-t0:.0f}s)')
    df_h = rows_to_df([rows_h])
    df_s = rows_to_df([rows_s])
    df_t = rows_to_df([rows_t])
    pd.concat([df_h, df_s, df_t], ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, 'hsgda_conv_curves.csv'), index=False)
    np.savez(os.path.join(RESULTS_DIR, 'hsgda_conv_gphi2.npz'),
             hsgda=np.array(gphi_h), ssgda=np.array(gphi_s),
             tsgda=np.array(gphi_t))
    # min_{t<T} 网格（HSGDA，对应定理左边 min_{0<=t<T} E||grad Phi||^2）
    grid = [50, 100, 200, 400, 800, 1600, T]
    rec = []
    for Tv in grid:
        for i, g in enumerate(gphi_h):
            rec.append(dict(T=Tv, seed=i, min_gphi2=float(np.nanmin(g[:Tv]))))
    pd.DataFrame(rec).to_csv(os.path.join(RESULTS_DIR, 'hsgda_conv_min.csv'),
                             index=False)
    # 精度对照表
    tab = []
    for df, name in [(df_h, 'HSGDA'), (df_s, 'SSGDA'), (df_t, 'TSGDA-1')]:
        last = df.groupby('seed').last()
        tab.append(dict(solver=name,
                        phi_pop_mean=last['phi_pop'].mean(),
                        phi_pop_std=last['phi_pop'].std(),
                        val_mean=last['val'].mean(), val_std=last['val'].std(),
                        test_mean=last['test'].mean(), test_std=last['test'].std(),
                        gap_mean=last['gap'].mean(), gap_std=last['gap'].std()))
    pd.DataFrame(tab).to_csv(os.path.join(RESULTS_DIR, 'hsgda_conv_accuracy.csv'),
                             index=False)
    print(pd.DataFrame(tab))
    return gphi_h, gphi_s, gphi_t


def _x0(prob, seed):
    rng = np.random.default_rng(500 + seed)
    return prob.proj_x(rng.standard_normal(prob.dx) * 2.0)


# ---------------- E2: T 与 K 的影响 ----------------

def exp_tk(args):
    from hsgda_core import run_tsgda1
    # (a) K 扫描：T=T_gen，同时对比 HSGDA 与 TSGDA-1（K 的泛化代价）
    rows = []
    for K in args.K_list:
        for seed in range(args.repeats):
            t0 = time.time()
            prob = make_prob()
            prob.make_meta(args.m1_gen, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=args.T_gen, K=K, eta=ETA_DEF,
                            gamma=GAMMA_DEF, alpha=GAMMA_DEF, seed=seed,
                            eval_every=args.eval_every_gen, track_phi=False,
                            x0=_x0(prob, seed))
            rows += records_to_rows(out['records'],
                                    dict(K=K, seed=seed, solver='HSGDA'))
            out = run_tsgda1(prob, T=args.T_gen, K=K, eta=ETA_DEF,
                             gamma=GAMMA_DEF, seed=seed,
                             eval_every=args.eval_every_gen, track_phi=False,
                             x0=_x0(prob, seed))
            rows += records_to_rows(out['records'],
                                    dict(K=K, seed=seed, solver='TSGDA-1'))
            print(f'[tk] K={K} seed={seed} done ({time.time()-t0:.0f}s)')
    df = rows_to_df([rows])
    df.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_tk_detail.csv'), index=False)
    tab = df.groupby(['solver', 'K', 'seed']).last().reset_index() \
            .groupby(['solver', 'K'])[['val', 'test', 'gap']] \
            .agg(['mean', 'std']).reset_index()
    tab.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_tk_final.csv'), index=False)
    print(tab)
    # (b) T 扫描：固定 K=1，取最终误差，观察 Gap 随外层迭代数的增长
    rowsT = []
    for Tv in args.T_list:
        for seed in range(args.repeats):
            prob = make_prob()
            prob.make_meta(args.m1_gen, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=Tv, K=1, eta=ETA_DEF, gamma=GAMMA_DEF,
                            alpha=GAMMA_DEF, seed=seed,
                            eval_every=max(Tv // 5, 1),
                            track_phi=False, x0=_x0(prob, seed))
            rowsT += records_to_rows(out['records'], dict(T=Tv, seed=seed))
            last = out['records']
            print(f'[T] T={Tv} seed={seed} val={last["val"][-1]:.4f} '
                  f'test={last["test"][-1]:.4f} gap={last["gap"][-1]:.4f}')
    dfT = rows_to_df([rowsT])
    dfT.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_T_detail.csv'), index=False)
    tabT = dfT.groupby(['T', 'seed']).last().reset_index() \
              .groupby('T')[['val', 'test', 'gap']].agg(['mean', 'std']).reset_index()
    tabT.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_T_final.csv'), index=False)
    print(tabT)


# ---------------- E3: meta 集大小 m1 的影响 ----------------

def exp_m1(args):
    rows = []
    for m1 in args.m1_list:
        for seed in range(args.repeats_m1):
            t0 = time.time()
            prob = make_prob()
            prob.make_meta(m1, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=args.T_m1, K=1, eta=ETA_DEF, gamma=GAMMA_DEF,
                            alpha=ALPHA_DEF, seed=seed,
                            eval_every=args.eval_every_m1, track_phi=False,
                            x0=_x0(prob, seed))
            rows += records_to_rows(out['records'], dict(m1=m1, seed=seed))
            last = out['records']
            print(f'[m1] m1={m1} seed={seed} val={last["val"][-1]:.4f} '
                  f'test={last["test"][-1]:.4f} gap={last["gap"][-1]:.4f} '
                  f'({time.time()-t0:.0f}s)')
    df = rows_to_df([rows])
    df.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_m1_detail.csv'), index=False)
    tab = df.groupby(['m1', 'seed']).last().reset_index() \
            .groupby('m1')[['val', 'test', 'gap']].agg(['mean', 'std']).reset_index()
    tab.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_m1_final.csv'), index=False)
    print(tab)


# ---------------- E4: 步长与调度的影响 ----------------

SCHEDS = [('fixed', 0.0, 0.005, 'Fixed eta=0.005'),
          ('fixed', 0.0, 0.01, 'Fixed eta=0.01'),
          ('fixed', 0.0, 0.02, 'Fixed eta=0.02'),
          ('exp', 0.95, 0.1, 'Decay 0.95'),
          ('exp', 0.85, 0.1, 'Decay 0.85')]


def exp_eta(args):
    rows = []
    for mode, decay, eta0, name in SCHEDS:
        for seed in range(args.repeats):
            t0 = time.time()
            prob = make_prob()
            prob.make_meta(args.m1_gen, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=args.T_eta, K=1, eta=eta0, gamma=GAMMA_DEF,
                            alpha=ALPHA_DEF, eta_mode=mode, eta_decay=decay,
                            seed=seed, eval_every=args.eval_every_eta,
                            track_phi=False, x0=_x0(prob, seed))
            rows += records_to_rows(out['records'], dict(sched=name, seed=seed))
            last = out['records']
            print(f'[eta] {name} seed={seed} val={last["val"][-1]:.4f} '
                  f'test={last["test"][-1]:.4f} gap={last["gap"][-1]:.4f} '
                  f'({time.time()-t0:.0f}s)')
    df = rows_to_df([rows])
    df.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_eta_detail.csv'), index=False)
    tab = df.groupby(['sched', 'seed']).last().reset_index() \
            .groupby('sched')[['val', 'test', 'gap']].agg(['mean', 'std']).reset_index()
    tab.to_csv(os.path.join(RESULTS_DIR, 'hsgda_gen_eta_final.csv'), index=False)
    print(tab)


# ---------------- 探针：快速确认 gap 可见性与步长稳定性 ----------------

def exp_probe(args):
    print('--- probe 1: 各步长下 HSGDA 短程稳定性 (T=600, m1=25) ---')
    for eta0 in (0.005, 0.01, 0.02, 0.05):
        prob = make_prob()
        prob.make_meta(25, np.random.default_rng(1000))
        out = run_hsgda(prob, T=600, K=1, eta=eta0, gamma=GAMMA_DEF,
                        alpha=ALPHA_DEF, eta_mode='fixed', seed=0,
                        eval_every=200, track_phi=True, x0=_x0(prob, 0))
        rec = out['records']
        print(f'  fixed eta={eta0}: phi {rec["phi_pop"][0]:.4f}->'
              f'{rec["phi_pop"][-1]:.4f} min|g|^2={out["min_gphi2"]:.2e} '
              f'val {rec["val"][-1]:.4f} test {rec["test"][-1]:.4f} '
              f'gap {rec["gap"][-1]:.4f}')
    print('--- probe 2: gap 可见性（m1 的影响, 6 seeds, T=2000） ---')
    for m1 in (25, 50, 100, 200, 400):
        gaps = []
        for seed in range(6):
            prob = make_prob()
            prob.make_meta(m1, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=2000, K=1, eta=ETA_DEF, gamma=GAMMA_DEF,
                            alpha=ALPHA_DEF, seed=seed, eval_every=1000,
                            track_phi=False, x0=_x0(prob, seed))
            rec = out['records']
            gaps.append((rec['val'][-1], rec['test'][-1], rec['gap'][-1]))
        g = np.array(gaps)
        print(f'  m1={m1}: val={g[:,0].mean():.4f}+-{g[:,0].std():.4f} '
              f'test={g[:,1].mean():.4f}+-{g[:,1].std():.4f} '
              f'gap={g[:,2].mean():.5f}+-{g[:,2].std():.5f}')
    print('--- probe 3: f 的逐样本波动幅度（决定 gap 信噪比） ---')
    prob = make_prob()
    prob.make_meta(25, np.random.default_rng(1000))
    x = _x0(prob, 0)
    w = prob.saddle_solve(x)
    fv = [prob.f_val(x, w[:prob.dy], w[prob.dy:], xi) for xi in prob.xi_pop[:5000]]
    print(f'  f std over xi = {np.std(fv):.4f}, mean = {np.mean(fv):.4f}')
    print('--- probe 4: K 的影响 (T=600, m1=25, 6 seeds) ---')
    for K in (1, 10):
        gaps = []
        for seed in range(6):
            prob = make_prob()
            prob.make_meta(25, np.random.default_rng(1000 + seed))
            out = run_hsgda(prob, T=600, K=K, eta=ETA_DEF, gamma=GAMMA_DEF,
                            alpha=ALPHA_DEF, seed=seed, eval_every=600,
                            track_phi=False, x0=_x0(prob, seed))
            rec = out['records']
            gaps.append((rec['val'][-1], rec['test'][-1], rec['gap'][-1]))
        g = np.array(gaps)
        print(f'  K={K}: val={g[:,0].mean():.4f} test={g[:,1].mean():.4f} '
              f'gap={g[:,2].mean():.5f}+-{g[:,2].std():.5f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', choices=['probe', 'conv', 'tk', 'm1', 'eta', 'all'],
                   default='all')
    p.add_argument('--m1', type=int, default=200)
    p.add_argument('--T_conv', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--repeats', type=int, default=16)
    p.add_argument('--k_tsgda', type=int, default=10)
    p.add_argument('--T_gen', type=int, default=2000)
    p.add_argument('--eval_every_gen', type=int, default=50)
    p.add_argument('--K_list', type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--T_list', type=int, nargs='+', default=[200, 800, 2000, 4000])
    p.add_argument('--m1_gen', type=int, default=25)
    p.add_argument('--T_eta', type=int, default=800)
    p.add_argument('--eval_every_eta', type=int, default=10)
    p.add_argument('--m1_list', type=int, nargs='+',
                   default=[25, 50, 100, 200, 400])
    p.add_argument('--T_m1', type=int, default=2000)
    p.add_argument('--eval_every_m1', type=int, default=20)
    p.add_argument('--repeats_m1', type=int, default=16)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    exps = ['probe', 'conv', 'tk', 'm1', 'eta'] if args.exp == 'all' else [args.exp]
    for e in exps:
        if e == 'probe':
            exp_probe(args)
        elif e == 'conv':
            exp_conv(args)
        elif e == 'tk':
            exp_tk(args)
        elif e == 'm1':
            exp_m1(args)
        elif e == 'eta':
            exp_eta(args)
    print(f'all done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
