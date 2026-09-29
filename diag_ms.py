# -*- coding: utf-8 -*-
"""诊断：真实任务上 MS-BMO 与 SSGDA 的收敛轨迹对比（调参用）。

用法： python diag_ms.py --T 100 --eta 1e-3 --beta 0.2 --m1 500 --seed 100
"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import run_ms_experiments as R

ap = argparse.ArgumentParser()
ap.add_argument('--T', type=int, default=100)
ap.add_argument('--eta', type=float, default=1e-3)
ap.add_argument('--beta', type=float, default=0.2)
ap.add_argument('--m1', type=int, default=500)
ap.add_argument('--seed', type=int, default=100)
ap.add_argument('--no_corr', action='store_true', help='关闭超梯度校正（消融）')
cli = ap.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64; a.beta_x=cli.beta; a.beta_w=cli.beta
a.cg_iters=10; a.damping=1e-3; a.grad_clip=10.0

t0=time.time()
import bmo_core
from bmo_core import MSBMO as _MS
_orig_init = _MS.__init__
if cli.no_corr:
    a.use_correction = False
else:
    a.use_correction = True
# run_ms 里不识别 use_correction，这里手动构造
import pandas as pd
from bmo_core import generate_simulation_data, make_dataloaders, BMOTask, StepSizeSchedule

def run_ms_flag(a, device, seed, m1, eta, T, eval_every, use_correction):
    data = generate_simulation_data(R.FRAMES_DIR, img_size=a.img_size,
                                    num_copies=a.num_copies, noise_std=a.noise_std,
                                    m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=a.batch_size, device=device)
    task = R.build_task(a, device)
    R.init_nets(task, seed)
    solver = _MS(task, eta_x=eta, eta_w=eta, eta_x_schedule='exp', eta_x_decay=0.95,
                 eta_w_schedule='exp', eta_w_decay=0.95, beta_x=a.beta_x, beta_w=a.beta_w,
                 cg_iters=a.cg_iters, damping=a.damping, grad_clip=a.grad_clip, seed=seed,
                 use_correction=use_correction)
    hist = solver.run(loaders['train'], loaders['meta'], loaders['test'], T=T, eval_every=eval_every)
    return pd.DataFrame(hist.records), solver

df, solver = run_ms_flag(a, device, cli.seed, cli.m1, cli.eta, cli.T, 10, not cli.no_corr)
print(('MS-BMO(no corr) ' if cli.no_corr else 'MS-BMO  '), 'val :', [f'{v:.4f}' for v in df['val_error']])
print('test:', [f'{v:.4f}' for v in df['test_error']])
print('gap :', [f'{v:.4f}' for v in df['gap']])
if 'grad_norm' in df.columns:
    print('|d| :', [f'{v:.2e}' for v in df['grad_norm'].dropna()])
print(f'({time.time()-t0:.0f}s)')

df2 = R.run_first_order(a, device, cli.seed, 'SSGDA', m1=cli.m1, T=cli.T, eval_every=10)
print('SSGDA   val :', [f'{v:.4f}' for v in df2['val_error']])
print('SSGDA   test:', [f'{v:.4f}' for v in df2['test_error']])
print('SSGDA   gap :', [f'{v:.4f}' for v in df2['gap']])
