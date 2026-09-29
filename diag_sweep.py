# -*- coding: utf-8 -*-
"""MS-BMO 超参扫描（收敛速度对比 SSGDA）。"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import run_ms_experiments as R
from bmo_core import generate_simulation_data, make_dataloaders, MSBMO, SSGDA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64; a.cg_iters=10; a.damping=1e-3; a.grad_clip=10.0
a.beta_x=0.2; a.beta_w=0.2

def run(eta, sched_x, sched_w, beta, T=100, seed=100):
    data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                    noise_std=0.1, m1=500, seed=seed)
    loaders = make_dataloaders(data, batch_size=64, device=device)
    task = R.build_task(a, device)
    R.init_nets(task, seed)
    s = MSBMO(task, eta_x=eta, eta_w=eta, eta_x_schedule=sched_x, eta_x_decay=0.95,
              eta_w_schedule=sched_w, eta_w_decay=0.95, beta_x=beta, beta_w=beta,
              cg_iters=10, damping=1e-3, grad_clip=10.0, seed=seed)
    h = s.run(loaders['train'], loaders['meta'], loaders['test'], T=T, eval_every=20)
    return h.records['val_error']

print('both-fixed b=0.2 :', [f'{v:.4f}' for v in run(1e-3, 'fixed', 'fixed', 0.2)])
print('both-fixed b=0.5 :', [f'{v:.4f}' for v in run(1e-3, 'fixed', 'fixed', 0.5)])
print('w-fixed x-exp b.2:', [f'{v:.4f}' for v in run(1e-3, 'exp', 'fixed', 0.2)])
print('both-fixed 3e-3  :', [f'{v:.4f}' for v in run(3e-3, 'fixed', 'fixed', 0.2)])
print('both-fixed 3e-3b5:', [f'{v:.4f}' for v in run(3e-3, 'fixed', 'fixed', 0.5)])
