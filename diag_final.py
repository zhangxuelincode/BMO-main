# -*- coding: utf-8 -*-
"""对比确认：相同步长下 SSGDA / MS-BMO(无校正) / MS-BMO(完整二阶)。"""
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

def env(seed=100, m1=500):
    data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                    noise_std=0.1, m1=m1, seed=seed)
    return make_dataloaders(data, batch_size=64, device=device)

def ms(run_corrections, eta=3e-3, beta=0.2, T=150, seed=100, m1=500):
    loaders = env(seed, m1); task = R.build_task(a, device); R.init_nets(task, seed)
    s = MSBMO(task, eta_x=eta, eta_w=eta, eta_x_schedule='fixed',
              eta_w_schedule='fixed', beta_x=beta, beta_w=beta,
              cg_iters=10, damping=1e-3, grad_clip=10.0, seed=seed,
              use_correction=run_corrections)
    h = s.run(loaders['train'], loaders['meta'], loaders['test'], T=T, eval_every=25)
    return h.records, s

def ssgda(eta=3e-3, T=150, seed=100, m1=500):
    loaders = env(seed, m1); task = R.build_task(a, device); R.init_nets(task, seed)
    s = SSGDA(task, eta=eta, gamma1=eta, gamma2=eta, eta_schedule='fixed',
              gamma_schedule='fixed', seed=seed)
    h = s.run(loaders['train'], loaders['meta'], loaders['test'], T=T, eval_every=25)
    return h.records

r1, s1 = ms(True)
print('MS-BMO(full)   val:', [f'{v:.4f}' for v in r1['val_error']])
print('               gap:', [f'{v:.4f}' for v in r1['gap']])
r2, s2 = ms(False)
print('MS-BMO(no corr)val:', [f'{v:.4f}' for v in r2['val_error']])
r3 = ssgda()
print('SSGDA          val:', [f'{v:.4f}' for v in r3['val_error']])
print('               gap:', [f'{v:.4f}' for v in r3['gap']])
