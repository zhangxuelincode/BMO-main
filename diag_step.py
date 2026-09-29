# -*- coding: utf-8 -*-
"""逐步对比：真 SSGDA（每步评估） vs 疑似等价的手写循环。"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import run_ms_experiments as R
from bmo_core import generate_simulation_data, make_dataloaders, BMOTask, SSGDA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64

def make_env(seed=100):
    data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                    noise_std=0.1, m1=500, seed=seed)
    loaders = make_dataloaders(data, batch_size=64, device=device)
    task = R.build_task(a, device)
    R.init_nets(task, seed)
    return loaders, task

print('===== true SSGDA, eval every step =====')
loaders, task = make_env()
s = SSGDA(task, eta=1e-3, gamma1=1e-3, gamma2=1e-3, eta_schedule='exp',
          eta_decay=0.95, gamma_schedule='fixed', seed=100)
s.evaluate(loaders['train'], loaders['meta'], loaders['test'], 0)
for t in range(10):
    mb = s._sample(loaders['meta']); trb = s._sample(loaders['train'])
    f_val, eta = s._update_upper(mb, t)
    s._update_lower_yz(trb, t)
    tr, v, te = s.evaluate(loaders['train'], loaders['meta'], loaders['test'], t+1)
    print(f't={t+1} f_batch={f_val:.4f} val={v:.4f} eta={eta:.2e}')
