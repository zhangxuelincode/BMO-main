# -*- coding: utf-8 -*-
"""微调：CG 阻尼/迭代数与动量对 MS-BMO 收敛的影响。"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import run_ms_experiments as R
from bmo_core import generate_simulation_data, make_dataloaders, MSBMO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64

def ms(cg=10, damp=1e-3, beta=0.2, eta=3e-3, T=150, seed=100, m1=500):
    data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                    noise_std=0.1, m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=64, device=device)
    task = R.build_task(a, device); R.init_nets(task, seed)
    s = MSBMO(task, eta_x=eta, eta_w=eta, eta_x_schedule='fixed',
              eta_w_schedule='fixed', beta_x=beta, beta_w=beta,
              cg_iters=cg, damping=damp, grad_clip=10.0, seed=seed)
    h = s.run(loaders['train'], loaders['meta'], loaders['test'], T=T, eval_every=25)
    cg = s.extra['cg_res']
    return h.records['val_error'], [f'{v:.1e}' for v in cg[::30]]

v, cg = ms(cg=20, damp=1e-2, beta=0.2)
print('cg=20 damp=1e-2 b=.2 :', [f'{x:.4f}' for x in v], 'cg:', cg)
v, cg = ms(cg=10, damp=1e-2, beta=0.5)
print('cg=10 damp=1e-2 b=.5 :', [f'{x:.4f}' for x in v], 'cg:', cg)
v, cg = ms(cg=10, damp=1e-3, beta=0.5, eta=1e-2)
print('cg=10 damp=1e-3 b=.5 eta=1e-2:', [f'{x:.4f}' for x in v], 'cg:', cg)
