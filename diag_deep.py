# -*- coding: utf-8 -*-
"""深度诊断：跟踪 MS-BMO 的下层 GAN 损失与动量范数。"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import run_ms_experiments as R
from bmo_core import generate_simulation_data, make_dataloaders, MSBMO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64; a.beta_x=0.2; a.beta_w=0.2
a.cg_iters=10; a.damping=1e-3; a.grad_clip=10.0

data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                noise_std=0.1, m1=500, seed=100)
loaders = make_dataloaders(data, batch_size=64, device=device)
task = R.build_task(a, device)
R.init_nets(task, 100)
solver = MSBMO(task, eta_x=1e-3, eta_w=1e-3, eta_x_schedule='exp', eta_x_decay=0.95,
               eta_w_schedule='exp', eta_w_decay=0.95, beta_x=0.2, beta_w=0.2,
               cg_iters=10, damping=1e-3, grad_clip=10.0, seed=100, use_correction=False)

# 手动跑 40 步，打印 g_y / g_z / |wy| / |wz| / f
train_loader, meta_loader = loaders['train'], loaders['meta']
for t in range(40):
    zeta1 = solver._sample(train_loader)
    zeta2 = solver._sample(train_loader)
    xi1 = solver._sample(meta_loader)
    xi2 = solver._sample(meta_loader)
    gy_cur, gz_cur = solver._lower_grads(zeta1)
    f_val, gx, pz = solver._upper_grads(xi1)
    eta = solver.eta_w_sched(t)
    with torch.no_grad():
        solver.wy = gy_cur  # 无校正、无动量的裸版本（诊断）
        solver.wz = gz_cur
        off = 0
        for p in solver.pG:
            n = p.numel(); p.add_(solver.wy[off:off+n].view_as(p), alpha=-eta); off += n
        off = 0
        for p in solver.pD:
            n = p.numel(); p.add_(solver.wz[off:off+n].view_as(p), alpha=eta); off += n
        off = 0
        for p in solver.pW:
            n = p.numel(); p.add_(gx[off:off+n].view_as(p), alpha=-solver.eta_x_sched(t)); off += n
    if (t+1) % 5 == 0:
        print(f't={t+1} f={f_val:.4f} g_y={gy_cur.norm():.3f} g_z={gz_cur.norm():.3f} '
              f'eta={eta:.2e}')
with torch.no_grad():
    print('final val:', task.eval_error(meta_loader), 'train:', task.train_error(train_loader, max_batches=8))
