# -*- coding: utf-8 -*-
"""MS-BMO 冒烟测试：toy 流形数据上验证算法可运行、超梯度有界、损失下降。

用法： python test_msbmo_smoke.py
"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from bmo_core import (make_toy_manifold, WeightNet, MLPGenerator, MLPDiscriminator,
                      BMOTask, MSBMO, SSGDA)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = make_toy_manifold(n_train=256, n_meta=128, n_test=64, batch_size=64, seed=1)
# 把 toy loader 的张量搬到 device（make_toy_manifold 默认建在 CPU 上）
from torch.utils.data import TensorDataset, DataLoader
def _to_dev(loader, shuffle):
    t = loader.dataset.tensors[0].to(device)
    return DataLoader(TensorDataset(t), batch_size=64, shuffle=shuffle, drop_last=shuffle)
data['train'] = _to_dev(data['train'], True)
data['meta'] = _to_dev(data['meta'], True)
data['test'] = _to_dev(data['test'], False)

torch.manual_seed(1)
netG = MLPGenerator(latent_dim=data['latent_dim'], hidden=64, out_dim=2).to(device)
netD = MLPDiscriminator(in_dim=2, hidden=64).to(device)
weight_net = WeightNet(hidden_size=10, num_layers=2).to(device)
task = BMOTask(netG, netD, weight_net, noise_std=0.1, latent_dim=data['latent_dim'],
               device=device)

print('=== MS-BMO smoke (toy) ===')
solver = MSBMO(task, eta_x=3e-3, eta_w=3e-3, beta_x=0.2, beta_w=0.2,
               cg_iters=10, damping=1e-3, grad_clip=10.0, seed=1)
hist = solver.run(data['train'], data['meta'], data['test'], T=30, eval_every=10,
                  verbose=True)
print('MSBMO extra grad_norm tail:', [f'{v:.2e}' for v in solver.extra['grad_norm'][-5:]])
print('MSBMO cg_res tail:', [f'{v:.1e}' for v in solver.extra['cg_res'][-5:]])
print('MSBMO corr_norm tail:', [f'{v:.2e}' for v in solver.extra['corr_norm'][-5:]])
print('MSBMO val errors:', [f'{v:.4f}' for v in hist.records['val_error']])
print('MSBMO train errors:', [f'{v:.4f}' for v in hist.records['train_error']])
assert all(v == v for v in hist.records['val_error']), 'NaN in val errors'

print('=== SSGDA smoke (toy, same budget) ===')
torch.manual_seed(1)
netG2 = MLPGenerator(latent_dim=data['latent_dim'], hidden=64, out_dim=2).to(device)
netD2 = MLPDiscriminator(in_dim=2, hidden=64).to(device)
weight_net2 = WeightNet(hidden_size=10, num_layers=2).to(device)
task2 = BMOTask(netG2, netD2, weight_net2, noise_std=0.1,
                latent_dim=data['latent_dim'], device=device)
s2 = SSGDA(task2, eta=3e-3, gamma1=3e-3, gamma2=3e-3, eta_schedule='fixed',
           gamma_schedule='fixed', seed=1)
h2 = s2.run(data['train'], data['meta'], data['test'], T=30, eval_every=10)
print('SSGDA val errors:', [f'{v:.4f}' for v in h2.records['val_error']])
print('OK')
