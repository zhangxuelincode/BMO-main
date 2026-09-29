# -*- coding: utf-8 -*-
"""隔离诊断：SSGDA 的动量与更新顺序哪个导致 meta loss 下降。"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import run_ms_experiments as R
from bmo_core import generate_simulation_data, make_dataloaders, BMOTask

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class A: pass
a = A(); a.img_size=32; a.latent_dim=16; a.width_g=64; a.width_d=16; a.noise_std=0.1
a.num_copies=20; a.batch_size=64

data = generate_simulation_data(R.FRAMES_DIR, img_size=32, num_copies=20,
                                noise_std=0.1, m1=500, seed=100)
loaders = make_dataloaders(data, batch_size=64, device=device)

def make(seed=100):
    task = R.build_task(a, device)
    R.init_nets(task, seed)
    return task

def clip(params, c=10.0):
    torch.nn.utils.clip_grad_norm_(params, c)

def variant(name, use_momentum=True, order='x_first', T=40):
    task = make()
    it_tr = iter(loaders['train']); it_me = iter(loaders['meta'])
    def sample(it):
        try: return next(it)
        except StopIteration:
            it2 = iter(loaders['train'] if it is it_tr else loaders['meta'])
            return next(it2)
    mW = None
    for t in range(T):
        eta = 1e-3 * 0.95**t
        tr = next(iter(loaders['train'])) if False else None
        if order == 'x_first':
            me = next(iter(loaders['meta'])) if False else None
        # 简化：每次从 loader 取一个 batch
        mb = next(iter(loaders['meta'])) if False else None
        def smp(loader, it_box):
            try: return next(it_box[0])
            except StopIteration:
                it_box[0] = iter(loader); return next(it_box[0])
        itb_tr = [it_tr]; itb_me = [it_me]
        tr_b = smp(loaders['train'], itb_tr)
        me_b = smp(loaders['meta'], itb_me)
        if order == 'x_first':
            # x 先（SSGDA 顺序）
            f = task.upper_loss(me_b)
            gW = torch.autograd.grad(f, list(task.weight_net.parameters()))
            gW = torch.cat([g.reshape(-1) for g in gW])
            with torch.no_grad():
                off = 0
                for p in task.weight_net.parameters():
                    n = p.numel()
                    if use_momentum:
                        p.add_(gW[off:off+n].view_as(p), alpha=-eta)
                    else:
                        p.add_(gW[off:off+n].view_as(p), alpha=-eta)
                    off += n
            gy, gz = task.lower_losses(tr_b)
            gY = torch.autograd.grad(gy, list(task.netG.parameters()))
            gZ = torch.autograd.grad(gz, list(task.netD.parameters()))
            with torch.no_grad():
                for p, g in zip(task.netG.parameters(), gY): p.add_(g, alpha=-eta)
                for p, g in zip(task.netD.parameters(), gZ): p.add_(g, alpha=eta)
        else:
            gy, gz = task.lower_losses(tr_b)
            gY = torch.autograd.grad(gy, list(task.netG.parameters()))
            gZ = torch.autograd.grad(gz, list(task.netD.parameters()))
            with torch.no_grad():
                for p, g in zip(task.netG.parameters(), gY): p.add_(g, alpha=-eta)
                for p, g in zip(task.netD.parameters(), gZ): p.add_(g, alpha=eta)
            f = task.upper_loss(me_b)
            gW = torch.autograd.grad(f, list(task.weight_net.parameters()))
            gW = torch.cat([g.reshape(-1) for g in gW])
            with torch.no_grad():
                off = 0
                for p in task.weight_net.parameters():
                    n = p.numel()
                    p.add_(gW[off:off+n].view_as(p), alpha=-eta)
                    off += n
        if (t+1) % 10 == 0:
            v = task.eval_error(loaders['meta'])
            print(f'[{name}] t={t+1} f_batch={f.item():.4f} val={v:.4f}')

variant('x_first(no mom)', use_momentum=False, order='x_first')
variant('lower_first(no mom)', use_momentum=False, order='lower_first')
