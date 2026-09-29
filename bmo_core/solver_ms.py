# -*- coding: utf-8 -*-
"""MS-BMO：二阶随机动量双层极小极大算法（new2.tex 算法 1）的实现与收敛验证。

对应 new2.tex 中的算法 1（MS-BMO），求解论文公式 (1)：
    min_x  E_ξ f(x, y*(x), z*(x); ξ)
    s.t.   (y*(x), z*(x)) ∈ arg min_y arg max_z E_ζ g(x, y, z; ζ)

变量与本仓库一致：x = 上层加权网络 WeightNet，y = 生成器 G（下层极小），
z = 判别器 D（下层极大）。

每次外层迭代 t（严格对应算法 1 的行 2-16）：

  1. 采样 ζ_t^(1), ζ_t^(2) ~ D_m2，ξ_t^(1), ξ_t^(2) ~ D_m1。

  2. 下层 STORM 同样本校正动量（行 4-6，Cutkosky–Orabona 形式：
     上一评估项同样乘 (1−β)，保证动量不动点为 ∇g）：
       ŵ^y_{t+1} = ∇_y g(w_t; ζ1) − (1−β_w)·∇_y g(w_{t-1}; ζ1) + (1−β_w) ŵ^y_t
       ŵ^z_{t+1} = ∇_z g(w_t; ζ1) − (1−β_w)·∇_z g(w_{t-1}; ζ1) + (1−β_w) ŵ^z_t
       y_{t+1} = y_t − η_w ŵ^y_{t+1}          z_{t+1} = z_t + η_w ŵ^z_{t+1}

  3. 鞍点逆估计（行 9，新鲜 Hessian oracle）：本仓库实现的下层目标
     g_y 与 g_z 分别前向（D 对 g_y 冻结、G 对 g_z detach，见 bmo_task.lower_losses），
     有效下层目标在 (y,z) 上可分，故 M = diag(∇²_yy g_y, ∇²_zz g_z)，M⁻¹ 的 z 块为
     (∇²_zz g_z)⁻¹ = −(−∇²_zz g_z)⁻¹ ≺ 0。按 new2.tex 引理 2.1 的分块实例化，
     z 块先解良定的正定系统再取负号：
       (−∇²_zz g_z + λI) ĉ = P_z,   P = ∇_{(y,z)} f(x_t, y_t, z_t; ξ2),
       û_z = −ĉ  ≈ (∇²_zz g_z)⁻¹ P_z   （即 M⁻¹P 的 z 块）
     阻尼 λ 与 CG 截断共同构成理论中的估计偏差 ε_H。
     （本任务 f 不依赖 y，P_y = 0，故 y 块跳过求解，u_y = 0。）

  4. 完整超梯度 oracle 与上层 STORM 校正（行 10-16，同样本、同一 u）：
       J^T u = ∇_x ⟨u, ∇_{(y,z)} g⟩        （权重网络对 x 可微，create_graph）
       Õ_t        = ∇_x f(w_t; ξ1) − J(w_t)^T u
       Õ_{t-1}^{(r)} = ∇_x f(w_{t-1}; ξ1) − J(w_{t-1})^T u
       d_{t+1} = Õ_t − (1−β_x)·Õ_{t-1}^{(r)} + (1−β_x) d_t
       x_{t+1} = x_t − η_x d_{t+1}

工程保护（满足有界 oracle 假设）：
- 各动量向量按 grad_clip 做范数裁剪；超梯度校正与 CG 解按 corr_clip 裁剪；
- CG 相对残差容忍 cg_tol，截断 cg_iters 步。
"""
import copy
import torch
import torch.nn.functional as F

from .bmo_task import BMOTask, MetricHistory
from .solvers import StepSizeSchedule, _clip_grads


def _flat(params):
    return torch.cat([p.reshape(-1) for p in params])


def _set_flat(params, vec):
    """把扁平向量写回参数组（in-place，不破坏计算图）。"""
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vec[offset:offset + n].view_as(p))
            offset += n


def _dot(a, b):
    return torch.dot(a, b)


class MSBMO(object):
    """二阶随机动量求解器（new2.tex 算法 1）。单循环、单时间尺度。"""

    solver_name = 'MS-BMO'

    def __init__(self, task: BMOTask,
                 eta_x=1e-3, eta_w=1e-3,
                 eta_x_schedule='exp', eta_x_decay=0.95,
                 eta_w_schedule='exp', eta_w_decay=0.95,
                 beta_x=0.2, beta_w=0.2,
                 cg_iters=10, damping=1e-3, cg_tol=1e-8,
                 grad_clip=10.0, corr_clip=1e3, seed=1,
                 use_correction=True):
        self.task = task
        self.device = task.device
        self.pG = list(task.netG.parameters())
        self.pD = list(task.netD.parameters())
        self.pW = list(task.weight_net.parameters())
        self.eta_x_sched = StepSizeSchedule(eta_x, eta_x_schedule, eta_x_decay)
        self.eta_w_sched = StepSizeSchedule(eta_w, eta_w_schedule, eta_w_decay)
        self.beta_x = beta_x
        self.beta_w = beta_w
        self.cg_iters = cg_iters
        self.damping = damping
        self.cg_tol = cg_tol
        self.grad_clip = grad_clip
        self.corr_clip = corr_clip
        self.use_correction = use_correction
        self.rng = torch.Generator().manual_seed(seed)
        self.t = 0
        # 动量状态（扁平向量）
        self.wy = torch.zeros(_flat(self.pG).numel(), device=self.device)
        self.wz = torch.zeros(_flat(self.pD).numel(), device=self.device)
        self.dx = torch.zeros(_flat(self.pW).numel(), device=self.device)
        # w_{t-1} 状态快照（G/D/W 三组，供同样本校正重评估）
        self.prev_flat = torch.cat([_flat(self.pG), _flat(self.pD), _flat(self.pW)]).detach().clone()
        self.history = MetricHistory()
        self.extra = {'step': [], 'grad_norm': [], 'cg_res': [], 'corr_norm': []}

    # ---------- 状态快照 ----------
    def _get_all(self):
        return torch.cat([_flat(self.pG), _flat(self.pD), _flat(self.pW)]).detach()

    def _set_all(self, vec):
        nG = _flat(self.pG).numel()
        nD = _flat(self.pD).numel()
        _set_flat(self.pG, vec[:nG])
        _set_flat(self.pD, vec[nG:nG + nD])
        _set_flat(self.pW, vec[nG + nD:])

    def _swap_to(self, vec, keep):
        """切到 vec 状态，返回当前状态用于恢复。"""
        saved = self._get_all()
        self._set_all(vec)
        return saved

    # ---------- 数据采样 ----------
    def _sample(self, loader):
        if not hasattr(self, '_iters'):
            self._iters = {}
        key = id(loader)
        if key not in self._iters or self._iters[key] is None:
            self._iters[key] = iter(loader)
        try:
            return next(self._iters[key])
        except StopIteration:
            self._iters[key] = iter(loader)
            return next(self._iters[key])

    # ---------- 下层梯度（权重按常量处理，与一阶求解器一致） ----------
    def _lower_grads(self, batch):
        """返回 (ŵ_y 目标, ŵ_z 目标) 的扁平向量。

        理论约定（new2.tex）：g = g_y − g_z^code，y 极小化 g_y^code，
        z 极大化 −g_z^code（即极小化判别器损失）。因此
            ∇_y g = ∇_y g_y^code            （不取负）
            ∇_z g = −∇_z g_z^code           （取负，见 _update_lower_yz 的 (-g_z) 约定）
        这里直接返回理论约定的两个梯度，供 STORM 动量跟踪。
        """
        g_y, g_z = self.task.lower_losses(batch)
        grad_y = torch.autograd.grad(g_y, self.pG, retain_graph=False, allow_unused=True)
        grad_z = torch.autograd.grad(g_z, self.pD, retain_graph=False, allow_unused=True)
        gy = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grad_y, self.pG)])
        gz = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grad_z, self.pD)])
        return gy, -gz

    # ---------- z 块 Hessian-向量积（A = ∇²_zz g_z^code + λI ≻ 0） ----------
    def _hess_z_matvec(self, batch, v_flat):
        """A v = ∇²_zz(g_z^code) v + λ v。

        理论鞍点 Hessian 的 z 块 D = ∇²_zz(g_theory) = −∇²_zz(g_z^code) ≺ 0，
        其逆的 PD 化系统为 (−D + λI) = (∇²_zz g_z^code + λI)，故这里取 +hv。
        解得 ĉ ≈ (∇²_zz g_z^code)⁻¹ P_z，而 (M⁻¹P)_z = D⁻¹P_z = −ĉ（run 中取负）。
        """
        g_z = self._g_z_only(batch)
        grad_z = torch.autograd.grad(g_z, self.pD, create_graph=True, allow_unused=True)
        gv = _dot(torch.cat([g.reshape(-1) for g in grad_z]), v_flat)
        hv = torch.autograd.grad(gv, self.pD, retain_graph=False, allow_unused=True)
        hv_flat = torch.cat([(h if h is not None else torch.zeros_like(p)).reshape(-1)
                             for h, p in zip(hv, self.pD)])
        return hv_flat + self.damping * v_flat

    def _g_z_only(self, batch):
        """g_z（D 的极大化目标），权重按常量。与 bmo_task.lower_losses 的 g_z 部分一致。"""
        task = self.task
        x_noisy = batch[0] if isinstance(batch, (tuple, list)) else batch
        n = x_noisy.size(0)
        fake_for_d = task.make_fake(n).detach()
        pred_fake = task.netD(fake_for_d)
        loss_d_fake_vec = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake), reduction='none')
        pred_real = task.netD(x_noisy)
        loss_d_real_vec = F.binary_cross_entropy_with_logits(
            pred_real, torch.ones_like(pred_real), reduction='none')
        w_fake = task._weights(loss_d_fake_vec)
        w_real = task._weights(loss_d_real_vec)
        return (w_fake * loss_d_fake_vec).mean() + (w_real * loss_d_real_vec).mean()

    # ---------- 阻尼共轭梯度 ----------
    def _cg(self, batch, b_flat):
        """解 A u = b，A = −∇²_zz g_z + λI（凸化后的鞍点逆的 z 块）。"""
        v = torch.zeros_like(b_flat)
        r = b_flat.clone()
        p = r.clone()
        rs = _dot(r, r)
        rs0 = rs.item() + 1e-32
        res = rs0 ** 0.5
        for _ in range(self.cg_iters):
            if res <= self.cg_tol * (rs0 ** 0.5):
                break
            Ap = self._hess_z_matvec(batch, p)
            alpha = rs / (_dot(p, Ap) + 1e-32)
            v = v + alpha * p
            r = r - alpha * Ap
            rs_new = _dot(r, r)
            p = r + (rs_new / (rs + 1e-32)) * p
            rs = rs_new
            res = rs ** 0.5
        return v, res / (rs0 ** 0.5)

    # ---------- 超梯度校正 J^T u（权重网络对 x 可微） ----------
    def _lower_losses_wgrad(self, batch):
        """g_y, g_z 且权重网络参数在图中（用于二阶校正）。冻结/分离模式与任务一致。"""
        task = self.task
        x_noisy = batch[0] if isinstance(batch, (tuple, list)) else batch
        n = x_noisy.size(0)
        fake_for_d = task.make_fake(n).detach()
        pred_fake = task.netD(fake_for_d)
        loss_d_fake_vec = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake), reduction='none')
        pred_real = task.netD(x_noisy)
        loss_d_real_vec = F.binary_cross_entropy_with_logits(
            pred_real, torch.ones_like(pred_real), reduction='none')
        w_fake = task._weights(loss_d_fake_vec, weight_grad=True)
        w_real = task._weights(loss_d_real_vec, weight_grad=True)
        g_z = (w_fake * loss_d_fake_vec).mean() + (w_real * loss_d_real_vec).mean()
        for p in task.netD.parameters():
            p.requires_grad_(False)
        try:
            fake_for_g = task.make_fake(n)
            pred_g = task.netD(fake_for_g)
            loss_g_vec = F.binary_cross_entropy_with_logits(
                pred_g, torch.ones_like(pred_g), reduction='none')
            w_g = task._weights(loss_g_vec, weight_grad=True)
            g_y = (w_g * loss_g_vec).mean()
        finally:
            for p in task.netD.parameters():
                p.requires_grad_(True)
        return g_y, g_z

    def _hyper_correction(self, batch, uz_flat):
        """J^T u = ∇_x ⟨u, ∇_{(y,z)} g⟩，本任务 u_y = 0，故只需 z 通道。"""
        g_y, g_z = self._lower_losses_wgrad(batch)
        grad_z = torch.autograd.grad(g_z, self.pD, create_graph=True, allow_unused=True)
        s = _dot(torch.cat([g.reshape(-1) for g in grad_z]), uz_flat)
        corr = torch.autograd.grad(s, self.pW, allow_unused=True)
        return torch.cat([(c if c is not None else torch.zeros_like(p)).reshape(-1)
                          for c, p in zip(corr, self.pW)])

    # ---------- 上层梯度 ∇_x f 与 P_z ----------
    def _upper_grads(self, meta_batch):
        f = self.task.upper_loss(meta_batch)
        grads = torch.autograd.grad(f, self.pW + self.pD, allow_unused=True)
        nW = sum(p.numel() for p in self.pW)
        gx = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads[:len(self.pW)], self.pW)])
        pz = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads[len(self.pW):], self.pD)])
        return f.item(), gx, pz

    # ---------- 评估（与一阶求解器一致） ----------
    @torch.no_grad()
    def evaluate(self, train_loader, meta_loader, test_loader, step_label):
        task = self.task
        train_err = task.train_error(train_loader, max_batches=8)
        val_err = task.eval_error(meta_loader)
        test_err = task.eval_error(test_loader)
        val_raw = task.eval_error(meta_loader, weighted=False)
        test_raw = task.eval_error(test_loader, weighted=False)
        self.history.log(step_label, train_err, val_err, test_err,
                         val_error_raw=val_raw, test_error_raw=test_raw)
        return train_err, val_err, test_err

    # ---------- 主循环 ----------
    def run(self, train_loader, meta_loader, test_loader, T=100,
            eval_every=10, verbose=False, probe=None):
        """probe: 可选回调 probe(task) -> float，在每个评估点计算（如超梯度范数）。"""
        self.evaluate(train_loader, meta_loader, test_loader, 0)
        for t in range(self.t, self.t + T):
            eta_x = self.eta_x_sched(t)
            eta_w = self.eta_w_sched(t)
            # 1. 采样（新鲜样本；ζ1 用于下层动量，ζ2 用于 CG，ξ1/ξ2 用于上层）
            zeta1 = self._sample(train_loader)
            zeta2 = self._sample(train_loader)
            xi1 = self._sample(meta_loader)
            xi2 = self._sample(meta_loader)

            cur = self._get_all()
            prev = self.prev_flat

            # 2. 下层 STORM 同样本校正：在 w_t 与 w_{t-1} 处用同一 ζ1 评估
            gy_cur, gz_cur = self._lower_grads(zeta1)
            saved = self._get_all()
            self._set_all(torch.cat([prev[:_flat(self.pG).numel() + _flat(self.pD).numel()],
                                     saved[_flat(self.pG).numel() + _flat(self.pD).numel():]]))
            gy_prev, gz_prev = self._lower_grads(zeta1)
            self._set_all(saved)

            self.wy = (gy_cur - (1.0 - self.beta_w) * gy_prev) + (1.0 - self.beta_w) * self.wy
            self.wz = (gz_cur - (1.0 - self.beta_w) * gz_prev) + (1.0 - self.beta_w) * self.wz
            if self.grad_clip:
                nwy = self.wy.norm().clamp(min=1e-12)
                nwz = self.wz.norm().clamp(min=1e-12)
                self.wy = self.wy * min(1.0, self.grad_clip / nwy.item())
                self.wz = self.wz * min(1.0, self.grad_clip / nwz.item())

            # 3. 鞍点逆估计：CG 解 (−∇²_zz g_z + λI) ĉ = P_z（P_z 来自 ξ2），
            #    û_z = −ĉ ≈ M⁻¹P 的 z 块（带自适应阻尼保护）
            _, gx_cur, pz = self._upper_grads(xi2)
            if self.use_correction:
                chat, cg_res = self._cg(zeta2, pz)
                tries = 0
                while cg_res > 1.0 and tries < 2:
                    self.damping = min(self.damping * 10.0, 1.0)
                    chat, cg_res = self._cg(zeta2, pz)
                    tries += 1
                uz = -chat
                ucap = uz.norm().clamp(min=1e-12)
                if ucap.item() > self.corr_clip:
                    uz = uz * (self.corr_clip / ucap.item())

                # 4a. 当前点超梯度校正（同一 u）：Õ = ∇_x f + ∇_x⟨û, ∇_z g⟩
                #     （û_z ≈ M⁻¹P 的 z 块，J^T(H̃P) = ∇_x⟨û, ∇_z g⟩ 的负号已并入 û）
                corr_cur = self._hyper_correction(zeta2, uz)
            else:
                cg_res, uz, corr_cur = float('nan'), None, torch.zeros_like(self.dx)
            fx1_cur, gx_xi1, _ = self._upper_grads(xi1)   # ∇_x f(w_t; ξ1)
            o_t = gx_xi1 + corr_cur

            # 4b. 同样本校正重评估：在 w_{t-1} 处用同一 ξ1/ξ2 与同一 u
            corr_prev = torch.zeros_like(self.dx)
            if self.use_correction:
                saved = self._swap_to(prev, keep=None)
                corr_prev = self._hyper_correction(zeta2, uz)
                self._set_all(saved)
            _, gx_prev_xi1, _ = self._upper_grads(xi1)
            o_prev = gx_prev_xi1 + corr_prev

            self.dx = (o_t - (1.0 - self.beta_x) * o_prev) + (1.0 - self.beta_x) * self.dx
            if self.grad_clip:
                ndx = self.dx.norm().clamp(min=1e-12)
                self.dx = self.dx * min(1.0, self.grad_clip / ndx.item())

            # 5. 下层更新（行 7-8）
            with torch.no_grad():
                off = 0
                for p in self.pG:
                    n = p.numel()
                    p.add_(self.wy[off:off + n].view_as(p), alpha=-eta_w)
                    off += n
                off = 0
                for p in self.pD:
                    n = p.numel()
                    p.add_(self.wz[off:off + n].view_as(p), alpha=eta_w)
                    off += n

            # 6. 上层更新（行 15-16）
            with torch.no_grad():
                off = 0
                for p in self.pW:
                    n = p.numel()
                    p.add_(self.dx[off:off + n].view_as(p), alpha=-eta_x)
                    off += n

            self.prev_flat = cur.detach().clone()
            self.extra['step'].append(t + 1)
            self.extra['grad_norm'].append(self.dx.norm().item())
            self.extra['cg_res'].append(float(cg_res))
            self.extra['corr_norm'].append(corr_cur.norm().item())

            if verbose and (t + 1) % eval_every == 0:
                print(f'[MS-BMO] t={t+1} f={fx1_cur:.4f} |d|={self.dx.norm():.3e} '
                      f'cg_res={cg_res:.1e} |corr|={corr_cur.norm():.2e}')
            if (t + 1) % eval_every == 0:
                if probe is not None:
                    self.extra.setdefault('hyper_norm', []).append(probe(self.task))
                self.evaluate(train_loader, meta_loader, test_loader, t + 1)
        self.t += T
        return self.history


def plugin_hypergrad_norm(task, meta_batch, train_batch, cg_iters=15, damping=1e-2,
                          clip=100.0):
    """在当前参数状态下估计插入式超梯度的范数 ‖𝒢(x,y,z)‖₂。

    𝒢 = ∇_x f − Jᵀ(H̃P)，其中 H̃P 的 z 块由阻尼 CG 解得（见 MSBMO 的符号约定）。
    用大 batch 评估时它是 ∥∇Φ∥ 的低噪估计，用于 ε-驻点性验证（new2.tex 定义 1.5）。
    对任意求解器的快照均可用（MS-BMO / SSGDA / TSGDA-1）。
    damping 取较大值并把 ĉ 范数截断，避免欠定 Hessian 方向的探针爆炸。
    """
    probe = MSBMO(task, cg_iters=cg_iters, damping=damping, use_correction=False)
    _, gx, pz = probe._upper_grads(meta_batch)
    chat, _ = probe._cg(train_batch, pz)
    nrm = chat.norm().clamp(min=1e-12)
    if nrm.item() > clip:
        chat = chat * (clip / nrm.item())
    corr = probe._hyper_correction(train_batch, -chat)   # uz = −ĉ
    g = gx + corr                                        # Õ 的期望形式（见 run() 推导）
    return g.norm().item()
