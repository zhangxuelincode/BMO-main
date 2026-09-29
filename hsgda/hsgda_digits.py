# -*- coding: utf-8 -*-
"""HSGDA 在真实数据（sklearn digits）上的预测精度验证：双层样本重加权。

任务设定（对应论文的 bilevel reweighting 场景， 上层 min， 下层 min_y max_z）：
    训练集 400 个样本，其中 30% 的标签被随机翻转（噪声样本）；
    meta（验证）集 200 个干净样本；测试集 1000 个干净样本。
    上层变量 x ∈ R^400 为逐样本权重 logits，w_i = sigmoid(x_i)；
    下层变量 Y ∈ R^{10×65} 为线性分类器（64 维特征加偏置），z ∈ R^10 为
    分组 DRO 对偶变量（按 10 个类别分组）。

    下层  g(Y, z) = beta * <z, L(Y, w)> + (lambda/2)‖Y‖_F^2 - (mu_z/2)‖z‖^2
            L_g(Y, w) = Σ_{i∈g} w_i · CE_i(Y)（未归一化的组内加权和），  分组 min_y max_z
    上层  f = (1/m1) Σ_{j∈meta} CE_j(Y) + (rho/2)‖x‖^2

    上层对 x 的直接依赖只有正则项 rho/2‖x‖^2，元损失完全通过下层解映射
    Y*(x) 传递。因此偏梯度 ∇_x f = rho x 不含任何重加权信息，naive 算法
    （SSGDA 与 TSGDA-1）的 x 会退回均匀权重，而 HSGDA 通过隐式梯度项
    J^T u 学到样本权重，从而在噪声标签下取得更高的测试精度。

所有梯度与 Hessian-向量积均为解析实现（全批 400 样本，向量化计算）。
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsgda_core import _sched


def softmax(logits):
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def _sigmoid(t):
    t = np.clip(t, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-t))


class DigitsBMO:
    """digits 双层重加权问题（解析梯度接口与 SyntheticBMO 一致）。"""

    def __init__(self, n_train=400, n_meta=200, n_test=1000, noise_rate=0.3,
                 lam=0.1, rho=0.01, n_class=10, mu_z=1.0, beta=0.1, seed=0):
        from sklearn.datasets import load_digits
        rng = np.random.default_rng(seed)
        data = load_digits()
        X = data.data / 32.0                    # 特征缩放，改善下层条件数
        y = np.asarray(data.target)
        idx = rng.permutation(len(X))
        tr, me, te = idx[:n_train], idx[n_train:n_train + n_meta], \
            idx[n_train + n_meta:n_train + n_meta + n_test]
        self.lam, self.rho, self.mu_z, self.beta = lam, rho, mu_z, beta
        self.dx, self.dy, self.dz = n_train, n_class * (X.shape[1] + 1), n_class
        self.n_class = n_class

        def build(ii, noisy):
            phi = np.hstack([X[ii], np.ones((len(ii), 1))])
            lab = y[ii].copy()
            flip = np.zeros(len(ii), dtype=bool)
            if noisy:
                flip = rng.random(len(ii)) < noise_rate
                for i in np.where(flip)[0]:
                    cand = [c for c in range(n_class) if c != lab[i]]
                    lab[i] = rng.choice(cand)
            grp = lab.copy()                     # DRO 按观测标签分组
            return phi, lab, grp, flip

        self.phi_tr, self.lab_tr, self.grp_tr, self.noisy_tr = build(tr, noisy=True)
        self.phi_me, self.lab_me, self.grp_me, _ = build(me, noisy=False)
        self.phi_te, self.lab_te, self.grp_te, _ = build(te, noisy=False)
        self.n_tr = len(self.lab_tr)
        self.n_me = len(self.lab_me)
        self.n_te = len(self.lab_te)
        self.n_g = np.array([(self.grp_tr == g).sum() for g in range(n_class)],
                            dtype=float)
        self.n_g[self.n_g == 0] = 1.0
        self.eye = np.eye(n_class)
        self.E_tr = self.eye[self.lab_tr]          # (n_tr, 10) one-hot 转置
        self.xi_meta = None

    # ---------- 抽样接口 ----------
    def set_meta(self, xi_meta):
        self.xi_meta = np.asarray(xi_meta)

    def make_meta(self, m1, rng):
        """meta 集：m1 个干净 meta 样本的下标（有放回抽样池）。"""
        idx = rng.choice(self.n_me, size=m1, replace=m1 > self.n_me)
        self.set_meta(idx)
        return idx

    def draw_meta(self, rng):
        return int(self.xi_meta[rng.integers(len(self.xi_meta))])

    def draw_lower(self, rng):
        return None                                # 下层为确定性全批目标

    def proj_x(self, x):
        return x

    # ---------- 基础量 ----------
    def _P(self, Y, phi):
        """logits：Y (10×65) 作用到样本特征 (n×65) → (10, n)。"""
        return Y @ phi.T

    def _w(self, x):
        return _sigmoid(x)

    def _ce_terms(self, P, lab):
        """逐样本 CE 与 one-hot 误差 (p − e)：P (10,n) → ce (n,), dP (10,n)。"""
        n = P.shape[1]
        lse = np.log(np.exp(P - P.max(axis=0, keepdims=True)).sum(axis=0)) \
            + P.max(axis=0)
        ce = lse - P[lab, np.arange(n)]
        dP = softmax(P)
        dP[lab, np.arange(n)] -= 1.0
        return ce, dP

    # ---------- 下层 ----------
    def lower_grad(self, x, y, z, zeta=None):
        Y = y.reshape(self.n_class, -1)
        w = self._w(x)
        P = self._P(Y, self.phi_tr)
        ce, dP = self._ce_terms(P, self.lab_tr)
        coef = self.beta * z[self.grp_tr] * w                    # (n_tr,)
        gY = (dP * coef) @ self.phi_tr + self.lam * Y
        Lg = np.array([w[self.grp_tr == g] @ ce[self.grp_tr == g]
                       for g in range(self.n_class)])
        gz = self.beta * Lg - self.mu_z * z
        return gY.reshape(-1), gz

    def hess_u(self, x, y, z, u_y, u_z):
        """H u（下层鞍点 Hessian 作用）。

        H_yy 块的逐样本 CE Hessian 为 M_i = diag(p_i) − p_i p_i^T（p 为 softmax
        概率），注意与梯度用的 (p_i − e_i) 区分。
        """
        Y = y.reshape(self.n_class, -1)
        V = u_y.reshape(self.n_class, -1)
        w = self._w(x)
        P = self._P(Y, self.phi_tr)
        pP = softmax(P)                                          # 概率 p_i
        _, dP = self._ce_terms(P, self.lab_tr)                   # dP = p − e
        B = V @ self.phi_tr.T                                    # (10, n)
        pB = (pP * B).sum(axis=0)                                # p_i^T b_i
        MB = pP * B - pP * pB                                    # M_i b_i
        coef = self.beta * z[self.grp_tr] * w
        Hu_y = (MB * coef) @ self.phi_tr
        wu = self.beta * u_z[self.grp_tr] * w
        Hu_y = Hu_y + (dP * wu) @ self.phi_tr + self.lam * V
        # <∇_y CE_i, u_y> = (p_i − e_i)^T b_i（dP 已含 −e，勿重复减）
        dPb = (dP * B).sum(axis=0)
        Hu_z = self.beta * np.array(
            [w[self.grp_tr == g] @ dPb[self.grp_tr == g]
             for g in range(self.n_class)]) - self.mu_z * u_z
        return Hu_y.reshape(-1), Hu_z

    def jac_T_u(self, x, y, z, u_y, u_z):
        """∇_x<u, ∇_w g> = J^T u（隐式项，逐样本解析）。"""
        Y = y.reshape(self.n_class, -1)
        U = u_y.reshape(self.n_class, -1)
        w = self._w(x)
        dsw = w * (1.0 - w)
        P = self._P(Y, self.phi_tr)
        ce, dP = self._ce_terms(P, self.lab_tr)
        r = U @ self.phi_tr.T                                    # (10, n)
        ry = (r * dP).sum(axis=0)                                # <u_y, ∇_y CE_i>
        jT = w * (1.0 - w) * self.beta * (z[self.grp_tr] * ry
                                          + u_z[self.grp_tr] * ce)
        return jT

    # ---------- 上层 ----------
    def upper_grads(self, x, y, z, xi):
        """xi 为 meta 样本下标，返回 (∇_x f, ∇_w f)（单样本随机梯度）。"""
        Y = y.reshape(self.n_class, -1)
        Pm = self._P(Y, self.phi_me[xi:xi + 1])
        _, dPm = self._ce_terms(Pm, self.lab_me[xi:xi + 1])
        gY = dPm @ self.phi_me[xi:xi + 1]
        gx = self.rho * x
        gw = np.concatenate([gY.reshape(-1), np.zeros(self.dz)])
        return gx, gw

    def f_val(self, x, y, z, xi):
        Y = y.reshape(self.n_class, -1)
        Pm = self._P(Y, self.phi_me[xi:xi + 1])
        ce, _ = self._ce_terms(Pm, self.lab_me[xi:xi + 1])
        return float(ce[0] + 0.5 * self.rho * (x @ x))

    # ---------- 评估 ----------
    def val_loss(self, x, y, z):
        Y = y.reshape(self.n_class, -1)
        P = self._P(Y, self.phi_me)
        ce, _ = self._ce_terms(P, self.lab_me)
        return float(ce.mean() + 0.5 * self.rho * (x @ x))

    def test_loss(self, x, y, z, n=None):
        Y = y.reshape(self.n_class, -1)
        P = self._P(Y, self.phi_te)
        ce, _ = self._ce_terms(P, self.lab_te)
        return float(ce.mean() + 0.5 * self.rho * (x @ x))

    def accuracy(self, y, split='test'):
        Y = y.reshape(self.n_class, -1)
        if split == 'test':
            P = self._P(Y, self.phi_te)
            lab = self.lab_te
        elif split == 'meta':
            P = self._P(Y, self.phi_me)
            lab = self.lab_me
        else:
            P = self._P(Y, self.phi_tr)
            lab = self.lab_tr
        return float((P.argmax(axis=0) == lab).mean())

    def weights(self, x):
        return _sigmoid(x)


# ==========================================
# 求解器（与 hsgda_core 中的版本一致， 仅下层噪声为空）
# ==========================================

def run_hsgda_digits(prob, T=300, K=1, eta=0.05, gamma=0.05, alpha=0.05,
                     seed=0, eval_every=10):
    rng = np.random.default_rng(seed)
    dy, dz = prob.dy, prob.dz
    x = np.zeros(prob.dx)
    y = 0.01 * rng.standard_normal(dy)
    z = np.zeros(dz)
    u_y = np.zeros(dy)
    u_z = np.zeros(dz)
    rec = {'step': [], 'val': [], 'test': [], 'gap': [], 'acc': [],
           'w_std': [], 'w_noise_mean': [], 'w_clean_mean': []}
    noisy = prob.noisy_tr
    for t in range(T):
        if K == 1:
            xi = prob.draw_meta(rng)
            gy, gz = prob.lower_grad(x, y, z, None)
            Hy, Hz = prob.hess_u(x, y, z, u_y, u_z)
            gx, gw = prob.upper_grads(x, y, z, xi)
            jvp = prob.jac_T_u(x, y, z, u_y, u_z)
            y, z = y - gamma * gy, z + gamma * gz
            u_y, u_z = u_y - alpha * (Hy - gw[:dy]), u_z + alpha * (Hz - gw[dy:])
            x = x - eta * (gx - jvp)
        else:
            for _ in range(K):
                xi = prob.draw_meta(rng)
                gy, gz = prob.lower_grad(x, y, z, None)
                Hy, Hz = prob.hess_u(x, y, z, u_y, u_z)
                _, gw = prob.upper_grads(x, y, z, xi)
                y, z = y - gamma * gy, z + gamma * gz
                u_y, u_z = u_y - alpha * (Hy - gw[:dy]), u_z + alpha * (Hz - gw[dy:])
            xi = prob.draw_meta(rng)
            gx, _ = prob.upper_grads(x, y, z, xi)
            jvp = prob.jac_T_u(x, y, z, u_y, u_z)
            x = x - eta * (gx - jvp)
        if (t + 1) % eval_every == 0 or t == 0:
            w = prob.weights(x)
            rec['step'].append(t + 1)
            rec['val'].append(prob.val_loss(x, y, z))
            rec['test'].append(prob.test_loss(x, y, z))
            rec['gap'].append(rec['test'][-1] - rec['val'][-1])
            rec['acc'].append(prob.accuracy(y))
            rec['w_std'].append(float(w.std()))
            rec['w_noise_mean'].append(float(w[noisy].mean()))
            rec['w_clean_mean'].append(float(w[~noisy].mean()))
    return {'records': rec, 'x': x, 'y': y, 'z': z, 'u_y': u_y, 'u_z': u_z}


def run_naive_digits(prob, T=300, K=1, eta=0.05, gamma=0.05, seed=0,
                     eval_every=10, solver='SSGDA'):
    """naive 基线：外层只有偏梯度 rho*x（不含重加权信息）。"""
    rng = np.random.default_rng(seed)
    dy, dz = prob.dy, prob.dz
    x = np.zeros(prob.dx)
    y = 0.01 * rng.standard_normal(dy)
    z = np.zeros(dz)
    rec = {'step': [], 'val': [], 'test': [], 'gap': [], 'acc': [],
           'w_std': [], 'w_noise_mean': [], 'w_clean_mean': []}
    noisy = prob.noisy_tr
    for t in range(T):
        xi = prob.draw_meta(rng)
        gx, _ = prob.upper_grads(x, y, z, xi)
        x = x - eta * gx
        if solver == 'SSGDA':
            gy, gz = prob.lower_grad(x, y, z, None)
            y = y - gamma * gy
            _, gz2 = prob.lower_grad(x, y, z, None)
            z = z + gamma * gz2
        else:                                   # TSGDA-1
            for k in range(K):
                gy, _ = prob.lower_grad(x, y, z, None)
                y = y - (gamma / (k + 1)) * gy
            for k in range(K):
                _, gz = prob.lower_grad(x, y, z, None)
                z = z + (gamma / (k + 1)) * gz
        if (t + 1) % eval_every == 0 or t == 0:
            w = prob.weights(x)
            rec['step'].append(t + 1)
            rec['val'].append(prob.val_loss(x, y, z))
            rec['test'].append(prob.test_loss(x, y, z))
            rec['gap'].append(rec['test'][-1] - rec['val'][-1])
            rec['acc'].append(prob.accuracy(y))
            rec['w_std'].append(float(w.std()))
            rec['w_noise_mean'].append(float(w[noisy].mean()))
            rec['w_clean_mean'].append(float(w[~noisy].mean()))
    return {'records': rec, 'x': x, 'y': y, 'z': z}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=20000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--eta', type=float, default=0.03)
    p.add_argument('--gamma', type=float, default=0.03)
    p.add_argument('--rho', type=float, default=0.002)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.probe:
        for rho in (0.02, 0.05, 0.1):
            prob = DigitsBMO(rho=rho, seed=0)
            prob.make_meta(200, np.random.default_rng(0))
            out = run_hsgda_digits(prob, T=400, eta=0.05, gamma=0.08,
                                   alpha=0.08, seed=0, eval_every=200)
            r = out['records']
            gy, gz = prob.lower_grad(out['x'], out['y'], out['z'])
            print(f'HSGDA rho={rho}: val={r["val"][-1]:.4f} '
                  f'test={r["test"][-1]:.4f} acc={r["acc"][-1]:.4f} '
                  f'w_std={r["w_std"][-1]:.3f} '
                  f'w(noisy)={r["w_noise_mean"][-1]:.3f} '
                  f'w(clean)={r["w_clean_mean"][-1]:.3f} '
                  f'track_res=({np.linalg.norm(gy):.2e},{gz_norm(out, prob):.2e})')
        prob = DigitsBMO(rho=0.05, seed=0)
        prob.make_meta(200, np.random.default_rng(0))
        for eta in (0.02, 0.05, 0.1):
            out = run_hsgda_digits(prob, T=400, eta=eta, gamma=0.08,
                                   alpha=0.08, seed=0, eval_every=100)
            r = out['records']
            print(f'HSGDA eta={eta}: val={r["val"][-1]:.4f} '
                  f'test={r["test"][-1]:.4f} acc={r["acc"][-1]:.4f} '
                  f'w_std={r["w_std"][-1]:.3f} '
                  f'w(noisy)={r["w_noise_mean"][-1]:.3f} '
                  f'w(clean)={r["w_clean_mean"][-1]:.3f}')
        for eta in (0.02, 0.05, 0.1):
            out = run_naive_digits(prob, T=400, eta=eta, gamma=0.08,
                                   seed=0, eval_every=100)
            r = out['records']
            print(f'SSGDA eta={eta}: val={r["val"][-1]:.4f} '
                  f'test={r["test"][-1]:.4f} acc={r["acc"][-1]:.4f} '
                  f'w_std={r["w_std"][-1]:.4f}')
        return

    run_all(args, results_dir)


def gz_norm(out, prob):
    return float(np.linalg.norm(out['z'] * 0 + _gz_val(prob, out['x'],
                                                       out['y'], out['z'])))


def _gz_val(prob, x, y, z):
    _, gz = prob.lower_grad(x, y, z, None)
    return gz


def run_all(args, results_dir):
    rows = []
    for seed in range(args.repeats):
        t0 = time.time()
        prob = DigitsBMO(rho=args.rho, seed=0)
        prob.make_meta(200, np.random.default_rng(1000 + seed))
        out = run_hsgda_digits(prob, T=args.T, eta=args.eta, gamma=args.gamma,
                               alpha=args.gamma, seed=seed, eval_every=500)
        r = out['records']
        for i, s in enumerate(r['step']):
            rows.append(dict(solver='HSGDA', seed=seed, step=s, **{k: r[k][i]
                             for k in ('val', 'test', 'gap', 'acc', 'w_std',
                                       'w_noise_mean', 'w_clean_mean')}))
        for name, fn in [('SSGDA', run_naive_digits),
                         ('TSGDA-1', lambda pr, **kw: run_naive_digits(
                             pr, solver='TSGDA-1', **kw))]:
            out = fn(prob, T=args.T, K=10, eta=args.eta, gamma=args.gamma,
                     seed=seed, eval_every=500)
            r = out['records']
            for i, s in enumerate(r['step']):
                rows.append(dict(solver=name, seed=seed, step=s,
                                 **{k: r[k][i] for k in ('val', 'test', 'gap',
                                                         'acc', 'w_std',
                                                         'w_noise_mean',
                                                         'w_clean_mean')}))
        print(f'[digits] seed={seed} done ({time.time()-t0:.0f}s)')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'hsgda_digits_detail.csv'), index=False)
    tab = df.groupby(['solver', 'seed']).last().reset_index() \
            .groupby('solver')[['val', 'test', 'gap', 'acc', 'w_std',
                                'w_noise_mean', 'w_clean_mean']] \
            .agg(['mean', 'std'])
    tab.to_csv(os.path.join(results_dir, 'hsgda_digits_final.csv'))
    print(tab)


if __name__ == '__main__':
    main()
