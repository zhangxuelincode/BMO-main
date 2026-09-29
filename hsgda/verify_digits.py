# -*- coding: utf-8 -*-
"""临时探针：有限差分验证 digits 的 hess_u 与 jac_T_u。"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsgda_digits import DigitsBMO

prob = DigitsBMO(rho=0.01, seed=0)
prob.make_meta(200, np.random.default_rng(0))
rng = np.random.default_rng(3)
dy, dz = prob.dy, prob.dz
x = rng.standard_normal(prob.dx) * 0.3
y = 0.05 * rng.standard_normal(dy)
z = 0.1 * rng.standard_normal(dz)


def lower_grad_w(w_vec):
    """把 w=(y,z) 打包的 ∇_w g。"""
    gy, gz = prob.lower_grad(x, y, z, None)
    return np.concatenate([gy, gz])


# --- 验证 hess_u：H u ≈ (∇_wg(w + eps*u) − ∇_wg(w − eps*u)) / (2 eps) ---
u_y = rng.standard_normal(dy)
u_z = rng.standard_normal(dz)
u = np.concatenate([u_y, u_z])
u /= np.linalg.norm(u)

eps = 1e-6
y2 = y + eps * u[:dy]
z2 = z + eps * u[dy:]
g1 = np.concatenate(prob.lower_grad(x, y2, z2, None))
y3 = y - eps * u[:dy]
z3 = z - eps * u[dy:]
g2 = np.concatenate(prob.lower_grad(x, y3, z3, None))
fd = (g1 - g2) / (2 * eps)

Hy, Hz = prob.hess_u(x, y, z, u[:dy], u[dy:])
an = np.concatenate([Hy, Hz])
print('hess_u check: |an|=%.4f |fd|=%.4f rel_err=%.2e'
      % (np.linalg.norm(an), np.linalg.norm(fd),
         np.linalg.norm(an - fd) / max(np.linalg.norm(an), 1e-12)))

# --- 验证 jac_T_u：J^T u ≈ d/dx <u, ∇_wg(x)> ---
def jvp_fd(xx):
    g = np.concatenate(prob.lower_grad(xx, y, z, None))
    return np.concatenate([u_y, u_z]) @ g

fd2 = np.zeros(prob.dx)
for l in range(0, prob.dx, 40):          # 抽查 10 个坐标
    e = np.zeros(prob.dx)
    e[l] = eps
    fd2[l] = (jvp_fd(x + e) - jvp_fd(x - e)) / (2 * eps)
an2 = prob.jac_T_u(x, y, z, u_y, u_z)
idx = np.arange(0, prob.dx, 40)
print('jac_T_u check (10 coords): max rel err = %.2e'
      % np.max(np.abs(an2[idx] - fd2[idx]) / np.maximum(np.abs(fd2[idx]), 1e-12)))

# --- 检查访问状态处 ||PH|| 与 sym(PH) 最小特征值（小规模代理：随机 20 个状态）---
print('随轨迹的 ||H||：')
for t_mult in (0, 50, 100, 200, 400):
    nn = []
    for _ in range(5):
        v_y = rng.standard_normal(dy)
        v_z = rng.standard_normal(dz)
        v = np.concatenate([v_y, v_z])
        v /= np.linalg.norm(v)
        Hy2, Hz2 = prob.hess_u(x, y, z, v[:dy], v[dy:])
        nn.append(np.linalg.norm(np.concatenate([Hy2, Hz2])))
    print('  |H| ~ %.3f' % max(nn))
