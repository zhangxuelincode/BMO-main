# -*- coding: utf-8 -*-
"""HSGDA 核心模块（纯 numpy 实现）：可解析验证的合成 BMO 问题、三个求解器与评估工具。

对应 new.tex 中的算法 HSGDA（Hypergradient-aided Stochastic Gradient Descent-Ascent），
求解论文公式 (2) 的 BMO 问题（上层 min over x，下层 min_y max_z）：

    min_x  Phi(x) := E_xi f(x, y*(x), z*(x); xi)
    s.t.   (y*(x), z*(x)) in argmin_y argmax_z E_zeta g(x, y, z; zeta)

合成问题设计（便于解析验证，满足 new.tex 假设 3 到 5）：
    下层  g = 0.5 y^T A(x) y + y^T B(x) z - 0.5 z^T C(x) z + p(x)^T y - q(x)^T z
            + zeta_y^T y - zeta_z^T z
        A(x) = mu I + U diag(softplus(Wa x)) U^T   （A ⪰ mu I）
        C(x) = mu I + V diag(softplus(Wc x)) V^T   （C ⪰ mu I）
        B(x) = Lb tanh(Wb x)（reshape 成 dy×dz）    （‖B‖ ≤ Lb）
        p(x) = L_pq tanh(Wp x),  q(x) = L_pq tanh(Wq x)   （一致有界）
    噪声 zeta 只以线性项进入，因此 E[∇²_ww g] = ∇²_ww g（σ_H = 0），满足假设 5。
    上层  f = (1/Kf) Σ_k [tanh(u_k^T x) + (u_k^T x) ξ_k] (v_k^T y + r_k^T z) + c^T x + (rho/2)‖x‖²
    ξ ~ N(0, σ_ξ² I_Kf) 为 meta 样本噪声，meta 集 D_{m1} 即 m1 个这样的 ξ。
    tanh 响应有界，lin_k(x) = v_k^T y*(x) + r_k^T z*(x) 由有界的 p, q 驱动也有界，
    因此 population 目标 Φ 强制协调（rho/2‖x‖² 支配），存在内点最优解。
    噪声通道 (u_k^T x) ξ_k 对 x 线性响应，使算法能够拟合 meta 集的有限样本噪声，
    从而产生与稳定性理论一致的 O(T^{c*}/m1) 泛化 Gap。
    population 量（对 ξ 取期望）全部闭式：
        E_ξ[f] = (1/Kf) Σ_k tanh(u_k^T x) lin_k + c^T x + rho/2‖x‖²。

真实超梯度（population，new.tex 式 (eq_hypergrad)，闭式无 MC 噪声）：
    ∇Φ(x) = ∇_x f̄(x, w*) − ∇²_wx G(x, w*)^T u*，   u* = H^{-1} ∇_w f̄(x, w*)
其中鞍点 w* 与 H 解析可求。
"""
import numpy as np


def softplus(t):
    return np.logaddexp(0.0, t)


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


class SyntheticBMO:
    """可解析验证的合成 BMO 问题。"""

    def __init__(self, dx=25, dy=12, dz=12, mu=1.0, L=4.0, Lb_ratio=0.5,
                 sigma_zeta=0.3, sigma_xi=1.0, Kf=2, rho=1.0, R_x=5.0,
                 L_f=2.0, c_norm=0.5, L_pq=2.0, n_pop=200000, seed=0):
        rng = np.random.default_rng(seed)
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dw = dy + dz
        self.mu, self.L = mu, L
        self.Lb = Lb_ratio * L                  # ‖B(x)‖ ≤ Lb
        self.L_pq = L_pq                        # p, q 的一致界
        self.sigma_zeta = sigma_zeta            # 下层噪声 std（zeta）
        self.sigma_xi = sigma_xi                # 上层 meta 噪声 std（xi）
        self.Kf = Kf                            # 上层分量数
        self.rho = rho                          # 上层二次正则
        self.R_x = R_x                          # x 的球约束半径（有界域 C_x）
        self.n_pop = n_pop

        def scale_w(M, cap):
            # 使 ‖M x‖ ≤ cap 对一切 ‖x‖ ≤ R_x 成立
            s = np.linalg.norm(M, 2)
            return M * (cap / (s * R_x))

        # 固定正交基（qr 分解得到）
        self.U, _ = np.linalg.qr(rng.standard_normal((dy, dy)))
        self.V, _ = np.linalg.qr(rng.standard_normal((dz, dz)))
        # 下层参数（尺度控制在 mu ≤ 块范数 ≤ L 附近）
        self.W_a = scale_w(rng.standard_normal((dy, dx)) / np.sqrt(dx), L - mu)
        self.W_c = scale_w(rng.standard_normal((dz, dx)) / np.sqrt(dx), L - mu)
        self.W_b = rng.standard_normal((dy * dz, dx)) / np.sqrt(dx)
        # p, q 的 tanh 自变量尺度：‖W x‖ ≤ 3 使 tanh 接近饱和且一致有界
        self.W_p = scale_w(rng.standard_normal((dy, dx)) / np.sqrt(dx), 3.0)
        self.W_q = scale_w(rng.standard_normal((dz, dx)) / np.sqrt(dx), 3.0)
        # 上层参数（u_k 归一到单位范数，v_k 与 r_k 行范数为 L_f）
        Uf = rng.standard_normal((Kf, dx))
        Uf /= np.linalg.norm(Uf, axis=1, keepdims=True)
        self.U_f = Uf
        Vf = rng.standard_normal((Kf, dy))
        Vf /= np.linalg.norm(Vf, axis=1, keepdims=True)
        Rf = rng.standard_normal((Kf, dz))
        Rf /= np.linalg.norm(Rf, axis=1, keepdims=True)
        self.V_f, self.R_f = Vf * L_f, Rf * L_f
        # 上层线性项（把 population 最优解推离原点，使噪声通道保持激活）
        c = rng.standard_normal(dx)
        self.c = c_norm * c / np.linalg.norm(c)
        # population 噪声库（固定，仅用于 test loss 的 MC 估计）
        self.xi_pop = rng.standard_normal((n_pop, Kf)) * sigma_xi
        self.xi_meta = None                     # set_meta 后由 draw_meta 抽样

    # ---------- meta 集与抽样接口 ----------
    def set_meta(self, xi_meta):
        self.xi_meta = np.asarray(xi_meta, dtype=float)

    def make_meta(self, m1, rng):
        """生成 meta 集 D_{m1}（m1 个 xi 样本）。"""
        xi = rng.standard_normal((m1, self.Kf)) * self.sigma_xi
        self.set_meta(xi)
        return xi

    def draw_meta(self, rng):
        return self.xi_meta[rng.integers(len(self.xi_meta))]

    def draw_lower(self, rng):
        return rng.standard_normal(self.dw) * self.sigma_zeta

    def proj_x(self, x):
        n = np.linalg.norm(x)
        if n > self.R_x:
            return x * (self.R_x / n)
        return x

    # ---------- 下层组件 ----------
    def A_mat(self, x):
        d = softplus(self.W_a @ x)
        return self.mu * np.eye(self.dy) + (self.U * d) @ self.U.T

    def C_mat(self, x):
        d = softplus(self.W_c @ x)
        return self.mu * np.eye(self.dz) + (self.V * d) @ self.V.T

    def B_mat(self, x):
        # tanh 的每个元素 |t|<1，除以 sqrt(dy*dz) 保证谱范数 ‖B(x)‖_2 ≤ Lb
        return (self.Lb / np.sqrt(self.dy * self.dz)) * np.tanh(
            (self.W_b @ x).reshape(self.dy, self.dz))

    def pq(self, x):
        """一致有界的下层驱动项 p, q。"""
        return (self.L_pq * np.tanh(self.W_p @ x),
                self.L_pq * np.tanh(self.W_q @ x))

    def H_mat(self, x):
        """鞍点 Hessian H = [[A, B], [B^T, -C]]。"""
        A, B, C = self.A_mat(x), self.B_mat(x), self.C_mat(x)
        H = np.zeros((self.dw, self.dw))
        H[:self.dy, :self.dy] = A
        H[:self.dy, self.dy:] = B
        H[self.dy:, :self.dy] = B.T
        H[self.dy:, self.dy:] = -C
        return H

    def lower_grad(self, x, y, z, zeta=None):
        """∇_w g 返回 (g_y, g_z)。"""
        A, B, C = self.A_mat(x), self.B_mat(x), self.C_mat(x)
        p, q = self.pq(x)
        gy = A @ y + B @ z + p
        gz = B.T @ y - C @ z - q
        if zeta is not None:
            gy = gy + zeta[:self.dy]
            gz = gz - zeta[self.dy:]
        return gy, gz

    def hess_u(self, x, y, z, u_y, u_z):
        """鞍点 Hessian 作用 H u = (A u_y + B u_z, B^T u_y − C u_z)（与 w、zeta 无关）。"""
        A, B, C = self.A_mat(x), self.B_mat(x), self.C_mat(x)
        return A @ u_y + B @ u_z, B.T @ u_y - C @ u_z

    def saddle_solve(self, x):
        """解析鞍点：H w* = [-p; q]（∇_y g = 0 与 ∇_z g = 0 的线性系统）。"""
        p, q = self.pq(x)
        rhs = np.concatenate([-p, q])
        return np.linalg.solve(self.H_mat(x), rhs)

    # ---------- 上层组件 ----------
    def _pre_lin(self, x, y, z):
        """返回 pre = U_f x（Kf 维）与 lin = V_f y + R_f z（Kf 维）。"""
        pre = self.U_f @ x
        lin = self.V_f @ y + self.R_f @ z
        return pre, lin

    def upper_grads(self, x, y, z, xi):
        """返回 (∇_x f, ∇_w f)（解析，单样本 xi）。

        f = (1/Kf) Σ_k [tanh(pre_k) + pre_k·ξ_k]·lin_k + c^T x + (rho/2)‖x‖²
        其中 pre = U_f x，tanh' = 1 − tanh²。
        """
        pre, lin = self._pre_lin(x, y, z)
        t = np.tanh(pre)
        dt = 1.0 - t ** 2
        gx = (self.U_f.T @ ((dt + xi) * lin)) / self.Kf + self.c + self.rho * x
        gw = np.concatenate([self.V_f.T @ (t + pre * xi),
                             self.R_f.T @ (t + pre * xi)]) / self.Kf
        return gx, gw

    def f_val(self, x, y, z, xi):
        pre, lin = self._pre_lin(x, y, z)
        t = np.tanh(pre)
        return float(((t + pre * xi) @ lin) / self.Kf + self.c @ x
                     + 0.5 * self.rho * (x @ x))

    # ---------- 隐式 Jacobian ----------
    def jacobian_wx(self, x, y, z):
        """显式 J(x, y, z) = ∂(∇_w g)/∂x ∈ R^{dw×dx}（对固定 (x, w)）。

        ∇_w g = [A(x)y + B(x)z + p(x); B(x)^T y − C(x)z − q(x)]，对 x_l 求导（y, z 固定）：
            ∂(∇_wg)/∂x_l = [∂A_l y + ∂B_l z + ∂p_l;  ∂B_l^T y − ∂C_l z − ∂q_l]
        其中 ∂p_l = L_pq·sech²(W_p x)·W_p[:,l]，∂q_l 同理。噪声项为线性，不影响 J。
        """
        dy, dz, dx = self.dy, self.dz, self.dx
        J = np.zeros((self.dw, dx))
        sig_a = sigmoid(self.W_a @ x)
        sig_c = sigmoid(self.W_c @ x)
        tanh_b = np.tanh(self.W_b @ x)
        sech2 = 1.0 - tanh_b ** 2
        b_scale = self.Lb / np.sqrt(self.dy * self.dz)
        sech_p = self.L_pq * (1.0 - np.tanh(self.W_p @ x) ** 2)
        sech_q = self.L_pq * (1.0 - np.tanh(self.W_q @ x) ** 2)
        y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
        for l in range(dx):
            dA = self.U @ np.diag(sig_a * self.W_a[:, l]) @ self.U.T
            dC = self.V @ np.diag(sig_c * self.W_c[:, l]) @ self.V.T
            dB = b_scale * (sech2 * self.W_b[:, l]).reshape(dy, dz)
            J[:dy, l] = dA @ y + dB @ z + sech_p * self.W_p[:, l]
            J[dy:, l] = dB.T @ y - dC @ z - sech_q * self.W_q[:, l]
        return J

    def jac_T_u(self, x, y, z, u_y, u_z):
        """∇_x⟨u, ∇_w g(x, w)⟩ = J^T u（算法外层更新的隐式项）。

        推导：∇_w g = [g_y; g_z]，⟨u, ∇_w g⟩ = u_y^T g_y + u_z^T g_z，
        故 ∇_x⟨u, ∇_w g⟩ = J^T [u_y; u_z]。
        """
        u = np.concatenate([u_y, u_z])
        return self.jacobian_wx(x, y, z).T @ u

    # ---------- 真实超目标（population，闭式） ----------
    def _pop_stats(self, x, n=None):
        """population 的响应值与导数（闭式，ξ 期望为零）。"""
        pre = self.U_f @ np.asarray(x, dtype=float)
        t = np.tanh(pre)
        return t, 1.0 - t ** 2

    def phi_grad(self, x, n=None):
        """解析真实超梯度 ∇Φ(x)（population 闭式，与 new.tex 式 (eq_hypergrad) 一致）。"""
        x = np.asarray(x, dtype=float)
        w_star = self.saddle_solve(x)
        y_s, z_s = w_star[:self.dy], w_star[self.dy:]
        s_bar, d_bar = self._pop_stats(x)
        lin = self.V_f @ y_s + self.R_f @ z_s
        gx_bar = (self.U_f.T @ (d_bar * lin)) / self.Kf + self.c + self.rho * x
        gw_bar = np.concatenate([self.V_f.T @ s_bar, self.R_f.T @ s_bar]) / self.Kf
        u_star = np.linalg.solve(self.H_mat(x), gw_bar)
        J = self.jacobian_wx(x, y_s, z_s)
        return gx_bar - J.T @ u_star

    def phi_val(self, x, n=None):
        """population Φ(x)（闭式，在解析鞍点 w*(x) 处取值）。"""
        x = np.asarray(x, dtype=float)
        w_star = self.saddle_solve(x)
        y_s, z_s = w_star[:self.dy], w_star[self.dy:]
        pre = self.U_f @ x
        lin = self.V_f @ y_s + self.R_f @ z_s
        return float((np.tanh(pre) @ lin) / self.Kf + self.c @ x
                     + 0.5 * self.rho * (x @ x))

    def u_star(self, x, n=None):
        """真实伴随解 u*(x) = H^{-1} ∇_w f̄(x, w*(x))（population，闭式）。"""
        x = np.asarray(x, dtype=float)
        s_bar, _ = self._pop_stats(x)
        gw_bar = np.concatenate([self.V_f.T @ s_bar, self.R_f.T @ s_bar]) / self.Kf
        return np.linalg.solve(self.H_mat(x), gw_bar)

    # ---------- 评估 ----------
    def val_loss(self, x, y, z):
        """验证误差 R_{D_{m1}}(x, y, z)：meta 集上 f 的平均。"""
        x = np.asarray(x, dtype=float)
        pre, lin = self._pre_lin(x, y, z)
        t = np.tanh(pre)
        xi_bar = self.xi_meta.mean(axis=0)                    # (Kf,)
        return float(((t + pre * xi_bar) @ lin) / self.Kf + self.c @ x
                     + 0.5 * self.rho * (x @ x))

    def test_loss(self, x, y, z, n=None):
        """测试误差 R(x, y, z)：population 的大样本 MC 估计。"""
        x = np.asarray(x, dtype=float)
        n = self.n_pop if n is None else n
        pre, lin = self._pre_lin(x, y, z)
        t = np.tanh(pre)
        xi_bar = self.xi_pop[:n].mean(axis=0)                 # (Kf,)
        return float(((t + pre * xi_bar) @ lin) / self.Kf + self.c @ x
                     + 0.5 * self.rho * (x @ x))


# ==========================================
# 三个求解器（与 new.tex 与主文的算法一一对应）
# ==========================================

def _sched(base, mode, decay, t):
    if mode == 'fixed':
        return base
    if mode == 'exp':
        return base * (decay ** t)
    if mode == 'inv':
        return base / np.sqrt(1.0 + t)
    raise ValueError(mode)


def _init_state(prob, rng, x0):
    if x0 is None:
        x = prob.proj_x(rng.standard_normal(prob.dx) * 0.5)
    else:
        x = np.asarray(x0, dtype=float).copy()
    y = np.zeros(prob.dy)
    z = np.zeros(prob.dz)
    u_y = np.zeros(prob.dy)
    u_z = np.zeros(prob.dz)
    return x, y, z, u_y, u_z


def run_hsgda(prob, T=3000, K=1, eta=0.01, gamma=0.02, alpha=0.02,
              eta_mode='fixed', eta_decay=0.95, gamma_mode='fixed',
              gamma_decay=0.95, seed=0, eval_every=10, track_phi=True,
              x0=None):
    """HSGDA（new.tex Algorithm 1）：同步鞍点跟踪 + 伴随跟踪 + 超梯度校正外层更新。

    K 为每个外层步的内层跟踪步数（K=1 即严格的单循环 Algorithm 1，
    外层方向的 JVP 严格使用更新前的 u^t 与旧点，与伪代码逐行一致）。
    """
    rng = np.random.default_rng(seed)
    dy, dz = prob.dy, prob.dz
    x, y, z, u_y, u_z = _init_state(prob, rng, x0)
    rec = {'step': [], 'val': [], 'test': [], 'gap': [], 'phi_pop': []}
    gphi2_all = np.full(T, np.nan)
    min_gphi2 = np.inf
    for t in range(T):
        eta_t = _sched(eta, eta_mode, eta_decay, t)
        gamma_t = _sched(gamma, gamma_mode, gamma_decay, t)
        alpha_t = _sched(alpha, 'fixed' if gamma_mode == 'fixed' else gamma_mode,
                         gamma_decay, t)
        if K == 1:
            # 严格按 Algorithm 1：所有量在旧点 (x^t, y^t, z^t) 处计算，外层用旧 u^t
            zeta = prob.draw_lower(rng)
            xi = prob.draw_meta(rng)
            gy, gz = prob.lower_grad(x, y, z, zeta)
            Hy, Hz = prob.hess_u(x, y, z, u_y, u_z)
            gx, gw = prob.upper_grads(x, y, z, xi)
            jvp = prob.jac_T_u(x, y, z, u_y, u_z)
            y, z = y - gamma_t * gy, z + gamma_t * gz
            u_y, u_z = u_y - alpha_t * (Hy - gw[:dy]), u_z + alpha_t * (Hz - gw[dy:])
            x = prob.proj_x(x - eta_t * (gx - jvp))
        else:
            # 内层跟踪 K 步后，外层方向在最新状态处计算（更充分跟踪的自然推广）
            for _ in range(K):
                zeta = prob.draw_lower(rng)
                xi = prob.draw_meta(rng)
                gy, gz = prob.lower_grad(x, y, z, zeta)
                Hy, Hz = prob.hess_u(x, y, z, u_y, u_z)
                _, gw = prob.upper_grads(x, y, z, xi)
                y, z = y - gamma_t * gy, z + gamma_t * gz
                u_y, u_z = u_y - alpha_t * (Hy - gw[:dy]), u_z + alpha_t * (Hz - gw[dy:])
            xi = prob.draw_meta(rng)
            gx, _ = prob.upper_grads(x, y, z, xi)
            jvp = prob.jac_T_u(x, y, z, u_y, u_z)
            x = prob.proj_x(x - eta_t * (gx - jvp))
        if track_phi:
            g2 = float(prob.phi_grad(x) @ prob.phi_grad(x))
            gphi2_all[t] = g2
            min_gphi2 = min(min_gphi2, g2)
        if (t + 1) % eval_every == 0 or t == 0:
            val = prob.val_loss(x, y, z)
            test = prob.test_loss(x, y, z)
            rec['step'].append(t + 1)
            rec['val'].append(val)
            rec['test'].append(test)
            rec['gap'].append(test - val)
            rec['phi_pop'].append(prob.phi_val(x))
    out = {'records': rec, 'gphi2_all': gphi2_all, 'min_gphi2': min_gphi2,
           'x': x, 'y': y, 'z': z, 'u_y': u_y, 'u_z': u_z,
           'meta': dict(T=T, K=K, eta=eta, gamma=gamma, alpha=alpha,
                        eta_mode=eta_mode, gamma_mode=gamma_mode, seed=seed)}
    return out


def run_ssgda(prob, T=3000, eta=0.01, gamma=0.02, eta_mode='fixed',
              eta_decay=0.95, gamma_mode='fixed', gamma_decay=0.95, seed=0,
              eval_every=10, track_phi=True, x0=None):
    """SSGDA（naive 基线）：外层只用偏梯度 ∇_x f，下层 y 交替 z 各更新一步。"""
    rng = np.random.default_rng(seed)
    x, y, z, _, _ = _init_state(prob, rng, x0)
    rec = {'step': [], 'val': [], 'test': [], 'gap': [], 'phi_pop': []}
    gphi2_all = np.full(T, np.nan)
    min_gphi2 = np.inf
    for t in range(T):
        eta_t = _sched(eta, eta_mode, eta_decay, t)
        gamma_t = _sched(gamma, gamma_mode, gamma_decay, t)
        xi = prob.draw_meta(rng)
        gx, _ = prob.upper_grads(x, y, z, xi)
        x = prob.proj_x(x - eta_t * gx)
        zeta = prob.draw_lower(rng)
        gy, gz = prob.lower_grad(x, y, z, zeta)
        y = y - gamma_t * gy
        _, gz2 = prob.lower_grad(x, y, z, zeta)
        z = z + gamma_t * gz2
        if track_phi:
            pg = prob.phi_grad(x)
            g2 = float(pg @ pg)
            gphi2_all[t] = g2
            min_gphi2 = min(min_gphi2, g2)
        if (t + 1) % eval_every == 0 or t == 0:
            val = prob.val_loss(x, y, z)
            test = prob.test_loss(x, y, z)
            rec['step'].append(t + 1)
            rec['val'].append(val)
            rec['test'].append(test)
            rec['gap'].append(test - val)
            rec['phi_pop'].append(prob.phi_val(x))
    return {'records': rec, 'gphi2_all': gphi2_all, 'min_gphi2': min_gphi2,
            'x': x, 'y': y, 'z': z,
            'meta': dict(T=T, eta=eta, gamma=gamma, eta_mode=eta_mode, seed=seed)}


def run_tsgda1(prob, T=1500, K=10, eta=0.01, gamma=0.02, eta_mode='fixed',
               eta_decay=0.95, seed=0, eval_every=10, track_phi=True, x0=None):
    """TSGDA-1（naive 基线）：外层偏梯度 + 每个外层步 K 步内层（先 y 后 z，内层步长衰减）。"""
    rng = np.random.default_rng(seed)
    x, y, z, _, _ = _init_state(prob, rng, x0)
    rec = {'step': [], 'val': [], 'test': [], 'gap': [], 'phi_pop': []}
    gphi2_all = np.full(T, np.nan)
    min_gphi2 = np.inf
    for t in range(T):
        eta_t = _sched(eta, eta_mode, eta_decay, t)
        xi = prob.draw_meta(rng)
        gx, _ = prob.upper_grads(x, y, z, xi)
        x = prob.proj_x(x - eta_t * gx)
        for k in range(K):
            zeta = prob.draw_lower(rng)
            gy, _ = prob.lower_grad(x, y, z, zeta)
            y = y - (gamma / (k + 1)) * gy
        for k in range(K):
            zeta = prob.draw_lower(rng)
            _, gz = prob.lower_grad(x, y, z, zeta)
            z = z + (gamma / (k + 1)) * gz
        if track_phi:
            pg = prob.phi_grad(x)
            g2 = float(pg @ pg)
            gphi2_all[t] = g2
            min_gphi2 = min(min_gphi2, g2)
        if (t + 1) % eval_every == 0 or t == 0:
            val = prob.val_loss(x, y, z)
            test = prob.test_loss(x, y, z)
            rec['step'].append(t + 1)
            rec['val'].append(val)
            rec['test'].append(test)
            rec['gap'].append(test - val)
            rec['phi_pop'].append(prob.phi_val(x))
    return {'records': rec, 'gphi2_all': gphi2_all, 'min_gphi2': min_gphi2,
            'x': x, 'y': y, 'z': z,
            'meta': dict(T=T, K=K, eta=eta, gamma=gamma, eta_mode=eta_mode, seed=seed)}


# ==========================================
# 数值校验工具
# ==========================================

def phi_grad_check(prob, x=None, eps=1e-5, n_mc=200000, rng_seed=123):
    """解析 ∇Φ 与有限差分的相对误差（用共同随机数）。"""
    rng = np.random.default_rng(rng_seed)
    if x is None:
        x = prob.proj_x(rng.standard_normal(prob.dx) * 2.0)
    x = np.asarray(x, dtype=float)
    g = prob.phi_grad(x, n=n_mc)
    fd = np.zeros(prob.dx)
    for l in range(prob.dx):
        e = np.zeros(prob.dx)
        e[l] = eps
        fd[l] = (prob.phi_val(x + e, n=n_mc) - prob.phi_val(x - e, n=n_mc)) / (2 * eps)
    denom = max(np.linalg.norm(g), np.linalg.norm(fd), 1e-12)
    return g, fd, float(np.linalg.norm(g - fd) / denom)


def gda_contraction(prob, x, gamma):
    """线性化迭代矩阵的算子范数 ‖I − γ P H‖₂（PH 非正规，看范数而非谱半径）。

    理论保证：sym(PH) ⪰ μI 时 ‖I − γPH‖₂ ≤ sqrt(1 − 2γμ + γ²‖PH‖²)，
    故 γ < 2μ/‖PH‖² 时严格收缩。
    """
    H = prob.H_mat(x)
    P = np.eye(prob.dw)
    P[prob.dy:, prob.dy:] = -1.0
    M = np.eye(prob.dw) - gamma * (P @ H)
    return float(np.linalg.norm(M, 2))


def tracking_check(prob, x, n_steps=500, gamma=0.02, alpha=0.02, rng_seed=7,
                   use_noise=True):
    """固定 x 时鞍点与伴随系统的跟踪误差衰减（验证收缩性）。

    use_noise=False 时为确定性收缩检验（误差应趋于 0）；
    use_noise=True 时误差收敛到由 zeta 噪声决定的稳态半径。
    """
    rng = np.random.default_rng(rng_seed)
    dy, dz = prob.dy, prob.dz
    y = np.zeros(dy) + 0.5
    z = np.zeros(dz) + 0.5
    u_y = np.zeros(dy)
    u_z = np.zeros(dz)
    w_star = prob.saddle_solve(x)
    u_star = prob.u_star(x)
    errs_w, errs_u = [], []
    for t in range(n_steps):
        zeta = prob.draw_lower(rng) if use_noise else None
        gy, gz = prob.lower_grad(x, y, z, zeta)
        Hy, Hz = prob.hess_u(x, y, z, u_y, u_z)
        # 伴随驱动项用 population 的 ∇_w f（在鞍点处）
        s_bar, _ = prob._pop_stats(x)
        gw = np.concatenate([prob.V_f.T @ s_bar, prob.R_f.T @ s_bar]) / prob.Kf
        y, z = y - gamma * gy, z + gamma * gz
        u_y, u_z = u_y - alpha * (Hy - gw[:dy]), u_z + alpha * (Hz - gw[dy:])
        if t % max(n_steps // 10, 1) == 0 or t == n_steps - 1:
            errs_w.append(float(np.linalg.norm(np.concatenate([y, z]) - w_star)))
            errs_u.append(float(np.linalg.norm(np.concatenate([u_y, u_z]) - u_star)))
    return errs_w, errs_u
