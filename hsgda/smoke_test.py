# -*- coding: utf-8 -*-
"""HSGDA 冒烟测试：解析正确性 + 收敛性 + 基线偏差现象。

通过标准：
1. 解析 ∇Φ 与有限差分相对误差 < 1e-4；
2. 固定 x 时鞍点跟踪与伴随跟踪误差衰减到 1e-2 以下（收缩性）；
3. HSGDA 在 T=2000 内 ‖∇Φ(x^t)‖² 显著下降且 Φ(x^t) 下降；
4. SSGDA 的 ‖∇Φ‖² 停留在明显高于 HSGDA 的偏差平台上。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsgda_core import (SyntheticBMO, run_hsgda, run_ssgda, run_tsgda1,
                        phi_grad_check, gda_contraction, tracking_check)


def main():
    t0 = time.time()
    prob = SyntheticBMO(seed=0)
    prob.make_meta(m1=200, rng=np.random.default_rng(0))

    print('===== 1) 解析超梯度 vs 有限差分 =====')
    worst = 0.0
    for i in range(5):
        rng = np.random.default_rng(100 + i)
        x = prob.proj_x(rng.standard_normal(prob.dx) * 2.0)
        g, fd, rel = phi_grad_check(prob, x)
        worst = max(worst, rel)
        print(f'  point {i}: |grad|={np.linalg.norm(g):.6f} '
              f'|fd|={np.linalg.norm(fd):.6f} rel_err={rel:.2e}')
    print(f'  worst rel_err = {worst:.2e} (require < 1e-4)')
    assert worst < 1e-4, '解析超梯度与有限差分不一致'

    print('===== 2) 步长收缩性诊断 =====')
    rng = np.random.default_rng(1)
    for gamma in (0.02, 0.03, 0.05):
        rho_max = max(gda_contraction(prob, prob.proj_x(rng.standard_normal(prob.dx) * 3.0),
                                      gamma) for _ in range(20))
        print(f'  gamma={gamma}: max_t rho(I-gamma*P*H) = {rho_max:.4f}')

    print('===== 3) 固定 x 的鞍点/伴随跟踪 =====')
    x_fix = prob.proj_x(np.random.default_rng(2).standard_normal(prob.dx) * 2.0)
    # 确定性收缩检验（无噪声时误差应趋于 0）
    errs_w, errs_u = tracking_check(prob, x_fix, n_steps=600, gamma=0.02,
                                    alpha=0.02, use_noise=False)
    print('  deterministic saddle err:', ' -> '.join(f'{e:.2e}' for e in errs_w))
    print('  deterministic adjoint err:', ' -> '.join(f'{e:.2e}' for e in errs_u))
    assert errs_w[-1] < 1e-2 and errs_u[-1] < 1e-2, '确定性跟踪未收缩'
    # 随机跟踪应收敛到噪声决定的稳态半径（E‖e‖² ~ γσ_ζ² dw/(2μ)）
    errs_w2, errs_u2 = tracking_check(prob, x_fix, n_steps=600, gamma=0.02,
                                      alpha=0.02, use_noise=True)
    floor = prob.sigma_zeta * np.sqrt(0.02 * prob.dw / (2 * prob.mu)) * 1.5
    print(f'  stochastic saddle err tail: {errs_w2[-1]:.3e} (noise floor x1.5 ~ {floor:.3e})')
    print(f'  stochastic adjoint err tail: {errs_u2[-1]:.3e}')
    assert errs_w2[-1] <= floor, '随机鞍点跟踪稳态半径异常'
    assert errs_u2[-1] < 1e-2, '随机伴随跟踪未收缩'

    print('===== 4) HSGDA 短程收敛（T=2000, K=1） =====')
    out = run_hsgda(prob, T=2000, K=1, eta=0.01, gamma=0.02, alpha=0.02,
                    seed=1, eval_every=100, track_phi=True)
    ga = out['gphi2_all']
    idx = np.isfinite(ga)
    show = [0, 9, 49, 99, 249, 499, 999, 1999]
    print('  ||grad Phi||^2:', ' '.join(f't={s+1}:{ga[s]:.3e}' for s in show))
    print(f'  min_t ||grad Phi||^2 = {out["min_gphi2"]:.3e}')
    rec = out['records']
    print(f'  phi_pop: {rec["phi_pop"][0]:.4f} -> {rec["phi_pop"][-1]:.4f}')
    print(f'  val: {rec["val"][0]:.4f} -> {rec["val"][-1]:.4f}; '
          f'test: {rec["test"][0]:.4f} -> {rec["test"][-1]:.4f}; '
          f'gap: {rec["gap"][0]:.4f} -> {rec["gap"][-1]:.4f}')
    assert ga[-1] < ga[0] * 1e-2, 'HSGDA 未充分下降'
    assert rec['phi_pop'][-1] < rec['phi_pop'][0], 'Phi 未下降'

    print('===== 5) SSGDA 偏差平台 =====')
    out_s = run_ssgda(prob, T=2000, eta=0.01, gamma=0.02, seed=1,
                      eval_every=100, track_phi=True)
    gs = out_s['gphi2_all']
    print('  ||grad Phi||^2:', ' '.join(f't={s+1}:{gs[s]:.3e}' for s in show))
    print(f'  phi_pop: {out_s["records"]["phi_pop"][0]:.4f} -> '
          f'{out_s["records"]["phi_pop"][-1]:.4f}')
    print(f'  SSGDA min||g|^2={out_s["min_gphi2"]:.3e}  vs  HSGDA min||g|^2='
          f'{out["min_gphi2"]:.3e}')
    assert out_s['min_gphi2'] > 10 * max(out['min_gphi2'], 1e-12), \
        'SSGDA 未表现出偏差平台（可能本问题隐式项太弱）'

    print('===== 6) TSGDA-1 短程 =====')
    out_t = run_tsgda1(prob, T=1000, K=10, eta=0.01, gamma=0.05, seed=1,
                       eval_every=100, track_phi=True)
    gt = out_t['gphi2_all']
    print('  ||grad Phi||^2:', ' '.join(f't={s+1}:{gt[s]:.3e}' for s in [0, 99, 499, 999]))
    print(f'  phi_pop: {out_t["records"]["phi_pop"][0]:.4f} -> '
          f'{out_t["records"]["phi_pop"][-1]:.4f}')

    print(f'ALL SMOKE TESTS PASSED in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
