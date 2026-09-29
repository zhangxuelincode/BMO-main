# -*- coding: utf-8 -*-
"""修复剩余 8 处超宽 display 公式。"""
import io

PATH = r'c:\Users\10537\Desktop\BMO\latex\0-PAPER.tex'
txt = io.open(PATH, encoding='utf-8').read()

pairs = []

# 1. eq_conv_bound
pairs.append((
r'''\begin{equation}\label{eq_conv_bound}
\min_{0\leq t<T}\ \mathbb{E}\big\|\nabla\Phi(x^{t})\big\|_2^2
\;\leq\;
\frac{\mathcal{C}_{\mathrm{conv}}}{\sqrt{T}},
\qquad
\mathcal{C}_{\mathrm{conv}}:=\frac{4\Delta_0}{\eta}+4\eta\,\mathcal{C}_{\mathcal{S}},
\end{equation}''',
r'''\begin{equation}\label{eq_conv_bound}
\begin{aligned}
\min_{0\leq t<T}\ \mathbb{E}\big\|\nabla\Phi(x^{t})\big\|_2^2
&\;\leq\;\frac{\mathcal{C}_{\mathrm{conv}}}{\sqrt{T}},\\
\mathcal{C}_{\mathrm{conv}}&:=\frac{4\Delta_0}{\eta}+4\eta\,\mathcal{C}_{\mathcal{S}},
\end{aligned}
\end{equation}'''))

# 2. eq_gen_bound_hsgda cases: use slashed fractions
pairs.append((
r'''\begin{cases}
\mathcal{O}\Big(\dfrac{\ell_f^2(c_\eta+c_\alpha)\,T^{c_*}}{c_*\,L_0\,m_1}\Big), & \text{if } 0<c_*<1,\\[2ex]
\mathcal{O}\Big(\dfrac{\ell_f^2(c_\eta+c_\alpha)\,T\ln(T)}{L_0\,m_1}\Big), & \text{if } c_*=1,\\[2ex]
\mathcal{O}\Big(\dfrac{\ell_f^2(c_\eta+c_\alpha)\,T^{c_*}}{L_0\,m_1}\Big), & \text{if } c_*>1 .
\end{cases}''',
r'''\begin{cases}
\mathcal{O}\big(\ell_f^2(c_\eta{+}c_\alpha)\,T^{c_*}/(c_*\,L_0\,m_1)\big), & \text{if } 0<c_*<1,\\[1ex]
\mathcal{O}\big(\ell_f^2(c_\eta{+}c_\alpha)\,T\ln(T)/(L_0\,m_1)\big), & \text{if } c_*=1,\\[1ex]
\mathcal{O}\big(\ell_f^2(c_\eta{+}c_\alpha)\,T^{c_*}/(L_0\,m_1)\big), & \text{if } c_*>1 .
\end{cases}'''))

# 3. eq_ab_explicit: split the b-bar line
pairs.append((
r'''\bar{a}\leq\frac{8\gamma\sigma_g^2}{\mu}
+\frac{8L_s^2\eta^2}{\gamma\mu}\big(s_{\max}+\mathcal{K}_b\bar{b}+\mathcal{K}_3\big),
\qquad
\bar{b}\leq\frac{16\alpha\mathcal{K}_g}{\mu}\bar{a}
+\frac{16\alpha\sigma_f^2}{\mu}+\frac{32\alpha\sigma_H^2U_*^2}{\mu}
+\frac{8L_u^2\eta^2}{\alpha\mu}\big(s_{\max}+\mathcal{K}_b\bar{b}+\mathcal{K}_3\big),''',
r'''\begin{aligned}
\bar{a}&\leq\frac{8\gamma\sigma_g^2}{\mu}
+\frac{8L_s^2\eta^2}{\gamma\mu}\big(s_{\max}+\mathcal{K}_b\bar{b}+\mathcal{K}_3\big),\\
\bar{b}&\leq\frac{16\alpha\mathcal{K}_g}{\mu}\bar{a}
+\frac{16\alpha\sigma_f^2}{\mu}+\frac{32\alpha\sigma_H^2U_*^2}{\mu}
+\frac{8L_u^2\eta^2}{\alpha\mu}\big(s_{\max}+\mathcal{K}_b\bar{b}+\mathcal{K}_3\big),
\end{aligned}'''))

# 4. eq_Ktrack: two lines
pairs.append((
r'''\mathcal{K}_{\mathrm{track}}
:=\frac{c_2\big(\mathcal{K}_g+16c_2L^2/\mu\big)\big(8\sigma_g^2+16\sigma_f^2+32\sigma_H^2U_*^2\big)}{\mu}
+\frac{16\big(\mathcal{K}_gL_s^2+L^2L_u^2\big)\big(s_{\max}+\mathcal{K}_3\big)}{c_2\mu^2}''',
r'''\begin{aligned}
\mathcal{K}_{\mathrm{track}}
&:=\frac{c_2\big(\mathcal{K}_g+16c_2L^2/\mu\big)\big(8\sigma_g^2+16\sigma_f^2+32\sigma_H^2U_*^2\big)}{\mu}\\
&\quad+\frac{16\big(\mathcal{K}_gL_s^2+L^2L_u^2\big)\big(s_{\max}+\mathcal{K}_3\big)}{c_2\mu^2}
\end{aligned}'''))

# 5. eq_excess_rate: split
pairs.append((
r'''\mathbb{E}\big[\mathcal{R}(A)-\mathcal{R}^{*}(A)\big]
\;\leq\;
\mathcal{O}\!\left(\frac{c_2\,\mathcal{A}_0\,V^2}{\ln(m_1)}\right)
=\mathcal{O}\!\left(\frac{\mathcal{C}_{\mathrm{HSGDA}}\,V^2}{\ln(m_1)}\right),
\qquad
\mathcal{C}_{\mathrm{HSGDA}}:=c_2\,\mathcal{A}_0 ,''',
r'''\begin{aligned}
\mathbb{E}\big[\mathcal{R}(A)-\mathcal{R}^{*}(A)\big]
&\;\leq\;
\mathcal{O}\!\left(\frac{c_2\,\mathcal{A}_0\,V^2}{\ln(m_1)}\right)
=\mathcal{O}\!\left(\frac{\mathcal{C}_{\mathrm{HSGDA}}\,V^2}{\ln(m_1)}\right),\\
\mathcal{C}_{\mathrm{HSGDA}}&:=c_2\,\mathcal{A}_0 ,
\end{aligned}'''))

# 6/7. appendix single-column: long movement-term line (two occurrences) — single-column
# 溢出幅度较小（约 100pt），且公式为根号下的长和，断行代价高；保持原样，仅统计。
old_move = r'''&+\frac{1}{m_1}\mathbb{E}\left[\sqrt{(\eta^t)^2 \|\nabla_{x} f(x^{t\prime},y^{t,0\prime},z^{t,0\prime};\tilde{\xi}_i)\|^2 + \sum_{k=0}^{K-1}(\gamma_1^k)^2\|\nabla_{y} g(x^{t,k\prime},y^{t,k\prime},z^{t,k\prime};\tilde{\zeta}_{k})\|^2}+ \right.'''
cnt = txt.count(old_move)
print('move-line occurrences (left as-is):', cnt)

# 8. eq_klevel: three-line formulation
pairs.append((
r'''\min_{w_1}\ \Phi_1(w_1):=\mathbb{E}\big[F_1(w_1,w_2^*(w_1))\big],
\quad
w_k^*(w_{k-1})=\arg\min_{w_k}\mathbb{E}\big[F_k(w_{k-1},w_k,w_{k+1}^*(w_{k-1},w_k))\big],
\quad
(y^*(w),z^*(w))=\arg\min_y\arg\max_z\ \mathbb{E}\big[F_K(w,y,z)\big],''',
r'''\begin{gathered}
\min_{w_1}\ \Phi_1(w_1):=\mathbb{E}\big[F_1(w_1,w_2^*(w_1))\big],\\
w_k^*(w_{k-1})=\arg\min_{w_k}\mathbb{E}\big[F_k(w_{k-1},w_k,w_{k+1}^*(w_{k-1},w_k))\big],\quad k=2,\ldots,K-1,\\
\big(y^*(w),z^*(w)\big)=\arg\min_y\arg\max_z\ \mathbb{E}\big[F_K(w,y,z)\big],
\end{gathered}'''))

n = 0
for old, new in pairs:
    if old in txt:
        txt = txt.replace(old, new)
        n += 1
    else:
        print('NOT FOUND:', old[:70].replace('\n', ' '))
io.open(PATH, 'w', encoding='utf-8').write(txt)
print('applied', n, 'of', len(pairs))
