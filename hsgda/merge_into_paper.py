# -*- coding: utf-8 -*-
"""将 new.tex 与 new2.tex 合并进 0-PAPER.tex 的脚本（含风格修正与校验）。"""
import io
import os
import re

LATEX = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'latex'))


def read(name):
    with io.open(os.path.join(LATEX, name), 'r', encoding='utf-8') as f:
        return f.read()


def write(name, text):
    with io.open(os.path.join(LATEX, name), 'w', encoding='utf-8') as f:
        f.write(text)


paper = read('0-PAPER.tex')
new = read('new.tex')
new2 = read('new2.tex')

# ---------- 1. 从 new.tex 提取三个内容块 ----------
i_algo = new.index('\\subsection{A Hypergradient-Corrected First-Order Algorithm}')
i_conv = new.index('\\subsection{Epsilon-Convergence')
i_app = new.index('\\section{Proofs for the HSGDA Algorithm}')
i_end = new.index('%<<<PART-II-END>>>')

chunk_algo = new[i_algo:i_conv].rstrip() + '\n\n'
chunk_theory = new[i_conv:i_app].rstrip()
# 去掉 theory 块尾部的注释横幅行
while chunk_theory.splitlines()[-1].lstrip().startswith('%'):
    chunk_theory = '\n'.join(chunk_theory.splitlines()[:-1]).rstrip()
chunk_theory += '\n\n'
chunk_app = new[i_app:i_end].rstrip() + '\n\n'

# ---------- 2. 风格修正：em-dash 连结改为行文 ----------
fixes_new = [
    ('tracks $u^*(x^t)$ geometrically fast --- this is the minimax counterpart'
     ' of the momentum/Neumann tracking in',
     'tracks $u^*(x^t)$ geometrically fast, which is the minimax counterpart'
     ' of the momentum and Neumann tracking in'),
    ('evaluations --- two gradient evaluations of $f$',
     'evaluations, namely two gradient evaluations of $f$'),
    ('backward pass --- the \\emph{oracle complexity}',
     'backward pass, so the \\emph{oracle complexity}'),
    ('The structural message is unchanged --- generalization degrades'
     ' polynomially in $T$ at the rate $T^{c_*}/m_1$ --- but the constant'
     ' reveals',
     'The structural message is unchanged in that generalization degrades'
     ' polynomially in $T$ at the rate $T^{c_*}/m_1$, while the constant'
     ' reveals'),
    ('weighted asymmetrically --- the standard two-timescale device',
     'weighted asymmetrically, which is the standard two-timescale device'),
    ('$\\ell=\\max\\{\\ell_f,\\ell_g\\}$ as in the main paper,',
     '$\\ell=\\max\\{\\ell_f,\\ell_g\\}$ as in the preliminaries,'),
    ('We repeatedly use the facts, established in the main paper, that',
     'We repeatedly use the facts, established in the preliminaries, that'),
    ('The algorithms analyzed in the preceding sections share a common'
     ' structural simplification',
     'The algorithms introduced above share a common structural'
     ' simplification'),
]


def apply_fixes(text, fixes, tag):
    for old, newtext in fixes:
        if old not in text:
            print('[warn] %s: pattern not found: %r...' % (tag, old[:60]))
            continue
        text = text.replace(old, newtext)
    return text


chunk_algo = apply_fixes(chunk_algo, fixes_new, 'algo')
chunk_theory = apply_fixes(chunk_theory, fixes_new, 'theory')
chunk_app = apply_fixes(chunk_app, fixes_new, 'appendix')

# ---------- 3. 附录 Lyapunov 证明中的 itemize 改为行文 ----------
def replace_block(text, start_marker, new_block):
    i0 = text.index(start_marker)
    i1 = text.index('\\end{itemize}', i0) + len('\\end{itemize}')
    return text[:i0] + new_block + text[i1:]


block_a = (
    '\\emph{$a$-coefficient} (to be dominated by the '
    '$\\lambda_a\\frac{\\gamma\\mu}{2}a_t$-drift of \\eqref{eq_A_rec}, '
    'leaving $\\lambda_a\\frac{\\gamma\\mu}{4}$). The term '
    '$\\frac{3\\eta}{2}\\mathcal{K}_g$ is dominated since '
    '$\\frac{3\\eta}{2}\\mathcal{K}_g\\leq\\lambda_a\\frac{\\gamma\\mu}{16}'
    '=\\frac{16c_2\\eta\\mathcal{K}_g}{\\mu}$, which holds because '
    '$\\frac{3}{2}\\leq\\frac{16c_2}{\\mu}$. The term '
    '$L_\\Phi\\eta^2\\mathcal{K}_a$ is dominated by the same slot through '
    '$\\eta\\leq\\frac{16c_2\\mathcal{K}_g}{\\mu L_\\Phi\\mathcal{K}_a}$ '
    '(in \\eqref{eq_step_conditions} with margin). The cross term '
    '$\\lambda_b\\frac{16\\alpha\\mathcal{K}_g}{\\mu}'
    '=\\frac{16c_2\\eta\\mathcal{K}_g}{\\mu}$ exhausts exactly one '
    '$\\lambda_a\\frac{\\gamma\\mu}{16}$-slot, so the three slots '
    '$\\lambda_a\\frac{\\gamma\\mu}{16}\\cdot3=\\lambda_a\\frac{3\\gamma\\mu}'
    '{16}$ leave the net drift '
    '$\\lambda_a(\\frac{\\gamma\\mu}{2}-\\frac{3\\gamma\\mu}{16})'
    '=\\lambda_a\\frac{5\\gamma\\mu}{16}>0$.\n')
block_b = (
    '\\emph{$b$-coefficient} (to be dominated by the '
    '$\\lambda_b\\frac{3\\alpha\\mu}{4}b_t$-drift of \\eqref{eq_B_rec}, '
    'leaving $\\lambda_b\\frac{3\\alpha\\mu}{8}$). The term '
    '$\\frac{3\\eta}{2}L^2$ is dominated through '
    '$\\eta\\leq\\frac{\\alpha\\mu}{32L^2}=\\frac{c_2\\eta\\mu}{32L^2}$, '
    'that is, $c_2\\geq\\frac{32L^2}{\\mu}$ (in \\eqref{eq_c2}). The term '
    '$L_\\Phi\\eta^2\\mathcal{K}_b$ is dominated by '
    '$\\eta\\leq\\frac{3c_2\\mu}{64L_\\Phi\\mathcal{K}_b}$ (in '
    '\\eqref{eq_step_conditions}). For the cross term '
    '$\\lambda_a\\cdot2\\big(1+\\tfrac{2}{\\gamma\\mu}\\big)L_s^2\\eta^2'
    '\\mathcal{K}_b$, using $(1+\\frac{2}{\\gamma\\mu})\\leq\\frac{3}'
    '{\\gamma\\mu}$, the left side is '
    '$\\frac{256\\mathcal{K}_g}{\\mu^2}\\cdot\\frac{6L_s^2\\eta^2\\mathcal{K}_b}'
    '{\\gamma\\mu}=\\frac{1536\\mathcal{K}_gL_s^2\\eta\\mathcal{K}_b}{\\mu^3}$ '
    'and the right side is $\\frac{3c_2\\eta\\mu}{32}$, where the $\\eta$ '
    'cancels and the condition '
    '$c_2\\geq\\frac{256\\mathcal{K}_gL_s\\sqrt{\\mathcal{K}_b}}{\\mu^2}$ '
    '(in \\eqref{eq_c2}, up to squaring: '
    '$\\frac{1536\\mathcal{K}_gL_s^2\\mathcal{K}_b}{\\mu^3}\\leq'
    '\\frac{3c_2\\mu}{32}\\Leftrightarrow c_2^2\\geq'
    '\\frac{16384\\mathcal{K}_g^2L_s^4\\mathcal{K}_b}{\\mu^4}\\Leftrightarrow '
    'c_2\\geq\\frac{128\\mathcal{K}_gL_s\\sqrt{\\mathcal{K}_b}}{\\mu^2}$) '
    'covers it. The last term $\\lambda_b\\cdot2\\big(1+\\tfrac{2}{\\alpha\\mu}'
    '\\big)L_u^2\\eta^2\\mathcal{K}_b$ is already absorbed in '
    '\\eqref{eq_B_rec}.\n')
block_s = (
    '\\emph{$s$-coefficient} (to leave a net drift '
    '$\\leq-\\frac{\\eta}{4}s_t$ from the $-\\frac{3\\eta}{8}s_t$ of '
    '\\eqref{eq_descent}, i.e., total $s$-forcing $\\leq\\frac{\\eta}{8}$). '
    'Here $L_\\Phi\\eta^2\\leq\\frac{\\eta}{16}$ holds by '
    '$\\eta\\leq\\frac{1}{16L_\\Phi}$, the term '
    '$\\lambda_a\\cdot2\\big(1+\\tfrac{2}{\\gamma\\mu}\\big)L_s^2\\eta^2$ '
    'equals $\\frac{256\\mathcal{K}_g}{\\mu^2}\\cdot'
    '\\frac{6L_s^2\\eta^2}{\\gamma\\mu}'
    '=\\frac{1536\\mathcal{K}_gL_s^2\\eta}{c_2\\mu^3}$ and is therefore '
    'dominated by the condition $c_2\\geq\\frac{24576\\,\\mathcal{K}_gL_s^2}'
    '{\\mu^3}$ (in \\eqref{eq_c2}), and the term '
    '$\\lambda_b\\cdot2\\big(1+\\tfrac{2}{\\alpha\\mu}\\big)L_u^2\\eta^2'
    '=\\frac{6L_u^2\\eta^2}{\\alpha\\mu}=\\frac{6L_u^2\\eta}{c_2\\mu}'
    '\\leq\\frac{\\eta}{16}$ amounts to $c_2\\geq\\frac{96L_u^2}{\\mu}$ '
    '(in \\eqref{eq_c2}).\n')

chunk_app = replace_block(chunk_app, '\\emph{$a$-coefficient}', block_a)
chunk_app = replace_block(chunk_app, '\\emph{$b$-coefficient}', block_b)
chunk_app = replace_block(chunk_app, '\\emph{$s$-coefficient}', block_s)

# ---------- 4. new2.tex 的措辞与交叉引用同步 ----------
i_sec = new2.index('\\section{Empirical Validation for HSGDA}')
chunk_new2 = new2[i_sec:].rstrip() + '\n\n'
fixes_new2 = [
    ('the protocol of Figures 1 to 4 of the main paper for HSGDA',
     'the protocol of Figures \\ref{fig_gen1} to \\ref{fig_stepsize} '
     'for HSGDA'),
    ('the tradeoff reported in Figure 1 of the main paper',
     'the tradeoff reported in Figure \\ref{fig_gen1}'),
    ('the meta set size experiment in Figure 2 of the main paper',
     'the meta set size experiment in Figure \\ref{fig:figure2}'),
    ('the HSGDA counterpart of Figure 3 and Figure 4 of the main paper',
     'the HSGDA counterpart of Figures \\ref{fig:figure3} and '
     '\\ref{fig_stepsize}'),
    ('which follows the BMO formulation of the main paper',
     'which follows the BMO formulation in \\eqref{bmo_problem}'),
]
chunk_new2 = apply_fixes(chunk_new2, fixes_new2, 'new2')

# ---------- 5. 插入 0-PAPER.tex ----------
inserts = [
    ('\\subsection{Excess Risk and Error Decomposition}', chunk_algo),
    ('\\section{Empirical Validation}', chunk_theory),
    ('\\section{Conclusion} \\label{section6}', chunk_new2),
    ('\\section{Limitations and Discussions}', chunk_app),
]
for anchor, chunk in inserts:
    assert paper.count(anchor) == 1, 'anchor not unique: ' + anchor
    paper = paper.replace(anchor, chunk + anchor)

write('0-PAPER.tex', paper)

# ---------- 6. 校验 ----------
# 6a. 引用键是否都在 bib 中
bib = read('example_paper.bib')
bib_keys = set(re.findall(r'@\w+\{([^,\s]+),', bib))
cited = set()
for m in re.finditer(r'\\cite[pt]?\{([^}]*)\}', paper):
    for k in m.group(1).split(','):
        cited.add(k.strip())
missing = sorted(k for k in cited if k not in bib_keys)
print('cited keys: %d, missing in bib: %s' % (len(cited), missing or 'none'))

# 6b. 重复标签检查
labels = re.findall(r'\\label\{([^}]+)\}', paper)
dup = sorted(l for l in set(labels) if labels.count(l) > 1)
print('duplicate labels: %s' % (dup or 'none'))

# 6c. 引用是否都有定义（含图形与公式）
refs = set()
for m in re.finditer(r'\\(?:eq)?ref\{([^}]+)\}', paper):
    refs.add(m.group(1))
undefined = sorted(r for r in refs if r not in set(labels))
print('undefined refs: %s' % (undefined or 'none'))

# 6d. 插入确认
for key in ['sec_hsgda_algo', 'thm_conv_hsgda', 'thm_gen_hsgda',
            'sec_appendix_hsgda', 'sec_exp_hsgda']:
    print(key, 'occurrences:', paper.count('\\label{' + key + '}'))
print('total lines:', paper.count('\n'))
