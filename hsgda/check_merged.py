# -*- coding: utf-8 -*-
"""合并后的最终校验：环境配对、新内容区 em-dash 残留、接缝抽查。"""
import io
import os
import re

LATEX = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'latex'))
with io.open(os.path.join(LATEX, '0-PAPER.tex'), 'r', encoding='utf-8') as f:
    paper = f.read()

# 1. 环境配对
begins = re.findall(r'\\begin\{(\w+\*?)\}', paper)
ends = re.findall(r'\\end\{(\w+\*?)\}', paper)
from collections import Counter
cb, ce = Counter(begins), Counter(ends)
bad = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}
print('environment imbalance:', bad or 'none')

# 2. 新增内容区是否还有 "---"（em-dash 连结）
i_hsgda = paper.index('\\subsection{A Hypergradient-Corrected')
i_lim = paper.index('\\section{Limitations and Discussions}')
new_zone = paper[i_hsgda:]
math_free = re.sub(r'\$[^$]*\$', '', new_zone)          # 去掉行内公式
dashes = [(m.start(), new_zone[max(0, m.start()-60):m.start()+60].replace('\n', ' '))
          for m in re.finditer(r'---', math_free)]
print('remaining --- in new content:', len(dashes))
for _, ctx in dashes:
    print('  ...', ctx, '...')

# 3. 新内容区正文分号（去公式、去注释后）
lines = []
for ln in new_zone.splitlines():
    s = ln.split('%')[0]
    s = re.sub(r'\$[^$]*\$', '', s)
    if ';' in s:
        lines.append(ln.strip()[:100])
print('prose semicolon lines:', len(lines))
for l in lines[:10]:
    print('  ', l)

# 4. 引用键（排除注释行）是否都在 bib 中
with io.open(os.path.join(LATEX, 'example_paper.bib'), 'r', encoding='utf-8') as f:
    bib = f.read()
bib_keys = set(re.findall(r'@\w+\{([^,\s]+),', bib))
body = '\n'.join(l for l in paper.splitlines() if not l.lstrip().startswith('%'))
cited = set()
for m in re.finditer(r'\\cite[pt]?\{([^}]*)\}', body):
    for k in m.group(1).split(','):
        cited.add(k.strip())
missing = sorted(k for k in cited if k not in bib_keys)
print('live citations missing in bib:', missing or 'none')

# 5. 接缝抽查（每个插入点后两行）
for anchor in ['\\subsection{A Hypergradient-Corrected',
               '\\subsection{Epsilon-Convergence',
               '\\section{Empirical Validation for HSGDA}',
               '\\section{Proofs for the HSGDA Algorithm}']:
    i = paper.index(anchor)
    before = paper[:i].rstrip().splitlines()[-1]
    print('SEAM before %-55r : %s' % (anchor[:45], before[:80]))
