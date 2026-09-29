# -*- coding: utf-8 -*-
"""精确检测新增内容中的正文分号：剥离显示公式与行内公式后检查。"""
import io
import os
import re

LATEX = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'latex'))
with io.open(os.path.join(LATEX, '0-PAPER.tex'), 'r', encoding='utf-8') as f:
    paper = f.read()

i_hsgda = paper.index('\\subsection{A Hypergradient-Corrected')
new_zone = paper[i_hsgda:]

# 去掉注释
lines = [l for l in new_zone.splitlines() if not l.lstrip().startswith('%')]
text = '\n'.join(lines)
# 去掉显示公式环境
for env in ['equation', 'equation*', 'align', 'align*', 'cases', 'aligned',
            'algorithm', 'algorithmic', 'figure*', 'figure', 'equation*']:
    text = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}',
                  ' DISPLAYMATH ', text, flags=re.S)
# 去掉 \[...\] 与 \(...\)
text = re.sub(r'\\\[.*?\\\]', ' DISPLAYMATH ', text, flags=re.S)
text = re.sub(r'\\\(.*?\\\)', ' MATH ', text, flags=re.S)
# 去掉行内公式 $...$ 与 \( \)、命令参数中的嵌套花括号内容简化处理
text = re.sub(r'\$[^$]*\$', ' MATH ', text)
# 去掉 \eqref/\ref/\cite 等
text = re.sub(r'\\(?:eq)?ref\{[^}]*\}', ' REF ', text)
text = re.sub(r'\\cite[pt]?\{[^}]*\}', ' CITE ', text)
text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})*', ' CMD ', text)

hits = []
for ln in text.splitlines():
    if ';' in ln:
        hits.append(ln.strip()[:110])
print('genuine prose semicolons:', len(hits))
for h in hits:
    print('  ', h)
