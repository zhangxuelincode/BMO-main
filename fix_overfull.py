# -*- coding: utf-8 -*-
"""批量修复 aligned 公式环境中过长的左端行：在 \\leq 前断行。"""
import re, io

PATH = r'c:\Users\10537\Desktop\BMO\latex\0-PAPER.tex'
txt = io.open(PATH, encoding='utf-8').read()

# 规则1：aligned 内 "&\mathbb{E}\left[\mathcal{R}...\right]" 首行过长，且随后 \leq
# 在 \leq 前断行： "...right]\n\leq" -> "...right]\\\n&\quad\leq"
pat1 = re.compile(r'(&\\mathbb\{E\}\\left\[\\mathcal\{R\}[^\n]{10,600}?\\right\])\s*\n(\s*)\\leq ')
n1 = 0
def r1(m):
    global n1
    n1 += 1
    return m.group(1) + ' \\\\\n&\\quad\\leq '
txt = pat1.sub(r1, txt)

# 规则2：单行 aligned 内 "&\mathbb{E}\left[\mathcal{R}...\right] \leq ..."（同一行）在 \leq 前断行
pat2 = re.compile(r'(&\\mathbb\{E\}\\left\[\\mathcal\{R\}[^\n]{10,600}?\\right\])\s*\\leq ')
n2 = 0
def r2(m):
    global n2
    n2 += 1
    return m.group(1) + ' \\\\\n&\\quad\\leq '
txt = pat2.sub(r2, txt)

io.open(PATH, 'w', encoding='utf-8').write(txt)
print('rule1 breaks:', n1, '| rule2 breaks:', n2)
