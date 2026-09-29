import re, io
txt = io.open(r'c:\Users\10537\Desktop\BMO\latex\0-PAPER.tex', encoding='utf-8').read()
labels = re.findall(r'\\label\{([^}]+)\}', txt)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', txt)
lset, rset = set(labels), set(refs)
missing = sorted(r for r in rset if r not in lset)
dups = sorted(set(l for l in labels if labels.count(l) > 1))
print('labels:', len(labels), 'refs:', len(refs))
print('missing refs:', missing)
print('duplicate labels:', dups)
for env in ['equation', 'figure', 'table', 'tabular', 'lemma', 'theorem',
            'assumption', 'remark', 'corollary', 'proof', 'algorithm',
            'algorithmic', 'aligned', 'cases', 'pmatrix', 'bmatrix']:
    b = len(re.findall(r'\\begin\{' + env + r'\}', txt))
    e = len(re.findall(r'\\end\{' + env + r'\}', txt))
    if b != e:
        print('MISMATCH', env, b, e)
print('env check done')
print('braces:', txt.count('{'), txt.count('}'))
# check cited bib keys exist
bib = io.open(r'c:\Users\10537\Desktop\BMO\latex\example_paper.bib', encoding='utf-8').read()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cites = set()
for m in re.findall(r'\\citep?\{([^}]+)\}', txt):
    for k in m.split(','):
        cites.add(k.strip())
miss_bib = sorted(c for c in cites if c not in keys)
print('missing bib keys:', miss_bib)
