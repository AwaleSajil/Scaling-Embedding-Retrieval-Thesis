"""Generate fig_cross_backbone.pdf: quality-retention heat map across backbones."""
import sys, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/nas/rhome/sawale/thesis')
from eval_v2.config.models import MODELS

DATA_DIR = Path('/nas/rhome/sawale/thesis/eval_v2/outputs/results/nanobeir')
FIG_DIR  = Path('/nas/rhome/sawale/thesis/docs/report/Figures')

with open(DATA_DIR / 'aggregate.json') as f:
    agg = json.load(f)

BACKBONES = [
    ('bge-base-en-v1.5', 'BGE'),
    ('roberta-base',     'RoBERTa'),
    ('mpnet-base',       'MPNet'),
]

def mrr10(key):
    return agg.get(key, {}).get('mean', {}).get('mrr@10')

def bytes_per_vec(spec):
    dim = spec.truncate_dim or 768
    if spec.similarity == 'hamming':
        return dim / 8
    if spec.pq_M is not None:
        return spec.pq_M * spec.pq_nbits / 8
    if spec.tq_bits is not None:
        return dim * spec.tq_bits / 8
    if spec.asigmoid_bits is not None:
        return dim * spec.asigmoid_bits / 8
    if spec.quant_bits is not None:
        return dim * spec.quant_bits / 8
    return dim * 4

ROWS = [
    ('MRL-256',                            'mrl-256'),
    ('INT8',                               'post-int8'),
    ('MRL-128',                            'mrl-128'),
    ('TQ-4bit',                            'tq-4bit'),
    ('MRL-64',                             'mrl-64'),
    ('TQ-2bit',                            'tq-2bit'),
    ('MRL-32',                             'mrl-32'),
    ('TQ-1bit',                            'tq-1bit'),
    (r'PQ ($M{=}96,n{=}8$)',               'pq-m96-n8'),
    ('Post-hoc binary',                    'post-binary'),
    ('BAT',                                'bat'),
    (r'BAT-atanh ($\gamma{=}0.1$)',        'atanh-gamma0.1'),
    ('MRL-512 + BAT',                      'mrl-512-bat'),
    (r'MRL-256 + PQ ($M{=}64,n{=}8$)',     'mrl-256-pq-m64-n8'),
    ('MRL-256 + BAT',                      'mrl-256-bat'),
    (r'MRL-256 + PQ ($M{=}32,n{=}8$)',     'mrl-256-pq-m32-n8'),
    (r'MRL-128 + PQ ($M{=}16,n{=}8$)',     'mrl-128-pq-m16-n8'),
    ('MRL-32 + post-binary',               'mrl-32-post-binary'),
]

baseline = {bb: mrr10(f'finetuned-{bb}') for bb, _ in BACKBONES}

retention = np.zeros((len(ROWS), len(BACKBONES)))
raw_mrr   = np.zeros_like(retention)
ratios    = []

for i, (label, suf) in enumerate(ROWS):
    row_ratios = []
    for j, (bb, _) in enumerate(BACKBONES):
        k = f'finetuned-{bb}-{suf}'
        spec = MODELS[k]
        v = mrr10(k)
        retention[i, j] = v / baseline[bb]
        raw_mrr[i, j]   = v
        row_ratios.append(3072.0 / bytes_per_vec(spec))
    assert len(set(row_ratios)) == 1, f'ratio mismatch on row {label}: {row_ratios}'
    ratios.append(row_ratios[0])

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

fig, ax = plt.subplots(figsize=(7.0, 7.6))

vmin, vmax = 0.65, 1.10
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
cmap = plt.get_cmap('RdBu_r').reversed()

im = ax.imshow(retention, aspect='auto', cmap=cmap, norm=norm)

ax.set_xticks(range(len(BACKBONES)))
ax.set_xticklabels([lab for _, lab in BACKBONES], fontsize=11, fontweight='bold')
ax.xaxis.tick_top()

row_labels = [f'{r:>4.0f}$\\times$   {lab}' for (lab, _), r in zip(ROWS, ratios)]
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels(row_labels, fontsize=9.5)

for i in range(len(ROWS)):
    for j in range(len(BACKBONES)):
        r = retention[i, j]
        txt_color = 'white' if (r < 0.78 or r > 1.05) else 'black'
        ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                fontsize=9, color=txt_color, fontweight='bold')

ax.set_xticks(np.arange(len(BACKBONES) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(ROWS) + 1) - 0.5, minor=True)
ax.grid(which='minor', color='white', lw=1.2)
ax.tick_params(which='minor', length=0)
ax.tick_params(which='major', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

prev_r = None
for i, r in enumerate(ratios):
    if prev_r is not None and r != prev_r:
        ax.axhline(i - 0.5, color='black', lw=0.9, alpha=0.7)
    prev_r = r

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label('Quality retention (compressed MRR@10 / float32 MRR@10)',
               rotation=270, labelpad=18, fontsize=10)
cbar.ax.axhline(1.0, color='black', lw=1.2)

fig.tight_layout()
out = FIG_DIR / 'fig_cross_backbone.pdf'
fig.savefig(out)
print(f'Saved -> {out}')
