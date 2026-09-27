"""
Agent information-flow figure for the MetaCrit paper  (slopegraph design).

A cumulative-improvement slopegraph: each dataset is a line tracing how the
mean cosine similarity of the pipeline's output to the ground-truth answer
changes, relative to the Brainstorming / Backgrounding-Educator baseline, as
the response passes through each agent:

    Educator       background  (object-level baseline, fixed at 0)
    Validator      validity    (monitoring,  A_1   -> phi_up)
    Critic         critique    (control,     phi_up -> phi_dn)
    Meta-Reviewer  analysis    (synthesis,   phi_dn -> Sigma)

All values are computed DIRECTLY from the experiment results in
``semantic_similarity_results_summary.json`` (mean cosine similarity via
text-embedding-3-small) so the figure can never drift from the underlying
data. No values are hard-coded.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

# -- Paths -----------------------------------------------------------
RESULTS_DIR = os.path.join(os.getcwd(), 'results')  # run from the repository root
SUMMARY_JSON = os.path.join(RESULTS_DIR, 'semantic_similarity_results_summary.json')

# -- Style -----------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'text.usetex': False,
})

# -- Data: compute per-stage deltas from the experiment results ------
# Display name -> key used inside semantic_similarity_results_summary.json
DATASET_KEYS = {
    'BBQ':         'bbq',
    'MMLU':        'mmlu',
    'CrowS-Pairs': 'crowspairs',
    'CREAK':       'creak',
    'CIAR':        'ciar',
    'TruthfulQA':  'truthfulqa',
}
# the four agents in pipeline order
AGENT_ORDER = ['background', 'validity', 'critique', 'analysis']

with open(SUMMARY_JSON, 'r', encoding='utf-8') as fh:
    summary = json.load(fh)

DELTA_VALUES = {}
for disp, key in DATASET_KEYS.items():
    mas = summary[key]['mean_answer_similarity']
    sims = [mas[a] for a in AGENT_ORDER]
    DELTA_VALUES[disp] = [sims[i + 1] - sims[i] for i in range(3)]

datasets = list(DELTA_VALUES.keys())

# Cumulative values (starting from 0 = the Brainstorming baseline)
cumulative = {}
for name, deltas in DELTA_VALUES.items():
    c = [0.0]
    s = 0.0
    for d in deltas:
        s += d
        c.append(s)
    cumulative[name] = c

# Echo the data the figure is built from (audit trail)
print('Cumulative values computed from', os.path.basename(SUMMARY_JSON))
for name in datasets:
    c = cumulative[name]
    print('  %-12s Educator=%+.4f  Validator=%+.4f  Critic=%+.4f  Meta-Reviewer=%+.4f'
          % (name, c[0], c[1], c[2], c[3]))

# -- Colors (colorblind-friendly, print-safe) ------------------------
colors = {
    'BBQ':         '#2166AC',
    'MMLU':        '#7570B3',
    'CrowS-Pairs': '#D62728',
    'CREAK':       '#E6AB02',
    'CIAR':        '#E07B39',
    'TruthfulQA':  '#1B9E77',
}

# -- Figure: cumulative-improvement slopegraph -----------------------
agent_labels = ['Educator', 'Validator', 'Critic', 'Meta-Reviewer']
x = np.arange(4)

fig, ax = plt.subplots(figsize=(9.4, 6.4))

# baseline reference
ax.axhline(0, color='#9aa0a6', ls=(0, (6, 4)), lw=1.6, zorder=1)

for name in datasets:
    y = cumulative[name]
    ax.plot(x, y, color=colors[name], marker='o', markersize=8.5,
            markeredgecolor='white', markeredgewidth=1.4,
            linewidth=2.6, zorder=3, label=name)

# value labels at every node, with centred vertical dodging
all_vals = [v for n in datasets for v in cumulative[n]]
span = max(all_vals) - min(all_vals)
gap = span * 0.062


def dodge(vals):
    """Push apart sorted values that are closer than `gap`, then re-centre."""
    out = list(vals)
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < gap:
            out[i] = out[i - 1] + gap
    shift = (sum(vals) - sum(out)) / len(out)
    return [o + shift for o in out]


for xi in range(4):
    if xi == 0:
        ax.text(-0.10, 0.0, '+0.0000', ha='right', va='center',
                fontsize=9, color='#444444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#bbbbbb', lw=1.1), zorder=5)
        continue
    items = sorted(((cumulative[n][xi], n) for n in datasets),
                   key=lambda t: t[0])
    label_y = dodge([v for v, _ in items])
    dx = 0.13 if xi == 3 else 0.0
    ha = 'left' if xi == 3 else 'center'
    for (val, name), ly in zip(items, label_y):
        if abs(ly - val) > gap * 0.35:          # leader line when displaced
            ax.plot([xi, xi + dx], [val, ly], color=colors[name],
                    lw=0.8, alpha=0.55, zorder=2)
        ax.annotate('%+.4f' % val, xy=(xi, val), xytext=(xi + dx, ly),
                    ha=ha, va='center', fontsize=8.5, color=colors[name],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.28', fc='white',
                              ec=colors[name], lw=1.3), zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(agent_labels)
ax.set_xlim(-0.45, 3.7)
ax.set_ylim(min(all_vals) - 0.013, max(all_vals) + 0.016)
ax.set_xlabel('Agent', labelpad=8)
ax.set_ylabel('Cumulative Improvement vs. Baseline', labelpad=8)
ax.set_title('Cumulative Similarity Improvement Through Pipeline\n'
             '(Relative to Backgrounding Educator Baseline)',
             fontweight='bold', pad=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.2f'))
ax.grid(axis='y', linewidth=0.5, alpha=0.5, color='#dddddd')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles = [Line2D([0], [0], color=colors[n], marker='o', markersize=8,
                   markeredgecolor='white', markeredgewidth=1.2,
                   linewidth=2.6, label=n) for n in datasets]
handles.append(Line2D([0], [0], color='#9aa0a6', ls=(0, (6, 4)),
                       lw=1.6, label='Baseline (Educator)'))
ax.legend(handles=handles, title='Dataset', loc='upper left',
          frameon=False, handlelength=2.4, borderpad=0.6,
          labelspacing=0.6)

plt.tight_layout()

# -- Save -------------
outputs = [
    os.path.join(RESULTS_DIR, 'figure_pipeline.png'),
    os.path.join(RESULTS_DIR, 'figure_pipeline.pdf'),
]
for path in outputs:
    plt.savefig(path, dpi=300, bbox_inches='tight',
                pad_inches=0.12, facecolor='white')
    print('Saved', path)

print('Done')
