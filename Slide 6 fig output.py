"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── PATH ──────────────────────────────────────────────────────
DATA = '/Users/vrisha/Desktop/BENG 203 project/'
OUT  = '/Users/vrisha/Desktop/BENG 203 project/'

# ── LOAD RANKED GENES ─────────────────────────────────────────
print("Loading ranked_biomarker_genes.csv...")
ranked = pd.read_csv(DATA + 'ranked_biomarker_genes.csv')
print(f"  {len(ranked)} genes loaded")

# ── BUILD FIGURE ──────────────────────────────────────────────
print("Building figure...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LEFT — p-value distribution across all 740 genes
# Top 50 highlighted in teal, rest in light grey
colors = ['#0891B2' if i < 50 else '#CBD5E1' for i in range(len(ranked))]

axes[0].bar(ranked['rank'], -np.log10(ranked['pvalue']),
            color=colors, width=1.0, alpha=0.85)
axes[0].axhline(y=-np.log10(0.05), color='#E11D48', linestyle='--',
                linewidth=1.5, label='p = 0.05')
axes[0].set_xlabel('Gene Rank (by DE p-value)', fontsize=11)
axes[0].set_ylabel('-log10(p-value)', fontsize=11)
axes[0].set_title('Differential Expression Ranking\n740 Biomarker Genes (R vs N patients)',
                  fontsize=12, fontweight='bold', color='#1A2B5E')
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Annotation arrow pointing to top 50 genes
axes[0].annotate('Top 50 genes\n(most significant)',
                 xy=(25, -np.log10(ranked['pvalue'].iloc[49])),
                 xytext=(120, 4.5),
                 fontsize=8.5, color='#0891B2',
                 arrowprops=dict(arrowstyle='->', color='#0891B2', lw=1.2))

# RIGHT — top 20 genes horizontal bar chart
top20 = ranked.head(20)
axes[1].barh(range(20), -np.log10(top20['pvalue'])[::-1],
             color='#0891B2', alpha=0.85)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20['gene_id'][::-1], fontsize=7.5)
axes[1].set_xlabel('-log10(p-value)', fontsize=11)
axes[1].set_title('Top 20 Most Differentially\nExpressed Biomarker Genes',
                  fontsize=12, fontweight='bold', color='#1A2B5E')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Biology-Guided Feature Selection — Mann-Whitney U Test (Zhou et al. PNAS 2019)',
             fontsize=9, color='#475569', y=0.02)
plt.tight_layout()

# ── SAVE ──────────────────────────────────────────────────────
output_path = OUT + 'slide6_feature_selection.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {output_path}")
print("Done — add this figure to the right side of Slide 6")
