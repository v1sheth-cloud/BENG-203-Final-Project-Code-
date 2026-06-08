import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

DATA = 'data/'
OUT  = 'data/'

X_t1 = pd.read_csv(DATA + 'task1_feature_matrix.csv', index_col=0)
y_t1 = pd.read_csv(DATA + 'task1_labels.csv', index_col=0).values.ravel()
X_t2 = pd.read_csv(DATA + 'task2_feature_matrix.csv', index_col=0)
y_t2 = pd.read_csv(DATA + 'task2_labels.csv', index_col=0).values.ravel()

gene_names = X_t1.columns.tolist()

rf_t1 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_t1.fit(X_t1.values, y_t1)
imp_t1 = pd.Series(rf_t1.feature_importances_, index=gene_names).sort_values(ascending=False)

rf_t2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_t2.fit(X_t2.values, y_t2)
imp_t2 = pd.Series(rf_t2.feature_importances_, index=gene_names).sort_values(ascending=False)

top20_t1 = imp_t1.head(20)
top20_t2 = imp_t2.head(20)

overlap = set(top20_t1.index) & set(top20_t2.index)
print(f"Top 5 Task 1 genes: {list(top20_t1.index[:5])}")
print(f"Top 5 Task 2 genes: {list(top20_t2.index[:5])}")
print(f"Overlap between top 20: {len(overlap)} genes")

NAVY  = '#1A2B5E'
CYAN  = '#0891B2'
GREEN = '#059669'
known = {'ESR1', 'ERBB2', 'MKI67', 'BRCA1', 'CCND1'}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Feature Importance & Biological Insights\nTop 20 Genes by Random Forest Importance Score',
             fontsize=13, fontweight='bold', color=NAVY)

colors_t1 = [GREEN if g in known else CYAN for g in top20_t1.index[::-1]]
axes[0].barh(range(20), top20_t1.values[::-1], color=colors_t1, alpha=0.85)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top20_t1.index[::-1], fontsize=8)
axes[0].set_xlabel('Feature Importance Score', fontsize=11)
axes[0].set_title('Task 1 — Cancer vs. Normal', fontsize=12, fontweight='bold', color=NAVY)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

colors_t2 = [GREEN if g in known else CYAN for g in top20_t2.index[::-1]]
axes[1].barh(range(20), top20_t2.values[::-1], color=colors_t2, alpha=0.85)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20_t2.index[::-1], fontsize=8)
axes[1].set_xlabel('Feature Importance Score', fontsize=11)
axes[1].set_title('Task 2 — Recurrence vs. Non-Recurrence', fontsize=12, fontweight='bold', color=NAVY)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

legend_elements = [Patch(facecolor=GREEN, label='Known BC marker'),
                   Patch(facecolor=CYAN,  label='Other biomarker gene')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT + 'slide12_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: slide12_feature_importance.png")
