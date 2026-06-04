"""

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── PATHS — update if needed ──────────────────────────────────
BASE = '/Users/vrisha/Desktop/BENG 203 project/'
OUT  = BASE  # figures saved here

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# Cancer TPM (genes x 96 samples, no header)
tpm_cancer = pd.read_csv(BASE + 'pnas_tpm_96_nodup.txt',
                         sep='\t', index_col=0, header=None)
tpm_cancer.columns = [f'C{i}' for i in range(1, 97)]
print(f"Cancer TPM shape: {tpm_cancer.shape}")

# Normal TPM (genes x 32 samples, has header)
tpm_normal = pd.read_csv(BASE + 'pnas_normal_tpm.txt',
                         sep='\t', index_col=0)
print(f"Normal TPM shape: {tpm_normal.shape}")

# Patient info
info = pd.read_csv(BASE + 'pnas_patient_info.csv')
labels_recur = info['recurStatus'].values  # 'R' or 'N'
print(f"Patient info shape: {info.shape}")
print(f"Recurrence counts: {pd.Series(labels_recur).value_counts().to_dict()}")

# Preselected 750 biomarker genes
with open(BASE + 'preselectedList.txt') as f:
    biomarker_genes = [l.strip() for l in f if l.strip()]
print(f"Biomarker genes loaded: {len(biomarker_genes)}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

def preprocess(tpm_df):
    """Log2(TPM + 1) transform, filter low-expression genes."""
    log_tpm = np.log2(tpm_df + 1)
    # Filter: keep genes with mean log-TPM >= 0.5
    keep = log_tpm.mean(axis=1) >= 0.5
    return log_tpm[keep]

log_cancer = preprocess(tpm_cancer)
log_normal = preprocess(tpm_normal)

# Filter to common genes
common_genes = log_cancer.index.intersection(log_normal.index)
log_cancer = log_cancer.loc[common_genes]
log_normal = log_normal.loc[common_genes]
print(f"Genes after filtering: {len(common_genes)}")

# Filter biomarker genes to those present in our data
bio_available = [g for g in biomarker_genes if g in common_genes]
print(f"Biomarker genes available in data: {len(bio_available)}")

# ─────────────────────────────────────────────────────────────
# 3. BIOLOGY-GUIDED FEATURE SELECTION (Mann-Whitney U)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Biology-guided feature selection")
print("=" * 60)

# Rank 750 biomarker genes by differential expression p-value
# between recurrent (R) and non-recurrent (N) patients
r_idx = np.where(labels_recur == 'R')[0]
n_idx = np.where(labels_recur == 'N')[0]

bio_tpm = log_cancer.loc[bio_available]

pvals = []
for gene in bio_available:
    r_vals = bio_tpm.loc[gene].iloc[r_idx].values
    n_vals = bio_tpm.loc[gene].iloc[n_idx].values
    _, p = mannwhitneyu(r_vals, n_vals, alternative='two-sided')
    pvals.append(p)

gene_pvals = pd.Series(pvals, index=bio_available).sort_values()
print(f"Top 10 most differentially expressed biomarker genes:")
print(gene_pvals.head(10))

# Ranked gene list (most significant first)
ranked_genes = gene_pvals.index.tolist()

# ─────────────────────────────────────────────────────────────
# 4. TASK 1: CANCER vs. NORMAL CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Task 1 — Cancer vs. Normal")
print("=" * 60)

# Combine cancer + normal, use top 200 biomarker genes
TOP_N_TASK1 = 200
genes_t1 = ranked_genes[:TOP_N_TASK1]

X_cancer = log_cancer.loc[genes_t1].T.values  # 96 x TOP_N
X_normal = log_normal.loc[[g for g in genes_t1 if g in log_normal.index]].T.values

# Align genes
genes_t1_avail = [g for g in genes_t1 if g in log_normal.index]
X_cancer_t1 = log_cancer.loc[genes_t1_avail].T.values
X_normal_t1 = log_normal.loc[genes_t1_avail].T.values

X_t1 = np.vstack([X_cancer_t1, X_normal_t1])
y_t1 = np.array([1]*96 + [0]*32)  # 1=cancer, 0=normal
print(f"Task 1 — X shape: {X_t1.shape}, y: {pd.Series(y_t1).value_counts().to_dict()}")

# Classifiers
classifiers_t1 = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                            random_state=42),
    'Logistic Reg (L2)': LogisticRegression(C=0.1, penalty='l2',
                                             max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB()
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_t1 = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#0891B2', '#7C3AED', '#059669']

for (name, clf), color in zip(classifiers_t1.items(), colors):
    train_aucs, val_aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for train_idx, val_idx in skf.split(X_t1, y_t1):
        X_tr, X_val = X_t1[train_idx], X_t1[val_idx]
        y_tr, y_val = y_t1[train_idx], y_t1[val_idx]

        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_prob_tr = clf.predict_proba(X_tr)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        val_auc = auc(fpr, tpr)
        train_auc = auc(*roc_curve(y_tr, y_prob_tr)[:2])

        val_aucs.append(val_auc)
        train_aucs.append(train_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(val_aucs)
    mean_train_auc = np.mean(train_aucs)
    gap = mean_train_auc - mean_auc

    results_t1[name] = {
        'train_auc': round(mean_train_auc, 3),
        'val_auc': round(mean_auc, 3),
        'gap': round(gap, 3)
    }

    axes[0].plot(mean_fpr, mean_tpr, color=color, lw=2,
                 label=f'{name} (AUC={mean_auc:.3f})')

axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('Task 1 — ROC Curves\nCancer vs. Normal', fontsize=12,
                  fontweight='bold', color='#1A2B5E')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Confusion matrix for best classifier (RF)
rf_t1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
from sklearn.model_selection import cross_val_predict
y_pred_t1 = cross_val_predict(rf_t1, X_t1, y_t1, cv=skf)
cm_t1 = confusion_matrix(y_t1, y_pred_t1)

im = axes[1].imshow(cm_t1, cmap='Blues')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Normal','Cancer']); axes[1].set_yticklabels(['Normal','Cancer'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title('Task 1 — Confusion Matrix\n(Random Forest)', fontsize=12,
                  fontweight='bold', color='#1A2B5E')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm_t1[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm_t1[i,j] > cm_t1.max()/2 else 'black')

plt.tight_layout()
plt.savefig(OUT + 'slide9_task1_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Task 1 results saved → slide9_task1_results.png")
print("Task 1 metrics:", results_t1)

# ─────────────────────────────────────────────────────────────
# 5. FEATURE SUBSPACE SWEEP (Slide 7)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature subspace comparison")
print("=" * 60)

subsets = [50, 100, 150, 200, 300, 500, len(bio_available)]
subset_aucs_rf = []
subset_aucs_lr = []

for n in subsets:
    top_genes = ranked_genes[:n]
    avail = [g for g in top_genes if g in log_normal.index]
    Xc = log_cancer.loc[avail].T.values
    Xn = log_normal.loc[avail].T.values
    X = np.vstack([Xc, Xn])
    y = np.array([1]*96 + [0]*32)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

    auc_rf = cross_val_score(rf, X, y, cv=skf, scoring='roc_auc').mean()
    auc_lr = cross_val_score(lr, X, y, cv=skf, scoring='roc_auc').mean()
    subset_aucs_rf.append(auc_rf)
    subset_aucs_lr.append(auc_lr)
    print(f"  Top {n} genes — RF AUC: {auc_rf:.3f}, LR AUC: {auc_lr:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(subsets, subset_aucs_rf, 'o-', color='#0891B2', lw=2,
        markersize=7, label='Random Forest')
ax.plot(subsets, subset_aucs_lr, 's--', color='#7C3AED', lw=2,
        markersize=7, label='Logistic Reg (L2)')
ax.set_xlabel('Number of Biomarker Genes', fontsize=12)
ax.set_ylabel('Mean AUC (5-fold CV)', fontsize=12)
ax.set_title('Feature Subspace Comparison\nAUC vs. Number of Genes',
             fontsize=13, fontweight='bold', color='#1A2B5E')
ax.legend(fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.axhline(y=max(subset_aucs_rf), color='#0891B2', linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(subsets)
plt.tight_layout()
plt.savefig(OUT + 'slide7_feature_subspace.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature subspace figure saved → slide7_feature_subspace.png")

best_n = subsets[np.argmax(subset_aucs_rf)]
print(f"Optimal gene count: {best_n} genes (AUC = {max(subset_aucs_rf):.3f})")

# ─────────────────────────────────────────────────────────────
# 6. TASK 2: RECURRENCE vs. NON-RECURRENCE (Slide 10)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Task 2 — Recurrence vs. Non-Recurrence")
print("=" * 60)

TOP_N_TASK2 = best_n
genes_t2 = ranked_genes[:TOP_N_TASK2]
X_t2 = log_cancer.loc[genes_t2].T.values
y_t2 = (labels_recur == 'R').astype(int)  # 1=recurrent, 0=non-recurrent
print(f"Task 2 — X shape: {X_t2.shape}")
print(f"Class balance: R={y_t2.sum()}, N={len(y_t2)-y_t2.sum()}")

classifiers_t2 = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5,
                                            random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Reg (L2)': LogisticRegression(C=0.01, penalty='l2',
                                             max_iter=1000, random_state=42)
}

results_t2 = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for (name, clf), color in zip(classifiers_t2.items(), colors):
    train_aucs, val_aucs = [], []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, val_idx in skf.split(X_t2, y_t2):
        X_tr, X_val = X_t2[train_idx], X_t2[val_idx]
        y_tr, y_val = y_t2[train_idx], y_t2[val_idx]

        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_prob_tr = clf.predict_proba(X_tr)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        val_auc = auc(fpr, tpr)
        train_auc = auc(*roc_curve(y_tr, y_prob_tr)[:2])

        val_aucs.append(val_auc)
        train_aucs.append(train_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(val_aucs)
    mean_train_auc = np.mean(train_aucs)

    results_t2[name] = {
        'train_auc': round(mean_train_auc, 3),
        'val_auc': round(mean_auc, 3),
        'gap': round(mean_train_auc - mean_auc, 3)
    }

    axes[0].plot(mean_fpr, mean_tpr, color=color, lw=2,
                 label=f'{name} (AUC={mean_auc:.3f})')

axes[0].plot([0,1],[0,1],'k--', alpha=0.4, label='Random chance')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('Task 2 — ROC Curves\nRecurrence vs. Non-Recurrence\n'
                  '(low AUC expected — biologically hard problem)',
                  fontsize=11, fontweight='bold', color='#1A2B5E')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Confusion matrix for RF Task 2
rf_t2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
y_pred_t2 = cross_val_predict(rf_t2, X_t2, y_t2, cv=skf)
cm_t2 = confusion_matrix(y_t2, y_pred_t2)

im = axes[1].imshow(cm_t2, cmap='Purples')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Non-Recurrent','Recurrent'])
axes[1].set_yticklabels(['Non-Recurrent','Recurrent'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title('Task 2 — Confusion Matrix\n(Random Forest)',
                  fontsize=12, fontweight='bold', color='#1A2B5E')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm_t2[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm_t2[i,j] > cm_t2.max()/2 else 'black')

plt.tight_layout()
plt.savefig(OUT + 'slide10_task2_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Task 2 results saved → slide10_task2_results.png")
print("Task 2 metrics:", results_t2)

# ─────────────────────────────────────────────────────────────
# 7. CLASSIFIER COMPARISON TABLE (Slide 11)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Classifier comparison table")
print("=" * 60)

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')

rows = []
for name in ['Random Forest', 'Logistic Reg (L2)', 'Naive Bayes']:
    t1 = results_t1.get(name, {})
    t2 = results_t2.get(name, {})
    rows.append([
        name,
        str(t1.get('train_auc','—')),
        str(t1.get('val_auc','—')),
        str(t1.get('gap','—')),
        str(t2.get('train_auc','—')),
        str(t2.get('val_auc','—')),
        str(t2.get('gap','—')),
    ])

cols = ['Classifier',
        'T1 Train AUC', 'T1 Val AUC', 'T1 Gap',
        'T2 Train AUC', 'T2 Val AUC', 'T2 Gap']

table = ax.table(cellText=rows, colLabels=cols,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Header styling
for j in range(len(cols)):
    table[0, j].set_facecolor('#1A2B5E')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best val AUC per task
best_t1 = max(range(3), key=lambda i: float(rows[i][2]))
best_t2 = max(range(3), key=lambda i: float(rows[i][5]))
table[best_t1+1, 2].set_facecolor('#BBF7D0')
table[best_t2+1, 5].set_facecolor('#BBF7D0')

ax.set_title('Classifier Comparison — Task 1 (Cancer/Normal) vs. Task 2 (Recurrence)',
             fontsize=12, fontweight='bold', color='#1A2B5E', pad=20)
plt.tight_layout()
plt.savefig(OUT + 'slide11_comparison_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("Comparison table saved → slide11_comparison_table.png")

# ─────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Slide 12)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature importance")
print("=" * 60)

rf_final = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_final.fit(X_t2, y_t2)
importances = pd.Series(rf_final.feature_importances_, index=genes_t2)
top_genes_imp = importances.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(len(top_genes_imp)), top_genes_imp.values[::-1],
               color='#0891B2', alpha=0.85)
ax.set_yticks(range(len(top_genes_imp)))
ax.set_yticklabels(top_genes_imp.index[::-1], fontsize=9)
ax.set_xlabel('Feature Importance Score', fontsize=11)
ax.set_title('Top 15 Genes by Random Forest Importance\n(Task 2 — Recurrence Classifier)',
             fontsize=12, fontweight='bold', color='#1A2B5E')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + 'slide12_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature importance saved → slide12_feature_importance.png")
print("\nTop 15 genes by importance:")
print(top_genes_imp)

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE — figures saved to:", OUT)
print("Files generated:")
print("  slide7_feature_subspace.png")
print("  slide9_task1_results.png")
print("  slide10_task2_results.png")
print("  slide11_comparison_table.png")
print("  slide12_feature_importance.png")
print("=" * 60)
