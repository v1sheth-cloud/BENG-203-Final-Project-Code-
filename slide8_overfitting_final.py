import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

DATA = 'data/'
OUT  = 'data/'

X_t2 = pd.read_csv(DATA + 'task2_feature_matrix.csv', index_col=0).values
y_t2 = pd.read_csv(DATA + 'task2_labels.csv', index_col=0).values.ravel()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_train_val_auc(clf, X, y):
    train_aucs, val_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        y_prob_val = clf.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob_val)
        val_aucs.append(auc(fpr, tpr))
        y_prob_tr = clf.predict_proba(X_tr)[:, 1]
        fpr_tr, tpr_tr, _ = roc_curve(y_tr, y_prob_tr)
        train_aucs.append(auc(fpr_tr, tpr_tr))
    return round(float(np.mean(train_aucs)),3), round(float(np.mean(val_aucs)),3), round(float(np.mean(train_aucs))-float(np.mean(val_aucs)),3)

lr_results = []
for c_val in [0.01, 0.1, 1.0]:
    clf = LogisticRegression(C=c_val, penalty='l2', max_iter=1000, random_state=42)
    tr, val, gap = get_train_val_auc(clf, X_t2, y_t2)
    lr_results.append({'Method': f'C={c_val}', 'Train AUC': tr, 'Val AUC': val, 'Gap': gap})
    print(f"LR C={c_val}: train={tr} | val={val} | gap={gap}")

rf_results = []
for depth, label in [(5,'depth=5'), (10,'depth=10'), (None,'depth=None')]:
    clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    tr, val, gap = get_train_val_auc(clf, X_t2, y_t2)
    rf_results.append({'Method': label, 'Train AUC': tr, 'Val AUC': val, 'Gap': gap})
    print(f"RF {label}: train={tr} | val={val} | gap={gap}")

NAVY  = '#1A2B5E'
CYAN  = '#0891B2'
GREEN = '#059669'
RED   = '#DC2626'
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Overfitting Mitigation & Cross-Validation\nTask 2: Recurrence vs. Non-Recurrence (5-fold CV)',
             fontsize=13, fontweight='bold', color=NAVY)

x = np.arange(3)
lr_train = [r['Train AUC'] for r in lr_results]
lr_val   = [r['Val AUC']   for r in lr_results]
axes[0].bar(x - w/2, lr_train, w, label='Train AUC', color=NAVY, alpha=0.85)
axes[0].bar(x + w/2, lr_val,   w, label='Val AUC',   color=CYAN, alpha=0.85)
for i, (tr, val) in enumerate(zip(lr_train, lr_val)):
    gap = round(tr - val, 3)
    axes[0].annotate(f'gap={gap}', xy=(x[i], max(tr,val)+0.02),
                     ha='center', fontsize=9, color=GREEN if gap < 0.25 else RED, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([r['Method'] for r in lr_results], fontsize=11)
axes[0].set_ylim(0, 1.18)
axes[0].set_ylabel('AUC', fontsize=11)
axes[0].set_title('Logistic Regression — L2 Regularization', fontsize=11, fontweight='bold', color=NAVY)
axes[0].legend(fontsize=10)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

x2 = np.arange(3)
rf_train = [r['Train AUC'] for r in rf_results]
rf_val   = [r['Val AUC']   for r in rf_results]
axes[1].bar(x2 - w/2, rf_train, w, label='Train AUC', color=NAVY,  alpha=0.85)
axes[1].bar(x2 + w/2, rf_val,   w, label='Val AUC',   color=GREEN, alpha=0.85)
for i, (tr, val) in enumerate(zip(rf_train, rf_val)):
    gap = round(tr - val, 3)
    axes[1].annotate(f'gap={gap}', xy=(x2[i], max(tr,val)+0.02),
                     ha='center', fontsize=9, color=GREEN if gap < 0.25 else RED, fontweight='bold')
axes[1].set_xticks(x2)
axes[1].set_xticklabels([r['Method'] for r in rf_results], fontsize=11)
axes[1].set_ylim(0, 1.18)
axes[1].set_ylabel('AUC', fontsize=11)
axes[1].set_title('Random Forest — Max Depth Tuning', fontsize=11, fontweight='bold', color=NAVY)
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT + 'slide8_overfitting.png', dpi=300, bbox_inches='tight')
print("Saved: slide8_overfitting.png")
