"""



import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os


BASE = '/Users/vrisha/Desktop/BENG 203 project/'
CODES = BASE + 'codes/'
OUT = BASE  # output files saved to project folder

print("=" * 60)
print("BENG203 Script 1: Preprocessing & Feature Selection")
print("=" * 60)


print("\n[Step 1] Loading raw data files...")

# Cancer TPM — 60,675 genes x 96 samples, no header row
tpm_cancer = pd.read_csv(BASE + 'pnas_tpm_96_nodup.txt',
                         sep='\t', index_col=0, header=None)
tpm_cancer.columns = [f'C{i}' for i in range(1, 97)]
print(f"  Cancer TPM loaded: {tpm_cancer.shape[0]:,} genes x {tpm_cancer.shape[1]} samples")

# Normal TPM — 60,675 genes x 32 samples, has header
tpm_normal = pd.read_csv(BASE + 'pnas_normal_tpm.txt',
                         sep='\t', index_col=0)
print(f"  Normal TPM loaded: {tpm_normal.shape[0]:,} genes x {tpm_normal.shape[1]} samples")

# Patient info — recurrence labels
info = pd.read_csv(BASE + 'pnas_patient_info.csv')
labels_recur = info['recurStatus'].values
r_count = (labels_recur == 'R').sum()
n_count = (labels_recur == 'N').sum()
print(f"  Patient info loaded: {r_count} recurrent, {n_count} non-recurrent")

# Preselected 750 biomarker genes
with open(BASE + 'preselectedList.txt') as f:
    biomarker_genes = [l.strip() for l in f if l.strip()]
print(f"  Biomarker gene list loaded: {len(biomarker_genes)} genes")

print("\n[Step 2] Normalizing and filtering...")

def log2_normalize_and_filter(tpm_df, min_mean=0.5):
    """
    Apply log2(TPM + 1) transform, then filter out genes
    with mean log-expression below threshold.
    """
    log_tpm = np.log2(tpm_df + 1)
    keep = log_tpm.mean(axis=1) >= min_mean
    return log_tpm[keep]

log_cancer = log2_normalize_and_filter(tpm_cancer)
log_normal = log2_normalize_and_filter(tpm_normal)

print(f"  Cancer: {tpm_cancer.shape[0]:,} → {log_cancer.shape[0]:,} genes after filter")
print(f"  Normal: {tpm_normal.shape[0]:,} → {log_normal.shape[0]:,} genes after filter")

# Find common genes between cancer and normal
common_genes = log_cancer.index.intersection(log_normal.index)
log_cancer = log_cancer.loc[common_genes]
log_normal = log_normal.loc[common_genes]
print(f"  Common genes (cancer ∩ normal): {len(common_genes):,}")

print("\n[Step 3] Filtering to biomarker genes...")

bio_available = [g for g in biomarker_genes if g in common_genes]
bio_missing = len(biomarker_genes) - len(bio_available)
print(f"  Biomarker genes in data: {len(bio_available)} / {len(biomarker_genes)}")
print(f"  Missing (filtered out as low expression): {bio_missing}")

bio_cancer = log_cancer.loc[bio_available]  # 740 genes x 96 samples
bio_normal = log_normal.loc[bio_available]  # 740 genes x 32 samples

print("\n[Step 4] Ranking genes by differential expression p-value...")
print("  Running Mann-Whitney U test (R vs N patients)...")

r_idx = np.where(labels_recur == 'R')[0]
n_idx = np.where(labels_recur == 'N')[0]

pvals = []
for gene in bio_available:
    r_vals = bio_cancer.loc[gene].iloc[r_idx].values
    n_vals = bio_cancer.loc[gene].iloc[n_idx].values
    _, p = mannwhitneyu(r_vals, n_vals, alternative='two-sided')
    pvals.append(p)

gene_rankings = pd.DataFrame({
    'gene_id': bio_available,
    'pvalue': pvals
}).sort_values('pvalue').reset_index(drop=True)
gene_rankings['rank'] = range(1, len(gene_rankings) + 1)

print(f"  Done. Top 5 most differentially expressed genes:")
for _, row in gene_rankings.head(5).iterrows():
    print(f"    Rank {int(row['rank'])}: {row['gene_id']} (p={row['pvalue']:.4f})")

ranked_genes = gene_rankings['gene_id'].tolist()


print("\n[Step 5] Building output matrices...")

# Task 1: Cancer vs Normal
# Use all 740 ranked biomarker genes
# Samples as rows, genes as columns (standard ML format)
X_cancer_t1 = bio_cancer.T          # 96 x 740
X_normal_t1  = bio_normal.T         # 32 x 740
X_task1 = pd.concat([X_cancer_t1, X_normal_t1], axis=0)
y_task1 = pd.Series(
    [1]*96 + [0]*32,
    index=X_task1.index,
    name='label'
)
print(f"  Task 1 matrix: {X_task1.shape} (1=cancer, 0=normal)")

# Task 2: Recurrence vs Non-recurrence
# Cancer samples only
X_task2 = bio_cancer.T              # 96 x 740
y_task2 = pd.Series(
    (labels_recur == 'R').astype(int),
    index=X_task2.index,
    name='label'
)
print(f"  Task 2 matrix: {X_task2.shape} (1=recurrent, 0=non-recurrent)")


print("\n[Step 6] Saving output files...")

# Processed TPM matrices
bio_cancer.T.to_csv(OUT + 'processed_cancer_tpm.csv')
bio_normal.T.to_csv(OUT + 'processed_normal_tpm.csv')
print(f"   processed_cancer_tpm.csv")
print(f"   processed_normal_tpm.csv")

# Task 1
X_task1.to_csv(OUT + 'task1_feature_matrix.csv')
y_task1.to_csv(OUT + 'task1_labels.csv')
print(f"   task1_feature_matrix.csv")
print(f"   task1_labels.csv")

# Task 2
X_task2.to_csv(OUT + 'task2_feature_matrix.csv')
y_task2.to_csv(OUT + 'task2_labels.csv')
print(f"   task2_feature_matrix.csv")
print(f"   task2_labels.csv")

# Ranked gene list
gene_rankings.to_csv(OUT + 'ranked_biomarker_genes.csv', index=False)
print(f"   ranked_biomarker_genes.csv")


RAW DATA
  Cancer samples:       96
  Normal samples:       32
  Total genes in TPM:   {tpm_cancer.shape[0]:,}

NORMALIZATION
  Method:               log2(TPM + 1)
  Low-expression filter: mean log-TPM < 0.5 removed

AFTER FILTERING
  Genes remaining:      {len(common_genes):,}
  Common genes (cancer ∩ normal): {len(common_genes):,}

BIOLOGY-GUIDED FEATURE SELECTION
  Biomarker list:       750 pre-curated breast cancer genes
                        (from Zhou et al. PNAS 2019 sample code)
  Available in data:    {len(bio_available)} / 750
  (10 removed due to low expression filtering)

DIFFERENTIAL EXPRESSION RANKING
  Method:               Mann-Whitney U test (two-sided)
  Groups compared:      Recurrent (R, n={r_count}) vs Non-recurrent (N, n={n_count})
  Genes ranked:         {len(bio_available)} (ascending p-value)

OUTPUT MATRICES
  Task 1 (Cancer vs Normal):         {X_task1.shape[0]} samples x {X_task1.shape[1]} features
  Task 2 (Recurrence vs Non-recur):  {X_task2.shape[0]} samples x {X_task2.shape[1]} features

OUTPUT FILES
  processed_cancer_tpm.csv     — log2-normalized cancer TPM
  processed_normal_tpm.csv     — log2-normalized normal TPM
  task1_feature_matrix.csv     — Task 1 feature matrix (samples x genes)
  task1_labels.csv             — Task 1 labels (1=cancer, 0=normal)
  task2_feature_matrix.csv     — Task 2 feature matrix (samples x genes)
  task2_labels.csv             — Task 2 labels (1=recurrent, 0=non-recurrent)
  ranked_biomarker_genes.csv   — 740 genes ranked by DE p-value
"""

with open(OUT + 'preprocessing_summary.txt', 'w') as f:
    f.write(summary)
print(f"  ✓ preprocessing_summary.txt")

print("\n" + "=" * 60)
print("Script 1 COMPLETE — all output files saved to:")
print(f"  {OUT}")
print("=" * 60)


