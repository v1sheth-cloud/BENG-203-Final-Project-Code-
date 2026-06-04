

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

DATA = '/Users/vrisha/Desktop/BENG 203 project/'
OUT  = '/Users/vrisha/Desktop/BENG 203 project/'



tpm_cancer = pd.read_csv(DATA + 'pnas_tpm_96_nodup.txt', sep='\t', index_col=0, header=None)
tpm_cancer.columns = [f'C{i}' for i in range(1, 97)]

tpm_normal = pd.read_csv(DATA + 'pnas_normal_tpm.txt', sep='\t', index_col=0)

info = pd.read_csv(DATA + 'pnas_patient_info.csv')
labels_recur = info['recurStatus'].values

with open(DATA + 'preselectedList.txt') as f:
    biomarker_genes = [line.strip() for line in f if line.strip()]

print(f"  Cancer TPM: {tpm_cancer.shape}")
print(f"  Normal TPM: {tpm_normal.shape}")
print(f"  Recurrent (R): {(labels_recur == 'R').sum()}, Non-recurrent (N): {(labels_recur == 'N').sum()}")
print(f"  Biomarker genes: {len(biomarker_genes)}")



log_cancer = np.log2(tpm_cancer + 1)
log_normal = np.log2(tpm_normal + 1)

log_cancer = log_cancer[log_cancer.mean(axis=1) >= 0.5]
log_normal = log_normal[log_normal.mean(axis=1) >= 0.5]

common_genes = log_cancer.index.intersection(log_normal.index)
log_cancer = log_cancer.loc[common_genes]
log_normal = log_normal.loc[common_genes]

print(f"  Genes after filtering: {len(common_genes):,}")




bio_available = [g for g in biomarker_genes if g in common_genes]
print(f"  Biomarker genes available: {len(bio_available)} / {len(biomarker_genes)}")

bio_cancer = log_cancer.loc[bio_available]
bio_normal = log_normal.loc[bio_available]



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
ranked_genes = gene_rankings['gene_id'].tolist()

print(f"  Top 5 genes by p-value:")
for _, row in gene_rankings.head(5).iterrows():
    print(f"    Rank {int(row['rank'])}: {row['gene_id']} (p={row['pvalue']:.4f})")



X_task1 = pd.concat([bio_cancer.T, bio_normal.T], axis=0)
y_task1 = pd.Series([1]*96 + [0]*32, index=X_task1.index, name='label')

X_task2 = bio_cancer.T
y_task2 = pd.Series((labels_recur == 'R').astype(int), index=X_task2.index, name='label')

print(f"  Task 1 matrix: {X_task1.shape}  (1=cancer, 0=normal)")
print(f"  Task 2 matrix: {X_task2.shape}  (1=recurrent, 0=non-recurrent)")


X_task1.to_csv(OUT + 'task1_feature_matrix.csv')
y_task1.to_csv(OUT + 'task1_labels.csv')
X_task2.to_csv(OUT + 'task2_feature_matrix.csv')
y_task2.to_csv(OUT + 'task2_labels.csv')
gene_rankings.to_csv(OUT + 'ranked_biomarker_genes.csv', index=False)
bio_cancer.T.to_csv(OUT + 'processed_cancer_tpm.csv')
bio_normal.T.to_csv(OUT + 'processed_normal_tpm.csv')

print("  task1_feature_matrix.csv  saved")
print("  task1_labels.csv          saved")
print("  task2_feature_matrix.csv  saved")
print("  task2_labels.csv          saved")
print("  ranked_biomarker_genes.csv saved")
print("  processed_cancer_tpm.csv  saved")
print("  processed_normal_tpm.csv  saved")

print("\nDONE — all files saved to:", OUT)

