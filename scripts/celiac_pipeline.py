# celiac_project
# Author : Amir Taghizadeh


# imports

import os
import pandas as pd
import re
import scanpy as sc
import numpy as np
import gseapy as gp
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

from sklearn.metrics import roc_curve, auc

# step 0: directories
base_dir = r"C:\Users\Asus\Desktop\projects\celiac_project"
raw_dir = os.path.join(base_dir, 'data', 'raw')
processed_dir = os.path.join(base_dir, 'data', 'processed')
results_dir = os.path.join(base_dir, 'results')
figures_dir = os.path.join(base_dir, 'figures')

# step 1: loading df

df_path = os.path.join(raw_dir, 'GSE164883_series_matrix.txt')
df = pd.read_csv(df_path, sep='\t', comment='!', index_col=0)
df = df.drop_duplicates()
df = df.dropna()


print(df.shape, df.columns)

# step 2: extracting metadata

meta_lines = []
with open(df_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!Sample_characteristics_ch1'):
            meta_lines.append(line.strip('\n'))

disease_lines = [line for line in meta_lines if 'disease' in line.lower()][0]
statuses = re.findall('"(.*?)"', disease_lines)

disease_labels = [
    'Celiac' if 'celiac disease' in s.lower()
    else 'Control' 
    for s in statuses
]
df_columns = disease_labels

print(pd.Series(disease_labels).value_counts())

# step 3: anndata object

adata = sc.AnnData(df.T)
adata.var_names_make_unique()
adata.obs['disease'] = disease_labels
sc.pp.calculate_qc_metrics(adata, inplace=True)


# step 4: normalisation

sc.pp.normalize_total(adata, target_sum=1e4)
adata.write(os.path.join(processed_dir, 'first_anndata.h5ad'))

adata.raw = adata.copy()

# step 5: DEA

sc.tl.rank_genes_groups(adata, groupby='disease', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25)

deg = sc.get.rank_genes_groups_df(adata, group='Celiac')  
deg.head()

deg_sig = deg[(abs(deg['logfoldchanges']) > 1) & (deg['pvals_adj'] < 0.05)]
deg_sig.head()

# step 6: Annotation

annot_file = os.path.join(raw_dir, 'GPL10558.annot')

table_lines = []
with open(annot_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        split_lines = line.strip().split('\t')
        if len(split_lines) > 1:
            table_lines.append(split_lines)

header = table_lines[0]
data = table_lines[1:]
annot_file = pd.DataFrame(data, columns=header)


prob_col = [c for c in annot_file.columns if 'id' in c.lower()][0]
symbol_col = [c for c in annot_file.columns if 'gene symbol' in c.lower()][0]

annot_file = annot_file[[prob_col ,symbol_col]].rename(columns={prob_col: 'ID', symbol_col: 'Gene'})
annot_file = annot_file[annot_file['Gene'].notna() & (annot_file['Gene'] != '')]

print(annot_file.columns)

cd_deg_annot = deg.merge(annot_file[['ID', 'Gene']], left_on='names', right_on='ID', how='left')
cd_deg_annot = cd_deg_annot.drop(columns=['ID', 'names'])
col = ['Gene'] + [c for c in cd_deg_annot.columns if c != 'Gene']
cd_deg_annot = cd_deg_annot[col]
cd_deg_annot_path = os.path.join(results_dir, 'cd_deg_annotation.csv')
cd_deg_annot.to_csv(cd_deg_annot_path, index=False)
print(cd_deg_annot.head(10))


# step 7: enrichment pathaway

sig_deg_annot = deg_sig.merge(annot_file[['ID', 'Gene']], left_on='names', right_on='ID', how='left')
sig_deg_annot = sig_deg_annot.drop(columns=['ID', 'names'])
col = ['Gene'] + [c for c in sig_deg_annot.columns if c != 'Gene']
sig_deg_annot = sig_deg_annot[col]
sig_deg_annot_path = os.path.join(results_dir, 'sig_deg_annotation.csv')
sig_deg_annot.to_csv(sig_deg_annot_path, index=False)
print(sig_deg_annot.head(10))

significant_degs = sig_deg_annot['Gene'].dropna().head(10).tolist() 



path_dir = os.path.join(results_dir, 'pathaway_initial_celiac_sig_genes')
os.makedirs(path_dir, exist_ok=True)
enrichr_libs = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human"
]
    
for l in enrichr_libs:
    gp.enrichr(
        gene_list=significant_degs,
        gene_sets=l,
        outdir=path_dir,
        no_plot=True,
        cutoff=0.05
    )


# step 8: annot for ml

expr_path = df_path
expr_df = pd.read_csv(expr_path, sep='\t', comment='!', header=0, dtype=str)
expr_df = expr_df.rename(columns={expr_df.columns[0]: 'ID'})
expr_df['ID'] = expr_df['ID'].str.strip()

print(expr_df.head(5))
annot_file = os.path.join(raw_dir, 'GPL10558.annot')

table_lines = []
with open((annot_file), 'r', encoding='latin1') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.startswith('!platform_table_begin'):
        table_start = i + 1
        break
table_str = ''.join(lines[table_start:])
annot = pd.read_csv(io.StringIO(table_str), sep='\t', dtype=str)

annot['ID'] = annot['ID'].str.strip()
annot['Gene symbol'] = annot['Gene symbol'].str.strip()

annot = annot[annot['Gene symbol'].notna() & (annot['Gene symbol'] != '')]
print(annot.shape, annot.columns)
merged_df = pd.merge(expr_df, annot[['ID','Gene symbol']], on='ID', how='inner')
sample_cols = merged_df.columns.difference(['ID', 'Gene symbol'])
merged_df[sample_cols] = merged_df[sample_cols].apply(pd.to_numeric, errors='coerce')
merged_agg = merged_df.groupby('Gene symbol')[sample_cols].mean()
print(merged_agg.head(5))

# step 9: creating new AnnData for ml

adata_celia = sc.AnnData(merged_agg.T)
adata_celia.obs['disease'] = disease_labels
adata_celia.var_names_make_unique()

adata_cd = adata_celia[adata_celia.obs['disease'].isin(['Celiac' , 'Control'])].copy()
print(adata_cd)
adata_cd.obs['disease'].value_counts()

adata_cd.write(os.path.join(processed_dir, 'final_anndata.h5ad'))

# step 10: gene based ml


X = adata_cd.X
y = adata_cd.obs['disease'].map({'Celiac': 0, 'Control': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

rf_class_report = classification_report(y_test, rf_y_pred)
rf_roc_auc_score = roc_auc_score(y_test, rf_y_pred)

print(f" RF Accuracy: {rf_class_report}")
print(f' RF ROC-AUC: {rf_roc_auc_score}')

rf_results = os.path.join(results_dir, 'RF_results.csv')

with open(rf_results, 'w') as f:
    f.write('\nclassification_report: \n')
    f.write(f'\n{rf_class_report}\n')
    f.write('\nroc_auc_score: \n')
    f.write(f'\n{rf_roc_auc_score}\n')

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

lr_class_report = classification_report(y_test, lr_y_pred)
lr_roc_auc_score = roc_auc_score(y_test, lr_y_pred)

print(f" LR Accuracy: {lr_class_report}")
print(f' LR ROC-AUC: {lr_roc_auc_score}')

lr_results = os.path.join(results_dir, 'LR_results.csv')

with open(lr_results, 'w') as f:
    f.write('\nclassification_report: \n')
    f.write(f'\n{lr_class_report}\n')
    f.write('\nroc_auc_score: \n')
    f.write(f'\n{lr_roc_auc_score}\n')

svm = LinearSVC(C=1, max_iter=5000)
svm.fit(X_train, y_train)
svm_y_pred = svm.predict(X_test)
svm_class_report = classification_report(y_test, svm_y_pred)
svm_roc_auc_score = roc_auc_score(y_test, svm_y_pred)

print(f" SVM Accuracy: {svm_class_report}")
print(f' SVM ROC-AUC: {svm_roc_auc_score}')

svm_results = os.path.join(results_dir, 'SVM_results.csv')

with open(svm_results, 'w') as f:
    f.write('\nclassification_report: \n')
    f.write(f'\n{svm_class_report}\n')
    f.write('\nroc_auc_score: \n')
    f.write(f'\n{svm_roc_auc_score}\n')

# step 11: CV

scores_rf = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
scores_lr = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
scores_svm = cross_val_score(svm, X, y, cv=5, scoring='roc_auc')


cv_rf = scores_rf.mean()
cv_lr = scores_lr.mean()
cv_svm = scores_svm.mean()


print(f" RF CV: {cv_rf}")
print(f" LR CV: {cv_lr}")
print(f" SVM CV: {cv_svm}")


cross_valid = os.path.join(results_dir, 'cv_models.txt')
with open(cross_valid, 'w') as f:
    f.write('\nRF Cross Validation: \n')
    f.write(f'\n{cv_rf}\n')
    f.write('\nLR Cross Validation: \n')
    f.write(f'\n{cv_lr}\n')
    f.write('\nSVM Cross Validation: \n')
    f.write(f'\n{cv_svm}\n')

# step 12: Bootstrap

le = LabelEncoder()
y_encoded = le.fit_transform(y)


n_bootstrap = 100
rf_scores = []
lr_scores = []
svm_scores = []


for i in range(n_bootstrap):
    X_resample, y_resample = resample(X, y_encoded, replace=True, n_samples=len(X), random_state=i)
    
    # RF
    rf.fit(X_resample, y_resample)
    rf_scores.append(rf.score(X, y_encoded))

    # LR
    lr.fit(X_resample, y_resample)
    lr_scores.append(lr.score(X, y_encoded))

    # SVM
    svm.fit(X_resample, y_resample)
    svm_scores.append(svm.score(X, y_encoded))




all_scores = np.array([rf_scores, lr_scores, svm_scores])

model_means = np.nanmean(all_scores, axis=1)
model_stds = np.nanstd(all_scores, axis=1)

print("RF Bootstrap:", model_means[0], "±", model_stds[0])
print("LR Bootstrap:", model_means[1], "±", model_stds[1])
print("SVM Bootstrap:", model_means[2], "±", model_stds[2])


final_mean = np.mean(model_means)
final_std = np.std(model_means)

print(f" Final mean Bootstrap accuracy across all models: {final_mean:.4f} ± {final_std:.4f}")


boot_valid = os.path.join(results_dir, 'Bootstrap_models.txt')
with open(boot_valid, 'w') as f:
    f.write('\nRF Bootstrap: \n')
    f.write(f'\n{model_means[0]} ± {model_stds[0]}\n')
    f.write('\nLR Bootstrap: \n')
    f.write(f'\n{model_means[1]} ± {model_stds[1]}\n')
    f.write('\nSVM Bootstrap: \n')
    f.write(f'\n{model_means[2]} ± {model_stds[2]}\n')
    f.write('\nFinal mean Bootstrap accuracy across all models: \n')
    f.write(f'\n{final_mean:.4f} ± {final_std:.4f}\n')

# step 13: feature importance

feature_importance_rf = pd.DataFrame(
    {'Gene': adata_cd.var_names, 'importance': rf.feature_importances_}
                                 ).sort_values('importance', ascending=False)
coefficients_lr = lr.coef_[0]
importance_lr = abs(coefficients_lr)
feature_importance_lr = pd.DataFrame(
    {'Gene': adata_cd.var_names, 'importance': importance_lr}
                                 ).sort_values('importance', ascending=False)

coefficients_svm = svm.coef_[0]
importance_svm = abs(coefficients_svm)
feature_importance_svm = pd.DataFrame(
    {'Gene': adata_cd.var_names, 'importance': importance_svm}
                                 ).sort_values('importance', ascending=False)



feature_importance_rf.head(500).to_csv(os.path.join(results_dir, 'CD_RF_top_genes.csv'), index=False)
feature_importance_lr.head(500).to_csv(os.path.join(results_dir, 'CD_LR_top_genes.csv'), index=False)
feature_importance_svm.head(500).to_csv(os.path.join(results_dir, 'CD_SVM_top_genes.csv'), index=False)

rf_top250 = feature_importance_rf.head(250)
lr_top250 = feature_importance_lr.head(250)
svm_top250 = feature_importance_svm.head(250)


overlap = set(rf_top250['Gene']) \
          .intersection(set(lr_top250['Gene'])) \
          .intersection(set(svm_top250['Gene']))

overlap_df = pd.DataFrame(list(overlap), columns=['Gene'])
overlap_df.to_csv(os.path.join(results_dir, 'model_genes_overlap_all.csv'), index=False)

print("Overlap count:", len(overlap))
print("Overlap genes:", overlap)



deg_overlap_genes = ['SMPDL3A', 'GBP4', 'IRF9', 'STAT1', 'ARHGDIB', 'MS4A10',
                     'AOC1', 'HLA-E', 'PSME1', 'HSP90B1', 'APOB', 'ADH4',
                     'HCP5', 'IRF1', 'SLC25A23', 'TAP1', 'HLA-B', 'AQP10',
                     'DHRS7', 'PRAP1', 'CCL5', 'EPSTI1', 'SERPINE2', 'FBP1']

deg_check = cd_deg_annot[cd_deg_annot['Gene'].isin(deg_overlap_genes)]
model_deg_overlap = deg_check.to_csv(
    os.path.join(results_dir, 'model_deg_genes_overlap.csv'), index=False)
genes_cleaned = deg_check.sort_values(['pvals_adj', 'logfoldchanges'], ascending=[True, False])
genes_cleaned = genes_cleaned.drop_duplicates(subset='Gene', keep='first')

final_model_deg_overlap = genes_cleaned.to_csv(
    os.path.join(results_dir, 'model_deg_genes_final_overlap.csv'), index=False)

print(genes_cleaned)

# step 14: pahtaway enrichment of overlap genes

final_significant_genes = genes_cleaned['Gene'].tolist()

path_directory = os.path.join(results_dir, 'final_overlap_pathaway_enrichment')
os.makedirs(path_directory, exist_ok=True)



gp.enrichr(
    gene_list=final_significant_genes,
    gene_sets= "GO_Biological_Process_2021",
    outdir=path_directory,
    no_plot=True,
    cutoff=0.05
    )


# step 15: final ml(XGB)

final_genes = ['SMPDL3A', 'GBP4', 'IRF9', 'STAT1', 'ARHGDIB', 'MS4A10',
                     'AOC1', 'HLA-E', 'PSME1', 'HSP90B1', 'APOB', 'ADH4',
                     'HCP5', 'IRF1', 'SLC25A23', 'TAP1', 'HLA-B', 'AQP10',
                     'DHRS7', 'PRAP1', 'CCL5', 'EPSTI1', 'SERPINE2', 'FBP1']

X_final = adata_cd[:, final_genes].X
y_final = adata_cd.obs['disease'].map({'Celiac': 0, 'Control': 1})

final_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
final_cv = cross_val_score(final_model, X_final, y_final, cv=5, scoring='accuracy')

cross_validation_final = os.path.join(results_dir, 'cv_final_model.txt')
with open(cross_validation_final, 'w') as f:
    f.write('\nfinal XGB Cross Validation and STD: \n')
    f.write(f'\n{final_cv.mean()}, ± {final_cv.std()}\n')
    


print("Final Model Accuracy:", final_cv.mean(), "±", final_cv.std())


# step 16: visualisation

## PCA & UMAP

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
plt.figure(figsize=(6,5))
sc.pl.pca(adata, color='total_counts', show=False)
pca_file = os.path.join(figures_dir, 'PCA_Celiac.png')
plt.savefig(pca_file, dpi=300, bbox_inches='tight')

print('Done creating pca image.')

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)
sc.tl.leiden(adata, resolution=0.5)

sc.tl.umap(adata)
plt.figure(figsize=(6,5))
sc.pl.umap(adata, color=['leiden', 'disease'], show=False)
umap_file = os.path.join(figures_dir, 'UMAP_Celiac.png')
plt.savefig(umap_file, dpi=300, bbox_inches='tight')

print('Done creating umap image.')


## SD DEG Volcano Plot

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=cd_deg_annot,
    x='logfoldchanges',
    y=-np.log10(cd_deg_annot['pvals']),
    hue=cd_deg_annot['pvals_adj'] < 0.05,
    palette={True: 'red', False: 'gray'},
    alpha=0.7
    )

plt.axvline(0, color="black", lw=1)
plt.xlabel("log2 Fold Change (CD vs Control)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot: Celiac vs Control")
volcano_path = os.path.join(figures_dir, 'Volcano_CD_deg.png')
plt.savefig(volcano_path, dpi=300, bbox_inches='tight')

print("Done with DEG volcano plot. ")


## CD enrichment barplot

for l in enrichr_libs:
    file_name = f'{l}.human.enrichr.reports.txt'
    file_path = os.path.join(path_dir, file_name)

    path_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    keep_cols = [c for c in ['Term', 'Adjusted P-value', 'Overlap', 'P-value', 'Combined Score'] if c in path_df.columns]
    path_df = path_df[keep_cols].sort_values(by=keep_cols[1]).head(50)

    out_file = os.path.join(path_dir, f"top20_{l}_enrichment.csv")
    path_df.to_csv(out_file, index=False)
    

libraries = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human"
]
for li in libraries:
    top_enr_path = os.path.join(path_dir, f"top20_{li}_enrichment.csv")
    enrich = pd.read_csv(top_enr_path)
    score_col = 'Adjusted P-value' if 'Adjusted P-value' in enrich.columns else 'P-value'
    enrich = enrich.sort_values(by=score_col).head(15)
    enrich['Term'] = enrich['Term'].astype(str)

    plt.figure(figsize=(8, 6))
    plt.barh(enrich['Term'], -enrich[score_col].apply(lambda x: np.log10(x)))
    plt.xlabel('-log10(p-value)')
    plt.ylabel('Enriched Term')
    plt.title(f'Top Enriched Terms: {li}')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # save figure
    plot_file = os.path.join(figures_dir, f"{li}_barplot.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

print("Done with enrichment barplot. ")


## models feature importances

rf_top20 = feature_importance_rf.head(20)
lr_top20 = feature_importance_lr.head(20)
svm_top20 = feature_importance_svm.head(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='Gene', data=rf_top20)
plt.title('Random Forest Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Gene')
plt.tight_layout()
rf_feature = os.path.join(figures_dir, 'RF_Top20_feature_importance.png')
plt.savefig(rf_feature, dpi=300, bbox_inches='tight')

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='Gene', data=lr_top20)
plt.title('Logistic Regression Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Gene')
plt.tight_layout()
lr_feature = os.path.join(figures_dir, 'LR_Top20_feature_importance.png')
plt.savefig(lr_feature, dpi=300, bbox_inches='tight')

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='Gene', data=svm_top20)
plt.title('SVM Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Gene')
plt.tight_layout()
svm_feature = os.path.join(figures_dir, 'SVM_Top20_feature_importance.png')
plt.savefig(svm_feature, dpi=300, bbox_inches='tight')


print("Done with feature importance visualisation. ")


## RF, LR and SVM overlap

rf_top250_genes = set(rf_top250['Gene'])
lr_top250_genes = set(lr_top250['Gene'])
svm_top250_genes = set(svm_top250['Gene'])

venn_fig_file = os.path.join(figures_dir, "venn_RF_LR_SVM.png")
plt.figure(figsize=(6,4))
venn3([rf_top250_genes,
       lr_top250_genes,
       svm_top250_genes], set_labels=('RF Top 250', 'LR Top 250', 'SVM Top 250'))
plt.title("Overlap Between RF, LR and SVM Top 250 Genes")
plt.figtext(0.5, -0.1, 'Overlaping Genes: SMPDL3A, GBP4, IRF9, STAT1, ARHGDIB, MS4A10, AOC1, HLA-E, PSME1, HSP90B1, APOB, ADH4, HCP5, IRF1, SLC25A23, TAP1, HLA-B, AQP10, DHRS7, PRAP1, CCL5, EPSTI1, SERPINE2, FBP1'
            ,wrap=True, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(venn_fig_file, dpi=300, bbox_inches='tight')

print("Done with models overlap visualisation. ")


## models and deg overlap

model_deg_overlap_path = os.path.join(results_dir, 'model_deg_genes_overlap.csv')
model_deg_overlap_df = pd.read_csv(model_deg_overlap_path)

plt.figure(figsize=(8,6))
sns.barplot(x='Gene', y='logfoldchanges', data=model_deg_overlap_df, color='skyblue', edgecolor='black')

plt.axhline(0, color='grey', linestyle='--')  
plt.ylabel('Log2 Fold Change')
plt.title('Top 24 Genes Overlap Significance')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
biomarker_out_logfc = os.path.join(figures_dir, 'final_biomarker_logfc.png')
plt.savefig(biomarker_out_logfc, dpi=300, bbox_inches='tight')

## ROC & AUC curve

y_pred_proba = cross_val_predict(
    final_model,
    X_final,
    y_final,
    cv=5,
    method='predict_proba'
)[:, 1]

fpr, tpr, _ = roc_curve(y_final, y_pred_proba)

roc_auc_value = auc(fpr, tpr)

print("ROC-AUC:", roc_auc_value)

roc_auc_final = os.path.join(results_dir, 'roc_auc_final_model.txt')
with open(roc_auc_final, 'w') as f:
    f.write('\nfinal XGB ROC & AUC: \n')
    f.write(f'\n{roc_auc_value}\n')

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc_value:.3f}')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final 24 Genes")
plt.legend(loc='lower right')
roc_auc_plot_path = os.path.join(figures_dir, 'final_ROC_AUC_curve.png')
plt.savefig(roc_auc_plot_path, dpi=300, bbox_inches='tight')

