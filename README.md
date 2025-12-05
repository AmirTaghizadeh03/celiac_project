# **Celiac Project**



This repository contains a full transcriptomic analysis pipeline for identifying differentially expressed genes, enriched pathways, and machine learning biomarkers in celiac disease using the GSE164883 dataset.





##### **Overview**



The workflow covers data loading, preprocessing, differential expression analysis, annotation, enrichment analysis, feature selection, multiple machine learning models, and visualization. The pipeline is implemented in Python using tools such as Scanpy, GSEAPY, scikit learn, and XGBoost.





##### **Main Steps**





###### 1\. Data Preparation



•Load raw GEO series matrix.

•Extract sample metadata and assign disease labels.

•Create an AnnData object for downstream analysis.

•Perform library-size normalization and QC metrics.







###### 2\. Differential Expression Analysis (DEA)



•Identify up  and down regulated genes between celiac and control samples.

•Export full and significant DEG tables.







###### 3\. Gene Annotation



•Map probe IDs to gene symbols using GPL10558 annotation.

•Generate annotated DEG tables.







###### 4\. Pathway Enrichment



•Enrich significant DEGs using GO Biological Process and KEGG Human libraries.

•Save reports and top enriched terms.







###### 5\. Machine Learning Analysis



Models used:



•Random Forest

•Logistic Regression

•Linear SVM

•XGBoost (final model)



Pipeline includes:



•Train/test split

•Cross validation

•Bootstrap validation

•Feature importance extraction

•Overlap analysis between models

All results are exported to the results/ directory.







###### 6\. Visualization



•PCA and UMAP plots for sample clustering

•Volcano plot of DEGs

•Enrichment barplots

•Saved in the figures/ directory





##### **Directory Structure**



celiac\_project/

├── data/

│   ├── raw/          # raw GEO data + platform annotation

│   └── processed/    # AnnData and intermediate processed files

├── results/          # DEG tables, ML outputs, enrichment results

├── figures/          # PCA, UMAP, volcano, pathway plots

├── scripts/          # (optional) analysis scripts

└── README.md





##### **Requirements**



•Python 3.9+

•scanpy

•pandas

•numpy

•gseapy

•scikit learn

•xgboost

•seaborn

•matplotlib

Install all dependencies:

pip install scanpy pandas numpy gseapy scikit-learn xgboost seaborn matplotlib





##### **How to Run**



1.Place GSE164883\_series\_matrix.txt and GPL10558.annot in data/raw/.

2.Run the main analysis script.

3.Processed data, plots, and results will appear in their respective directories.





##### **Notes**

•Raw and processed data directories are intentionally excluded from Git due to size.

•This project aims to identify robust biomarkers for celiac disease using multi step statistical and ML based methods.



For any issues or improvements, feel free to open an issue or submit a pull request.





