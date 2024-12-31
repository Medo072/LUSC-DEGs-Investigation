import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import false_discovery_control
from scipy.stats import shapiro, wilcoxon, ranksums
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import os


# Helper function to check directory
def check_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Volcano plot function
def create_volcano_plot(pvalues_file, log2fc_file, output_file='volcano_plot.png'):
    # Load the data
    pvalues_df = pd.read_csv(pvalues_file)
    log2fc_df = pd.read_csv(log2fc_file)

    # Merge the DataFrames on 'Gene_Name' column
    merged_df = pd.merge(log2fc_df, pvalues_df, on='Gene_Name')

    # Replace zero p-values to avoid taking log of zero
    merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

    # Calculate negative log10 p-value
    merged_df['negLog10PValue'] = -np.log10(merged_df['P_Value'])

    # Set thresholds
    pvalue_threshold = 0.05
    log2fc_threshold = 1  # Corresponds to 2-fold change

    # Add a column to indicate significance
    merged_df['Significant'] = 'Not Significant'
    merged_df.loc[
        (abs(merged_df['log2FC']) >= log2fc_threshold) &
        (merged_df['P_Value'] <= pvalue_threshold),
        'Significant'
    ] = 'Significant'

    # Count the number of significant DEGs
    num_significant_degs = merged_df['Significant'].eq('Significant').sum()

    print(f"The number of significant DEGs: {num_significant_degs}")

    # Create the volcano plot
    plt.figure(figsize=(10, 8))

    # Scatter plot for non-significant genes
    plt.scatter(
        merged_df['log2FC'][merged_df['Significant'] == 'Not Significant'],
        merged_df['negLog10PValue'][merged_df['Significant'] == 'Not Significant'],
        color='grey',
        alpha=0.5,
        label='Not Significant'
    )

    # Scatter plot for significant genes
    plt.scatter(
        merged_df['log2FC'][merged_df['Significant'] == 'Significant'],
        merged_df['negLog10PValue'][merged_df['Significant'] == 'Significant'],
        color='red',
        alpha=0.7,
        label='Significant'
    )

    # Add threshold lines
    plt.axhline(y=-np.log10(pvalue_threshold), color='blue', linestyle='--', linewidth=1)
    plt.axvline(x=log2fc_threshold, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=-log2fc_threshold, color='green', linestyle='--', linewidth=1)

    # Labels and Title
    plt.title('Volcano Plot of Differentially Expressed Genes')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 (P-Value)')
    plt.legend()

    # Add the number of significant DEGs as a text label
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f'Number of Significant DEGs: {num_significant_degs}',
        transform=ax.transAxes,
        fontsize=10,
        color='black',
        verticalalignment='top'
    )

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.show()

# GSEA analysis function
def perform_gsea_analysis(log2fc_file, pvalues_file, gene_sets, output_dir='GSEA_results'):
    # Load the data
    pvalues_df = pd.read_csv(pvalues_file)
    log2fc_df = pd.read_csv(log2fc_file)

    # Merge the DataFrames on the common 'Gene_Name' column
    merged_df = pd.merge(log2fc_df, pvalues_df, on='Gene_Name')

    # Prepare ranked data for GSEA
    ranked_data = merged_df[['Gene_Name', 'log2FC']].sort_values(by='log2FC', ascending=False)
    ranked_data.columns = ['gene', 'score']

    # Save ranked data to a temporary file for GSEA
    ranked_data_file = 'ranked_gene_list.rnk'
    ranked_data.to_csv(ranked_data_file, sep='\t', index=False, header=False)

    # Run GSEA
    gsea_results = gp.prerank(
        rnk=ranked_data_file,  # Use the ranked gene list file
        gene_sets=gene_sets,  # Replace with your desired gene set
        outdir=output_dir,  # Output directory for results
        permutation_num=100,  # Number of permutations
    )

    # Save summary of results
    gsea_results.res2d.to_csv(f'{output_dir}/gsea_summary.csv', index=False)


def check_normality(healthy, tumor):
    _, p_value = shapiro(healthy - tumor)
    return p_value

def drop_genes_on_condition(df_h, df_t, threshold=0.5):
    num_subjects = len(df_h.columns) - 1
    assert len(df_h.columns) == len(df_t.columns)
    mask_h = (df_h.eq(0).sum(axis=1) >= num_subjects * 0.5)
    mask_t = (df_t.eq(0).sum(axis=1) >= num_subjects * 0.5)
    mask_any = mask_h | mask_t
    df_h_filtered = df_h[~mask_any]
    df_t_filtered = df_t[~mask_any]
    return df_h_filtered, df_t_filtered

def extract_non_zeros(df_h, df_t):
    mask_h = (df_h != 0)
    mask_t = (df_t != 0)
    mask_any = mask_h & mask_t
    df_h_non_zeros = df_h[mask_any]
    df_t_non_zeros = df_t[mask_any]
    return df_h_non_zeros, df_t_non_zeros

def calc_p_values(healthy, tumor, sample_type):
    output_genes = []
    p_values = []
    statistics = []
    output_with_statistic = []
    for gene in healthy.index:
        if sample_type == "paired":
            statistic, p_value_wilcoxon = wilcoxon(healthy.loc[gene], tumor.loc[gene])
        else:
            statistic, p_value_wilcoxon = ranksums(healthy.loc[gene], tumor.loc[gene])
        p_values.append(p_value_wilcoxon)
        statistics.append(statistic)
        output_genes.append({"Gene_Name": gene, "P_Value": p_value_wilcoxon})
        output_with_statistic.append({"Gene_Name": gene, "Statistic": statistic})
    return output_genes, p_values, output_with_statistic

def find_degs(healthy, tumor, sample_type):
    alpha = 0.05
    normal_genes = []
    non_normal_genes = []
    p_values = []
    for gene in healthy.index:
        p_value = check_normality(healthy.loc[gene], tumor.loc[gene])
        if p_value > alpha:
            normal_genes.append(gene)
        else:
            non_normal_genes.append(gene)
    genes_and_p_values, p_values, genes_and_statistic = calc_p_values(healthy, tumor, sample_type=sample_type)
    p_values_after_fdr = false_discovery_control(p_values)
    DEGs = [gene for gene, p_value_ in zip(healthy.index, p_values_after_fdr) if p_value_ > .05]
    return DEGs, genes_and_p_values, genes_and_statistic

def save_list_to_csv(list, header, file_name, index=False):
    df_DEGs = pd.DataFrame({header: list})
    df_DEGs.to_csv(file_name, index=index)

def drop_CNA_with_zeros(df):
    num_subjects = len(df)
    is_zero = df == 0
    zero_counts = is_zero.sum()
    condition = zero_counts < num_subjects * 0.5
    df_filtered = df.loc[:, condition]
    return df_filtered

def get_common_subjects(data_1, data_2):
    set1, set2 = set(data_1), set(data_2)
    common_elements = set1 & set2
    return common_elements

def rank_degs_by_p_value(data: pd.DataFrame):
    sorted_data = data.sort_values(by="P_Value", ascending=False)
    return sorted_data

def rank_degs_by_fold_change(data: pd.DataFrame):
    data["log2FC"] = np.log2(data)
    data["abslog2FC"] = np.abs(data["log2FC"])
    fold_change = data
    fold_change.columns.values[0] = "Fold Change"
    fold_change_sorted = fold_change.sort_values(by="abslog2FC", ascending=False)
    return fold_change_sorted

def rank_degs_by_statistic(data: pd.DataFrame):
    data["abs"] = np.abs(data["Statistic"])
    ranked = data.sort_values(by="abs", ascending=False)
    return ranked

def get_columns_list(df, n=5):
    return df.columns[:n].tolist()

def get_sub_dfs(df, column_names):
    subset_df = df[column_names]
    dfs = {}
    for col_name in column_names:
        current_df = df.drop(columns=column_names).join(subset_df[col_name])
        dfs[col_name] = current_df
    return dfs

def check_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
        
        
# def create_volcano_plot():

#     # Load the data
#     pvalues_df = pd.read_csv('results/csv files/DEGs_with_p_values.csv')   # Replace with your p-values file name
#     log2_fc_df = pd.read_csv('results/csv files/lusc_degs_ranked_by_log2fc.csv')  # Replace with your log2 fold change file name

#     # Merge the DataFrames on 'Gene' column
#     merged_df = pd.merge(log2_fc_df, pvalues_df, on='Gene_Name')

#     # Replace zero p-values to avoid taking log of zero
#     merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

#     # Calculate negative log10 p-value
#     merged_df['negLog10PValue'] = -np.log10(merged_df['P_Value'])

#     # Set thresholds
#     pvalue_threshold = 0.05
#     log2fc_threshold = 1  # Corresponds to 2-fold change

#     # Add a column to indicate significance
#     merged_df['Significant'] = 'Not Significant'
#     merged_df.loc[
#         (abs(merged_df['log2FC']) >= log2fc_threshold) &
#         (merged_df['P_Value'] <= pvalue_threshold),
#         'Significant'
#     ] = 'Significant'

#     # Create a boolean mask using your criteria
#     significant_mask = (
#         (merged_df['P_Value'] <= pvalue_threshold) &
#         (abs(merged_df['log2FC']) >= log2fc_threshold)
#     )

#     # Add a column to indicate significance
#     merged_df['Significant'] = 'Not Significant'
#     merged_df.loc[significant_mask, 'Significant'] = 'Significant'

#     # Count the number of significant DEGs
#     num_significant_degs = significant_mask.sum()

#     print(f"The number of significant DEGs: {num_significant_degs}")
#     # Create the volcano plot
#     plt.figure(figsize=(10, 8))

#     # Scatter plot for non-significant genes
#     plt.scatter(
#         merged_df['log2FC'][merged_df['Significant'] == 'Not Significant'],
#         merged_df['negLog10PValue'][merged_df['Significant'] == 'Not Significant'],
#         color='grey',
#         alpha=0.5,
#         label='Not Significant'
#     )

#     # Scatter plot for significant genes
#     plt.scatter(
#         merged_df['log2FC'][merged_df['Significant'] == 'Significant'],
#         merged_df['negLog10PValue'][merged_df['Significant'] == 'Significant'],
#         color='red',
#         alpha=0.7,
#         label='Significant'
#     )

#     # Add threshold lines
#     plt.axhline(y=-np.log10(pvalue_threshold), color='blue', linestyle='--', linewidth=1)
#     plt.axvline(x=log2fc_threshold, color='green', linestyle='--', linewidth=1)
#     plt.axvline(x=-log2fc_threshold, color='green', linestyle='--', linewidth=1)

#     # Labels and Title
#     plt.title('Volcano Plot of Differentially Expressed Genes')
#     plt.xlabel('Log2 Fold Change')
#     plt.ylabel('-Log10 (P-Value)')
#     plt.legend()

#     # Add the number of significant DEGs as a text label
#     ax = plt.gca()
#     ax.text(
#         0.05, 0.95,
#         f'Number of Significant DEGs: {num_significant_degs}',
#         transform=ax.transAxes,
#         fontsize=6,
#         color='black',
#         verticalalignment='top'
#     )

#     # Show the plot
#     plt.show()

#     # Optional: Save the plot
#     plt.savefig('volcano_plot.png', dpi=300)
#     return