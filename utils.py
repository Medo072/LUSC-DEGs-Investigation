import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from scipy.stats import shapiro, wilcoxon, ranksums
from scipy.stats import t
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
    """
    Check the normality of the difference between two data columns using the Shapiro-Wilk test.

    This function assesses whether the difference between two datasets (e.g., healthy and tumor) follows a normal distribution.

    Parameters:
        healthy (array-like): The data column representing the healthy condition.
        tumor (array-like): The data column representing the tumor condition.

    Returns:
        float: The p-value obtained from the Shapiro-Wilk test.

    Notes:
        The Shapiro-Wilk test is a statistical test used to evaluate the normality of a dataset.
        It tests the null hypothesis that the data follows a normal distribution.
        A low p-value (typically below 0.05) suggests that the data significantly deviates from normality.

    Example:
        p_value = check_normality(healthy_data, tumor_data)
        if p_value < 0.05:
            print("The data significantly deviates from a normal distribution.")
        else:
            print("The data may follow a normal distribution.")
    """
    # Perform Shapiro-Wilk test on the difference between healthy and tumor data
    _, p_value = shapiro(healthy - tumor)
    return p_value


def drop_genes_on_condition(df_h, df_t, threshold=0.5):
    """
    Drop genes from two DataFrames based on a specified threshold condition.

    Parameters:
    - df_h (DataFrame): The first DataFrame containing gene data.
    - df_t (DataFrame): The second DataFrame containing gene data (should have the same structure as df_h).
    - threshold (float): The threshold value to determine if a gene should be dropped.

    Returns:
    - Tuple[DataFrame, DataFrame]: A tuple containing two filtered DataFrames (df_h_filtered, df_t_filtered).
    """
    num_subjects = len(df_h.columns) - 1
    assert len(df_h.columns) == len(df_t.columns)

    mask_h = (df_h.eq(0).sum(axis=1) >= num_subjects * 0.5)
    mask_t = (df_t.eq(0).sum(axis=1) >= num_subjects * 0.5)
    mask_any = mask_h | mask_t

    df_h_filtered = df_h[~mask_any]
    df_t_filtered = df_t[~mask_any]

    return df_h_filtered, df_t_filtered


def extract_non_zeros(df_h, df_t):
    """
    Filter zeros out  from two DataFrames.
    Parameters:
    - df_h (DataFrame): The first DataFrame containing gene data.
    - df_t (DataFrame): The second DataFrame containing gene data (should have the same structure as df_h).

    Returns:
    - Tuple[DataFrame, DataFrame]: A tuple containing two zero filtered DataFrames (df_h_filtered, df_t_filtered).

    """

    # masks for the dataframes to filter out zeros
    mask_h = (df_h != 0)
    mask_t = (df_t != 0)

    # a slot is taken when it is non zero in both masks
    mask_any = mask_h & mask_t

    df_h_non_zeros = df_h[mask_any]
    df_t_non_zeros = df_t[mask_any]

    return df_h_non_zeros, df_t_non_zeros


def calc_p_values(healthy, tumor, sample_type):
    """
    Compare the gene expression between healthy and tumor samples to identify differentially expressed genes.

    This function calculates the statistical significance of the difference in gene expression
    between healthy and tumor samples for each gene using the Wilcoxon Signed Rank-Sum Test.

    Parameters:
        healthy (pandas.Series): Gene expression data for healthy samples.
        tumor (pandas.Series): Gene expression data for tumor samples.

    Returns:
        list of dict: A list of dictionaries containing gene names and their associated p-values
                     indicating differential expression. If no genes are differentially expressed,
                     an empty list is returned. Each dictionary includes the following keys:
                     - "Gene_Name" (str): The name of the gene.
                     - "P_Value" (float): The p-value of the Wilcoxon test.

    Notes:
        - The Wilcoxon Signed Rank-Sum Test is applied to each gene to compare gene expression
          between healthy and tumor samples.
        - Genes with p-values less than 0.05 (alpha) are considered differentially expressed and included
          in the output list.
        - If no genes are differentially expressed, the function returns an empty list.

    Examples:
         healthy_data = pd.Series([1.2, 1.4, 1.6, 1.8, 2.0])
         tumor_data = pd.Series([2.1, 2.3, 2.5, 2.7, 2.9])
         result = calc_p_values(healthy_data, tumor_data)
         print(result)
        [{'Gene_Name': 'Gene1', 'P_Value': 0.023}, {'Gene_Name': 'Gene2', 'P_Value': 0.042}]

    """
    # Initialize empty lists to store genes with significant differential expression and their p-values
    output_genes = []
    p_values = []
    statistics = []
    output_with_statistic = []
    # Iterate over each gene in the input data
    for gene in healthy.index:
        if sample_type == "paired":
            # Calculate the Wilcoxon p-value for the gene's expression between healthy and tumor samples
            statistic, p_value_wilcoxon = wilcoxon(healthy.loc[gene], tumor.loc[gene])
        else:
            statistic, p_value_wilcoxon = ranksums(healthy.loc[gene], tumor.loc[gene])

        # Append the gene name and p-value to the output lists
        p_values.append(p_value_wilcoxon)
        statistics.append(statistic)
        output_genes.append({"Gene_Name": gene, "P_Value": p_value_wilcoxon})
        output_with_statistic.append({"Gene_Name": gene, "Statistic": statistic})
    # Return the list of genes with significant differential expression
    return output_genes, p_values, output_with_statistic


def find_degs(healthy, tumor, sample_type):
    """
    Find differentially expressed genes (DEGs) between healthy and tumor samples.

    This function compares gene expression between healthy and tumor samples and identifies DEGs.
    It performs a series of statistical tests, including the assessment of normality and
    comparison using the Wilcoxon Rank-Sum Test based on the data distribution.

    Parameters:
        healthy (pandas.DataFrame): Gene expression data for healthy samples.
        tumor (pandas.DataFrame): Gene expression data for tumor samples.

    Returns:
        list: A list of gene names that are differentially expressed between healthy and tumor samples.

    Notes:
        - The function first assesses the normality of each gene's expression using the
          `check_normality` function.
        - It categorizes genes as 'normal' or 'not normal' based on a significance level (alpha) of 0.05.
        - DEGs are identified by comparing gene expression using the Wilcoxon Rank-Sum Test,
          which is suitable for non-normally distributed data.
    """
    # Define the significance level (alpha)
    alpha = 0.05

    # Define empty lists for both normally distributed genes and non-normally distributed genes
    normal_genes = []
    non_normal_genes = []
    p_values = []

    # Loop over each gene and check its normality, categorize genes as normal or not normal
    for gene in healthy.index:
        # Check the normality of the gene's expression using the `check_normality` function
        p_value = check_normality(healthy.loc[gene], tumor.loc[gene])

        # Based on the p-value, categorize the gene as 'normal' or 'not normal'
        if p_value > alpha:
            normal_genes.append(gene)
        else:
            non_normal_genes.append(gene)

    # Calculate p-values for all genes and identify DEGs using false discovery rate control
    genes_and_p_values, p_values, genes_and_statistic = calc_p_values(healthy, tumor, sample_type=sample_type)

    # Apply false discovery rate control to the p-values
    p_values_after_fdr = false_discovery_control(p_values)

    # Identify DEGs based on the adjusted p-values
    DEGs = [gene for gene, p_value_ in zip(healthy.index, p_values_after_fdr) if p_value_ > .05]

    # Return the list of differentially expressed genes (DEGs)
    return DEGs, genes_and_p_values, genes_and_statistic


def save_list_to_csv(list, header, file_name, index=False):
    """
    Save a list of differentially expressed genes (DEGs) to a CSV file.

    Parameters:
        list (list): List of DEG names.
        header (str): The header of the list.
        file_name (str): Name of the CSV file to save the DEGs.

    Returns:
        None
    """
    df_DEGs = pd.DataFrame({header: list})
    df_DEGs.to_csv(file_name, index=index)


def drop_CNA_with_zeros(df):
    """
    Drop columns with excessive zero values from a DataFrame.

    This function takes a DataFrame as input and removes columns where more than half of
    their values are zeros (0).

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing data.

    Returns:
        pandas.DataFrame: A filtered DataFrame with columns that do not exceed half of
        their values as zeros.

    Notes:
        - This function is useful for data preprocessing when dealing with datasets
          where columns with many zero values may not provide significant information.
    """
    # Calculate the number of subjects (samples) in the DataFrame
    num_subjects = len(df)

    # Create a Boolean DataFrame where True represents zero values
    is_zero = df == 0

    # Calculate the sum of True values (zeros) in each column
    zero_counts = is_zero.sum()

    # Create a condition that checks if the count of zeros in each column is less than half of the total subjects
    condition = zero_counts < num_subjects * 0.5
    # Use DataFrame indexing to select columns based on the condition
    df_filtered = df.loc[:, condition]

    return df_filtered


def get_common_subjects(data_1, data_2):
    """
    Get the common elements between two sequences.

    This function takes two input sequences, 'data_1' and 'data_2', and returns a set
    containing the common elements that exist in both sequences.

    Parameters:
        data_1 (iterable): The first sequence of elements.
        data_2 (iterable): The second sequence of elements.

    Returns:
        set: A set containing the common elements shared between 'data_1' and 'data_2'.

    Example:
        data1 = [1, 2, 3, 4, 5]
        data2 = [3, 4, 5, 6, 7]
        common_elements = get_common_subjects(data1, data2)
        print(common_elements)  # Output: {3, 4, 5}
    """
    # Create sets from the input sequences to remove duplicates and find unique elements
    set1, set2 = set(data_1), set(data_2)

    # Use set intersection (&) to find common elements between set1 and set2
    common_elements = set1 & set2

    # Return the set of common elements
    return common_elements


def rank_degs_by_p_value(data: pd.DataFrame):
    """
    Rank differentially expressed genes (DEGs) by p-value in descending order.

    This function takes a pandas DataFrame 'data' containing information about differentially
    expressed genes (DEGs) and their associated p-values. It sorts the DataFrame in descending
    order based on the 'P_Value' column, effectively ranking the DEGs from highest to lowest p-value.

    Parameters:
        data (pd.DataFrame): A DataFrame containing DEGs and their associated p-values.

    Returns:
        pd.DataFrame: A new DataFrame with DEGs ranked by p-value in descending order.
    """
    # Sort the input DataFrame 'data' by the 'P_Value' column in descending order
    sorted_data = data.sort_values(by="P_Value", ascending=False)

    # Return the sorted DataFrame
    return sorted_data


def rank_degs_by_fold_change(data: pd.DataFrame):
    """
    Rank differentially expressed genes (DEGs) by fold change in descending order.

    This function takes a pandas DataFrame 'data' containing information about differentially
    expressed genes (DEGs) and their fold change values. It calculates the absolute fold change,
    ranks the DEGs by their absolute fold change in descending order, and returns a sorted DataFrame.

    Parameters:
        data (pd.DataFrame): A DataFrame containing DEGs and their fold change values.

    Returns:
        pd.DataFrame: A new DataFrame with DEGs ranked by absolute fold change in descending order.
    """
    # Calculate the logarithm (base 2) of fold change and store it in a new column 'log2FC'
    data["Gene_Name"] = data.index
    data.set_index("Gene_Name", inplace=True)
    data["log2FC"] = np.log2(data)
    print(data.head())
    # Calculate the absolute value of 'log2FC' and store it in a new column 'abslog2FC'
    data["abslog2FC"] = np.abs(data["log2FC"])

    # Create a new DataFrame 'fold_change' to hold the calculated fold change values
    fold_change = data

    # Rename the first column (presumably 'Gene_Name') to 'Fold Change'
    fold_change.columns.values[0] = "Fold Change"

    # Sort the 'fold_change' DataFrame by the 'abslog2FC' column in descending order
    fold_change_sorted = fold_change.sort_values(by="abslog2FC", ascending=False)

    # Return the sorted DataFrame
    return fold_change_sorted


def rank_degs_by_statistic(data: pd.DataFrame):
    data["abs"] = np.abs(data["Statistic"])
    ranked = data.sort_values(by="abs", ascending=False)
    return ranked


def get_columns_list(df, n=5):
    """
    Retrieve a list of column names from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame from which column names will be extracted.
        n (int, optional): The number of column names to retrieve. Default is 5.

    Returns:
        list: A list containing the names of the first 'n' columns of the DataFrame.
    """
    return df.columns[:n].tolist()


def get_sub_dfs(df, column_names):
    """
    Create sub-DataFrames by combining selected columns with all other columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame from which sub-DataFrames will be created.
        column_names (list): A list of column names to create sub-DataFrames for.

    Returns:
        dict: A dictionary where keys are column names and values are corresponding sub-DataFrames.
    """
    subset_df = df[column_names]
    dfs = {}

    for col_name in column_names:
        # Create a DataFrame for the current column
        current_df = df.drop(columns=column_names).join(subset_df[col_name])

        # Store the DataFrame in the dictionary with the column name as the key
        dfs[col_name] = current_df

    return dfs


# def check_directory(directory_path):
#     """
#     Check if a directory exists, and if not, create it.

#     This function checks whether a directory exists at the specified 'directory_path'. If the directory does not
#     exist, it creates the directory including all intermediate directories as needed.

#     Parameters:
#         directory_path (str): The path of the directory to check or create.

#     Returns:
#         None

#     Example:
#         check_directory('output/results')  # Checks if 'output/results' directory exists and creates it if not.
#     """
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)

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
