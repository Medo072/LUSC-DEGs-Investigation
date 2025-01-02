import warnings

import pandas as pd

from utils import (
    find_degs,
    save_list_to_csv,
    drop_genes_on_condition,
    extract_non_zeros,
    drop_CNA_with_zeros,
    get_common_subjects,
    rank_degs_by_fold_change,
    rank_degs_by_statistic,
    get_columns_list,
    check_directory,
    create_volcano_plot,
    perform_gsea_analysis,
)

warnings.filterwarnings("ignore")


def main():
    # Read the healthy gene expression data
    GE_lusc_healthy = pd.read_csv(
        "data/lusc-rsem-fpkm-tcga_paired.txt", delimiter="\t", keep_default_na=False
    )
    print(f"The Gene Expression of healthy lung:\n{GE_lusc_healthy}")

    # Read the tumor gene expression data
    GE_lusc_tumor = pd.read_csv(
        "data/lusc-rsem-fpkm-tcga-t_paired.txt",
        delimiter="\t",
        keep_default_na=False,
    )
    print(f"The Gene Expression of tumor lung:\n{GE_lusc_tumor}")

    # Set Hugo_Symbol column the index of all dataframes
    GE_lusc_healthy.set_index("Hugo_Symbol", inplace=True)
    GE_lusc_tumor.set_index("Hugo_Symbol", inplace=True)

    data = [(GE_lusc_healthy, GE_lusc_tumor)]

    rslt = []
    for h, t in data:
        rslt.append(drop_genes_on_condition(h, t)) #check its implementation

    GE_lusc_healthy, GE_lusc_tumor = extract_non_zeros(rslt[0][0], rslt[0][1])

    GE_lusc_healthy, GE_lusc_tumor = GE_lusc_healthy.dropna(), GE_lusc_tumor.dropna()

    assert GE_lusc_healthy.isna().sum(axis=1).sum() == 0
    assert GE_lusc_tumor.isna().sum(axis=1).sum() == 0

    assert GE_lusc_healthy.eq(0).sum(axis=1).sum() == 0
    assert GE_lusc_tumor.eq(0).sum(axis=1).sum() == 0

    # Calculate the mean gene expression values for healthy and tumor samples for LUSC, then create a DataFrame
    GE_lusc_healthy_means = GE_lusc_healthy.iloc[:, 1:].mean(axis=1)
    GE_lusc_tumor_means = GE_lusc_tumor.iloc[:, 1:].mean(axis=1)
    df_lusc_means = pd.DataFrame(
        {"Healthy": GE_lusc_healthy_means, "Tumor": GE_lusc_tumor_means}
    )
    check_directory("results/")
    GE_lusc_ratio = GE_lusc_tumor_means / GE_lusc_healthy_means

    # Print the DataFrame containing mean gene expression values for lusc
    print(f"Mean gene expression values for LUSC dataset:\n{df_lusc_means}")

    # Find the list of DEGs of the LUSC dataset, in addition to the list of p_value_lusc
    DEGs_lusc, p_values_lusc, statistics_lusc = find_degs(GE_lusc_healthy.iloc[:, 1:], GE_lusc_tumor.iloc[:, 1:], "paired")
    DEGs_lusc_ind, p_values_lusc_ind, statistics_lusc_ind = find_degs(GE_lusc_healthy.iloc[:, 1:], GE_lusc_tumor.iloc[:, 1:], "independent")

    # Save the list of LUSC DEGs to DEGs_lusc.csv file
    save_list_to_csv(DEGs_lusc, "Gene_Name", "results/DEGs_lusc.csv")
    print("List of DEGs in LUSC dataset is saved as DEGs_lusc.csv")

    save_list_to_csv(DEGs_lusc_ind, "Gene_Name", "results/DEGs_lusc_ind.csv")
    print("List of DEGs in LUSC dataset is saved as DEGs_lusc.csv")

    # Convert the p_value_lusc to a dataframe and save it to p_values_lusc.csv
    p_values_lusc_df = pd.DataFrame(p_values_lusc)
    p_values_lusc_df.to_csv("results/p_values_lusc.csv", index=False)
    print("List of p_value of each gene in LUSC dataset is saved as p_values_lusc.csv")

    statistics_lusc_df = pd.DataFrame(statistics_lusc)
    statistics_lusc_df.to_csv("results/statistic_lusc.csv", index=False)
    print("List of statistic of each gene in LUSC dataset is saved as statistic_lusc.csv")

    p_values_lusc_ind_df = pd.DataFrame(p_values_lusc_ind)
    p_values_lusc_ind_df.to_csv("results/p_values_lusc_ind.csv", index=False)
    print("List of p_value of each gene in LUSC dataset is saved as p_values_lusc_ind.csv")

    statistics_lusc_ind_df = pd.DataFrame(statistics_lusc_ind)
    statistics_lusc_ind_df.to_csv("results/statistic_lusc_ind.csv", index=False)
    print("List of statistic of each gene in LUSC dataset is saved as statistic_lusc_ind.csv")

    DEGs_lusc_df = pd.DataFrame({"Gene_Name": DEGs_lusc})
    statistics_lusc_df.set_index("Gene_Name", inplace=True)

    DEGs_lusc_ind_df = pd.DataFrame({"Gene_Name": DEGs_lusc})
    statistics_lusc_ind_df.set_index("Gene_Name", inplace=True)
    
    DEGs_with_p_values = p_values_lusc_df.loc[DEGs_lusc_df.index]
    DEGs_with_p_values.to_csv("results/DEGs_with_p_values.csv", index=False)
    
    lusc_CNA = pd.read_csv(
        "data/lusc_CNV_core.txt", delimiter="\t", keep_default_na=False
    )

    GE_lusc_with_statistic = statistics_lusc_df.loc[DEGs_lusc_df["Gene_Name"]]
    GE_lusc_with_statistic_ind = statistics_lusc_ind_df.loc[DEGs_lusc_ind_df["Gene_Name"]]
    # Set the index of CNA data to be the 'feature' column
    lusc_CNA.set_index("feature", inplace=True)

    # Filter out columns with more than 50% zeros for  lusc CNA DataFrame
    lusc_CNA = drop_CNA_with_zeros(lusc_CNA)

    # Drop the last column in each dataframe (Unnamed column, corrupted column)
    lusc_CNA.pop(lusc_CNA.columns[-1])

    # Find common subjects (individuals/samples) between gene expression and CNA data
    lusc_subjects = get_common_subjects(GE_lusc_tumor.columns[1:], lusc_CNA.index)
    # Select and set the index for tumor data based on common subjects
    lusc_tumor_data = GE_lusc_tumor[list(lusc_subjects)]

    lusc_CNA_common = lusc_CNA.loc[list(lusc_subjects)]

    GE_lusc_ratio = pd.DataFrame(GE_lusc_ratio, index=GE_lusc_ratio.index)
    # Rank DEGs by fold change values
    print(GE_lusc_ratio)
    lusc_degs_ranked_by_log2FC = rank_degs_by_fold_change(
        GE_lusc_ratio
    )
    lusc_degs_ranked_by_log2FC.to_csv("results/lusc_degs_ranked_by_log2fc.csv", index=True)

    lusc_degs_ranked_by_statistic = rank_degs_by_statistic(
        GE_lusc_with_statistic
    )
    
    lusc_degs_ranked_by_statistic_ind = rank_degs_by_statistic(
        GE_lusc_with_statistic_ind
    )

    # Print the ranked DEGs based on fold change values

    # lusc_final_form_FC = pd.concat(
    #     [
    #         lusc_tumor_data.loc[lusc_degs_ranked_by_log2FC.head().index].transpose(),
    #         pd.DataFrame(lusc_CNA_common),
    #     ],
    #     axis=1,
    # )
    
    # lusc_final_form_hyp = pd.concat(
    #     [
    #         lusc_tumor_data.loc[lusc_degs_ranked_by_statistic.head().index].transpose(),
    #         pd.DataFrame(lusc_CNA_common),
    #     ],
    #     axis=1,
    # ) 
    
    # lusc_final_form_hyp_ind = pd.concat(
    #     [
    #         lusc_tumor_data.loc[lusc_degs_ranked_by_statistic_ind.head().index].transpose(),
    #         pd.DataFrame(lusc_CNA_common),
    #     ],
    #     axis=1,
    # ) 

    # Get the list of most significant genes
    lusc_genes = get_columns_list(lusc_degs_ranked_by_log2FC.transpose(), 10)
    print(lusc_genes)
    
    lusc_genes_stat = get_columns_list(lusc_degs_ranked_by_statistic.transpose(), 10)
    print(lusc_genes_stat)
    
    lusc_genes_stat_ind = get_columns_list(lusc_degs_ranked_by_statistic_ind.transpose(), 10)
    print(lusc_genes_stat_ind)
    # Create a volcano plot
    create_volcano_plot(
        'results/DEGs_with_p_values.csv',
        'results/lusc_degs_ranked_by_log2fc.csv'
    )

    # Perform GSEA analysis
    perform_gsea_analysis(
        'results/lusc_degs_ranked_by_log2fc.csv',
        'results/DEGs_with_p_values.csv',
        'Cancer_Cell_Line_Encyclopedia'
    )


if __name__ == "__main__":
    main()
