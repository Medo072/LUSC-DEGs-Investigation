# import pandas as pd
# import numpy as np
# import gseapy as gp
# import matplotlib.pyplot as plt

# # Load the data
# log2fc_file = 'lusc_degs_ranked_by_log2fc.csv'
# pvalues_file = 'DEGs_with_p_values.csv'

# log2fc_df = pd.read_csv(log2fc_file)
# pvalues_df = pd.read_csv(pvalues_file)

# # Merge the DataFrames on 'Gene_Name'
# merged_df = pd.merge(log2fc_df, pvalues_df, on='Gene_Name')

# # Replace zero P-Values to avoid log(0)
# merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

# # Prepare ranked data
# ranked_data = merged_df[['Gene_Name', 'log2FC']].sort_values(by='log2FC', ascending=False)
# ranked_data.columns = ['gene', 'score']

# # Save ranked data for GSEA
# ranked_data_file = 'ranked_gene_list.rnk'
# ranked_data.to_csv(ranked_data_file, sep='\t', index=False, header=False)

# # Create a custom gene set file (.gmt)
# custom_gene_set_file = 'custom_gene_set.gmt'
# with open(custom_gene_set_file, 'w') as gmt:
#     # Example: Group genes based on thresholds
#     high_expression = merged_df.loc[merged_df['log2FC'] > 1, 'Gene_Name'].tolist()
#     low_expression = merged_df.loc[merged_df['log2FC'] < -1, 'Gene_Name'].tolist()
    
#     # Write each gene set to the .gmt file
#     gmt.write(f"High_Expression\tHigh log2FC genes\t" + "\t".join(high_expression) + "\n")
#     gmt.write(f"Low_Expression\tLow log2FC genes\t" + "\t".join(low_expression) + "\n")

# # Run GSEA with custom gene set
# gsea_results = gp.prerank(
#     rnk=ranked_data_file,
#     gene_sets=custom_gene_set_file,
#     outdir='GSEA_results',  # Directory to save results
#     permutation_num=100,  # Number of permutations
# )

# # Inspect results
# results_df = gsea_results.res2d
# results_csv = 'GSEA_results/gsea_summary.csv'
# results_df.to_csv(results_csv, index=False)

# # Plot top pathways
# top_pathways = results_df.sort_values('FDR q-val').head(10)
# print("Top pathways:")
# print(top_pathways)



# # Ensure all values in 'FDR q-val' are numeric, replacing invalid ones with a high value
# top_pathways['FDR q-val'] = pd.to_numeric(top_pathways['FDR q-val'], errors='coerce')
# top_pathways['FDR q-val'] = top_pathways['FDR q-val'].fillna(1.0)  # Replace NaN with 1.0 (non-significant)

# # Avoid log(0) by setting a small floor value
# top_pathways['FDR q-val'] = top_pathways['FDR q-val'].replace(0, 1e-10)

# # Plot top pathways
# plt.barh(
#     top_pathways['Term'], 
#     -np.log10(top_pathways['FDR q-val']), 
#     color='skyblue'
# )
# plt.xlabel('-log10(FDR q-value)')
# plt.ylabel('Pathway')
# plt.title('Top Enriched Pathways')
# plt.tight_layout()

# # Save and show the plot
# plt.savefig('top_pathways.png', dpi=300)
# plt.show()

import pandas as pd
import numpy as np
import gseapy as gp

# Load the data
pvalues_file = 'degs/DEGs_with_p_values.csv'
log2fc_file = 'degs/lusc_degs_ranked_by_log2fc.csv'

# Read the CSV files
pvalues_df = pd.read_csv(pvalues_file)
log2fc_df = pd.read_csv(log2fc_file)

# Merge the DataFrames on the common 'Gene_Name' column
merged_df = pd.merge(log2fc_df, pvalues_df, on='Gene_Name')

# Replace zero p-values to avoid log(0)
merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

# Prepare ranked data for GSEA
ranked_data = merged_df[['Gene_Name', 'log2FC']].sort_values(by='log2FC', ascending=False)
ranked_data.columns = ['gene', 'score']

# Save the ranked data to a file (optional)
ranked_data_file = 'ranked_gene_list.rnk'
ranked_data.to_csv(ranked_data_file, sep='\t', index=False, header=False)

# Run GSEA
gsea_results = gp.prerank(
    rnk=ranked_data_file,  # Use the ranked gene list file
    gene_sets='KEGG_2021_Human',  # Replace with your desired gene set
    outdir='GSEA_results',  # Output directory for results
    permutation_num=100,  # Number of permutations
)

# Inspect the results
print(gsea_results.res2d.head())

# Save summary of results
gsea_results.res2d.to_csv('gsea_summary.csv', index=False)

print("GSEA analysis completed and results saved!")
