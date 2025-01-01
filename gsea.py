import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Debugging helper function
def debug_print(message, value):
    print(f"{message}: {value}")

# Load the data
pvalues_file = './results/csv files/DEGs_with_p_values.csv'
log2fc_file = './results/csv files/lusc_degs_ranked_by_log2fc.csv'

# Read the CSV files
pvalues_df = pd.read_csv(pvalues_file)
log2fc_df = pd.read_csv(log2fc_file)

debug_print("P-values DataFrame head", pvalues_df.head())
debug_print("Log2FC DataFrame head", log2fc_df.head())

# Merge the DataFrames on the common 'Gene_Name' column
merged_df = pd.merge(log2fc_df, pvalues_df, on='Gene_Name')
debug_print("Merged DataFrame head", merged_df.head())

# Replace zero p-values to avoid log(0)
merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

# Prepare ranked data for GSEA
ranked_data = merged_df[['Gene_Name', 'log2FC']].sort_values(by='log2FC', ascending=False)
ranked_data.columns = ['gene', 'score']
debug_print("Ranked data head", ranked_data.head())

# Save the ranked data to a file (optional)
ranked_data_file = 'rnk_files/ranked_gene_list.rnk'
ranked_data.to_csv(ranked_data_file, sep='\t', index=False, header=False)

# Run GSEA
gsea_results = gp.prerank(
    rnk=ranked_data_file,  # Use the ranked gene list file
    gene_sets='Cancer_Cell_Line_Encyclopedia',  # Replace with your desired gene set
    outdir='GSEA_results',  # Output directory for results
    permutation_num=100,  # Number of permutations
)

# Inspect the results
debug_print("Column names in GSEA results", gsea_results.res2d.columns)
debug_print("GSEA results head", gsea_results.res2d.head())

# Save summary of results
gsea_results.res2d.to_csv('GSEA_summary/gsea_summary.csv', index=False)



# Prepare data for visualization
top_pathways = gsea_results.res2d.head(20)  
debug_print("Top pathways DataFrame head", top_pathways)

# Ensure all values in 'FDR q-val' are numeric, replacing invalid ones with a high value
# Check and adjust columns for Bar Graph
if 'Term' not in gsea_results.res2d.columns:
    debug_print("Missing column 'Term'", "Using 'Name' column instead.")
    heatmap_index = 'Name'
else:
    heatmap_index = 'Term'

# Create bar graph
plt.figure(figsize=(10, 6))
sns.barplot(data=top_pathways, x='NES', y=heatmap_index, palette="viridis")
plt.title('Top 10 GSEA Pathways NES Scores')
plt.xlabel('Normalized Enrichment Score (NES)')
plt.ylabel('Pathways')
plt.tight_layout()
plt.show()

print("GSEA analysis completed and visualizations generated!")
