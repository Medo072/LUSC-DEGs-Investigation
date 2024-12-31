import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
pvalues_df = pd.read_csv('results/csv files/DEGs_with_p_values.csv')   # Replace with your p-values file name
log2_fc_df = pd.read_csv('results/csv files/lusc_degs_ranked_by_log2fc.csv')  # Replace with your log2 fold change file name

# Merge the DataFrames on 'Gene' column
merged_df = pd.merge(log2_fc_df, pvalues_df, on='Gene_Name')

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

# Create a boolean mask using your criteria
significant_mask = (
    (merged_df['P_Value'] <= pvalue_threshold) &
    (abs(merged_df['log2FC']) >= log2fc_threshold)
)

# Add a column to indicate significance
merged_df['Significant'] = 'Not Significant'
merged_df.loc[significant_mask, 'Significant'] = 'Significant'

# Count the number of significant DEGs
num_significant_degs = significant_mask.sum()

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
    fontsize=6,
    color='black',
    verticalalignment='top'
)

# Show the plot
plt.show()

# Optional: Save the plot
plt.savefig('volcano_plot.png', dpi=300)