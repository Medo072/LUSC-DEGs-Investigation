import pandas as pd
import numpy as np
import plotly.express as px

# Load the data
pvalues_df = pd.read_csv('results/csv files/DEGs_with_p_values.csv')  # Replace with your p-values file name
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
significant_mask = (
    (abs(merged_df['log2FC']) >= log2fc_threshold) &
    (merged_df['P_Value'] <= pvalue_threshold)
)
merged_df.loc[significant_mask, 'Significant'] = 'Significant'

# Count the number of significant DEGs
num_significant_degs = significant_mask.sum()
print(f"The number of significant DEGs: {num_significant_degs}")

# Create the volcano plot using Plotly
fig = px.scatter(
    merged_df,
    x='log2FC',
    y='negLog10PValue',
    color='Significant',
    color_discrete_map={
        'Significant': 'red',
        'Not Significant': 'grey'
    },
    hover_data=['Gene_Name', 'log2FC', 'P_Value'],
    title='Interactive Volcano Plot of Differentially Expressed Genes',
    labels={
        'log2FC': 'Log2 Fold Change',
        'negLog10PValue': '-Log10 (P-Value)'
    },
    template='plotly_white'
)

# Add threshold lines
fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", line_color="blue", annotation_text="P-Value Threshold", annotation_position="bottom left")
fig.add_vline(x=log2fc_threshold, line_dash="dash", line_color="green", annotation_text="+Log2 FC Threshold", annotation_position="top right")
fig.add_vline(x=-log2fc_threshold, line_dash="dash", line_color="green", annotation_text="-Log2 FC Threshold", annotation_position="top left")

# Show the plot
fig.show()

# Optional: Save the plot to an HTML file
fig.write_html('volcano_plot_interactive.html')
