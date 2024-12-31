from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QWidget, QLabel, QMessageBox, QLineEdit, QHBoxLayout)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import sys


class VolcanoPlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Volcano Plot Viewer")
        self.setGeometry(100, 40, 1500, 1000)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Instruction label
        self.label = QLabel("Select two CSV files to generate the volcano plot:")
        self.label.setMaximumSize(1500, 40)
        self.label.setStyleSheet("font-family: 'Times New Roman'; font-size: 32px;")
        self.layout.addWidget(self.label)

        # Buttons
        self.file1_button = QPushButton("Select CSV for P-Values")
        self.file1_button.clicked.connect(self.select_pvalues_file)
        self.file1_button.setStyleSheet("font-family: 'Times New Roman'; font-size: 20px;")
        self.layout.addWidget(self.file1_button)

        self.file2_button = QPushButton("Select CSV for Log2 Fold Changes")
        self.file2_button.clicked.connect(self.select_log2fc_file)
        self.file2_button.setStyleSheet("font-family: 'Times New Roman'; font-size: 20px;")
        self.layout.addWidget(self.file2_button)

        # Input fields for thresholds
        self.threshold_layout = QHBoxLayout()
        self.pvalue_label = QLabel("P-Value Threshold:")
        self.pvalue_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 18px;")
        self.pvalue_input = QLineEdit("0.05")  # Default value
        self.pvalue_input.setMaximumWidth(150)
        self.pvalue_input.setStyleSheet("font-family: 'Times New Roman'; font-size: 22px;")


        self.log2fc_label = QLabel("Log2 FC Threshold:")
        self.log2fc_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 18px;")
        self.log2fc_input = QLineEdit("1")  # Default value
        self.log2fc_input.setMaximumWidth(150)
        self.log2fc_input.setStyleSheet("font-family: 'Times New Roman'; font-size: 22px;")


        self.threshold_layout.addWidget(self.pvalue_label)
        self.threshold_layout.addWidget(self.pvalue_input)
        self.threshold_layout.addWidget(self.log2fc_label)
        self.threshold_layout.addWidget(self.log2fc_input)
        self.layout.addLayout(self.threshold_layout)

        self.plot_button = QPushButton("Generate Volcano Plot")
        self.plot_button.setStyleSheet("font-family: 'Times New Roman'; font-size: 20px;")
        self.plot_button.clicked.connect(self.generate_plot)
        self.layout.addWidget(self.plot_button)

        # Label to show the count of significant DEGs
        self.sig_degs_label = QLabel("Significant DEGs: 0")
        self.sig_degs_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 18px;")
        self.sig_degs_label.setMaximumSize(1500, 40)
        self.layout.addWidget(self.sig_degs_label)

        # Web view for the plot
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)

        # File paths
        self.pvalues_file = None
        self.log2fc_file = None

    def select_pvalues_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select CSV for P-Values", "", "CSV Files (*.csv)")
        if file:
            self.pvalues_file = file
            self.file1_button.setText(f"P-Values File: {file}")

    def select_log2fc_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select CSV for Log2 Fold Changes", "", "CSV Files (*.csv)")
        if file:
            self.log2fc_file = file
            self.file2_button.setText(f"Log2FC File: {file}")

    def generate_plot(self):
        if not self.pvalues_file or not self.log2fc_file:
            QMessageBox.warning(self, "Error", "Please select both CSV files before generating the plot.")
            return

        try:
            # Parse thresholds from input fields
            try:
                pvalue_threshold = float(self.pvalue_input.text())
                log2fc_threshold = float(self.log2fc_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid numerical values for thresholds.")
                return

            # Load the data
            pvalues_df = pd.read_csv(self.pvalues_file)
            log2_fc_df = pd.read_csv(self.log2fc_file)

            # Merge the DataFrames on 'Gene' column
            merged_df = pd.merge(log2_fc_df, pvalues_df, on='Gene_Name')

            # Replace zero p-values to avoid taking log of zero
            merged_df['P_Value'] = merged_df['P_Value'].replace(0, 1e-10)

            # Calculate negative log10 p-value
            merged_df['negLog10PValue'] = -np.log10(merged_df['P_Value'])

            # Add a column to indicate significance
            merged_df['Significant'] = 'Unchanged'
            significant_mask = (
                    (abs(merged_df['log2FC']) >= log2fc_threshold) &
                    (merged_df['P_Value'] <= pvalue_threshold)
            )
            merged_df.loc[significant_mask & (merged_df['log2FC'] > 0), 'Significant'] = 'Increased'
            merged_df.loc[significant_mask & (merged_df['log2FC'] < 0), 'Significant'] = 'Decreased'

            # Count significant DEGs
            significant_count = significant_mask.sum()
            self.sig_degs_label.setText(f"Significant DEGs: {significant_count}")

            # Create the volcano plot using Plotly
            fig = px.scatter(
                merged_df,
                x='log2FC',
                y='negLog10PValue',
                color='Significant',
                color_discrete_map={
                    'Increased': 'red',
                    'Decreased': 'blue',
                    'Unchanged': 'grey'
                },
                hover_data=['Gene_Name', 'log2FC', 'P_Value'],
                title='=Volcano Plot of Differentially Expressed Genes (DEGs)',
                labels={
                    'log2FC': 'Fold Change (Log2)',
                    'negLog10PValue': 'P-Value (-Log10)'
                },
                template='plotly_white'
            )

            # Add threshold lines
            fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", line_color="blue", annotation_text="P-Value Threshold", annotation_position="bottom left")
            fig.add_vline(x=log2fc_threshold, line_dash="dash", line_color="green", annotation_text="+Log2 FC Threshold", annotation_position="top right")
            fig.add_vline(x=-log2fc_threshold, line_dash="dash", line_color="green", annotation_text="-Log2 FC Threshold", annotation_position="top left")

            # Save the plot to a temporary HTML file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            fig.write_html(temp_file.name)
            temp_file.close()

            # Load the temporary file in the web view
            self.web_view.setUrl(QUrl.fromLocalFile(temp_file.name))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while generating the plot: {str(e)}")
            print("Error", f"An error occurred while generating the plot: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = VolcanoPlotApp()
    main_window.show()
    sys.exit(app.exec_())
