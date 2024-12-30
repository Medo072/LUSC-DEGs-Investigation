import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import false_discovery_control
from scipy.stats import shapiro, wilcoxon
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


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


def calc_p_values(healthy, tumor):
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
        # Calculate the Wilcoxon p-value for the gene's expression between healthy and tumor samples
        statistic, p_value_wilcoxon = wilcoxon(healthy.loc[gene], tumor.loc[gene])

        # Append the gene name and p-value to the output lists
        p_values.append(p_value_wilcoxon)
        statistics.append(statistic)
        output_genes.append({"Gene_Name": gene, "P_Value": p_value_wilcoxon})
        output_with_statistic.append({"Gene_Name": gene, "Statistic": statistic})
    # Return the list of genes with significant differential expression
    return output_genes, p_values, output_with_statistic


def calc_p_values(healthy, tumor):
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
        # Calculate the Wilcoxon p-value for the gene's expression between healthy and tumor samples
        statistic, p_value_wilcoxon = wilcoxon(healthy.loc[gene], tumor.loc[gene])

        # Append the gene name and p-value to the output lists
        p_values.append(p_value_wilcoxon)
        statistics.append(statistic)
        output_genes.append({"Gene_Name": gene, "P_Value": p_value_wilcoxon})
        output_with_statistic.append({"Gene_Name": gene, "Statistic": statistic})
    # Return the list of genes with significant differential expression
    return output_genes, p_values, output_with_statistic


def find_degs(healthy, tumor):
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
    genes_and_p_values, p_values, genes_and_statistic = calc_p_values(healthy, tumor)

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
    data["log2FC"] = np.log2(data)

    # Calculate the absolute value of 'log2FC' and store it in a new column 'abslog2FC'
    data["abslog2FC"] = np.abs(data["log2FC"])

    # Create a new DataFrame 'fold_change' to hold the calculated fold change values
    fold_change = data

    # Rename the first column (presumably 'Gene_Name') to 'Fold Change'
    fold_change.columns.values[0] = "Fold Change"

    # Sort the 'fold_change' DataFrame by the 'abslog2FC' column in descending order
    fold_change_sorted = fold_change.sort_values(by="abslog2FC", ascending=False)

    # Return the sorted DataFrame
    return fold_change_sorted["abslog2FC"]


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


def scikit_model(df):
    """
    Train a scikit-learn Linear Regression model and return it along with data.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing both independent (X) and dependent (Y) variables.

    Returns:
        tuple: A tuple containing the trained Linear Regression model, independent variables (X), and dependent variables (Y).
    """
    # Separate independent (X) and dependent (Y) variables
    df_X = df.iloc[:, :-1]
    df_Y = df.iloc[:, -1]

    # Train a Linear Regression model
    model = LinearRegression().fit(df_X, df_Y)
    return model, df_X, df_Y


def scikit_model_summary(model, df_X, df_Y, output_file, threshold=0.5):
    """
    Calculate summary statistics for a scikit-learn Linear Regression model, including coefficients, errors,
    and accuracy, and save the results to a text file.

    Parameters:
        model (sklearn.linear_model.LinearRegression): The trained Linear Regression model.
        df_X (pandas.DataFrame): The independent variables (X) used for training.
        df_Y (pandas.Series): The dependent variable (Y) used for training.
        output_file (str): The name of the output text file where the summary will be saved.
        threshold (float, optional): The decision threshold for classification. Default is 0.5.

    Prints:
        None: Prints summary statistics to the console.

    Writes:
        None: Writes summary statistics to the specified output file.

    Notes:
        - The function calculates regression coefficients, R-squared, adjusted R-squared, mean squared error (MSE),
          root mean squared error (RMSE), and accuracy (for classification).
        - The results are saved in a text file with the specified name.
    """
    # Perform regression analysis
    n = len(df_Y)
    p = df_X.shape[1]
    predictions = model.predict(df_X)
    residuals = df_Y - predictions
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)

    # Calculate adjusted R-squared
    r_sq = model.score(df_X, df_Y)
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1)

    # Calculate standard errors and t-values
    se = np.sqrt(np.sum(residuals ** 2) / (n - p - 1))
    t_values = model.coef_ / se

    # Calculate p-values for coefficients
    p_values = 2 * (1 - t.cdf(np.abs(t_values), df=n - p - 1))

    # Calculate accuracy for classification
    predicted_labels = (predictions >= threshold).astype(int)
    actual_labels = (df_Y >= threshold).astype(int)
    accuracy = np.mean(predicted_labels == actual_labels)

    feature_names = df_X.columns.tolist()

    with open(output_file, "w") as file:
        file.write("Regression coefficients: \r")
        for i, coef in enumerate(model.coef_):
            file.write(
                f"    Coefficient for {feature_names[i]}\t:{coef:.4f}\t(p-value: {p_values[i]:.4f})\r",
            )
        file.write("\rIntercept = %0.4f" % (model.intercept_))
        file.write("\rR-squared = %0.4f" % (r_sq))
        file.write("\rAdjusted R-squared = %0.4f" % (adj_r_sq))
        file.write(f"\rMean Squared Error (MSE) = {mse:.4f}")
        file.write(f"\rRoot Mean Squared Error (RMSE) = {rmse:.4f}")
        file.write(f"\rAccuracy = {accuracy:.4f}")


def statsmodels_model(df):
    """
    Perform Ordinary Least Squares (OLS) regression using the StatsModels library.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing both independent and dependent variables.

    Returns:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The trained OLS regression model.
        df_X (pandas.DataFrame): The independent variables (features).
        df_Y (pandas.Series): The dependent variable.

    Notes:
        - This function fits an OLS regression model using the StatsModels library to the input data.
        - It returns the trained model, the independent variables (features), and the dependent variable.
    """
    # Extract the number of samples from the DataFrame
    n_samples, _ = df.shape

    # Separate the independent variables (features) and the dependent variable
    df_X = df.iloc[:, :-1]  # All columns except the last one
    df_Y = df.iloc[:, -1]  # The last column

    # Add an intercept (constant) term to the independent variables
    X_intercept = np.ones((n_samples, 1))
    X_sm = np.hstack((df_X, X_intercept))

    # Create and fit the OLS regression model
    mod = sm.OLS(df_Y, X_sm)
    model = mod.fit()

    # Return the trained model, independent variables, and dependent variable
    return model, df_X, df_Y


def check_directory(directory_path):
    """
    Check if a directory exists, and if not, create it.

    This function checks whether a directory exists at the specified 'directory_path'. If the directory does not
    exist, it creates the directory including all intermediate directories as needed.

    Parameters:
        directory_path (str): The path of the directory to check or create.

    Returns:
        None

    Example:
        check_directory('output/results')  # Checks if 'output/results' directory exists and creates it if not.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def statsmodels_model_summary(model, df_X, output_file, threshold=0.05):
    """
    Print a summary of the StatsModels regression model, including coefficients and p-values.

    Parameters:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The trained StatsModels regression model.
        df_X (pandas.DataFrame): The independent variables (X) used for training.
        output_file (str): The name of the output text file where the summary will be saved.
        threshold (float, optional): The decision threshold for features selection. Default is 0.05.

    Prints:
        None: Prints various statistics related to the model to the console.

    Writes:
        None: Writes various statistics related to the model to the specified output file.
    """
    with open(output_file, "w") as file:
        # Print the summary of the regression model
        model_summary = model.summary().as_text()
        file.write(f"{model_summary}\r")

        # Get the features names list, add const as the last element
        feature_names = df_X.columns.tolist()
        feature_names.append("const")

        # Get the p-values for each coefficient
        p_values = model.pvalues
        p_values.index = feature_names
        file.write(f"\rThe p-values are:\r")
        for feature, pvalue in p_values.iteritems():
            file.write(f"\t{feature}\t:{pvalue}\r")

        # Get the regression coefficients
        coefficients = model.params
        coefficients.index = feature_names
        file.write(f"\rThe regression coefficients are:\r")
        for feature, reg_coeff in coefficients.iteritems():
            file.write(f"\t{feature}\t:{reg_coeff}\r")

        # Filter coefficients based on p-values (e.g., keep only those with p-value <= threshold)
        filtered_coefficients = coefficients[p_values <= threshold]
        file.write(
            f"\rThe most significant features coefficients are:\r",
        )
        for feature, reg_coeff in filtered_coefficients.iteritems():
            file.write(f"\t{feature}:{reg_coeff:.2f}\r")


def create_models(dfs, models, output_directory, prefix):
    """
    Create and summarize machine learning models for a collection of dataframes.

    Parameters:
        dfs (dict): A dictionary of dataframes where keys are gene names and values are dataframes.
        models (dict): A dictionary to store the generated models.
        output_directory (str): The directory where model summary files will be saved.
        prefix (str): A prefix to include in the output file names.

    Returns:
        None: The function populates the 'models' dictionary with machine learning models and saves model summaries to text files.

    Prints:
        Model summaries for each gene using both Scikit-learn and StatsModels.
    """
    output_directory += f"/{prefix}"

    sklearn_directory = output_directory + "/Sklearn models"
    check_directory(sklearn_directory)

    statsmodels_directory = output_directory + "/Statsmodels"
    check_directory(statsmodels_directory)

    for name, df in dfs.items():
        # Make the scikit model for each gene
        sci_model, df_X, df_Y = scikit_model(df)
        models["scikit"][name] = sci_model
        scikit_output_file = os.path.join(
            sklearn_directory, f"{prefix}_{name}_scikit_model_summary.txt"
        )
        scikit_model_summary(sci_model, df_X, df_Y, scikit_output_file)
        print(
            f"{name} Scikit Model Summary was saved in {output_directory}/{prefix}_{name}_scikit_model_summary.txt successfully."
        )
        # Make the statsmodels model for each gene
        stats_model, _, _ = statsmodels_model(df)
        models["statsmodels"][name] = stats_model
        statsmodels_output_file = os.path.join(
            statsmodels_directory, f"{prefix}_{name}_statsmodels_model_summary.txt"
        )
        statsmodels_model_summary(stats_model, df_X, statsmodels_output_file)
        print(
            f"{name} Statsmodels Model Summary was saved in {output_directory}/{prefix}_{name}_statsmodels_model_summary.txt successfully."
        )


def to_x_y(df):
    """
    Split a DataFrame into features and target variables.

    This function separates a given DataFrame into two parts: features and target variables.
    It assumes that the first five columns of the DataFrame are target variables,
    and the remaining columns are features.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be split.

    Returns:
    - tuple: A tuple containing two DataFrames:
        - The first DataFrame contains the features.
        - The second DataFrame contains the target variables.

    Examples:
    If your DataFrame `df` has the following structure:

    ```
       Target1  Target2  Target3  Target4  Target5  Feature1  Feature2  Feature3
    0        1        0        1        0        1       0.1       0.2       0.3
    1        0        1        0        1        0       0.4       0.5       0.6
    2        1        0        1        0        1       0.7       0.8       0.9
    ```

    Calling `to_x_y(df)` would return a tuple with two DataFrames as follows:

    - The first DataFrame (features):
    ```
       Feature1  Feature2  Feature3
    0       0.1       0.2       0.3
    1       0.4       0.5       0.6
    2       0.7       0.8       0.9
    ```

    - The second DataFrame (targets):
    ```
       Target1  Target2  Target3  Target4  Target5
    0        1        0        1        0        1
    1        0        1        0        1        0
    2        1        0        1        0        1
    ```
    """
    # Get the column names of the input DataFrame
    cols = df.columns

    # Extract the first five columns as target variables
    targets = df.loc[:, cols[:5]]

    # Extract the remaining columns as features
    features = df.loc[:, cols[5:]]

    # Return the features and target variables as a tuple
    return features, targets


def train_models(data, model_name):
    """
    Train regression models (Ridge or Lasso) on the provided dataset and evaluate their performance.

    Parameters:
        data (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing two DataFrames:
            - The first DataFrame contains the feature variables.
            - The second DataFrame contains the target variables.
        model_name (str): The name of the regression model to be trained, either "Ridge" or "Lasso".

    Returns:
        Tuple[dict, dict]: A tuple containing two dictionaries:
            - The first dictionary ('models') stores the trained regression models.
            - The second dictionary ('model_results') contains evaluation results for each model.

    Raises:
        ValueError: If 'model_name' is not "Ridge" or "Lasso".

    Notes:
        - This function trains regression models (Ridge or Lasso) on the provided dataset.
        - It standardizes the feature variables.
        - Hyperparameter tuning is performed using randomized search for each target variable.
        - Model performance is evaluated in terms of R-squared (R2) and mean squared error (MSE).

    Example:
        data = (features_dataframe, targets_dataframe)
        model_name = "Ridge"
        models, model_results = train_models(data, model_name)
    """

    # Standardize the feature variables
    features, targets = data
    features = StandardScaler().fit_transform(features)

    # Initialize dictionaries to store models and evaluation results
    model_results = {}
    models = {}

    # Loop over each target variable
    for target_name, target in targets.items():
        if model_name == "Ridge":
            # Define the parameter grid for Ridge
            param_grid = {
                'alpha': np.logspace(-2, 2, 100)
            }

            # Create a Ridge model
            ridge_model = Ridge(max_iter=10000)

            # Perform randomized search for hyperparameter tuning
            randomized_search = RandomizedSearchCV(ridge_model, param_distributions=param_grid, n_iter=10, cv=5,
                                                   scoring='r2', random_state=42)
            randomized_search.fit(features, target)

            # Get the best Ridge model
            best_ridge_model = randomized_search.best_estimator_

            # Store the trained model
            models["ridge_model_" + target_name] = best_ridge_model

            # Make predictions and evaluate the model
            y_pred = best_ridge_model.predict(features)
            r2 = r2_score(target, y_pred)
            mse = mean_squared_error(target, y_pred)

            # Store evaluation results
            model_results["ridge_model_" + target_name] = {("r2", "mse"): [r2, mse]}

        elif model_name == "Lasso":
            # Define the parameter grid for Lasso
            param_grid = {
                'alpha': np.logspace(-2, 2, 100)  # Creates a range of alpha values from 0.01 to 100
            }

            # Create a Lasso model
            lasso_model = Lasso(max_iter=10000)

            # Perform randomized search for hyperparameter tuning
            randomized_search = RandomizedSearchCV(lasso_model, param_distributions=param_grid, n_iter=10, cv=5,
                                                   scoring='r2', random_state=42)
            randomized_search.fit(features, target)

            # Get the best Lasso model
            best_lasso_model = randomized_search.best_estimator_

            # Store the trained model
            models["lasso_model_" + target_name] = best_lasso_model

            # Make predictions and evaluate the model
            y_pred = best_lasso_model.predict(features)
            r2 = r2_score(target, y_pred)
            mse = mean_squared_error(target, y_pred)

            # Store evaluation results
            model_results["lasso_model_" + target_name] = {("r2", "mse"): [r2, mse]}

        else:
            raise ValueError("Use 'Ridge' or 'Lasso' for model_name")

    return models, model_results


def features_selection(features, models_dict, results, output_directory, method):
    """
    Perform feature selection based on regression model coefficients
    and save selected features and coefficients to files.

    This function takes a set of regression models, their corresponding results, and a set of features.
    For each model, it identifies the non-zero coefficients, selects the corresponding feature names and coefficients,
    and stores them in a dictionary.
    Additionally, it saves this information to individual text files, one per model, in the specified output directory.

    Parameters:
        features (numpy.ndarray or pandas.Series): An array or Series containing the names of all features.
        models_dict (dict): A dictionary where keys are model names and values are trained regression models
                            (e.g., Ridge or Lasso models).
        results (dict): A dictionary containing evaluation results for each model.
                        Typically, this includes metrics like R-squared (R2) and mean squared error (MSE).
        output_directory (str): The directory where output text files will be saved.
        method (str): A label indicating the type of regression method used in the output files.

    Returns:
        dict: A dictionary containing the selected features and coefficients for each model.
              The keys are model names, and the values are dictionaries with 'selected_features' and 'coefficients'.

    Notes:
        - This function performs feature selection by identifying non-zero coefficients from regression models.
        - It creates a separate text file for each model in the 'output_directory' to save the selected features
          and their corresponding coefficients.
        - The output files are named after the model names with a '.txt' extension.
        - Each output file includes a header indicating the regression method used, followed by the selected features
          and their coefficients. It also includes R-squared (R2) and mean squared error (MSE) metrics from 'results'.

    Example:
        features = np.array(['Feature1', 'Feature2', 'Feature3'])
        models = {'ridge_model_1': <Ridge model object>, 'lasso_model_2': <Lasso model object>}
        evaluation_results = {'ridge_model_1': {('r2', 'mse'): [0.85, 120.0]},
                              'lasso_model_2': {('r2', 'mse'): [0.75, 150.0]}}
        output_dir = 'output/'
        method = 'Ridge'

        selected_features = features_selection(features, models, evaluation_results, output_dir, method)

    """
    selected_results = {}  # Initialize a dictionary to store selected features and coefficients
    check_directory(output_directory)  # Create the output directory if it doesn't exist

    # Loop over each model and its corresponding name in the 'models_dict'
    for name, model in models_dict.items():
        # Find the indices of non-zero coefficients in the model
        non_zero_indices = np.where(model.coef_ != 0)

        # Extract the names of selected features and their corresponding coefficients
        selected_feature_names = features[non_zero_indices]
        selected_coefficients = model.coef_[non_zero_indices]

        # Store the selected features and coefficients in the 'selected_results' dictionary
        selected_results[name] = {
            'selected_features': selected_feature_names,
            'coefficients': selected_coefficients
        }

        # Define the output file path
        output_file = os.path.join(
            output_directory, f"{name}.txt"
        )

        # Write the results to the output file
        with open(output_file, "w") as file:
            file.write(f"{method} regression results\n")

            # Write each selected feature and its coefficient to the file
            for feature, coeff in zip(selected_feature_names, selected_coefficients):
                file.write(f"\t{feature}:{coeff:.2f}\r")

            # Write R-squared (R2) and mean squared error (MSE) from 'results' to the file
            file.write(f'''\nR_squared = {results[f"{name}"][('r2', 'mse')][0]}''')
            file.write(f'''\nMSE = {results[f"{name}"][('r2', 'mse')][1]}''')

    return selected_results  # Return the dictionary of selected features and coefficients
