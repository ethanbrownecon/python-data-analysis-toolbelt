import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def printAmtNA(df):
    """prints number of missing values in each column

    Args:
        df: dataframe to check for missing values
    """
    for column in df:
        print(f"{column}: {df[column].isna().sum()} missing values")

def findCategoricalandNumericalCols(df):
    """finds the categorial and numerical columns in a dataframe

    Args:
        df (pandas dataframe): the dataframe to analyze
    """

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    return categorical_cols, numerical_cols


findCategorialandNumericalCols = findCategoricalandNumericalCols


def makeHist(df, col):
    """
    Make a histogram for a given column in a dataframe.
    
    :param df: dataframe to use for the histogram
    :param col: column name to create the histogram for
    """
    plt.hist(df[col],
         color='darkseagreen',
         edgecolor='white')

    plt.title(f'Distribution for {col}')
    plt.xlabel('Value')
    plt.ylabel('Amount')

    plt.show()

def get_normal_dist(data, column_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    plt.figure(figsize=(8, 5))
    x = np.linspace(data[column_name].min(), data[column_name].max(), 100)
    mu, sigma = norm.fit(data[column_name])
    plt.plot(x, norm.pdf(x, mu, sigma), "b-", label=f"Fitted Normal (μ={mu:.2f}, σ={sigma:.2f})")
    plt.xlabel(column_name)
    plt.ylabel("Probability Density")
    plt.title(f"Normal Distribution Fitted to {column_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def remove_outliers_zscore(df, column, threshold):
    """
    Remove outliers from a dataframe based on z-score.

    :param df: dataframe to remove outliers from
    :param column: column name to check for outliers
    :param threshold: z-score threshold to identify outliers
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    standard_deviation = df[column].std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return df.copy()

    z_scores = (df[column] - df[column].mean()).abs() / standard_deviation
    keep_rows = z_scores.isna() | z_scores.le(threshold)
    zOutlierDF = df.loc[keep_rows].copy()
    print("Original DataFrame Shape:", df.shape)
    print("DataFrame Shape after Removing Outliers:", zOutlierDF.shape)
    return zOutlierDF

def calculate_iqr(df, column):
    """
    Calculate the interquartile range (IQR) for a given column in a dataframe.

    :param df: dataframe to calculate IQR for
    :param column: column name to calculate IQR for
    :return: IQR value
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return IQR

def imputeMean(dataFrame, column):
    """Imputes any missing values using the mean in the given column.

    Args:
        dataFrame (DataFrame): the source DataFrame
        column (String): String of the column name to replace missing values
    """
    if dataFrame[column].notna().sum() == 0:
        raise ValueError(f"cannot impute {column!r}: the column has no non-missing values")
    dataFrame[column] = dataFrame[column].fillna(dataFrame[column].mean())


def labelEncodeWithOrder(aDF, aCatList, column):
    """Label encodes for any individual categorical column with specific order
    
    Args:
        aDF(any): The dataframe being encoded
        aCatList(any): The categorical list with the values being encoded
            - Must create this list with the order you want (ie. [Jan., Feb., Mar.])
        column(str): The name of the column being encoded.
        """
    encoding = {category: index for index, category in enumerate(aCatList)}
    aDF[column] = [encoding.get(value, value) for value in aDF[column]]
    
def oneHotEncode(df, column):
    """One-hot encodes a categorical column, creating a new 0/1 column for each unique value.
    Use this when categories have no natural order (like country or city names).

    Args:
        df (DataFrame): the source DataFrame
        column (String): the column name to one-hot encode

    Returns:
        DataFrame: the DataFrame with the column one-hot encoded
    """
    df = pd.get_dummies(df, columns=[column], dtype=int)
    return df
